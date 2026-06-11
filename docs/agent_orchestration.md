# Agent Orchestration — LangGraph State Machine

> **[← Architecture](./architecture.md)** | **[← README](../README.md)**

---

## Overview

The agent layer is the decision-making brain of AURA. Built on **LangGraph**, it operates as a stateful, cyclic tool-calling agent: it decides which tools to invoke, maintains session memory across turns, and synthesises a final coherent response from all tool outputs.

---

## Module Map

```
src/agents/
├── orchestrator.py   — LangGraph graph definition, compilation, memory checkpointer
└── tools.py          — Tool definitions: rag_search, get_kpis, generate_report_sections

src/api/
└── server.py         — FastAPI endpoints; injects system instructions into agent queries
```

---

## LangGraph Graph Structure

```python
# State: a simple message list that grows with each turn
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Nodes
graph_builder.add_node("chatbot", chatbot_node)   # LLM decision node
graph_builder.add_node("tools",   ToolNode)        # Tool execution node

# Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools)  # tool_calls? → tools | END
graph_builder.add_edge("tools", "chatbot")                   # tool output → back to LLM

# Compilation with in-memory persistence
memory = MemorySaver()
app_graph = graph_builder.compile(checkpointer=memory)
```

This creates a **ReAct-style loop**: the LLM can invoke tools repeatedly until it decides no more tools are needed, at which point it returns a final response.

---

## Tool Definitions (`tools.py`)

### `rag_search`

```python
@tool
def rag_search(
    query: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> str:
    """Search earnings call transcripts using hybrid RAG retrieval."""
```

**Key behaviours:**
- Automatically scales `k` for multi-entity queries (`n_entities × 6`, capped at 24)
- Calls `get_answer()` from `qa_chain.py` with `retrieval_mode="auto"`
- Caches retrieved source documents in `ACTIVE_RETRIEVED_DOCS` for the frontend citation panel
- Returns formatted answer text with inline citations

**LLM Tool Description includes:**
> *"If the user asks about 'all companies', comparisons, or multiple entities, DO NOT specify the `company` argument. The underlying retrieval system automatically detects and handles multi-entity queries natively."*

This prevents the agent from incorrectly filtering to a single company when a comparative query is submitted.

---

### `get_kpis`

```python
@tool
def get_kpis(
    company: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None
) -> str:
    """Retrieve structured financial KPIs from the database."""
```

Executes a SQLAlchemy query against `data/finance_kpis.db`. Returns JSON-serialised rows with: `period`, `revenue_b`, `eps`, `gross_margin_pct`, `net_income_b`, `revenue_growth_yoy_pct`, `guidance_revenue_low_b`, `guidance_revenue_high_b`.

Used for: exact reported numbers, quarter-over-quarter comparisons, guidance ranges.

---

### `generate_report_sections`

```python
@tool
def generate_report_sections(company: str, year: int, quarter: str) -> str:
    """Retrieves all necessary information to generate an investment brief."""
```

Aggregates both `get_kpis` and `rag_search` results for a specific company/period into a single structured context block, which the LLM then synthesises into a full investment research brief.

---

## Session Memory & History Filtering

### Thread-Based Session Isolation

Each browser session generates a unique `thread_id` UUID. Every `run_agent_query()` call passes this as a LangGraph config:

```python
config = {"configurable": {"thread_id": thread_id}}
app_graph.stream(inputs, config=config, stream_mode="values")
```

The `MemorySaver` checkpointer automatically persists the full message thread per thread ID, enabling multi-turn conversation continuity.

### History Filtering — Anti-Hallucination

Without filtering, the LangGraph checkpointer feeds the **entire historical message thread** to the LLM on each turn. This includes old `ToolMessage` objects containing previous query results. The LLM sees these and incorrectly assumes it doesn't need to call tools again — recycling stale data.

**Solution: `filter_messages_for_llm()`**

```python
def filter_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip tool call artifacts from past turns; keep only current turn + clean history."""
    last_human_idx = find_last_human_message_index(messages)
    
    filtered = []
    for i, msg in enumerate(messages):
        if i >= last_human_idx:
            filtered.append(msg)          # Keep ALL messages from current turn
        elif isinstance(msg, HumanMessage):
            filtered.append(msg)          # Keep human queries from past turns
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            filtered.append(msg)          # Keep only final AI answers from past turns
        # ToolMessages from past turns are EXCLUDED
    return filtered
```

This forces the LLM to approach every new query with fresh tool calls while still maintaining conversational context (pronoun resolution, follow-up questions).

---

## Agent System Instruction Architecture

The `/api/chat` endpoint in `server.py` wraps every user query in a comprehensive system instruction before passing it to the agent:

```python
full_query = (
    "SYSTEM INSTRUCTION: You are a highly detailed financial assistant."
    
    # Response length must scale with context size received
    "CRITICAL RULE ON LENGTH: ..."
    
    # All inline citations from tool outputs must be preserved
    "CRITICAL RULE ON CITATIONS: ..."
    
    # Multi-company queries require equal per-company sections
    "CRITICAL RULE ON MULTI-ENTITY COVERAGE: ..."
    
    # Tables are mandatory for comparison queries
    "CRITICAL RULE ON COMPARISON TABLES: ..."
    
    # Pipe chars in citations cannot go inside table cells
    "TABLE CITATION SAFETY: ..."
    
    # Source document lists from tools must be preserved verbatim
    "IMPORTANT: If any tool provides a 'Source Documents' list..."
    
    f"USER QUERY: {req.query}"
)
```

### Why Two Layers of Prompt Instructions?

AURA has **two independently-instructed LLMs** in the pipeline:

| LLM | Location | Instruction Source |
|---|---|---|
| Inner RAG LLM | `qa_chain.py` → `get_answer()` | `SYSTEM_PROMPT` + `RAG_QA_PROMPT` in `prompts.py` |
| Agent Synthesis LLM | `orchestrator.py` → `chatbot_node` | System instruction prepended in `server.py` |

The inner RAG LLM generates structured, cited context passages from the retrieved documents. The agent LLM receives these as `ToolMessage` outputs and synthesises the final user-facing response. If the agent LLM's instruction doesn't include table and coverage rules, it freely reformats the inner LLM's output — dropping tables, skipping companies, etc.

Both layers must enforce the same rules.

---

## Error Handling

```python
def run_agent_query(query: str, thread_id: str = "default") -> str:
    try:
        ...
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "Rate limit reached" in error_msg:
            # Extract retry-after time from Groq error message
            return f"🛑 Execution Paused: API Quota Reached. Try again in {time_msg}."
        return f"⚠️ System Error: {error_msg}"
```

Groq API rate limit errors are handled gracefully with user-friendly messages including the suggested retry time extracted from the error payload.

---

## Prompt Engineering — `prompts.py`

### Citation Format
All factual claims must be cited as:
```
[Company | Quarter | Year | Section]
e.g.: [Apple | Q3 | 2024 | transcript]
```

### Table Citation Safety Rule
The citation format contains `|` pipe characters which are also markdown table delimiters. Placing full citations inside table cells destroys the table structure. The prompt enforces:

```
Inside table cells: use [1], [2], [3] numeric refs
Below the table: include a "Citation Key" section mapping [1] → [Company | Q | Y | Section]
```

### Multi-Entity Coverage Rule
```
When context passages are provided for multiple companies, you MUST address EVERY company 
with equal depth and a dedicated section. Do NOT skip or under-represent any company 
whose context passages appear below.
```
