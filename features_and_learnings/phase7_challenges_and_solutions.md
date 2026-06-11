# Phase 7: RAG Quality & Multi-Entity Intelligence — Challenges & Solutions

This document captures the engineering challenges, root cause analyses, and resolutions implemented during Phase 7 of the **Financial Earnings Intelligence Platform**. Phase 7 focuses on post-dockerization quality improvements: LLM output formatting, multi-entity retrieval fairness, citation rendering, and agent synthesis correctness.

---

## 🚀 Key Features Added

1. **Balanced Multi-Entity Retrieval with Round-Robin Allocation:**
   * Redesigned the multi-entity retrieval path in `qa_chain.py` to guarantee equal chunk representation per company *after* reranking using a two-pass entity-aware allocator and a round-robin fill-up mechanism.

2. **Entity-Grouped Context Ordering:**
   * Final retrieved documents are now sorted and grouped by company (all Apple chunks → all Microsoft chunks → all Nvidia chunks) before being sent to the LLM, reducing the chance the model skips a company that appears sparsely distributed across the context.

3. **Auto-Scaling `k` for Multi-Entity Queries in the Agent:**
   * The `rag_search` tool in `tools.py` now detects how many companies a query targets and automatically multiplies the base `top-k` value accordingly, ensuring each entity gets a minimum of 6 context chunks.

4. **Comparison Chart / Table Generation:**
   * The RAG prompt and the agent system instruction were both upgraded to require a structured markdown comparison table for any multi-entity or comparison query.

5. **Table-Safe Citation Format (Numeric Refs + Citation Key):**
   * Introduced the `TABLE CITATION SAFETY` prompt rule: inline citations containing pipe characters (`|`) are forbidden inside markdown table cells. The LLM must use numeric shorthand `[1]`, `[2]`, `[3]` inside cells and emit a separate **Citation Key** section below each table.

6. **Dual-Layer Prompt Enforcement (Inner RAG + Agent Synthesiser):**
   * All formatting and coverage rules are now enforced at *both* prompt layers: the inner RAG chain (`prompts.py`) and the final agent synthesis instruction (`server.py`), eliminating the gap that allowed the agent to silently ignore table and coverage rules.

---

## 🛠️ Challenges Faced & Resolutions

---

### 1. Reranker Starvation — Phase 2 Fix Regressing Under Higher `k`

**The Challenge:**
After Phase 2 introduced the per-entity guaranteed retrieval pool (each entity retrieved `k_per_entity = max(6, exact_per_entity + 2)` chunks before reranking), the fix worked at low `k` values. However, at higher user-selected `k` values (e.g., `k=19`), `exact_per_entity` became `6` and `k_per_entity` was only `8` — still far too small a candidate pool for the cross-encoder reranker to produce balanced output. Apple dominated because its chunks scored higher on general risk vocabulary, leaving Nvidia with zero survivors in the top-k cut.

**Root Cause:**
The retrieval buffer formula `max(6, exact_per_entity + 2)` provided only a **+2 margin** above the quota. This is insufficient when the reranker is free to eliminate all of an entity's candidates in favour of another entity's higher-scoring chunks.

**The Resolution:**
Changed the buffer formula across both `get_answer` and `get_answer_streaming` in [`qa_chain.py`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/generation/qa_chain.py):

```python
# Before (Phase 2 — fragile at high k)
k_per_entity = max(6, exact_per_entity + 2)

# After (Phase 7 — 3× buffer ensures reranker survival)
k_per_entity = max(8, exact_per_entity * 3)
```

By fetching **3× the quota** per entity, the candidate pool going into the reranker is rich enough that each entity retains sufficient survivors even after aggressive score-based elimination.

---

### 2. Uncontrolled Second-Pass Fill Destroying Allocation Balance

**The Challenge:**
The Phase 2 two-pass allocator had a subtle bug in its second pass. After the first pass allocated `exact_per_entity` chunks per company, any remaining budget slots were filled *greedily* from the full reranked list — with no entity awareness. This meant the second pass always added more Apple chunks (highest rerank score) to fill leftover budget, regardless of how many Apple chunks were already present.

**The Root Cause:**
```python
# Bug: Second pass has no company awareness
if len(source_docs) < k:
    for doc in reranked_all:
        if doc not in source_docs:
            source_docs.append(doc)   # Could be 5 more Apple docs
```

**The Resolution:**
Replaced the uncontrolled greedy fill with an **entity-aware round-robin** mechanism:

```python
# Phase 7: Round-robin across all detected entities
entity_overflow_pools = {
    ent: [d for d in entity_ranked[ent] if id(d) not in used_set]
    for ent in detected_cos
}
rr_idx = 0
entities_list = list(detected_cos)
while added < remaining_budget:
    ent = entities_list[rr_idx % len(entities_list)]
    pool = entity_overflow_pools[ent]
    if pool:
        doc = pool.pop(0)
        source_docs.append(doc)
        added += 1
    rr_idx += 1
    if all(len(p) == 0 for p in entity_overflow_pools.values()):
        break
```

This distributes any remaining budget slots cyclically across all entities, preventing any single company from monopolising the overflow allocation.

---

### 3. LLM Skipping Entities Despite Context Being Present

**The Challenge:**
Even after fixing the retrieval layer to deliver a balanced context pool (e.g., 6 Apple + 6 Microsoft + 6 Nvidia chunks), the LLM would still produce a response that covered Apple with 4 bullet points, Microsoft with 2, and Nvidia with a single line saying *"No specific risks identified in the retrieved context."* — despite Nvidia chunks being clearly present in the formatted context block.

**Root Cause Analysis:**
Two compounding causes:

1. **Context ordering was interleaved.** When the final `source_docs` list was sorted by rerank score, Apple chunks (highest scorers) appeared at positions 1-6, Microsoft at 7-10, and Nvidia at 11-18. LLMs exhibit **primacy bias** — they read and utilise the earliest context passages most thoroughly. Nvidia chunks appearing last in a long context were systematically under-used.

2. **The prompt lacked an explicit equal-coverage mandate.** The system prompt said "cite every claim with its source" but never explicitly said *"address EVERY company with equal depth"*. The LLM correctly cited sources it used, but had discretion over *which* sources to draw from.

**The Resolution — Two-Part Fix:**

*Part 1 — Entity-Grouped Context Ordering (`qa_chain.py`):*

After allocation, docs are re-sorted by entity group so Apple chunks appear together, then Microsoft, then Nvidia — giving each company a coherent contiguous block in the prompt:

```python
# Group final docs by entity for LLM coherence
grouped_docs = []
for ent in detected_cos:
    for doc in source_docs:
        if doc.metadata.get("company") == ent:
            grouped_docs.append(doc)
source_docs = grouped_docs
```

*Part 2 — Explicit Multi-Entity Coverage Mandate (`prompts.py`):*

Added the `MULTI-ENTITY COVERAGE` rule to the system prompt:

> *"When context passages are provided for multiple companies, you MUST address EVERY company with equal depth and a dedicated section. Do NOT skip or under-represent any company whose context passages appear below."*

---

### 4. Comparison Tables Broken by Citation Pipe Characters

**The Challenge:**
When the LLM generated markdown comparison tables and placed inline citations like `[Apple | Q3 | 2023 | summary]` inside a table cell, the markdown renderer interpreted each `|` in the citation as a **column separator**. This turned a single table cell into 4+ phantom columns, completely mangling the table structure.

Example of broken table output:
```
| Risk Category | Apple          | Microsoft       |
|---|---|---|
| FX Headwinds  | 4% impact [Apple | Q3 | 2023 | summary] | 150bps impact |
```
The renderer sees 6 columns instead of 3 — the citation splits into extra cells.

**The Resolution:**
Added the `TABLE CITATION SAFETY` rule to both the inner RAG prompt (`prompts.py`) and the agent synthesis instruction (`server.py`):

> *"NEVER place inline citations like [Apple | Q3 | 2023 | summary] inside a markdown table cell — the pipe characters will break the table. Instead, inside table cells use numeric shorthand refs like [1], [2], [3], and then place a Citation Key section directly below the table."*

**Example of correct post-fix output:**

```markdown
| Metric | Nvidia | Microsoft |
|---|---|---|
| CapEx Growth | Not disclosed [6] | 23–24% YoY in Q4 2023 [9] |

**Citation Key:**
[6] = [Nvidia | Q2 | 2023 | transcript]
[9] = [Microsoft | Q4 | 2023 | transcript]
```

---

### 5. Agent Synthesis Layer Ignoring Table & Coverage Rules

**The Challenge:**
After fixing the inner RAG prompt (`prompts.py`) to require comparison tables and equal entity coverage, the issue persisted in the final user-facing response. The agent would correctly retrieve balanced data via `rag_search`, but then reformatted the output into numbered prose sections — silently dropping the table.

**Root Cause — The Two-LLM Architecture Gap:**
The pipeline uses two LLMs in sequence:

```
User Query
    ↓
Agent LLM (orchestrator, uses server.py instruction)  ← makes tool call decision
    ↓
rag_search tool → get_answer() → Inner RAG LLM (uses prompts.py)  ← generates table
    ↓  returns formatted text with table
Agent LLM → Rewrites final response for user  ← ⚠️ had NO table or coverage rules
```

The inner RAG LLM correctly generated a comparison table. But the **agent LLM** received that tool output and then synthesised a new final response using only the `server.py` system instruction — which contained no mention of tables, equal coverage, or citation safety. The agent felt free to reformat everything in a style it preferred.

**The Resolution:**
Added three new rules to the agent synthesis instruction in [`server.py`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/api/server.py):

```python
full_query = (
    ...
    "CRITICAL RULE ON MULTI-ENTITY COVERAGE: If the query involves multiple "
    "companies or 'all companies', your final response MUST include a dedicated "
    "section for EACH company with equal depth. Do NOT skip or under-represent any company. "
    "CRITICAL RULE ON COMPARISON TABLES: If the query asks for a comparison, "
    "side-by-side analysis, or involves multiple companies, you MUST include "
    "a clean markdown comparison table summarising key metrics. "
    "TABLE CITATION SAFETY: NEVER place citations like [Apple | Q3 | 2023 | summary] "
    "inside table cells — the pipe characters break table formatting. Inside table "
    "cells use [1], [2], [3] numeric refs only, then list the full citation key below the table. "
    ...
)
```

**The Learning:**
In agentic pipelines where a tool-calling LLM synthesises output from tool responses, every formatting and quality rule must be applied to **both** the tool's internal LLM *and* the synthesising agent's instruction. Rules at only one layer are silently overridden by the other.

---

### 6. `GLOBAL_K = 6` Starving Multi-Entity Queries at the Agent Level

**The Challenge:**
The `rag_search` tool in `tools.py` passed `GLOBAL_K` (default `6`) directly to `get_answer()` as the `k` parameter. For a 3-company comparison query, the balanced allocator would divide `k=6` by 3, giving each company only **2 chunks**. Two chunks is far too little to find risk-specific or CapEx-specific content for each company — leading to Nvidia returning "no data found" even though its data existed in the index.

**Root Cause:**
The `k` parameter was designed as a user-facing "response richness" dial, but its semantics change completely for multi-entity queries. A `k=6` single-entity query gets 6 chunks for one company. A `k=6` three-company query gets 2 chunks per company — a 3× reduction in per-company context that was never visible or adjustable to the user.

**The Resolution:**
Added `_count_entities_in_query()` to `tools.py` that detects multi-entity queries and auto-scales `k` before passing it to `get_answer()`:

```python
def _count_entities_in_query(query: str) -> int:
    """Estimate how many companies are referenced in a query."""
    q = query.lower()
    if any(p in q for p in _ALL_COMPANY_PHRASES):
        return 3  # "all companies", "compare", "vs" etc.
    count = 0
    if any(x in q for x in ["apple", "aapl"]): count += 1
    if any(x in q for x in ["microsoft", "msft", "azure"]): count += 1
    if any(x in q for x in ["nvidia", "nvda", "jensen"]): count += 1
    return max(1, count)

# In rag_search:
n_entities = _count_entities_in_query(query)
effective_k = GLOBAL_K
if n_entities > 1:
    # Minimum 6 chunks per entity, capped at 24 total
    effective_k = min(24, max(GLOBAL_K, n_entities * 6))
```

For a 3-company query with `GLOBAL_K=6`, this produces `effective_k=18` — 6 chunks per company, regardless of what the slider is set to.

---

## 📐 Architectural Learnings

### L1 — The Two-LLM Synthesis Gap is a Systemic Risk in Agentic RAG
Any system that uses a tool-calling agent to synthesise output from an inner RAG LLM has two independently instructed models. Every quality rule — formatting, citation, coverage — must be explicitly duplicated in both the inner model's prompt template and the outer agent's system instruction. Assumptions that the outer model will "preserve" the inner model's structured output are consistently violated.

### L2 — Retrieval `k` Semantics Break in Multi-Entity Contexts
A single `k` parameter cannot serve dual purposes (total chunks for single-entity; per-entity chunks for multi-entity) without explicit scaling logic. The correct design is to expose total `k` to the user but internally reinterpret it as `k_per_entity × n_entities` when multi-entity intent is detected.

### L3 — LLM Primacy Bias Demands Entity-Grouped Context
LLMs consistently over-utilise the earliest passages in their context window. When retrieved chunks from multiple companies are interleaved by rerank score, the highest-scoring company's chunks cluster at the top and receive disproportionate attention. Grouping chunks by entity in the final context block is a simple and effective mitigation that requires no change to the retrieval or reranking logic.

### L4 — Markdown Syntax Conflicts Must Be Anticipated in Prompt Design
The citation format `[Company | Quarter | Year | Section]` is a good human-readable label but contains the markdown table delimiter `|`. Embedding it directly in table cells is a structural impossibility. Prompt design must anticipate the rendering environment and provide format-safe alternatives (numeric refs inside tables, full keys outside).

### L5 — Buffer Size is the Most Impactful Reranker Tuning Lever
The cross-encoder reranker is deterministic — it scores what it receives. The most impactful variable for multi-entity retrieval quality is the **size of the candidate pool fed to the reranker per entity**, not the reranker's score thresholds or temperature. A 3× buffer (`k_per_entity = exact_per_entity * 3`) consistently outperformed a fixed +2 margin across all query types tested.

---

### 7. Groq Daily Quota (TPD) Exhaustion Masquerading as Latency

**The Challenge:**
The application began experiencing massive 3-6 minute hangs when processing multi-entity queries, culminating in "API Quota Reached" errors. Initial debugging assumed it was a per-minute rate limit (TPM) and attempted to auto-sleep and retry. However, the wait times returned by Groq were progressively increasing (from 15s to 400s+).

**Root Cause:**
The `qwen/qwen3-32b` model on Groq's free tier has a hard **Tokens Per Day (TPD)** limit of 500,000 tokens. The intensive testing of multi-entity queries (which execute 2 orchestrator calls + 1 heavy inner RAG call per query) completely exhausted the daily budget. The "wait time" was actually Groq calculating the seconds remaining until midnight UTC when the daily quota resets.

**The Resolution:**
1. Re-architected the `_execute` exception handler in `orchestrator.py` to cap auto-retries at 30 seconds. If `Retry-After` exceeds 30s, it returns a formatted error instantly, preventing silent UI hangs.
2. Verified that API key rotation instantly restores the 500,000 TPD allowance for continued testing without altering the architecture.

---

### 8. The "Agent Loophole" — Passive Tool Failures Halting Execution

**The Challenge:**
When queried for NVIDIA's profitability, the orchestrator successfully called the `get_kpis` tool. Because NVIDIA's structured KPIs were missing from the SQL database, the tool returned a passive string: `"No KPIs found for NVIDIA"`. The orchestrator LLM read this, assumed the data didn't exist anywhere, and apologized to the user—despite knowing it *could* theoretically run `rag_search` to find it in the transcripts.

**Root Cause:**
LLMs often lack the autonomous agency to "try another tool" unless explicitly instructed. A passive failure message confirms to the LLM that the data is absent.

**The Resolution:**
Rewrote the fallback logic inside `tools.py`'s `get_kpis` to return an active, hard directive rather than a passive failure:
`"No KPIs found... CRITICAL INSTRUCTION: Do not give up. You MUST immediately call the rag_search tool to find these financial metrics inside the transcript vector embeddings."`
This closed the agent loophole and forced recursive tool chaining.

---

### 9. Precision vs Recall Trade-off in Forward-Looking Guidance

**The Challenge:**
When the user queried *"Summarize Q3 guidance for Apple"* with `k=6`, the system failed to find the data. When `k=16`, it succeeded and generated a highly detailed response. 

**Root Cause:**
Earnings call summaries have extremely high keyword density for terms like "Q3" and "Apple," dominating the top vector similarity ranks. However, actual forward-looking guidance (CFO projections) is sparsely distributed deep within the raw transcript chunks. At `k=6`, the high-level summaries pushed the transcript chunks entirely out of the context window (Context Starvation).

**The Resolution (Active RAG):**
1. **Auto-Boosting `k`:** In `qa_chain.py`, added a precision filter that detects forward-looking words ("guidance", "outlook", "project") and automatically boosts `k=16` to ensure transcript chunks survive the cut.
2. **Recursive AI Retrieval:** Exposed an optional `k_override` parameter in the `rag_search` tool signature. Added a `RETRY LOGIC` prompt instructing the LLM to actively re-call the tool with `k_override=24` or `32` if its initial retrieval attempt fails to find the required depth.

---

### L6 — Active RAG is Required for Deep Narrative Extraction
Static retrieval depths (`k`) are fundamentally flawed because different types of knowledge require different context window sizes. High-level summaries can be extracted at `k=4`, while deep CFO guidance metrics might require `k=16+`. Exposing retrieval parameters directly to the LLM (Active RAG) allows the agent to self-correct and iteratively widen its net when it encounters context starvation.
