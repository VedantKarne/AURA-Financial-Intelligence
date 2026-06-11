# Engineering Challenges & Solutions — All Phases

> **[← Architecture](./architecture.md)** | **[← README](../README.md)**

This document is a consolidated reference of every significant engineering challenge encountered across all 7 development phases of AURA, along with their root causes, solutions, and architectural learnings. Detailed per-phase logs are in the [`features_and_learnings/`](../features_and_learnings/) directory.

---

## Phase 1 — RAG Foundation

| Challenge | Root Cause | Resolution |
|---|---|---|
| Single-line transcript formatting | 40–60KB files stored without newlines | 2-step RCTS: split at `[ ` section marker first, then apply sentence-priority separators |
| Windows Unicode encoding crash | PowerShell defaults to legacy CP1252 — raw Unicode arrows (`→`) throw `UnicodeEncodeError` | Replaced all console symbols with ASCII equivalents |
| Streamlit API key cache miss | `load_dotenv` evaluated at import time; Streamlit caches module-level state | Moved `load_dotenv(override=True)` inside `get_llm()` function body |
| Groq model decommissioned | `llama-3.1-70b-versatile` removed from active endpoints | Centralised model config in `config.yaml`; migrated to `qwen/qwen3-32b` |
| `<think>` monologue tokens in output | Qwen3 reasoning model outputs internal chain-of-thought wrapped in `<think>` tags | Stateful stream parser: buffers tokens, detects tags, suppresses yield while inside reasoning block |
| Citation cards disappearing | Streamlit `st.rerun()` resets execution — citations rendered only during query block vanish | Decoupled query execution from rendering; citations stored in `session_state.chat_history` |

📄 [Full Phase 1 Document →](../features_and_learnings/Phase_1challenges_and_learnings.md)

---

## Phase 2 — Advanced Hybrid Retrieval

| Challenge | Root Cause | Resolution |
|---|---|---|
| Low-value boilerplate in chunks | Operator greetings, Safe Harbor disclaimers rank high on frequent financial terms | Sentence-level `clean_transcript_text()` filter removes 41 junk chunks at ingestion |
| Duplicate chunk leakage | IR forward-looking disclaimers duplicated at document start and mid-document | Boilerplate filter catches all instances globally before chunking |
| Reranker entity starvation | High-signal entity (Nvidia AI chunks) dominated shared candidate pool; reranker scored out all others | Per-entity guaranteed retrieval (`top_k=6` per company) + pool merge before reranking |
| LaTeX rendering crash in Streamlit | Markdown parser interprets `$69.7B` as LaTeX math delimiters | System prompt enforces `USD X billion` format; regex replacement in stream processor |
| Temporal scope narrowing | Query rewriter compressed *"performance so far"* to specific quarter | Modified condensation prompt to preserve broad historical ranges and multi-period context |
| SQLite connection lock (Windows) | Multiple Streamlit processes sharing same ChromaDB SQLite connection | Terminate conflicting processes; added explicit exception logging to `vector_store.py` |

📄 [Full Phase 2 Document →](../features_and_learnings/phase2_challenges_and_solutions.md)

---

## Phase 3 & 4 — Structured KPI Extraction + Agent Orchestration

| Challenge | Root Cause | Resolution |
|---|---|---|
| Unstructured metric parsing | Financial summaries contain prose paragraphs, not clean tables | Groq structured output with Pydantic `EarningsKPI` model validation |
| LangGraph tool-call loop | Agent calls tools in infinite loop if tool outputs don't satisfy LLM | Added `route_tools` conditional edge: `tool_calls` → tools, else → END |
| Multi-turn context hallucination | `MemorySaver` feeds entire history including old `ToolMessage` objects to LLM | `filter_messages_for_llm()`: strips historical tool artifacts, keeps only final AI responses from past turns |
| FastAPI async tool invocation | Langchain tools are synchronous; wrapping in async FastAPI endpoint caused thread blocking | Used `asyncio.run_in_executor()` or structured synchronous invocation with proper async context |

📄 [Full Phase 3-4 Document →](../features_and_learnings/phase3_4__challenges_and_resolutions.md)

---

## Phase 5 — Premium Frontend (Next.js Cockpit)

| Challenge | Root Cause | Resolution |
|---|---|---|
| Font loader subset failure | Next.js font loader doesn't support `"display"` subset for `Plus_Jakarta_Sans` | Restricted subset to `["latin"]` in `layout.tsx` |
| RAG candidate pool capped at 20 | `candidate_pool_limit = 20` hardcoded in `qa_chain.py` | Dynamic scaling: `max(20, k + 10)` |
| Stale cache hallucinations in multi-turn | LLM recycled old `ToolMessage` outputs instead of calling tools freshly | `filter_messages_for_llm()` introduced (see Phase 4) |
| Hydration mismatch errors | `crypto.randomUUID()` and `localStorage` are client-only; SSR renders different markup | Wrapped in `useEffect` with `mounted` guard; `suppressHydrationWarning` on `<body>` |

📄 [Full Phase 5 Document →](../features_and_learnings/phase5_challenges_and_solutions.md)

---

## Phase 6 — Dockerization

| Challenge | Root Cause | Resolution |
|---|---|---|
| ChromaDB compilation failure in Docker | `python:3.11-slim` has no C++ compiler; `hnswlib` requires native compilation | Added `build-essential gcc g++ make python3-dev` to `backend.Dockerfile` |
| 15-minute slow builds | PyTorch (1 GB) + Hugging Face (300 MB) downloaded on every build; large context sent to daemon | `.dockerignore` excludes `.venv/`, `node_modules/`, `data/`; Docker layer caching handles library reuse |
| WSL2 VHDX disk never shrinks | WSL2 auto-expands `.vhdx` but never compacts after pruning | Manual `diskpart compact vdisk` after `wsl --shutdown` |
| Frontend can't reach backend inside Docker | `localhost:8000` inside a container refers to that container's own loopback | Docker Compose internal DNS: frontend uses `http://backend:8000` (service name) |

📄 [Full Phase 6 Document →](../features_and_learnings/phase6_challenges_and_solutions.md)

---

## Phase 7 — RAG Quality & Multi-Entity Intelligence

| Challenge | Root Cause | Resolution |
|---|---|---|
| Phase 2 entity fix regresses at high `k` | Buffer formula `exact + 2` gave too small a margin at k=19+ | Changed to `exact * 3` — 3× buffer guarantees reranker survivor rate per entity |
| Second-pass fill destroys balance | Greedy overflow fill added only highest-scoring (Apple) chunks | Round-robin fill cycles across all entities for overflow budget allocation |
| LLM skips entities despite present context | Primacy bias: highest-scoring chunks (Apple) cluster at position 1–6; LLM under-reads late context | Entity-grouped context ordering: all Apple → all MSFT → all Nvidia in final context block |
| Comparison tables broken by citation pipes | `[Apple \| Q3 \| 2023 \| summary]` pipe chars split table cells into phantom columns | `TABLE CITATION SAFETY` rule: numeric `[1]`, `[2]` refs inside cells; Citation Key section below table |
| Agent synthesis ignores table rules | Two-LLM architecture gap: inner RAG LLM follows `prompts.py` rules, but agent LLM follows only `server.py` instruction | Added comparison table + coverage + citation safety rules to agent system instruction in `server.py` |
| `k=6` gives only 2 chunks/company | `GLOBAL_K` is a total budget; dividing by 3 companies leaves 2 chunks each — too sparse for risk analysis | Auto-scaling: `effective_k = min(24, max(GLOBAL_K, n_entities * 6))` in `tools.py` |
| Groq Daily Quota (TPD) exhaustion | Qwen3-32B 500k token daily limit exhausted by intensive testing, causing multi-minute Retry-After hangs | Capped auto-retries at 30s in orchestrator; returning formatted wait times to prevent UI hangs |
| Agent "Loophole" on missing DB data | SQL DB returned passive "Not found", causing agent to give up instead of searching embeddings | Returned hard "CRITICAL INSTRUCTION: MUST use rag_search" directive from failed DB queries |
| Forward-looking guidance starved at k=6 | High keyword density of summaries pushed sparse transcript chunks out of context | Auto-boosted k=16 for "guidance" queries + exposed k_override to LLM for recursive retry |

📄 [Full Phase 7 Document →](../features_and_learnings/phase7_challenges_and_solutions.md)

---

## Recurring Architectural Learnings

### L1 — Never Assume Text File Format
Transcripts came as single-line 60KB blobs. Always audit raw corpus structure before writing parsers.

### L2 — Centralise All Configs
Three models were decommissioned during development. Every model name, path, and threshold lives in `config/config.yaml`.

### L3 — Two-LLM Pipelines Have a Synthesis Gap
In agentic RAG where an outer agent LLM synthesises inner RAG LLM tool outputs, every formatting rule must be enforced at **both** layers independently.

### L4 — Retrieval K Semantics Break for Multi-Entity Queries
`k=6` for 3 companies = 2 chunks per company. The correct design auto-scales: `k_effective = n_entities * k_per_entity`.

### L5 — Reranker Starvation is a Retrieval Problem, Not a Prompt Problem
Fixing entity gaps at the prompt level treats symptoms. The root cause is always upstream — guarantee balanced representation **before** the reranker sees the candidate pool.

### L6 — LLM Primacy Bias Demands Entity-Grouped Context
LLMs consistently over-utilise the first passages in their context window. Group context by entity (all Apple → all MSFT → all Nvidia) so no company is systematically under-read.

### L7 — Markdown Syntax Conflicts Must Be Anticipated
`[Apple | Q3 | 2023 | summary]` contains `|` which breaks markdown tables. Prompt design must provide a format-safe alternative for constrained rendering environments.

### L8 — Active RAG is Required for Deep Narrative Extraction
Static retrieval depths (`k`) fail for sparse narrative data. Exposing retrieval parameters directly to the LLM allows it to actively self-correct and widen its net when it encounters context starvation.

### L9 — Close Agent Loopholes with Active Directives
Passive failure messages ("No data found") cause LLM agents to halt execution. Tools must return active directives ("CRITICAL INSTRUCTION: Try X instead") to force recursive tool chaining across the system architecture.
