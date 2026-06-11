# Session Context Transfer — Finance RAG Project
> Read this BEFORE continuing. This file captures the *why*, the state of the project, recent actions, and the scalability roadmap forward.

---

## 🚀 Current Project State

* **Status**: **Phase 4 Complete**. The platform is a fully functional **Multi-Agent RAG System** featuring:
  * Autonomous **LangGraph** orchestrator state machine.
  * Stateful **Memory Checkpointers** (`MemorySaver`) bound to conversational threads using client-side generated UUID `threadId` tokens.
  * **Dual-Storage Retrieval:** Dense vectors in ChromaDB (MiniLM-L6-v2) + Sparse lexical index in BM25 fused via Reciprocal Rank Fusion (RRF).
  * **Cross-Encoder Reranking:** Candidate rescoring via `ms-marco-MiniLM-L-6-v2` CE.
  * **Structured extraction:** Relational SQLite metrics database (`data/kpi_store.db` mapped to SQLAlchemy models) populated using Pydantic schemas from unstructured text summaries.
  * **Interactive Interfaces:** Reactive Next.js dashboard + FastAPI server stream gateway.
* **Running Processes:**
  * **FastAPI Server:** Running on [http://localhost:8000](http://localhost:8000) (Active background task started via `python -m src.api.server` in `c:\Users\ADMIN\Documents\3rd_Year_Projects\Finance_RAG_Project`).
  * **Next.js Client:** Running on [http://localhost:3000](http://localhost:3000) (Active background task started via `npm run dev` in `frontend/`).

---

## 🛠️ Accomplishments of Current Session

1. **RAG Architecture Flowchart Corrections:**
   * Spotted and resolved a logical redundancy in the detailed RAG retrieval mindmap (which checked target company conditions twice and incorrectly merged single/multi-company pipelines before RRF execution).
   * Updated [walkthrough.md](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/walkthrough.md#L145-L200) to separate single-company (Standard Hybrid -> RRF -> Rerank -> Slice) and multi-company (Parallel Hybrid -> per-entity RRF -> Merge -> Rerank -> Quota Allocation) tracks cleanly.
2. **Master README Upgrades:**
   * Overwrote the outdated Phase 1 `README.md` with a complete, production-grade [README.md](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/README.md) showing the system ecosystem, processed pipelines, file maps, setup steps, and key engineering solutions.
3. **Scaling & Infrastructure Tradeoff Analysis:**
   * Conducted a detailed architectural study of scaling database limits from 3 companies to 50+ major corporations (~120k vectors over a 3-year horizon).
   * Detailed findings comparing Vector DBs (Chroma vs. Qdrant), Relational DBs (SQLite vs. PostgreSQL), and the unified PostgreSQL + `pgvector` approach are saved inside the brain directory at [scalability_analysis.md](file:///C:/Users/ADMIN/.gemini/antigravity-ide/brain/206ee0f0-743d-48ee-8feb-60baee266954/scalability_analysis.md).

---

## 📂 Codebase State & Directories

All core components are structured as follows:

| Component | Directory / Path | Purpose |
| :--- | :--- | :--- |
| **Ingestion** | [`src/ingestion/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/ingestion/) | File parsing, sentence-boundary chunking, embedding, and indexing. |
| **Retrieval** | [`src/retrieval/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/retrieval/) | Chroma, BM25, RRF, Cross-Encoder reranker, router, and query transformer. |
| **Generation** | [`src/generation/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/generation/) | Prompts repository and multi-company balanced quota search logic. |
| **Extraction** | [`src/extraction/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/extraction/) | SQLite schemas (SQLAlchemy) and LLM-assisted KPI extractor. |
| **Agents & API** | [`src/agents/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/agents/) & [`src/api/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/src/api/) | LangGraph graph definition, agent tools, and FastAPI streaming server. |
| **Frontend UI** | [`frontend/`](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/frontend/) | React chat, comparison grids, and PDF report generator. |

---

## 📈 Next Steps: Project Scaling Roadmap

For the next session, you can choose between these three production-grade directions:

1. **Option A: Production Systems & Infrastructure** *(Highly Recommended for Academic/CS Core)*
   * **PostgreSQL + pgvector Migration:** Transition SQLite and ChromaDB into a single PostgreSQL service (run in Docker), enabling unified relational joins and semantic index searches in single queries.
   * **Background Task Queue:** Set up **Celery + Redis** to handle transcript uploads and ingestion processes asynchronously.
2. **Option B: Advanced RAG & LLMOps**
   * **Automated Evaluation:** Build pipeline validations using **Ragas** or **TruLens** (measuring Faithfulness, Answer Relevance, and Context Recall).
   * **Observability Dashboard:** Bind **LangSmith** or **Phoenix** tracing to track agent execution steps, tokens, and response latencies in real-time.
   * **Semantic Caching:** Build a cache layer in Redis to store and retrieve queries based on semantic embedding similarity.
3. **Option C: Agentic Self-Correction**
   * **Corrective Retrieval Loops:** Add feedback loops to LangGraph where the agent rewrites queries if retrieved documents fail a relevance test, or triggers web search as fallback.

---

## ⚠️ Critical Constraints to Maintain
* **No Dollar Signs ($) in UI Output:** Prompt templates must convert all math-triggering `$` symbols into `USD ` to prevent markdown parser rendering failures in the UI.
* **Strict Quota Allocation:** Preserve the `k // len(detected_cos)` equal-allocation retrieval schema in `qa_chain.py` to prevent multi-company queries from starving target entity context.
* **Thread-ID Persistence:** Ensure the FastAPI API matches the unique `threadId` header/payload provided by the client React mount to keep state preservation active.
