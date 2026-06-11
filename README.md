<div align="center">

# 📊 AURA — Financial Earnings Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

> **Not a ChatGPT wrapper. Not a demo. A precision-engineered, Multi-Agent RAG system that solves the hallucination and entity-starvation problems that break standard LLM pipelines — built for Apple, Microsoft & Nvidia earnings intelligence (Q1 2023 – Q4 2024)(Kaggle dataset).**

</div>

---

## 📺 Platform Preview

> A premium dark-luxury financial intelligence cockpit featuring AI chat, KPI analytics, and investment brief generation.

![AURA Intelligence Platform Cockpit](images/Chatbot_Page.png)

---

## 🌟 What Makes This a Production-Grade MVP

AURA transcends the typical "GenAI wrapper" by engineering robust, original solutions to the most critical failure modes in production financial AI — the same failure modes that affect LLM integrations at Bloomberg, Goldman Sachs, and large-scale enterprise RAG deployments:

- ⚖️ **Entity Starvation Elimination via 3-Layer Quota Allocation**: Standard RAG collapses for multi-company comparison queries, routing 12 chunks to Apple and 0 to Nvidia because the cross-encoder reranker scores based on signal density, not fairness. AURA's 3× per-entity retrieval buffer ensures each company enters the reranker with a rich candidate pool, a strict `k // n_entities` first-pass quota fills slots symmetrically, and a round-robin overflow mechanism distributes remaining budget cyclically — completely eliminating starvation.

- 🧠 **LLM Primacy Bias Mitigation via Entity-Grouped Context Ordering**: Even after retrieving equal chunks per company, standard LLMs still under-represent entities whose context appears at the *bottom* of the prompt. AURA counteracts this by regrouping the final context block by entity (all Apple → all Microsoft → all Nvidia), ensuring each company gets a coherent, contiguous reading window in the LLM's attention field.

- 🔗 **Two-LLM Synthesis Gap Resolution (Dual-Layer Prompt Enforcement)**: AURA uses two sequentially-called LLMs — an inner RAG chain LLM and an outer Agent Synthesiser LLM. A formatting rule in only one layer is silently overridden by the other. All quality rules (table generation, citation safety, equal entity coverage) are explicitly duplicated in both `prompts.py` and `server.py`, sealing the inter-LLM compliance gap.

- 🛡️ **Anti-Hallucination Stateful Memory Filter**: In multi-turn agentic systems, the LLM sees previous `ToolMessage` artifacts in history and "reuses" stale cached data instead of calling tools afresh — producing confident hallucinations. AURA's `filter_messages_for_llm()` actively strips all prior tool artifacts from the LangGraph state before each LLM call, forcing fresh data retrieval on every turn while preserving natural conversation flow.

- 📊 **Table-Safe Citation Architecture**: The citation format `[Apple | Q3 | 2023 | summary]` contains `|` — the markdown table column separator. Embedding it in table cells destroys table structure by adding phantom columns.

- 🔀 **Scale-Invariant Hybrid Search with Adaptive Candidate Pool Scaling**: Standard RAG hardcodes candidate pool limits (e.g., `20`), causing the cross-encoder reranker to receive an insufficient pool when users request higher context richness. AURA dynamically scales the pool: `candidate_pool_limit = max(20, k + 10)`, ensuring the reranker always has a sufficiently rich pool regardless of the user's slider position.

- 🗄️ **Quantitative vs. Qualitative Routing (Dual-Store Architecture)**: LLMs hallucinate reported financial numbers when they're treated as plain text in vector embeddings. AURA routes all hard KPI queries (Revenue, EPS, Gross Margin) directly to a strict Pydantic-validated SQLite database, isolating quantitative extraction from vector search entirely and guaranteeing provably correct numerical answers.

- 🧹 **Domain-Specific Corpus Noise Elimination**: Earnings transcripts are polluted with Safe Harbor legal disclaimers, operator introductions, and call setup boilerplate — all of which contain high-frequency financial terms (`revenue`, `guidance`, `Apple`) that BM25 and cosine similarity both rank highly, injecting junk context into the LLM. AURA's `clean_transcript_text()` applies sentence-level pattern filtering during ingestion, eliminating 41 junk chunks (from 1,475 to 1,434) while retaining 100% of analytical content.

- 💡 **LLM Length-Scaling via Psychological Prompt Instruction**: LLMs naturally compress large context windows into broad summaries regardless of context volume. AURA engineers a prompt instruction that *tethers* response length to the quantity of incoming context: *"If you receive many source chunks, you MUST extract distinct insights from each and write a proportionally longer response."* — transforming the model from a "summariser" into a "detailed extractor" dynamically.

---

## 📑 Table of Contents

- [🌟 What Makes This a Production-Grade MVP](#-what-makes-this-a-production-grade-mvp)
- [📺 Platform Preview](#-platform-preview)
- [🛑 The Enterprise Information Bottleneck](#-the-enterprise-information-bottleneck)
- [📥 Quick Start](#-quick-start)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📚 Documentation](#-documentation)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Deployment](#️-deployment)
- [📈 Phase Changelog](#-phase-changelog)
- [💡 Engineering Highlights](#-engineering-highlights)
- [📁 Project Structure](#-project-structure)

---

## 🛑 The Enterprise Information Bottleneck

**The Problem:** Financial analysts and hedge funds drown in information overload during earnings season. When trying to use standard LLMs to automate research, two critical failure modes emerge:

- 🔴 **LLM Hallucination** — Models confidently fabricate revenue figures, EPS values, and guidance numbers that never appeared in any transcript.
- 🔴 **Entity Starvation** — When asked to compare Apple, Microsoft, and Nvidia simultaneously, standard RAG pipelines return 12 Apple chunks, 3 Microsoft chunks, and 0 Nvidia chunks — producing a dangerously biased analysis.

**The Solution:** AURA was engineered specifically to eliminate both failure modes. A **Multi-Agent RAG Architecture** isolates qualitative analysis (Hybrid RAG + Cross-Encoders) from quantitative extraction (strict SQLite KPI database), while a custom 3-Layer Quota Allocator guarantees equal, unbiased context representation for every company.

👉 **[Read the Full Problem Statement & Market Value Here](docs/problem_statement.md)**

---

## 📥 Quick Start

**Prerequisites:** Python 3.11+, Node.js 20, a free [Groq API key](https://console.groq.com/)

```bash
# 1. Clone & configure
git clone <your-repo-url> && cd Finance_RAG_Project
echo "GROQ_API_KEY=gsk_your_key_here" > config/.env

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run data ingestion (~2–4 min, one-time)
python -m src.ingestion.pipeline

# 4. Start backend API
python -m src.api.server

# 5. Start frontend (new terminal)
cd frontend && npm install && npm run dev
```

🌐 Open [http://localhost:3000](http://localhost:3000)

> **Prefer Docker?** → [One-command deployment](#docker-compose-one-command)

---

## ✨ Features

### 🔍 Hybrid Retrieval Engine

- **Dual-index search**: Dense vector (ChromaDB + `all-MiniLM-L6-v2`) + sparse keyword (BM25Okapi) running in parallel
- **Reciprocal Rank Fusion**: Merges ranked candidate lists without score normalization — scale-invariant and consistently outperforms linear combination
- **Cross-Encoder Reranking**: Local `ms-marco-MiniLM-L-6-v2` scores `(query, passage)` pairs at deep interaction level

### 🤖 Multi-Agent Orchestration

AURA divides cognitive labor among specialized, autonomous sub-systems rather than relying on a single prompt:

- **The Orchestrator Agent (Manager)**: The LangGraph state machine using the ReAct paradigm to reason and route queries to specialized tool agents.
- **The Research Agent (`rag_search`)**: Handles qualitative data. It uses its own Query Router, Query Transformer, and a dedicated Synthesis LLM to format markdown tables and enforce citation rules independently of the Manager.
- **The Data Analyst Agent (`get_kpis`)**: Interacts directly with the SQLite database via Pydantic-validated Groq structured outputs to fetch precise quantitative metrics without hallucination.
- **The Writer Agent (`generate_report_sections`)**: Coordinates with both the Research and Analyst agents to synthesize comprehensive investment briefs.
- **Anti-Hallucination History Filter**: Strips past `ToolMessage` artifacts from the shared state memory before each LLM call to ensure fresh tool invocations.

### ⚖️ Fair Multi-Entity RAG

- **Strict quota allocation**: Guarantees `k // n_companies` document slots per company — prevents entity starvation
- **3× retrieval buffer**: Fetches `exact_per_entity × 3` candidates per entity before reranking
- **Round-robin overflow fill**: Remaining budget allocated cyclically across all companies
- **Entity-grouped context**: Apple → MSFT → Nvidia ordering combats LLM primacy bias

### 📊 Structured KPI Intelligence

- **Groq structured output**: Extracts Revenue, EPS, Gross Margin, Guidance, Net Income via Pydantic-validated LLM calls
- **SQLite database**: `data/finance_kpis.db` stores all structured metrics for instant query
- **KPI Analytics Dashboard**: YoY chevron indicators, quarterly metric cards, company selector

### 💅 Premium Cockpit UI

- **Three-panel interface**: Intelligence Chat · KPI Analytics · Investment Brief Generator
- **Animated sun icon**: 8s idle spin → 3s active spin + expanding pulse halo when typing
- **Citation bubble tooltips**: Hover any `[Company | Q | Year | Section]` reference for source snippet
- **Markdown comparison tables**: Enforced via dual-layer prompt engineering for multi-entity queries
- **Response Richness slider**: 1–30 references; auto-scaled internally for multi-entity queries

---

## 🏗️ Architecture

The platform is built as four sequential layers:

```mermaid
flowchart TB
    classDef ingestion fill:#e0f2fe,stroke:#0288d1,stroke-width:2px,color:#0369a1,font-weight:bold
    classDef storage fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#065f46,font-weight:bold
    classDef logic fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f,font-weight:bold
    classDef ui fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-weight:bold

    subgraph Data_Ingestion_Pipeline [1. Data Ingestion Pipeline]
        direction TB
        A1[Raw processed transcripts<br/>raw_data/**/*.txt] --> A2[File Parser<br/>file_parser.py]
        A2 --> A3[Semantic Chunker & Boilerplate Cleaner<br/>chunker.py]
        A3 --> A4[Local Embeddings Generator<br/>embedder.py]
    end
    class Data_Ingestion_Pipeline ingestion

    subgraph Dual_Storage_Layer [2. Dual Storage Layer]
        direction TB
        B1[(ChromaDB Vector Store<br/>data/chroma_db)]
        B2[(BM25 Lexical Index<br/>data/bm25_index)]
        B3[(SQL KPI Database<br/>data/finance_kpis.db)]
    end
    class Dual_Storage_Layer storage

    A4 -->|Vector + Metadata Chunks| B1
    A3 -->|Lexical Corpus| B2
    A2 -->|Summary Section Text| A5[Structured LLM KPI Extractor<br/>kpi_extractor.py]
    A5 -->|Pydantic ORM Models| B3

    subgraph Intelligence_Core [3. RAG & Agentic Intelligence Core]
        direction TB
        C1[LangGraph Orchestrator<br/>orchestrator.py]
        C2[Agent Tools Wrapper<br/>tools.py]
        C3[Query Router & Transformer<br/>router.py & query_transformer.py]
        C4[RAG Execution Engine<br/>qa_chain.py]
        C5[Local Reranker<br/>reranker.py]
        C6[Groq LPU LLM Core<br/>qwen/qwen3-32b]
    end
    class Intelligence_Core logic

    B1 <-->|Cosine Vector Similarity| C4
    B2 <-->|Lexical Keyword Search| C4
    B3 <-->|SQLAlchemy Queries| C2
    C1 <-->|Tool Executions| C2
    C2 <-->|get_answer| C4
    C4 <-->|Intent Classification| C3
    C4 <-->|Candidate Rescoring| C5
    C4 <-->|Prompt Synthesis & Stream| C6

    subgraph User_Facing [4. User Facing Application]
        direction TB
        D1[FastAPI Server Gateway<br/>server.py]
        D2[Next.js React Client<br/>frontend/src/app]
    end
    class User_Facing ui

    D2 <-->|API requests / JSON| D1
    D1 <-->|run_agent_query| C1
    D1 <-->|get_kpis / generate-report| C2

    class A1,A2,A3,A4,A5 ingestion
    class B1,B2,B3 storage
    class C1,C2,C3,C4,C5,C6 logic
    class D1,D2 ui
    linkStyle default stroke:#334155,stroke-width:2px
```

📖 **[Read the full Architecture Deep-Dive →](docs/architecture.md)**

---

## 🛠️ Deployment

### Local Development

See the [Quick Start](#-quick-start) section above.

### Docker Compose (One-Command)

```bash
# Ensure Docker Desktop is running
docker compose up --build
```

- First build: ~10–20 minutes (PyTorch + HuggingFace model downloads)
- Subsequent starts: <10 seconds (Docker layer cache)
- Open [http://localhost:3000](http://localhost:3000)

**Run ingestion inside Docker** (first-time only if `data/` is empty):
```bash
docker compose run --rm backend python -m src.ingestion.pipeline
```

📖 **[Full Deployment Guide →](docs/deployment.md)**

---

## 📚 Documentation

Explore the detailed technical documentation in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| 📐 [Architecture Deep-Dive](docs/architecture.md) | Full system design: all 4 layers, design decisions, data flows |
| 🔍 [Retrieval Engine](docs/retrieval_engine.md) | BM25, RRF, Cross-Encoder, Query Router, multi-entity quota algorithm |
| 🤖 [Agent Orchestration](docs/agent_orchestration.md) | LangGraph state machine, tools, memory, prompt engineering |
| 💅 [Frontend Cockpit](docs/frontend.md) | Next.js design system, components, animations, API integration |
| 🚀 [Deployment Guide](docs/deployment.md) | Local setup, Docker, troubleshooting, performance tuning |
| 🛠️ [Engineering Challenges](docs/engineering_challenges.md) | All-phases challenge log: root causes, solutions, learnings |

**Per-Phase Engineering Logs:**

| Phase | Focus | Document |
|---|---|---|
| 1 | RAG Foundation | [Phase 1 →](features_and_learnings/Phase_1challenges_and_learnings.md) |
| 2 | Hybrid Retrieval | [Phase 2 →](features_and_learnings/phase2_challenges_and_solutions.md) |
| 3–4 | KPI Extraction + Agent | [Phase 3–4 →](features_and_learnings/phase3_4__challenges_and_resolutions.md) |
| 5 | Premium Frontend | [Phase 5 →](features_and_learnings/phase5_challenges_and_solutions.md) |
| 6 | Dockerization | [Phase 6 →](features_and_learnings/phase6_challenges_and_solutions.md) |
| 7 | RAG Quality & Multi-Entity | [Phase 7 →](features_and_learnings/phase7_challenges_and_solutions.md) |

**Master Walkthrough:** [walkthrough.md →](walkthrough.md)

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM Inference** | Groq LPU + `qwen/qwen3-32b` | ~500 tok/s deterministic generation |
| **Agent Framework** | LangGraph 0.2+ | Stateful cyclic tool-calling graph |
| **Vector Store** | ChromaDB | Local persistent dense embedding index |
| **Sparse Retrieval** | BM25Okapi | Exact keyword and ticker matching |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim local embeddings, zero API cost |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Local cross-encoder passage reranking |
| **KPI Database** | SQLite + SQLAlchemy | Structured financial metrics ORM |
| **Backend API** | FastAPI + Uvicorn | Async streaming JSON endpoints |
| **Frontend** | Next.js 14 + TypeScript | App Router, SSR, React 18 |
| **Markdown** | react-markdown + remark-gfm | Tables, citations, code rendering |
| **Containerization** | Docker + Docker Compose | Multi-stage builds, bridge networking |

---

## 📈 Phase Changelog

| Phase | Focus | Key Deliverables |
|---|---|---|
| **1 — RAG Foundation** | Core pipeline | ChromaDB ingestion, `<think>` token filter, Streamlit UI, Qwen3 integration |
| **2 — Hybrid Retrieval** | Retrieval quality | BM25 index, RRF fusion, cross-encoder reranker, entity starvation fix v1 |
| **3 — KPI Extraction** | Structured data | Groq structured output, Pydantic ORM, SQLite schema, KPI dashboard |
| **4 — Agent Orchestration** | Agentic loop | LangGraph state machine, MemorySaver, multi-turn history filter |
| **5 — Premium Frontend** | UI overhaul | Next.js cockpit, citation bubbles, query history, agent stepper |
| **6 — Dockerization** | Production ops | Multi-stage Docker, Compose networking, volume persistence |
| **7 — RAG Quality** | Intelligence accuracy | 3× buffer retrieval, round-robin overflow, entity-grouped context, table citation safety, dual-layer prompts |

---

## 💡 Engineering Highlights

Building a production-grade AI system requires moving beyond simple API wrappers to solve the edge-case challenges that break standard implementations. This section outlines the core architectural hurdles encountered while building AURA, and the specific engineering solutions developed to guarantee enterprise-level reliability, fairness, and zero-hallucination accuracy.

📄 **Handling Single-Line Transcripts**

Source files are 40–60KB single-line blobs. The chunker splits at the `[ ` section boundary marker first, then applies sentence-priority RCTS separators (`. ` → `? ` → `! ` → `; `) to preserve financial figure context across chunk boundaries.

⚖️ **Preventing Company Starvation**

For *"Compare risks across Apple, Microsoft, Nvidia"* — naive retrieval returns 12 Apple chunks, 3 Microsoft, 0 Nvidia. The fix: **3× per-entity retrieval buffer → per-entity reranking → strict quota allocation → round-robin overflow fill → entity-grouped context ordering.**

🔗 **Two-LLM Synthesis Gap**

The inner RAG LLM and outer agent synthesis LLM are independently instructed. All formatting rules (table generation, citation safety, equal entity coverage) are duplicated at both `prompts.py` and `server.py` — rules at only one layer are silently overridden by the other.

📊 **Table-Safe Citations**

`[Apple | Q3 | 2023 | summary]` contains `|` which markdown renders as table column separators. Inside table cells, the system uses `[1]`, `[2]` numeric refs with a Citation Key section below — enforced via both prompt layers.

📈 **Auto-Scaling K for Multi-Entity Queries**

`k=6` with 3 companies gives 2 chunks per company — critically sparse for fair coverage. The agent auto-scales dynamically: `effective_k = min(24, max(GLOBAL_K, n_entities × 6))`.

📖 **[See all engineering challenges →](docs/engineering_challenges.md)**

---

## 📂 Project Structure

```
Finance_RAG_Project/
├── src/                        # Python backend modules
│   ├── ingestion/              # Transcript parsing & indexing pipeline
│   ├── retrieval/              # Vector, BM25, RRF, reranker, router
│   ├── generation/             # Prompts, RAG chain, quota allocation
│   ├── extraction/             # KPI extractor, SQLite ORM schema
│   ├── agents/                 # LangGraph tools & orchestrator
│   └── api/                    # FastAPI server & system instructions
├── frontend/                   # Next.js 14 premium cockpit UI
│   └── src/app/                # page.tsx, layout.tsx, globals.css
├── docs/                       # 📚 Detailed technical documentation
│   ├── architecture.md         # System design overview
│   ├── retrieval_engine.md     # Retrieval deep-dive
│   ├── agent_orchestration.md  # LangGraph agent reference
│   ├── frontend.md             # UI component documentation
│   ├── deployment.md           # Setup & deployment guide
│   └── engineering_challenges.md  # All-phases challenge log
├── features_and_learnings/     # Per-phase engineering logs (Phases 1–7)
├── config/
│   ├── config.yaml             # All system parameters (models, paths, thresholds)
│   └── .env                    # GROQ_API_KEY (gitignored)
├── data/                       # Generated at ingestion (gitignored)
│   ├── chroma_db/              # ChromaDB vector store
│   ├── bm25_index/             # Serialized BM25 corpus
│   └── finance_kpis.db         # SQLite KPI database
├── backend.Dockerfile          # Python container
├── docker-compose.yml          # Full-stack orchestration
└── requirements.txt            # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome — whether it's extending the dataset to more companies, adding new retrieval strategies, improving the frontend, or writing evaluation benchmarks.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com/) — LPU inference infrastructure enabling ~500 tok/s generation
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Stateful agent graph compilation
- [ChromaDB](https://www.trychroma.com/) — Local vector store with metadata filtering
- [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` embeddings & `ms-marco-MiniLM-L-6-v2` reranker
- Financial transcript dataset covering Q1 2023 – Q4 2024 earnings calls

---

<div align="center">

**Seven phases of iteration. One mission: financial intelligence without hallucination.**

[Documentation](docs/) • [Walkthrough](walkthrough.md) • [Engineering Logs](features_and_learnings/) • [Detailed RAG Architecture](docs/detailed_rag_architecture.md)

</div>
