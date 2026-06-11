# Financial Earnings Intelligence Platform — Implementation Plan

## Project Vision

Build a **Financial Earnings Call Intelligence Platform** for Apple, Microsoft, and Nvidia (Q1 2023 – Q4 2024) that evolves in 4 phases:

```
Phase 1 → RAG Foundation        (Core retrieval + Q&A with citations)
Phase 2 → Advanced RAG          (Hybrid search + reranking + evaluation)
Phase 3 → Intelligence Layer    (KPI extraction + structured DB + comparison UI)
Phase 4 → Agentic Layer         (Multi-agent orchestration + automation workflows)
```

Each phase produces a **fully working, demonstrable system** before the next begins.

---

## Dataset Facts (From Audit)

| Dimension | Value |
|---|---|
| Files | 23 `.txt` files across Apple, Microsoft, Nvidia |
| Time span | Q1 2023 – Q4 2024 |
| Total size | ~1.15 MB / ~289K tokens |
| Sentences per file | ~300–460 (period-delimited) |
| File structure | All text on a **single line** (no newlines) |
| Section separator | `[ Good afternoon` marks the start of the live transcript |
| Sections | **Summary** (analytical prose, pos 0) + **Transcript** (prepared remarks + Q&A) |
| Speaker markers | Embedded as `{Name}executive` or `[ {Name}` patterns |
| Metadata location | **Filename only**: `{year}_Q{quarter}_{ticker}_processed.txt` |

> [!IMPORTANT]
> The single-line file structure is the most critical engineering challenge. Standard text splitters will not work. A custom sentence-boundary chunker is required.

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Industry standard for AI/ML |
| **LLM** | Groq API — `llama-3.1-70b-versatile` | Free tier, 131K context, 10–25× faster than OpenAI via LPU hardware |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Runs fully locally, zero cost, 384-dim, strong retrieval quality |
| **Vector DB** | ChromaDB (persistent, local) | Zero infrastructure, 384-dim compatible, easy to inspect |
| **BM25** | `rank_bm25` | Lightweight, pure Python, no server needed |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, runs locally — consistent HuggingFace stack |
| **RAG Framework** | LangChain | Best balance of RAG + Agent support; most recognized in interviews |
| **Agent Orchestration** | LangGraph | State-machine agents, built by LangChain team, production-grade |
| **Structured DB** | SQLite + SQLAlchemy | Zero setup, file-based, perfect for KPI storage |
| **Evaluation** | RAGAS | Industry-standard RAG evaluation framework |
| **UI — Phase 1–2** | Streamlit | Ship fast; focus on RAG correctness first |
| **UI — Phase 3+** | Next.js + FastAPI | Professional product-grade UI for KPI dashboard + full intelligence layer |
| **Config** | `python-dotenv` + YAML | Clean environment management |

> [!NOTE]
> **Total running cost: $0.00.** Groq free tier handles all LLM inference. HuggingFace MiniLM runs locally for embeddings. Cross-encoder reranker runs locally. ChromaDB is file-based. This is a fully offline-capable, zero-cost AI system — a genuine portfolio differentiator.

---

## Project Directory Structure

```
Finance_RAG_Project/
│
├── dataset_2/                              # Raw data (existing)
│   └── Earning_Call_Transcripts/
│       └── cleaned_ECTs_dataset/
│           ├── Apple/
│           ├── Microsoft/
│           └── Nvidia/
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/                          # Phase 1
│   │   ├── __init__.py
│   │   ├── file_parser.py                  # Filename → metadata extraction
│   │   ├── chunker.py                      # Custom sentence-boundary chunker
│   │   ├── embedder.py                     # Embedding generation + batching
│   │   └── pipeline.py                     # End-to-end ingestion orchestrator
│   │
│   ├── retrieval/                          # Phase 1 → 2
│   │   ├── __init__.py
│   │   ├── vector_store.py                 # ChromaDB wrapper
│   │   ├── bm25_retriever.py               # BM25 index + retrieval
│   │   ├── hybrid_retriever.py             # RRF fusion of vector + BM25
│   │   └── reranker.py                     # Cross-encoder reranker
│   │
│   ├── generation/                         # Phase 1
│   │   ├── __init__.py
│   │   ├── prompts.py                      # All prompt templates
│   │   └── qa_chain.py                     # RAG chain with citation
│   │
│   ├── extraction/                         # Phase 3
│   │   ├── __init__.py
│   │   ├── kpi_extractor.py                # LLM-based structured KPI extraction
│   │   ├── schema.py                       # SQLAlchemy ORM models
│   │   └── db_manager.py                   # SQLite read/write interface
│   │
│   ├── agents/                             # Phase 4
│   │   ├── __init__.py
│   │   ├── tools.py                        # Tool definitions wrapping src/* components
│   │   ├── analysis_agent.py               # Cross-quarter analysis agent
│   │   ├── extraction_agent.py             # Automated KPI extraction agent
│   │   ├── report_agent.py                 # Report generation agent
│   │   └── orchestrator.py                 # LangGraph state machine
│   │
│   ├── evaluation/                         # Phase 2
│   │   ├── __init__.py
│   │   ├── test_set.py                     # Golden Q&A pairs for evaluation
│   │   └── ragas_eval.py                   # RAGAS metric runner
│   │
│   └── ui/                                 # Phase 1 (basic) → Phase 3 (full)
│       ├── __init__.py
│       ├── app.py                          # Main Streamlit application
│       ├── components/
│       │   ├── chat.py                     # Chat interface component
│       │   ├── sidebar.py                  # Filters: company, quarter, year
│       │   ├── kpi_dashboard.py            # Phase 3: KPI table + charts
│       │   └── agent_panel.py              # Phase 4: Agent workflow panel
│       └── utils.py
│
├── data/                                   # Generated artifacts
│   ├── chroma_db/                          # Persistent vector store
│   ├── bm25_index/                         # Serialized BM25 index
│   └── kpi_store.db                        # SQLite KPI database
│
├── evaluation/
│   ├── golden_dataset.json                 # Hand-crafted Q&A pairs
│   └── results/                            # Evaluation run outputs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_evaluation_analysis.ipynb
│
├── config/
│   ├── config.yaml                         # All tuneable parameters
│   └── .env                                # API keys (gitignored)
│
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_kpi_extractor.py
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Phase 1: RAG Foundation

**Goal**: A working Q&A system over all 23 transcripts with accurate citations.

**Deliverable**: Streamlit app where you can ask any question and get an answer with source (company, quarter, section).

---

### 1.1 — File Parser & Metadata Extractor

**File**: `src/ingestion/file_parser.py`

Parse metadata entirely from the filename using regex:

```
Pattern: {year}_Q{quarter}_{ticker}_processed.txt
Example: 2024_Q3_aapl_processed.txt
```

Metadata schema per document:

```python
{
    "company":    "Apple",           # Human-readable name
    "ticker":     "AAPL",            # Uppercase ticker
    "year":       2024,              # int
    "quarter":    "Q3",              # string
    "period":     "2024-Q3",         # Compound key for filtering
    "section":    "summary"|"transcript",
    "source_file": "2024_Q3_aapl_processed.txt"
}
```

**Ticker → Company map**: `{"aapl": "Apple", "msft": "Microsoft", "nvda": "Nvidia"}`

---

### 1.2 — Chunking Strategy

**File**: `src/ingestion/chunker.py`

> [!IMPORTANT]
> Files are single-line blobs (~42–63KB, zero newlines). **Selected strategy: LangChain `RecursiveCharacterTextSplitter` with sentence-boundary-first custom separators.** This was chosen over pure fixed-size chunking after a full tradeoff analysis — see rationale below.

**Why Not Fixed-Size Chunking**

Pure fixed-size splitting (every N characters) cuts mid-word at chunk boundaries, directly degrading embedding quality. For financial transcripts, this risks splitting values like `"$22.1 billi"` / `"on, up 265%"` across chunks — making individual chunks semantically incomplete. RCTS with sentence-boundary separators eliminates this at zero additional complexity.

**Selected Strategy: RCTS with Custom Separators**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=[". ", "? ", "! ", "; ", " ", ""],
    chunk_size=1000,      # characters (~250 tokens)
    chunk_overlap=200,    # characters (~50 tokens)
    length_function=len,
)
```

RCTS tries separators in order: sentence endings first, then word spaces, then character boundaries as absolute last resort. On these files, the vast majority of splits land at `". "` — preserving complete financial sentences within chunks.

**Step 1 — Section Split**
Split each file at the `[ ` marker into two named sections:
- `summary`: Everything before `[ ` (the analytical prose summary, ~2–4KB)
- `transcript`: Everything from `[ ` onward (prepared remarks + full Q&A, ~40–60KB)

Both sections are chunked independently so section metadata is preserved accurately.

**Step 2 — Chunking Each Section**
Apply the RCTS splitter to each section independently.

**Step 3 — Chunk Metadata**
Each chunk gets full metadata attached:

```python
{
    # Document-level (from file_parser)
    "company":      "Apple",
    "ticker":       "AAPL",
    "year":         2024,
    "quarter":      "Q3",
    "period":       "2024-Q3",
    "section":      "transcript",   # or "summary"
    "source_file":  "2024_Q3_aapl_processed.txt",

    # Chunk-level
    "chunk_id":     "2024_Q3_aapl_transcript_042",
    "chunk_index":  42,
}
```

**Expected output**: ~1,265 total chunks across all 23 files (~55 chunks/file average).
**Embedding cost**: ~316K tokens ≈ **$0.013 one-time** with `text-embedding-3-small`.

---

### 1.3 — Embedder

**File**: `src/ingestion/embedder.py`

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, runs locally on CPU)
- Library: `langchain-huggingface` → `HuggingFaceEmbeddings`
- No API calls, no rate limits, no batching complexity
- First run downloads the model (~80MB, cached permanently after)
- Cost: **$0.00** — fully local inference

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

> [!NOTE]
> **Upgrade path**: If you want higher retrieval quality in Phase 2+, swap to `BAAI/bge-small-en-v1.5` (same 384 dims, ranks higher on MTEB benchmarks for retrieval tasks). One line change.

---

### 1.4 — Vector Store

**File**: `src/retrieval/vector_store.py`

ChromaDB persistent collection with:
- Collection name: `earnings_transcripts`
- Embedding function: `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, **384 dimensions**)
- Metadata fields: All fields from chunk metadata above
- Persistence path: `data/chroma_db/`

**Key ChromaDB filtering pattern** (enables metadata-filtered retrieval):
```python
# Example: Only Apple transcripts from 2024
results = collection.query(
    query_texts=["What was gross margin guidance?"],
    where={"$and": [{"ticker": "AAPL"}, {"year": 2024}]},
    n_results=5
)
```

---

### 1.5 — Basic RAG Chain

**File**: `src/generation/qa_chain.py`

LangChain `RetrievalQA` chain with:
- Retriever: ChromaDB similarity search (top-5 chunks)
- LLM: Groq `llama-3.1-70b-versatile` via `ChatGroq`
- Output: Answer + source citations (company, quarter, section, verbatim excerpt)

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.0,      # deterministic for financial facts
    max_tokens=1024,
)
```

**Prompt template** (`src/generation/prompts.py`):
```
You are a financial analyst assistant specializing in earnings call transcripts
for Apple, Microsoft, and Nvidia.

Use ONLY the provided context to answer the question.
If the context does not contain enough information, say so clearly.
Never fabricate financial figures or guidance.

For every claim, cite its source as: [Company | Quarter | Year | Section]

Context:
{context}

Question: {question}

Answer:
```

---

### 1.6 — UI Strategy

**Phase 1–2: Streamlit** (`src/ui/app.py`)

Focus is on RAG correctness, not UI polish. Streamlit ships fast and demonstrates the intelligence clearly:
- Company / Year / Quarter filters (sidebar)
- Chat interface with streaming responses
- Citation cards per answer (company, quarter, section, verbatim excerpt)
- Session memory (last 5 exchanges)
- Custom CSS theming (dark finance aesthetic) to elevate above default Streamlit look

**Phase 3+: Next.js + FastAPI** (replaces Streamlit)

When the KPI dashboard and comparison engine are built, the product needs a proper UI that matches the intelligence underneath. Plan:
- **FastAPI backend** (`src/api/`) wraps all Phase 1–3 components as REST endpoints
- **Next.js frontend** (`frontend/`) with Tailwind CSS — dark mode, animated charts (Recharts/Chart.js), professional finance aesthetic
- The FastAPI layer also becomes the foundation for Phase 4 agent API endpoints

> This is the standard production pattern: build the intelligence first, build the product shell second. The FastAPI backend you write in Phase 3 is permanent — Next.js just plugs into it.

---

### Phase 1 Verification

Run the ingestion pipeline and verify:
- [ ] All 23 files parsed with correct metadata
- [ ] ~2,000–2,500 chunks created with no empty chunks
- [ ] ChromaDB collection queryable
- [ ] Q&A works with accurate citations
- [ ] Filters correctly restrict retrieval by company/quarter
- [ ] Groq API responses are deterministic (`temperature=0`) for financial figures
- [ ] HuggingFace model loads and caches correctly on first run

**Manual test questions for Phase 1**:
1. *"What was Apple's gross margin in Q3 2024?"*
2. *"What did Jensen Huang say about sovereign AI?"*
3. *"What was Microsoft's Azure revenue growth in Q4 2024?"*

---

## Phase 2: Advanced RAG

**Goal**: Dramatically improve retrieval quality. Add hybrid search, reranking, query routing, query rewriting/expansion, and quantified evaluation.

---

### 2.1 — BM25 Retriever

**File**: `src/retrieval/bm25_retriever.py`

- Library: `rank_bm25` (`BM25Okapi`)
- Index built over all chunk texts at ingestion time
- Serialized to `data/bm25_index/` with `pickle` for fast reload
- Returns top-20 candidates (before reranking)

**Why BM25 matters here**: Queries like *"What was the $24 billion revenue guidance?"* or *"DRIVE Orin"* are exact-match queries where BM25 outperforms vector search.

---

### 2.2 — Hybrid Retriever with RRF

**File**: `src/retrieval/hybrid_retriever.py`

Combine vector search + BM25 using **Reciprocal Rank Fusion (RRF)**:

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def fuse_results(vector_results, bm25_results) -> list:
    scores = {}
    for rank, chunk_id in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score(rank)
    for rank, chunk_id in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score(rank)
    return sorted(scores, key=scores.get, reverse=True)
```

- Vector search: top-20 candidates
- BM25 search: top-20 candidates
- RRF fusion → top-20 merged candidates
- Pass to reranker for final top-5

---

### 2.3 — Cross-Encoder Reranker

**File**: `src/retrieval/reranker.py`

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally, free)
- Takes the 20 hybrid candidates + original query
- Scores each (query, chunk) pair
- Returns top-5 highest-scored chunks
- Adds `rerank_score` to result metadata for transparency

**Expected retrieval quality improvement**: 15–25% precision increase over naive vector-only search.

---

### 2.4 — Query Routing Layer

**File**: `src/retrieval/router.py` [NEW]

- Evaluates query structure and keywords to select the optimal retrieval mode (rule-based and LLM-assisted options).
- **LLM-assisted Router**: Uses a fast LLM (`llama-3.1-8b-instant`) to classify user intent.
- **Routing Rules**:
  - *Metric / Exact term*: routes to BM25-heavy hybrid or `bm25_only`.
  - *Broad strategy/outlook*: routes to Vector-heavy hybrid or `vector_only`.
  - *Comparison / multi-company query*: routes to `multi_query_hybrid` (multi-query expansion loop).
  - *Transcripts summaries*: routes to section-aware search (only querying `summary` section chunks).
- **UI Integration**: Displays the active retrieval path selected by the router (e.g. `"Selected Path: Multi-Query Hybrid"`) for full transparency.

---

### 2.5 — Query Rewrite & Multi-Query Expansion

**File**: `src/retrieval/query_transformer.py` [NEW]

- **Query Rewrite (Conversational Memory)**: Resolves vague pronouns or implicit context for follow-up queries using the last 3 turns of chat history.
  - *Example*: User asks `"What did they say about margins?"` -> Rewritten to `"What did Apple say about margins and gross margin guidance in Q3 2024?"` based on previous context.
- **Multi-Query Expansion**: For complex comparison or broad questions, decomposes the query into 3 distinct financial search queries to maximize recall.
  - *Example*: `"Nvidia AI demand outlook"` -> 
    1. `"Nvidia GPU and H100 capacity constraints"`
    2. `"Nvidia data center generative AI growth commentary"`
    3. `"Nvidia AI inference revenue and software demand"`
  - Executes retrieval on each variant and fuses candidates using RRF.

**Pipeline Flow**:
```
User Query
    │
    ▼
Query Router ──[Comparison/Follow-up]──► Query Transformer (Rewrite/Multi-Query)
    │                                                   │
    ▼                                                   ▼
Apply Filters ─────────────────────────► Parallel Retrieval (Vector + BM25)
                                                        │
                                                        ▼
                                                    RRF Fusion
                                                        │
                                                        ▼
                                              Cross-Encoder Rerank
```

---

### 2.6 — Evaluation Harness

**File**: `src/evaluation/ragas_eval.py`

**Step 1 — Build Golden Dataset** (`evaluation/golden_dataset.json`)

Hand-craft 20–30 question-answer pairs covering:
- Exact financial figures (tests retrieval precision)
- Cross-quarter comparisons (tests temporal handling)
- Qualitative strategy questions (tests semantic search)
- Multi-company questions (tests scope handling)

**Step 2 — RAGAS Metrics**

| Metric | What It Measures |
|---|---|
| `faithfulness` | Is the answer supported by retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are retrieved chunks relevant to the question? |
| `context_recall` | Does the context contain the answer? |

**Step 3 — Baseline vs. Advanced Comparison**
Run evaluation on: (a) vector-only, (b) hybrid, (c) hybrid + reranker, (d) routed/transformed retrieval.
Document the improvement at each stage. This comparison **is your evaluation section** for any presentation or interview.

---

### Phase 2 Verification

- [ ] BM25 index builds successfully and reloads from disk
- [ ] Hybrid retrieval returns different (better) results than vector-only for exact-match queries
- [ ] Reranker improves top-5 chunk quality (measure manually on 10 test queries)
- [ ] Query router correctly classifies intent (vector vs. bm25 vs. multi-query)
- [ ] Query rewriter resolves pronouns in conversational memory test queries
- [ ] Multi-query expansion retrieves non-overlapping candidate chunks successfully
- [ ] RAGAS evaluation runs end-to-end and produces all 4 metric scores
- [ ] Baseline vs. hybrid vs. hybrid+rerank comparison documented in `evaluation/results/`

---

## Phase 3: Intelligence Layer

**Goal**: Add structured financial intelligence. Extract KPIs into a SQL database. Enable cross-quarter comparison. Build the full polished UI.

---

### 3.1 — KPI Extraction Pipeline

**File**: `src/extraction/kpi_extractor.py`

For each transcript, run an LLM extraction pass to populate structured data:

**Target KPIs per document**:

```python
class EarningsKPI(Base):
    __tablename__ = "earnings_kpis"

    id              = Column(Integer, primary_key=True)
    ticker          = Column(String)          # "AAPL"
    company         = Column(String)          # "Apple"
    year            = Column(Integer)         # 2024
    quarter         = Column(String)          # "Q3"
    period          = Column(String)          # "2024-Q3"

    # Financial Actuals
    revenue_b       = Column(Float)           # Revenue in $B
    eps_diluted     = Column(Float)           # Diluted EPS
    gross_margin_pct = Column(Float)          # Gross margin %
    net_income_b    = Column(Float)           # Net income in $B
    op_cash_flow_b  = Column(Float)           # Operating cash flow in $B

    # Guidance (next quarter)
    guidance_revenue_low_b  = Column(Float)
    guidance_revenue_high_b = Column(Float)
    guidance_gm_low_pct     = Column(Float)
    guidance_gm_high_pct    = Column(Float)

    # Growth Rates (YoY)
    revenue_growth_yoy_pct  = Column(Float)
    eps_growth_yoy_pct      = Column(Float)

    # Segment highlights (JSON string)
    segment_notes   = Column(Text)            # JSON: key segment metrics
```

**Extraction method**: LLM with structured output (Pydantic model) using a specific extraction prompt. Falls back to `None` for fields not found (never fabricates).

---

### 3.2 — Cross-Quarter Comparison Engine

Add a query type to the system: **Comparison Queries**

```
"Compare Apple's gross margin across all quarters in 2024"
"How did Nvidia's revenue growth change from Q1 2023 to Q4 2024?"
"Which company had the highest EPS growth in Q3 2024?"
```

These route to the SQL store (not vector search) for structured data, then the LLM synthesises the comparison narrative.

**Routing logic**: Keyword detection in query (`"compare"`, `"trend"`, `"change over"`, `"which quarter"`) → SQL path vs. qualitative narrative path → vector RAG path.

---

### 3.3 — Full Streamlit UI

**File**: `src/ui/app.py` (upgraded)

**Page 1 — Intelligence Chat**
- Full hybrid RAG + reranker Q&A
- Company / quarter / year filters
- Citation cards with expandable source text
- Query routing indicator (RAG / SQL / Hybrid)

**Page 2 — KPI Dashboard**
- Company selector
- Time series charts: Revenue, EPS, Gross Margin across quarters
- Side-by-side company comparison table
- Guidance vs. Actuals tracker

**Page 3 — Transcript Explorer**
- Select company + quarter → see full document
- Highlighted chunk viewer (shows which chunks were retrieved for a given query)

---

### Phase 3 Verification

- [ ] KPI extraction runs on all 23 files without errors
- [ ] SQLite DB populated with structured data for all company-quarter combinations
- [ ] Comparison queries route to SQL and return correct structured data
- [ ] KPI dashboard renders charts correctly
- [ ] Guidance vs. Actuals comparison works for at least 2 companies

---

## Phase 4: Advanced Agentic Layer & Next.js/FastAPI Shell

**Goal**: Transform the system from a prototype into a production-grade portfolio application. We will wrap the RAG/SQL logic into LangGraph agents, serve them via a FastAPI backend, and build a premium, highly responsive Next.js frontend.

---

### 4.1 — LangGraph Agentic Layer (Backend)

**File**: `src/api/agents.py`

Convert all Phase 1–3 components into a multi-agent orchestration system using LangGraph:
- **Router Agent**: Classifies query intent.
- **RAG Agent**: Hybrid retrieval + generation for qualitative questions.
- **SQL Agent**: KPI database queries for quantitative questions.
- **Report Generator Agent**: An end-to-end automated workflow that retrieves narrative passages, fetches cross-quarter KPIs, and synthesizes a professional investment research brief for a specific company and quarter.

### 4.2 — FastAPI Backend Shell

**File**: `src/api/server.py`

Build a robust API to serve the agents to the frontend:
- `POST /api/chat`: Streaming endpoint for chat queries (handles routing and yielding LLM tokens).
- `GET /api/kpis`: Returns structured KPI data for the dashboard visualizations.
- `POST /api/generate-report`: Triggers the Report Generator Agent and returns a comprehensive markdown brief.

### 4.3 — Premium Next.js Frontend

**Directory**: `frontend/`

Replace the Streamlit UI with a stunning, highly responsive Next.js web application built for your portfolio.
- **Technology Stack**: Next.js (App Router), TailwindCSS (for rapid premium styling), Framer Motion (for micro-animations), and Recharts (for KPI visualizations).
- **Aesthetics**: Dark mode default, glassmorphism elements, vibrant gradient accents, and smooth transitions.
- **Features**:
  - **Intelligence Chat Interface**: Streaming chat with expandable citation cards and entity filters.
  - **Interactive KPI Dashboard**: Dynamic charts comparing Revenue, EPS, and Margins with smooth animations.
  - **Earnings Briefs**: A dedicated view to read the generated investment reports.

---

### Phase 4 Verification

- [ ] FastAPI server successfully exposes chat, KPI, and report generation endpoints.
- [ ] LangGraph agents route queries accurately and produce investment briefs without hallucinations.
- [ ] Next.js frontend compiles and renders a premium, dark-mode design.
- [ ] Frontend successfully communicates with the FastAPI backend (streaming chat and data fetching).

---

## Open Questions

> [!IMPORTANT]
> **TailwindCSS**: The frontend plan utilizes TailwindCSS to quickly build the premium design. Are you comfortable with using Tailwind v3.4 for this, or do you strictly prefer Vanilla CSS?

> [!IMPORTANT]
> **Agentic Scope**: Do you want the Report Generator Agent to run automatically when a user asks for a "summary" in the chat, or should it be a separate feature triggered by a dedicated "Generate Brief" button in the UI?

---

## Verification Plan

### Automated
- Unit tests for chunker (correct metadata, no empty chunks, correct section split)
- Unit tests for retrieval (metadata filtering works correctly)
- RAGAS evaluation pipeline producing quantified improvement scores

### Manual Demonstration Queries (by phase)
| Phase | Query | Validates |
|---|---|---|
| 1 | *"What was Apple's gross margin in Q3 2024?"* | Basic RAG + citation |
| 1 | *"What did Tim Cook say about Apple Intelligence?"* | Semantic retrieval |
| 2 | *"What was the exact revenue guidance Nvidia gave for Q1 2024?"* | Hybrid search (exact match) |
| 2 | *"Compare retrieval precision before and after reranking"* | Evaluation harness |
| 3 | *"Show me Apple's gross margin trend across 2023 and 2024"* | SQL + chart |
| 3 | *"Which company had the strongest EPS growth in Q4 2024?"* | Cross-company SQL |
| 4 | *"Generate a full earnings brief for Nvidia Q4 2024"* | Multi-agent workflow |
| 4 | *"How did management tone on AI demand change from Q1 2023 to Q4 2024?"* | Multi-step analysis |

---

## Build Timeline Estimate

| Phase | Estimated Time | Key Output |
|---|---|---|
| Phase 1 | 3–4 days | Working RAG chatbot with citations |
| Phase 2 | 2–3 days | Hybrid search + reranking + RAGAS evaluation |
| Phase 3 | 3–4 days | KPI dashboard + comparison engine |
| Phase 4 | 4–5 days | Multi-agent system + brief generator |
| **Total** | **~2–3 weeks** | Full production-grade system |

---

*Ready to execute upon approval. Phase 1 can begin immediately.*
