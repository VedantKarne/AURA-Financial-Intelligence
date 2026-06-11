# Detailed RAG Architecture Mindmap

> **[← Back to Architecture Overview](./architecture.md)**

This document provides a holistic blueprint of the AURA Financial Intelligence Platform, tracking the data flow from raw unstructured transcripts through the storage, retrieval, orchestration, and presentation layers.

---

## The System Ecosystem Flowchart

The following mindmap illustrates the 4 core subsystems of AURA and how they interconnect to provide zero-hallucination, multi-agent financial intelligence.

```mermaid
flowchart TB
    classDef ingestion fill:#e0f2fe,stroke:#0288d1,stroke-width:2px,color:#0369a1,font-weight:bold
    classDef storage fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#065f46,font-weight:bold
    classDef logic fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f,font-weight:bold
    classDef ui fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-weight:bold

    subgraph Data_Ingestion_Pipeline ["1. Data Ingestion Pipeline"]
        direction TB
        A1["Raw processed transcripts<br/>raw_data/**/*.txt"] --> A2["File Parser<br/>file_parser.py"]
        A2 --> A3["Semantic Chunker & Boilerplate Cleaner<br/>chunker.py"]
        A3 --> A4["Local Embeddings Generator<br/>embedder.py"]
    end
    class Data_Ingestion_Pipeline ingestion

    subgraph Dual_Storage_Layer ["2. Dual Storage Layer"]
        direction TB
        B1[("ChromaDB Vector Store<br/>data/chroma_db")]
        B2[("BM25 Lexical Index<br/>data/bm25_index")]
        B3[("SQL KPI Database<br/>data/finance_kpis.db")]
    end
    class Dual_Storage_Layer storage

    A4 -->|"Vector + Metadata Chunks"| B1
    A3 -->|"Lexical Corpus"| B2
    A2 -->|"Summary Section Text"| A5["Structured LLM KPI Extractor<br/>kpi_extractor.py"]
    A5 -->|"Pydantic ORM Models"| B3

    subgraph Intelligence_Core ["3. RAG & Agentic Intelligence Core"]
        direction TB
        C1["LangGraph Orchestrator<br/>orchestrator.py"]
        C2["Agent Tools Wrapper<br/>tools.py"]
        C3["Query Router & Transformer<br/>router.py & query_transformer.py"]
        C4["RAG Execution Engine<br/>qa_chain.py"]
        C5["Local Reranker<br/>reranker.py"]
        C6["Groq LPU LLM Core<br/>qwen/qwen3-32b"]
    end
    class Intelligence_Core logic

    B1 <-->|"Cosine Vector Similarity"| C4
    B2 <-->|"Lexical Keyword Search"| C4
    B3 <-->|"SQL Alchemy Queries"| C2
    C1 <-->|"Tool Executions"| C2
    C2 <-->|"get_answer"| C4
    C4 <-->|"Intent Classification"| C3
    C4 <-->|"Candidate Rescoring"| C5
    C4 <-->|"Prompt Synthesis & Stream"| C6

    subgraph User_Facing ["4. User Facing Application"]
        direction TB
        D1["FastAPI Server Gateway<br/>server.py"]
        D2["Next.js React Client<br/>frontend/src/app"]
    end
    class User_Facing ui

    D2 <-->|"API requests / JSON streams"| D1
    D1 <-->|"run_agent_query"| C1
    D1 <-->|"get_kpis / generate-report"| C2

    class A1,A2,A3,A4,A5 ingestion
    class B1,B2,B3 storage
    class C1,C2,C3,C4,C5,C6 logic
    class D1,D2 ui
    linkStyle default stroke:#334155,stroke-width:2px;
```

---

## Architectural Layer Breakdown

### 1. Data Ingestion Pipeline (Blue)
The ingestion layer is responsible for converting unstructured walls of text into structured, searchable data.
- **File Parser (`file_parser.py`)**: Identifies the ticker, company, year, and quarter from filenames.
- **Chunker (`chunker.py`)**: Slices the massive transcripts using RCTS (Recursive Character Text Splitting) prioritized for sentence terminators (`.`, `?`, `!`) to avoid splitting mid-sentence and ruining financial context. It actively scrubs forward-looking "Safe Harbor" boilerplate language to clean the dataset.
- **Extractor (`kpi_extractor.py`)**: Uses a separate LLM call mapped strictly to a Pydantic schema to extract the hard numeric KPIs from the "Summary" section of the earnings call, ensuring numbers are completely isolated from text context.

### 2. Dual Storage Layer (Green)
AURA deliberately does NOT put all data in one place. It routes data to where it performs best:
- **ChromaDB**: Holds the `all-MiniLM-L6-v2` dense vectors. Excellent for conceptual questions like *"What is their AI strategy?"*
- **BM25 Lexical Index**: Holds the sparse lexical index. Excellent for exact-match questions like *"How much Azure revenue did Satya Nadella mention?"*
- **SQL Database**: Holds the rigid numbers extracted during ingestion. Completely sidesteps LLM hallucination for questions like *"What was Microsoft's EPS in Q2 2024?"*

### 3. RAG & Agentic Intelligence Core (Yellow)
This is the "Brain" of the platform, built using LangGraph.
- **LangGraph Orchestrator (`orchestrator.py`)**: Maintains conversational memory using a `MemorySaver` checkpointer. Critically, it scrubs old context out to prevent multi-turn hallucination.
- **Query Router (`router.py`)**: Dynamically decides which retrieval strategy to use (e.g., should I run an SQL comparison, or should I do a hybrid search?).
- **RAG Execution Engine (`qa_chain.py`)**: Uses a **3-Layer Quota Allocator** for multi-entity queries. It pulls from both Chroma and BM25, merges the lists using Reciprocal Rank Fusion, and then aggressively reranks them using `ms-marco-MiniLM-L-6-v2` before giving them to the `qwen/qwen3-32b` generation model.

### 4. User-Facing Application (Purple)
The client delivery mechanism that ties the AI architecture to the human user.
- **FastAPI Gateway (`server.py`)**: Translates HTTP REST endpoints into internal LangGraph graph executions and streams the JSON chunks back to the client. It also acts as the secondary prompt enforcer for the `TABLE CITATION SAFETY` rule.
- **Next.js React Client**: A premium dark-luxury frontend that renders the markdown streams (including complex tables), manages the query history via `localStorage`, and displays real-time agent progression steppers.
