# Phase 1 Walkthrough, Challenges & Key Learnings — Finance RAG Project

This document serves as the comprehensive archive for **Phase 1 (RAG Foundation)** of the Financial Earnings Intelligence Platform. It combines the original execution walkthrough, file changes, system metrics, output verifications, engineering challenges encountered, and key architectural solutions implemented.

---

## 📋 1. Goal Alignment Analysis

| Goal / Requirement | Status | Implementation Details |
| :--- | :---: | :--- |
| **Ingestion Pipeline** | ✅ Complete | Parses, chunks, and indexes all 23 transcripts cleanly. Deduplicates automatically on rerun. |
| **Custom Chunking** | ✅ Complete | Solved single-line text issues by using boundary markers to split `summary` and `transcript` segments before chunking. |
| **Metadata Filtering** | ✅ Complete | Dynamically builds metadata filters for ChromaDB query matching based on UI selectors (Company, Year, Quarter, Section). |
| **Dynamic API Key Loading** | ✅ Complete | Bypasses Streamlit resource caching by reading `.env` inside `get_llm()` with `override=True` to ensure key updates reflect immediately. |
| **Groq Integration** | ✅ Complete | Migrated from deprecated `llama-3.1` model to `qwen/qwen3-32b`. |
| **Stateful Thinking Filter** | ✅ Complete | Custom stream parser catches and suppresses `<think>...</think>` tokens, producing clean, professional responses immediately. |
| **Strict Financial Prompting** | ✅ Complete | System prompt explicitly enforces a professional financial analyst persona, forbids hallucination/fabrication, and forces precise citations. |
| **Streamlit User Interface** | ✅ Complete | High-fidelity dark finance-themed interface showing chat history, metadata filters, vector status, and source citation cards. |

---

## 📁 2. Files Created & Modified

| File Path | Purpose |
| :--- | :--- |
| [requirements.txt](requirements.txt) | Relaxed version pins to ensure Python 3.14 compatibility and prevent binary compilation failures. |
| [config/config.yaml](config/config.yaml) | Updated LLM model target to `qwen/qwen3-32b`. |
| [src/ingestion/chunker.py](src/ingestion/chunker.py) | Two-step RCTS chunker with boundary-first sentence segmentation. |
| [src/ingestion/pipeline.py](src/ingestion/pipeline.py) | Solved Unicode encoding errors on Windows terminal by using ASCII arrows instead of raw Unicode symbols. |
| [src/retrieval/vector_store.py](src/retrieval/vector_store.py) | Integrated backward-compatible fallbacks for ChromaDB 1.x collection APIs. |
| [src/generation/prompts.py](src/generation/prompts.py) | Restructured system prompt to enforce direct, professional responses and prohibit conversational filler. |
| [src/generation/qa_chain.py](src/generation/qa_chain.py) | Moved env loading inside the initialization routine and implemented a stateful parser to strip LLM reasoning monologue. |
| [src/ui/app.py](src/ui/app.py) | Streamlit interface with dark finance aesthetic, interactive filters, and side-by-side source panels. |

---

## 📊 3. Ingestion & Retrieval Performance

* **Files Ingested:** 23 / 23 transcripts parsed.
* **Total Chunks Created:** **1,475 chunks** (~289K tokens total).
* **Indexing Time:** 36.4 seconds using local CPU-based embedding extraction (`all-MiniLM-L6-v2`).
* **Vector Database:** ChromaDB storage at `data/chroma_db/`.
* **Embedding Cost:** **$0.00** (fully local).

---

## 🛠️ 4. Challenges Faced & Resolutions

### A. Single-Line Transcript Formatting
* **Challenge:** Every transcript source file is structured as a single massive line of text (40–60KB) without newlines or paragraph breaks. Standard splitters like `RecursiveCharacterTextSplitter` with default parameters failed, either chopping financial figures mid-sentence or creating monolithic chunks.
* **Solution:** We designed a two-step parsing strategy inside `src/ingestion/chunker.py`:
  1. We split the document at the section boundary `"[ "` to divide it into a prose `summary` segment and a live Q&A `transcript` segment.
  2. We applied RCTS with customized separators prioritizing sentence-boundary punctuation: `[". ", "? ", "! ", "; ", " ", ""]`.
* **Learning:** Never assume text dataset files have uniform newline delimiters. Always audit the raw corpus structure before writing chunking pipelines.

### B. Windows Unicode Encoding Errors
* **Challenge:** Printing pipeline progress using visual arrow indicators (like `→` and `✔`) threw a `UnicodeEncodeError: 'charmap' codec can't encode characters` in PowerShell. This is because PowerShell defaults to legacy ANSI/CP1252 character maps for stdout streams.
* **Solution:** We replaced all decorative Unicode symbols in the console logging pipeline with ASCII equivalents (`-->`, `[OK]`, `[ERROR]`) and explicitly configured the startup server environment variables.
* **Learning:** Cross-platform CLI utilities should stick to standard ASCII or explicitly capture encoding wrappers to prevent execution crashes on Windows machines.

### C. Streamlit Cache vs. Dynamic API Key Updates
* **Challenge:** Streamlit caches module-level imports. When a user updated their Groq API key in `config/.env`, the changes did not reflect in the application without restarting the entire server process.
* **Solution:** We modified `get_llm()` inside `src/generation/qa_chain.py` to evaluate `load_dotenv` dynamically with `override=True` at runtime upon every function call, instead of loading it at module import time.
* **Learning:** In interactive or server environments (like Streamlit or FastAPI), credentials and environment configurations must be loaded dynamically at the function execution level rather than evaluated once globally.

### D. Decommissioned API Model Endpoint
* **Challenge:** The originally planned `llama-3.1-70b-versatile` model was decommissioned from Groq's active API endpoints, causing query requests to fail.
* **Solution:** We decoupled model configurations in `config/config.yaml` and shifted the system to use the active `qwen/qwen3-32b` reasoning model.
* **Learning:** Third-party API lifecycles are brief. Always centralize model configurations in standard YAML/JSON structures rather than hardcoding string identifiers inside source code files.

### E. Reasoning Model Internal Monologues (`<think>` tags)
* **Challenge:** Qwen3 is a reasoning model that outputs internal monologues (e.g., *"Okay, let's tackle this..."*) which were returned as text and displayed in the UI.
* **Solution:** We implemented output-level filtering:
  - **For standard queries:** We added a regex strip `re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)` to clean the final string.
  - **For streaming queries:** We wrote a stateful generator loop that buffers incoming text, detects the `<think>` tag, suppresses yielding while inside it, and starts streaming normally once the `</think>` tag is detected.
* **Learning:** Modern reasoning models require dedicated cleaning wrappers on both synchronous and streaming channels to isolate clean final output from reasoning traces.

### F. Disappearing Source Citation Cards
* **Challenge:** In Streamlit, calling `st.rerun()` resets the execution flow. When citation cards were drawn only within the block that processed a submitted query, they disappeared immediately after the rerun completed, leaving the panel blank.
* **Solution:** We refactored `src/ui/app.py` to decouple query execution from rendering. Citations are stored inside `st.session_state.chat_history`. A dedicated rendering loop reads the last assistant message in history on every rerun and populates the source panel.
* **Learning:** Always separate data-fetching side-effects from interface rendering states in reactive web runtimes.

---

## 🧪 5. Q&A Output Verification (Qwen3-32B)

### Query 1: actual vs guidance (Q3 2024 Gross Margin)
* **Question:** *"What was Apple's gross margin in Q3 2024?"*
* **Response:** *"The provided context does not contain sufficient information to answer this question. The available passages cover Apple's Q4 2023, Q1 2024, and Q2 2024 results, with Q3 2024 only mentioned in Q2 2024 guidance (projecting gross margin between 45.5% and 46.5%) but no actual Q3 2024 gross margin figures reported. [Apple | Q2 | 2024 | summary]"*
* **Verification Result:** Passed. Recognizes that only forward-looking guidance exists in the dataset and avoids fabricating the actual number.

### Query 2: open-ended synthesis (Apple Market Position)
* **Question:** *"Tell me about the current market position of Apple Company"*
* **Response:** *"Apple's current market position reflects strong performance in key segments and strategic expansion. The Services business and iPad revenue are projected to maintain double-digit growth [Apple | Q2 | 2024 | summary]. The Mac segment achieved $7.5 billion in revenue, a 4% year-over-year increase, driven by new MacBook Air models [Apple | Q2 | 2024 | summary]..."*
* **Verification Result:** Passed. Delivers direct, cited statements from transcripts without conversational intro filler or thinking monologue.

---

## 🎨 6. Verified UI Interface

Below is the verified state of the Q&A interface using `qwen/qwen3-32b` with all thinking tokens filtered and citations rendering persistently on the right side:

![Verified Clean UI with Qwen3 Q&A](./assets/qwen3_citations_ui.png)

---

## 🚀 7. Transitioning to Phase 2

All Phase 1 features are fully stable. The next phase will elevate the system from a naive retrieval pipeline to an advanced hybrid platform:

1. **BM25 Retrieval:** Introduce `src/retrieval/bm25_retriever.py` for exact keyword/ticker searches.
2. **Hybrid RRF Search:** Implement reciprocal rank fusion to merge vector semantic matching with BM25 keyword matching.
3. **Cross-Encoder Reranking:** Add `src/retrieval/reranker.py` to rank retrieved chunks using a local mini LM cross-encoder.
4. **Automated Evaluation:** Build `src/evaluation/ragas_eval.py` to calculate Faithfulness, Answer Relevance, and Context Recall.
