# Phase 2 Challenges, Solutions, & Architectural Learnings

During the implementation of Phase 2 (Advanced Hybrid Retrieval, Evaluation, 
and Routing), several critical engineering challenges were encountered. This 
document captures those issues, the solutions engineered, and the architectural 
lessons learned.

---

## 1. Low-Value Boilerplate & Operator Noise (Red Flag #1)

**The Problem**: Earnings transcripts contain substantial greeting boilerplate,
operator call setups, participant introductions, and Safe Harbor legal 
disclaimers. Because keyword (BM25) and semantic (vector) searches match 
high-frequency terms like "Apple", "revenue", and "guidance", these boilerplate 
chunks were ranked highly, diluting the quality of context fed to the LLM.

**The Solution**: Created a sentence-level preprocessing step 
`clean_transcript_text` in `chunker.py`. This splits transcript texts into 
sentences and filters out any sentence matching common operator, introduction, 
or Safe Harbor patterns.

**The Outcome**: Running the ingestion pipeline with `--force-reset` rebuilt 
the search databases with cleaned text, shrinking the total index size from 
**1,475** to **1,434** chunks (wiping out **41** junk chunks) while retaining 
100% of analytical facts.

---

## 2. Duplicate Chunk Leakage (Red Flag #2)

**The Problem**: The raw transcripts (e.g. `2024_Q3_aapl_processed.txt`) 
contained duplicated prepared remarks (the investor relations setup and 
forward-looking statement disclaimer were duplicated at the beginning and right 
before the Q&A section). This caused identical safe harbor chunks to appear 
twice under different indices, wasting context space.

**The Solution**: The sentence-level boilerplate filtering resolved this by 
removing these sentences globally, preventing them from being chunked under 
any section.

**The Learning**: Structural anomalies and duplications are common in 
transcripts. Preprocessing text *before* split boundaries is much more 
effective than relying on vector deduplication alone.

---

## 3. Reranker Entity Starvation in Multi-Entity Queries (Red Flag #3)

**The Problem**: In multi-entity comparison queries (e.g., "What are the key 
insights about all companies?"), one dominant company (e.g. Nvidia, due to 
high-value AI signal density in chunks) would crowd out other companies during 
the initial retrieval step. The cross-encoder reranker would then assign lower 
relevance scores to Microsoft/Apple chunks and push them below the cutoff rank, 
causing the LLM to incorrectly report "no sufficient information available" for 
those companies — even though their data existed in the index.

This was first observed when testing with `top_k=12`, where Microsoft was 
completely absent from the final context despite being a known indexed entity. 
Increasing to `top_k=20` partially mitigated it but did not structurally 
resolve the root cause.

**Root Cause Analysis**: The cross-encoder reranker was functioning correctly 
— it was scoring chunks by relevance signal strength. The issue was 
architectural: a single shared retrieval pool allowed high-signal entities to 
dominate before the reranker even had a chance to balance representation. The 
problem was not in the reranker, but in feeding it an already-biased pool.

**The Solution**: Implemented a two-layer fix:

*Layer 1 — Per-Entity Guaranteed Retrieval (Retrieval Layer)*

When the router detects a multi-entity or comparative query, the system 
retrieves `top_k=6` chunks independently per entity, then merges all pools 
before passing to the cross-encoder reranker. This guarantees minimum 
representation for every detected company before reranking occurs.

```python
# Multi-entity path
entities = ["Apple", "Nvidia", "Microsoft"]
per_entity_chunks = [
    retriever.get_top_k(query, filter=entity, k=6) 
    for entity in entities
]
merged_pool = flatten(per_entity_chunks)
reranked = cross_encoder.rerank(merged_pool, query)

# Single-entity path (untouched)
chunks = retrieve_standard(query, k=20)
```

*Layer 2 — Conditional Router Branch (Orchestration Layer)*

Extended the existing query router to detect multi-entity intent — both 
explicit (named companies) and implicit ("all companies", "compare", 
"across the board") — and branch retrieval strategy accordingly. 
Single-entity and general queries continue using the standard pipeline 
with zero changes to existing behavior.

```python
def detect_entities(query: str) -> list[str]:
    known_companies = ["Apple", "AAPL", "Nvidia", "NVDA", "Microsoft", "MSFT"]
    return [c for c in known_companies if c.lower() in query.lower()]

detected = detect_entities(query)
is_multi_entity = len(detected) > 1 or any(
    phrase in query.lower() 
    for phrase in ["all companies", "compare", "across companies"]
)
```

*Layer 3 — Coverage Verification (Generation Layer)*

Added a post-retrieval coverage check before passing chunks to the response 
agent. If any detected entity has zero chunks in the final pool, it is 
explicitly flagged rather than silently skipped or hallucinated.

**The Outcome**: Balanced context representation guaranteed before reranking. 
The diagnostics panel now shows **Retrieval Coverage (Before Rerank)** vs. 
**Reranker Selection (After Rerank)** counts per entity for full transparency.

**The Learning**: Reranker starvation is a structural retrieval problem, not 
a prompt problem. Fixing it at the prompt/generation layer is treating a 
symptom. The correct fix is always upstream — guarantee representation before 
the reranker sees the pool. The router is the right decision hub for this 
branch, since it already classifies query intent.

---

## 4. LaTeX Stream-Parsing Conflicts in Streamlit UI

**The Problem**: Streamlit's markdown renderer interprets raw dollar signs 
(`$69.7 billion`) as LaTeX inline math delimiters. When the LLM outputs dollar 
values, Streamlit wraps the characters vertically or outputs formatting errors.

**The Solution**:
1. Updated the generation `SYSTEM_PROMPT` in `prompts.py` to enforce writing 
   currency as `USD X billion` or `USD Y million` instead of using the `$` 
   sign, and explicitly forbid LaTeX math wrappers.
2. Implemented regex replacements in the stream processor within 
   `get_answer_streaming` in `qa_chain.py` to replace raw `$` signs with 
   `USD ` dynamically.

---

## 5. Temporal Scope Narrowing in Query Condensation

**The Problem**: The query rewriter in `query_transformer.py` would compress 
conversational questions like *"What about the performance so far?"* into a 
single specific quarter (e.g. *"What was Nvidia's revenue in Q3 2024?"*), 
narrowing the user's intent.

**The Solution**: Modified the condensation prompt to instruct the transformer 
to preserve broad historical ranges, trends, and multi-period queries 
(e.g. "so far", "all periods").

---

## 6. Windows SQLite Database Locks

**The Problem**: Multiple instances of the Streamlit server running in parallel 
on Windows locked the file-based ChromaDB SQLite connection. This caused 
`vector_store.count()` to fail, leading the UI to report 
`-1 Chunks (Vector Store Empty)`.

**The Solution**: Terminated conflicting background python processes and added 
explicit exception logging to `vector_store.py` to prevent silent failures.