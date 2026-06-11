"""
src/generation/qa_chain.py
==========================
RAG chain with Groq LLM + ChromaDB retrieval + citation support.

Architecture (Phase 1 — vector-only):
  Query
    │
    ▼
  ChromaDB similarity_search (top-5 chunks, optional metadata filter)
    │
    ▼
  Format context block (number passages + add citations)
    │
    ▼
  Groq qwen/qwen3-32b (temperature=0, max_tokens=1024)
    │
    ▼
  Response + source_documents list

This module exposes a single public function: get_answer()
which is called by the Streamlit UI and by evaluation scripts.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.vector_store import EarningsVectorStore, build_metadata_filter
from src.generation.prompts import (
    RAG_QA_PROMPT,
    format_context_block,
    format_source_citation,
)
from src.retrieval.bm25_retriever import EarningsBM25Retriever
from src.retrieval.reranker import EarningsReranker

logger = logging.getLogger(__name__)

# .env paths searched in order (absolute path first for reliability)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATHS = [
    _PROJECT_ROOT / "config" / ".env",
    _PROJECT_ROOT / ".env",
    Path("config/.env"),
    Path(".env"),
]

def ensure_env_loaded() -> None:
    """Ensure environment variables are loaded from the configured .env paths."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        for _env_path in _ENV_PATHS:
            if _env_path.exists():
                load_dotenv(dotenv_path=_env_path, override=True)
                logger.info(f"Loaded .env from: {_env_path}")
                break

# ---------------------------------------------------------------------------
# Cached singletons for BM25, Reranker, Router, Transformer
# ---------------------------------------------------------------------------
_bm25_retriever: Optional[EarningsBM25Retriever] = None
_reranker: Optional[EarningsReranker] = None
_router: Optional[object] = None
_query_transformer: Optional[object] = None


def get_bm25_retriever() -> EarningsBM25Retriever:
    """Return the shared BM25 retriever instance, loading index from disk."""
    global _bm25_retriever
    if _bm25_retriever is None:
        index_dir = _PROJECT_ROOT / "data" / "bm25_index"
        _bm25_retriever = EarningsBM25Retriever(index_dir=str(index_dir))
        success = _bm25_retriever.load()
        if not success:
            logger.warning("BM25 index file not found. Please run ingestion pipeline first.")
    return _bm25_retriever


def get_reranker() -> EarningsReranker:
    """Return the shared Cross-Encoder reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = EarningsReranker(device="cpu")
    return _reranker


def get_router():
    """Return the shared QueryRouter instance."""
    global _router
    if _router is None:
        ensure_env_loaded()
        from src.retrieval.router import QueryRouter
        # Always use the fast, high-rate-limit qwen/qwen3-32b model for metadata routing
        api_key = os.getenv("GROQ_API_KEY")
        llm_router = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.0,
            api_key=api_key
        )
        _router = QueryRouter(llm=llm_router)
    return _router


def get_query_transformer():
    """Return the shared QueryTransformer instance."""
    global _query_transformer
    if _query_transformer is None:
        ensure_env_loaded()
        from src.retrieval.query_transformer import QueryTransformer
        api_key = os.getenv("GROQ_API_KEY")
        llm_transformer = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.0,
            api_key=api_key
        )
        _query_transformer = QueryTransformer(llm=llm_transformer)
    return _query_transformer


from langchain_core.documents import Document

def fuse_multiple_doc_lists(lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Fuse multiple Document lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    ranks_info: dict[str, dict[str, int]] = {}

    for doc_list in lists:
        for rank, doc in enumerate(doc_list):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
            if chunk_id not in ranks_info:
                ranks_info[chunk_id] = {}
            if "vector_rank" in doc.metadata and doc.metadata["vector_rank"] != -1:
                ranks_info[chunk_id]["vector_rank"] = doc.metadata["vector_rank"]
            if "bm25_rank" in doc.metadata and doc.metadata["bm25_rank"] != -1:
                ranks_info[chunk_id]["bm25_rank"] = doc.metadata["bm25_rank"]

    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_docs: list[Document] = []
    for final_rank, chunk_id in enumerate(sorted_chunk_ids, start=1):
        original_doc = doc_map[chunk_id]
        meta = original_doc.metadata.copy()
        meta["rrf_score"] = float(rrf_scores[chunk_id])
        meta["rrf_rank"] = final_rank
        meta["vector_rank"] = ranks_info[chunk_id].get("vector_rank", -1)
        meta["bm25_rank"] = ranks_info[chunk_id].get("bm25_rank", -1)
        fused_docs.append(Document(page_content=original_doc.page_content, metadata=meta))
        
    return fused_docs


# ---------------------------------------------------------------------------
# Load environment variables (Groq API key)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_MODEL       = "qwen/qwen3-32b"   # Groq model
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 1024

# ---------------------------------------------------------------------------
# Singleton LLM instance
# ---------------------------------------------------------------------------
_llm: Optional[ChatGroq] = None


def get_llm() -> ChatGroq:
    """Return the shared Groq LLM instance (initialised on first call)."""
    global _llm
    if _llm is None:
        ensure_env_loaded()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set or still placeholder. "
                "Please add your key to config/.env:\n"
                "  GROQ_API_KEY=gsk_your_actual_key_here\n"
                "Get a free key at: https://console.groq.com/"
            )
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=api_key,
        )
        logger.info(f"Groq LLM initialised: model={LLM_MODEL}, temperature={LLM_TEMPERATURE}")
    return _llm


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def get_answer(
    question: str,
    vector_store: EarningsVectorStore,
    k: int = 12,
    retrieval_mode: str = "auto",
    company: Optional[str] = None,
    ticker: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    section: Optional[str] = None,
    return_context_only: bool = False,
    chat_history: Optional[list[dict]] = None,
) -> dict:
    """
    Run a RAG query against the earnings transcript vector store.

    Parameters
    ----------
    question : str
        The user's natural language question.
    vector_store : EarningsVectorStore
        Initialised vector store (must be populated via pipeline.py first).
    k : int
        Number of context chunks to retrieve (default: 5).
    retrieval_mode : str
        Retrieval mode: "auto", "vector", "bm25", "hybrid", or "rerank".
    company : str | None
        Optional filter: "Apple", "Microsoft", or "Nvidia".
    ticker : str | None
        Optional filter: "AAPL", "MSFT", "NVDA".
    year : int | None
        Optional filter: 2023 or 2024.
    quarter : str | None
        Optional filter: "Q1", "Q2", "Q3", or "Q4".
    section : str | None
        Optional filter: "summary" or "transcript".
    chat_history : list | None
        Optional conversational chat history list.

    Returns
    -------
    dict with keys:
        "answer"          : str  — the LLM's response
        "source_documents": list — retrieved Document objects
        "citations"       : list[str] — formatted citation strings
        "context_block"   : str  — the context sent to the LLM
        "question"        : str  — the original question
        "filter_applied"  : dict | None — the metadata filter used
        "routing_mode"     : str — active mode used
        "routing_strategy" : str — strategy selected
        "routing_reason"   : str — routing reasoning
        "rewritten_query"  : str | None — rewritten query if history is resolved
        "multi_queries"    : list[str] | None — query variations
    """
    # Build metadata filter from UI selections
    metadata_filter = build_metadata_filter(
        company=company,
        ticker=ticker,
        year=year,
        quarter=quarter,
        section=section,
    )

    # 1. Query Rewrite (conversational context)
    search_query = question
    rewritten = False
    if chat_history:
        # Exclude current question if it is in the history dict
        history_to_pass = [h for h in chat_history if h.get("content") != question]
        if history_to_pass:
            transformer = get_query_transformer()
            search_query = transformer.rewrite_query(question, history_to_pass)
            rewritten = (search_query != question)

    # 2. Query Routing
    active_mode = retrieval_mode
    routing_strategy = "fixed"
    routing_reason = "Manual override"

    if retrieval_mode == "auto":
        router = get_router()
        route_info = router.route_query(search_query)
        active_mode = route_info["mode"]
        routing_strategy = route_info["strategy"]
        routing_reason = route_info.get("reason", "Dynamic routing classification")

    # If routing decided summary section, restrict section search to "summary"
    if routing_strategy == "summary_section" and not section:
        metadata_filter = build_metadata_filter(
            company=company, ticker=ticker, year=year, quarter=quarter, section="summary"
        )

    router_inst = get_router()

    # Resolve entity filters for single_entity_* strategies if no explicit company filter is selected in the UI
    if routing_strategy in ["single_entity_financial_summary", "single_entity_financial_metric", "single_entity_risk_analysis"]:
        target_co = company
        if not target_co:
            cos = router_inst.detect_entities(search_query)
            if cos and len(cos) == 1:
                target_co = cos[0]
        if target_co:
            if routing_strategy == "single_entity_financial_summary":
                # Do not hard-clear year and quarter if the user explicitly provided them.
                metadata_filter = build_metadata_filter(
                    company=target_co,
                    ticker=ticker,
                    year=year,
                    quarter=quarter,
                    section=section,
                )
            else:
                # Specific metric or risk, keep user's year/quarter filters
                metadata_filter = build_metadata_filter(
                    company=target_co,
                    ticker=None,
                    year=year,
                    quarter=quarter,
                    section=section,
                )

    # 3. Document Retrieval
    source_docs = []
    coverage_before = {}
    queries_run = [search_query]

    # Setup retrieval_query focusing on summary/risk terms
    retrieval_query = search_query
    if routing_strategy == "single_entity_financial_summary":
        retrieval_query = f"{search_query} revenue growth segment guidance"
    elif routing_strategy in ["multi_entity_risk_analysis", "single_entity_risk_analysis"]:
        retrieval_query = f"{search_query} risk factors headwinds challenges regulatory risk competition macroeconomic"

    # Run entity detection
    is_multi, detected_cos = router_inst.is_multi_entity(search_query)

    # If the user has explicitly filtered by a company in the sidebar, restrict to that company and disable multi-entity retrieval
    if company:
        detected_cos = [company]
        is_multi = False

    if is_multi and len(detected_cos) > 1:
        # Per-entity retrieval -> merge -> rerank with guaranteed equal representation
        exact_per_entity = max(1, k // len(detected_cos))
        # Use a much larger buffer (3x quota) so each entity has enough candidates
        # to survive reranking without being starved
        k_per_entity = max(8, exact_per_entity * 3)
        merged_candidates = []
        seen_chunk_ids = set()
        # Track per-entity candidate pools (before merging) for fallback
        entity_candidate_pools: dict[str, list] = {ent: [] for ent in detected_cos}
        
        bm25 = get_bm25_retriever()
        from src.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(vector_store, bm25)
        
        for ent in detected_cos:
            ent_filter = build_metadata_filter(
                company=ent,
                ticker=None,
                year=year,
                quarter=quarter,
                section=section,
            )
            
            # Retrieve chunks for this entity using the active mode
            if active_mode == "vector":
                ent_docs = vector_store.similarity_search(
                    query=retrieval_query,
                    k=k_per_entity,
                    filter=ent_filter,
                )
                for idx, d in enumerate(ent_docs):
                    d.metadata["vector_rank"] = idx + 1
            elif active_mode == "bm25":
                ent_docs = bm25.retrieve(
                    query=retrieval_query,
                    k=k_per_entity,
                    filter_dict=ent_filter,
                )
            else:  # hybrid or rerank
                ent_docs = hybrid.retrieve(
                    query=retrieval_query,
                    k=k_per_entity,
                    candidate_count=max(30, k_per_entity + 10),
                    filter_dict=ent_filter,
                )
                
            for doc in ent_docs:
                chunk_id = doc.metadata.get("chunk_id") or doc.page_content
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_candidates.append(doc)
                entity_candidate_pools[ent].append(doc)
                    
        # Rerank all merged candidates (full pool) to score them
        rerank_model = get_reranker()
        coverage_before = {}
        for d in merged_candidates:
            co = d.metadata.get("company", "Unknown")
            coverage_before[co] = coverage_before.get(co, 0) + 1
        # Score all candidates but keep all of them
        reranked_all = rerank_model.rerank(
            query=search_query,
            docs=merged_candidates,
            top_n=len(merged_candidates),
        )
        
        # Build a per-entity ranked list from reranked_all
        entity_ranked: dict[str, list] = {ent: [] for ent in detected_cos}
        for doc in reranked_all:
            co = doc.metadata.get("company", "Unknown")
            if co in entity_ranked:
                entity_ranked[co].append(doc)

        # --- Strict equal allocation ---
        # First pass: take top `exact_per_entity` chunks per entity (by rerank score)
        source_docs = []
        entity_used: dict[str, list] = {ent: [] for ent in detected_cos}
        for ent in detected_cos:
            pool = entity_ranked[ent][:exact_per_entity]
            entity_used[ent] = pool
            source_docs.extend(pool)
            logger.info(f"[multi-entity] {ent}: first-pass allocated {len(pool)}/{exact_per_entity} chunks")

        # Second pass: fill remaining budget via entity-aware round-robin
        # (cycling through entities to keep allocation balanced)
        remaining_budget = k - len(source_docs)
        if remaining_budget > 0:
            used_set = set(id(d) for d in source_docs)
            entity_overflow_pools = {
                ent: [d for d in entity_ranked[ent] if id(d) not in used_set]
                for ent in detected_cos
            }
            # Round-robin across entities
            rr_idx = 0
            entities_list = list(detected_cos)
            added = 0
            while added < remaining_budget:
                ent = entities_list[rr_idx % len(entities_list)]
                pool = entity_overflow_pools[ent]
                if pool:
                    doc = pool.pop(0)
                    if id(doc) not in used_set:
                        source_docs.append(doc)
                        used_set.add(id(doc))
                        added += 1
                rr_idx += 1
                # Safety: if all entity pools are empty, break
                if all(len(p) == 0 for p in entity_overflow_pools.values()):
                    break

        # Group final docs by entity for LLM coherence (Apple block, then MSFT, then NVDA)
        grouped_docs = []
        for ent in detected_cos:
            for doc in source_docs:
                if doc.metadata.get("company") == ent:
                    grouped_docs.append(doc)
        source_docs = grouped_docs

        final_coverage = {}
        for d in source_docs:
            co = d.metadata.get("company", "Unknown")
            final_coverage[co] = final_coverage.get(co, 0) + 1
        logger.info(f"[multi-entity] Final coverage after allocation: {final_coverage}")
        routing_strategy = f"multi_entity_retrieval ({', '.join(detected_cos)})"
    else:
        if routing_strategy == "multi_query" or routing_strategy == "comparison_query":
            # Multi-query expansion route
            transformer = get_query_transformer()
            queries_run = transformer.generate_multi_queries(search_query)
            bm25 = get_bm25_retriever()
            from src.retrieval.hybrid_retriever import HybridRetriever
            hybrid = HybridRetriever(vector_store, bm25)

            all_candidates = []
            for q in queries_run:
                cands = hybrid.retrieve(
                    query=q,
                    k=max(20, k + 10),
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
                all_candidates.append(cands)

            fused_candidates = fuse_multiple_doc_lists(all_candidates, k=60)
            rerank_model = get_reranker()
            coverage_before = {}
            for d in fused_candidates:
                co = d.metadata.get("company", "Unknown")
                coverage_before[co] = coverage_before.get(co, 0) + 1
            source_docs = rerank_model.rerank(
                query=search_query,
                docs=fused_candidates,
                top_n=k,
            )
        else:
            # Standard retrieval routes
            if active_mode == "vector":
                source_docs = vector_store.similarity_search(
                    query=retrieval_query,
                    k=k,
                    filter=metadata_filter,
                )
                for idx, doc in enumerate(source_docs):
                    doc.metadata["vector_rank"] = idx + 1
            elif active_mode == "bm25":
                bm25 = get_bm25_retriever()
                source_docs = bm25.retrieve(
                    query=retrieval_query,
                    k=k,
                    filter_dict=metadata_filter,
                )
            elif active_mode == "hybrid":
                bm25 = get_bm25_retriever()
                from src.retrieval.hybrid_retriever import HybridRetriever
                hybrid = HybridRetriever(vector_store, bm25)
                source_docs = hybrid.retrieve(
                    query=retrieval_query,
                    k=k,
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
            elif active_mode == "rerank":
                bm25 = get_bm25_retriever()
                from src.retrieval.hybrid_retriever import HybridRetriever
                hybrid = HybridRetriever(vector_store, bm25)
                candidates = hybrid.retrieve(
                    query=retrieval_query,
                    k=max(20, k + 10),
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
                rerank_model = get_reranker()
                coverage_before = {}
                for d in candidates:
                    co = d.metadata.get("company", "Unknown")
                    coverage_before[co] = coverage_before.get(co, 0) + 1
                source_docs = rerank_model.rerank(
                    query=search_query,
                    docs=candidates,
                    top_n=k,
                )
            else:
                raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    if not coverage_before and source_docs:
        for d in source_docs:
            co = d.metadata.get("company", "Unknown")
            coverage_before[co] = coverage_before.get(co, 0) + 1

    if not source_docs:
        return {
            "answer": "No relevant context found. Please adjust your filters.",
            "source_documents": [],
            "citations":        [],
            "context_block":    "",
            "question":         question,
            "filter_applied":   metadata_filter,
            "routing_mode":     active_mode,
            "routing_strategy": routing_strategy,
            "routing_reason":   routing_reason,
            "rewritten_query":  search_query if rewritten else None,
            "multi_queries":    queries_run if routing_strategy in ["multi_query", "comparison_query"] else None,
            "detected_entities": detected_cos if is_multi else None,
            "coverage_before_rerank": coverage_before,
        }

    context_block = format_context_block(source_docs)
    
    if return_context_only:
        answer = context_block
    else:
        llm = get_llm()
        chain = (
            RunnablePassthrough()
            | RAG_QA_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({
            "context":  context_block,
            "question": search_query,
        })
        # Clean up think tags and format dollar values
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        answer = answer.replace("<think>", "").replace("</think>", "").strip()
        answer = answer.replace("$", "USD ")
        answer = answer.replace("USD USD ", "USD ")
        answer = answer.replace("USD  USD ", "USD ")
    
    citations = [format_source_citation(doc.metadata) for doc in source_docs]

    return {
        "answer":           answer.strip(),
        "source_documents": source_docs,
        "citations":        citations,
        "context_block":    context_block,
        "question":         question,
        "filter_applied":   metadata_filter,
        "routing_mode":     active_mode,
        "routing_strategy": routing_strategy,
        "routing_reason":   routing_reason,
        "rewritten_query":  search_query if rewritten else None,
        "multi_queries":    queries_run if routing_strategy in ["multi_query", "comparison_query"] else None,
        "detected_entities": detected_cos if is_multi else None,
        "coverage_before_rerank": coverage_before,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Streaming variant (used by Streamlit for real-time output)
# ---------------------------------------------------------------------------
def get_answer_streaming(
    question: str,
    vector_store: EarningsVectorStore,
    k: int = 12,
    retrieval_mode: str = "auto",
    company: Optional[str] = None,
    ticker: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    section: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
):
    """
    Generator version of get_answer() that yields answer tokens for
    Streamlit's st.write_stream() interface, filtering out <think> blocks.
    Also resolves conversational rewrites and dynamic query routing.

    Yields
    ------
    str
        Token chunks from the LLM stream.
    """
    # Build filter
    metadata_filter = build_metadata_filter(
        company=company,
        ticker=ticker,
        year=year,
        quarter=quarter,
        section=section,
    )

    # 1. Query Rewrite (conversational context)
    search_query = question
    rewritten = False
    if chat_history:
        # Exclude current question if it is in the history dict
        history_to_pass = [h for h in chat_history if h.get("content") != question]
        if history_to_pass:
            transformer = get_query_transformer()
            search_query = transformer.rewrite_query(question, history_to_pass)
            rewritten = (search_query != question)

    # 2. Query Routing
    active_mode = retrieval_mode
    routing_strategy = "fixed"
    routing_reason = "Manual override"

    if retrieval_mode == "auto":
        router = get_router()
        route_info = router.route_query(search_query)
        active_mode = route_info["mode"]
        routing_strategy = route_info["strategy"]
        routing_reason = route_info.get("reason", "Dynamic routing classification")

    # If routing decided summary section, restrict section search to "summary"
    if routing_strategy == "summary_section" and not section:
        metadata_filter = build_metadata_filter(
            company=company, ticker=ticker, year=year, quarter=quarter, section="summary"
        )

    router_inst = get_router()

    # Resolve entity filters for single_entity_* strategies if no explicit company filter is selected in the UI
    if routing_strategy in ["single_entity_financial_summary", "single_entity_financial_metric", "single_entity_risk_analysis"]:
        target_co = company
        if not target_co:
            cos = router_inst.detect_entities(search_query)
            if cos and len(cos) == 1:
                target_co = cos[0]
        if target_co:
            if routing_strategy == "single_entity_financial_summary":
                # Broad summary across all quarters/years, ignore year/quarter
                metadata_filter = build_metadata_filter(
                    company=target_co,
                    ticker=None,
                    year=None,
                    quarter=None,
                    section=section,
                )
            else:
                # Specific metric or risk, keep user's year/quarter filters
                metadata_filter = build_metadata_filter(
                    company=target_co,
                    ticker=None,
                    year=year,
                    quarter=quarter,
                    section=section,
                )

    # 3. Document Retrieval
    source_docs = []
    coverage_before = {}
    queries_run = [search_query]

    # Setup retrieval_query focusing on summary/risk terms
    retrieval_query = search_query
    if routing_strategy == "single_entity_financial_summary":
        retrieval_query = f"{search_query} revenue growth segment guidance"
    elif routing_strategy in ["multi_entity_risk_analysis", "single_entity_risk_analysis"]:
        retrieval_query = f"{search_query} risk factors headwinds challenges regulatory risk competition macroeconomic"

    # Run entity detection
    is_multi, detected_cos = router_inst.is_multi_entity(search_query)

    # If the user has explicitly filtered by a company in the sidebar, restrict to that company and disable multi-entity retrieval
    if company:
        detected_cos = [company]
        is_multi = False

    if is_multi and len(detected_cos) > 1:
        # Per-entity retrieval -> merge -> rerank with guaranteed equal representation
        exact_per_entity = max(1, k // len(detected_cos))
        # Use a much larger buffer (3x quota) so each entity has enough candidates
        # to survive reranking without being starved
        k_per_entity = max(8, exact_per_entity * 3)
        merged_candidates = []
        seen_chunk_ids = set()
        
        bm25 = get_bm25_retriever()
        from src.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(vector_store, bm25)
        
        for ent in detected_cos:
            ent_filter = build_metadata_filter(
                company=ent,
                ticker=None,
                year=year,
                quarter=quarter,
                section=section,
            )
            
            # Retrieve chunks for this entity using the active mode
            if active_mode == "vector":
                ent_docs = vector_store.similarity_search(
                    query=retrieval_query,
                    k=k_per_entity,
                    filter=ent_filter,
                )
                for idx, d in enumerate(ent_docs):
                    d.metadata["vector_rank"] = idx + 1
            elif active_mode == "bm25":
                ent_docs = bm25.retrieve(
                    query=retrieval_query,
                    k=k_per_entity,
                    filter_dict=ent_filter,
                )
            else:  # hybrid or rerank
                ent_docs = hybrid.retrieve(
                    query=retrieval_query,
                    k=k_per_entity,
                    candidate_count=max(30, k_per_entity + 10),
                    filter_dict=ent_filter,
                )
                
            for doc in ent_docs:
                chunk_id = doc.metadata.get("chunk_id") or doc.page_content
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_candidates.append(doc)
                    
        # Rerank all merged candidates (full pool) to score them
        rerank_model = get_reranker()
        coverage_before = {}
        for d in merged_candidates:
            co = d.metadata.get("company", "Unknown")
            coverage_before[co] = coverage_before.get(co, 0) + 1
        # Score all candidates but keep all of them
        reranked_all = rerank_model.rerank(
            query=search_query,
            docs=merged_candidates,
            top_n=len(merged_candidates),
        )

        # Build a per-entity ranked list from reranked_all
        entity_ranked: dict[str, list] = {ent: [] for ent in detected_cos}
        for doc in reranked_all:
            co = doc.metadata.get("company", "Unknown")
            if co in entity_ranked:
                entity_ranked[co].append(doc)

        # --- Strict equal allocation ---
        # First pass: take top `exact_per_entity` chunks per entity (by rerank score)
        source_docs = []
        for ent in detected_cos:
            pool = entity_ranked[ent][:exact_per_entity]
            source_docs.extend(pool)
            logger.info(f"[stream multi-entity] {ent}: first-pass allocated {len(pool)}/{exact_per_entity} chunks")

        # Second pass: fill remaining budget via entity-aware round-robin
        remaining_budget = k - len(source_docs)
        if remaining_budget > 0:
            used_set = set(id(d) for d in source_docs)
            entity_overflow_pools = {
                ent: [d for d in entity_ranked[ent] if id(d) not in used_set]
                for ent in detected_cos
            }
            rr_idx = 0
            entities_list = list(detected_cos)
            added = 0
            while added < remaining_budget:
                ent = entities_list[rr_idx % len(entities_list)]
                pool = entity_overflow_pools[ent]
                if pool:
                    doc = pool.pop(0)
                    if id(doc) not in used_set:
                        source_docs.append(doc)
                        used_set.add(id(doc))
                        added += 1
                rr_idx += 1
                if all(len(p) == 0 for p in entity_overflow_pools.values()):
                    break

        # Group final docs by entity for LLM coherence
        grouped_docs = []
        for ent in detected_cos:
            for doc in source_docs:
                if doc.metadata.get("company") == ent:
                    grouped_docs.append(doc)
        source_docs = grouped_docs

        final_coverage = {}
        for d in source_docs:
            co = d.metadata.get("company", "Unknown")
            final_coverage[co] = final_coverage.get(co, 0) + 1
        logger.info(f"[stream multi-entity] Final coverage after allocation: {final_coverage}")
        routing_strategy = f"multi_entity_retrieval ({', '.join(detected_cos)})"
    else:
        if routing_strategy == "multi_query" or routing_strategy == "comparison_query":
            # Multi-query expansion route
            transformer = get_query_transformer()
            queries_run = transformer.generate_multi_queries(search_query)
            bm25 = get_bm25_retriever()
            from src.retrieval.hybrid_retriever import HybridRetriever
            hybrid = HybridRetriever(vector_store, bm25)

            all_candidates = []
            for q in queries_run:
                cands = hybrid.retrieve(
                    query=q,
                    k=max(20, k + 10),
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
                all_candidates.append(cands)

            fused_candidates = fuse_multiple_doc_lists(all_candidates, k=60)
            rerank_model = get_reranker()
            coverage_before = {}
            for d in fused_candidates:
                co = d.metadata.get("company", "Unknown")
                coverage_before[co] = coverage_before.get(co, 0) + 1
            source_docs = rerank_model.rerank(
                query=search_query,
                docs=fused_candidates,
                top_n=k,
            )
        else:
            # Standard retrieval routes
            if active_mode == "vector":
                source_docs = vector_store.similarity_search(
                    query=retrieval_query,
                    k=k,
                    filter=metadata_filter,
                )
                for idx, doc in enumerate(source_docs):
                    doc.metadata["vector_rank"] = idx + 1
            elif active_mode == "bm25":
                bm25 = get_bm25_retriever()
                source_docs = bm25.retrieve(
                    query=retrieval_query,
                    k=k,
                    filter_dict=metadata_filter,
                )
            elif active_mode == "hybrid":
                bm25 = get_bm25_retriever()
                from src.retrieval.hybrid_retriever import HybridRetriever
                hybrid = HybridRetriever(vector_store, bm25)
                source_docs = hybrid.retrieve(
                    query=retrieval_query,
                    k=k,
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
            elif active_mode == "rerank":
                bm25 = get_bm25_retriever()
                from src.retrieval.hybrid_retriever import HybridRetriever
                hybrid = HybridRetriever(vector_store, bm25)
                candidates = hybrid.retrieve(
                    query=retrieval_query,
                    k=max(20, k + 10),
                    candidate_count=max(20, k + 10),
                    filter_dict=metadata_filter,
                )
                rerank_model = get_reranker()
                coverage_before = {}
                for d in candidates:
                    co = d.metadata.get("company", "Unknown")
                    coverage_before[co] = coverage_before.get(co, 0) + 1
                source_docs = rerank_model.rerank(
                    query=search_query,
                    docs=candidates,
                    top_n=k,
                )
            else:
                raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    if not coverage_before and source_docs:
        for d in source_docs:
            co = d.metadata.get("company", "Unknown")
            coverage_before[co] = coverage_before.get(co, 0) + 1

    citations = [format_source_citation(doc.metadata) for doc in source_docs]

    # Store metadata in session state for UI to grab
    try:
        import streamlit as st
        st.session_state.last_stream_metadata = {
            "source_documents": source_docs,
            "citations": citations,
            "routing_mode": active_mode,
            "routing_strategy": routing_strategy,
            "routing_reason": routing_reason,
            "rewritten_query": search_query if rewritten else None,
            "multi_queries": queries_run if routing_strategy in ["multi_query", "comparison_query"] else None,
            "detected_entities": detected_cos if is_multi else None,
            "coverage_before_rerank": coverage_before,
        }
    except Exception:
        pass

    if not source_docs:
        yield "No relevant context found. Please run the ingestion pipeline first."
        return

    context_block = format_context_block(source_docs)
    llm = get_llm()

    # Format the prompt messages
    messages = RAG_QA_PROMPT.format_messages(
        context=context_block,
        question=search_query,
    )

    # Stream tokens statefully to hide Qwen3's <think> block
    buffer = ""
    in_think = False
    
    for chunk in llm.stream(messages):
        token = chunk.content
        buffer += token
        
        # If we see <think> or are already thinking
        if "<think>" in buffer or "think" in buffer.lower() and not in_think and len(buffer) < 15:
            # We suspect a think tag is forming or has formed
            if "<think>" in buffer:
                in_think = True
                parts = buffer.split("<think>", 1)
                before = parts[0]
                buffer = ""
                if before:
                    yield before.replace("$", "USD ").replace("USD USD ", "USD ")
            continue
            
        if in_think:
            if "</think>" in buffer:
                parts = buffer.split("</think>", 1)
                after = parts[1]
                buffer = after
                in_think = False
            else:
                # periodically discard to save memory
                if len(buffer) > 2000:
                    buffer = ""
                continue
                
        # If we are not thinking and have content, yield it
        if not in_think and len(buffer) >= 4:
            yield buffer.replace("$", "USD ").replace("USD USD ", "USD ")
            buffer = ""
            
    # Yield remaining buffer
    if not in_think and buffer:
        yield buffer.replace("$", "USD ").replace("USD USD ", "USD ")


# ---------------------------------------------------------------------------
# CLI helper — quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Requires ingestion pipeline to have been run first
    store = EarningsVectorStore()

    if store.count() == 0:
        print("Vector store is empty. Run: python -m src.ingestion.pipeline")
        sys.exit(1)

    test_questions = [
        "What was Apple's gross margin in Q3 2024?",
        "What did Jensen Huang say about sovereign AI?",
        "What was Microsoft's Azure revenue growth in Q4 2024?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = get_answer(q, store)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['citations']}")
