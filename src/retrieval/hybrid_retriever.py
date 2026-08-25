"""
src/retrieval/hybrid_retriever.py
=================================
Hybrid retriever integrating vector semantic search and BM25 exact keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge candidate rank lists from both
retrieval sources, producing a single, re-ranked retrieval list.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

from src.retrieval.vector_store import EarningsVectorStore
from src.retrieval.bm25_retriever import EarningsBM25Retriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Combine document result lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    vector_docs : list[Document]
        List of Documents from semantic vector search.
    bm25_docs : list[Document]
        List of Documents from BM25 keyword search.
    k : int
        RRF constant factor (default: 60).

    Returns
    -------
    list[Document]
        Sorted list of fused Documents containing combined ranking details in metadata.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    ranks_info: dict[str, dict[str, int]] = {}

    # Rank documents retrieved via vector search
    for rank, doc in enumerate(vector_docs):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue
        doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        ranks_info[chunk_id] = {"vector_rank": rank + 1}

    # Rank documents retrieved via BM25
    for rank, doc in enumerate(bm25_docs):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue
        doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        if chunk_id in ranks_info:
            ranks_info[chunk_id]["bm25_rank"] = rank + 1
        else:
            ranks_info[chunk_id] = {"bm25_rank": rank + 1}

    # Sort candidates by combined RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Convert sorted candidates back to Document list, augmenting metadata
    fused_docs: list[Document] = []
    for final_rank, chunk_id in enumerate(sorted_chunk_ids, start=1):
        original_doc = doc_map[chunk_id]
        meta = original_doc.metadata.copy()

        # Add fusion metadata details
        meta["rrf_score"] = float(rrf_scores[chunk_id])
        meta["rrf_rank"] = final_rank
        meta["vector_rank"] = ranks_info[chunk_id].get("vector_rank", -1)
        meta["bm25_rank"] = ranks_info[chunk_id].get("bm25_rank", -1)

        fused_docs.append(
            Document(
                page_content=original_doc.page_content,
                metadata=meta,
            )
        )

    return fused_docs


class HybridRetriever:
    """
    Coordinates vector store and BM25 retrievers to execute RRF hybrid queries.
    """

    def __init__(
        self,
        vector_store: EarningsVectorStore,
        bm25_retriever: EarningsBM25Retriever,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_count: int = 20,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """
        Query vector and BM25 indices, fuse results with RRF, and return top-k.

        Parameters
        ----------
        query : str
            Search query string.
        k : int
            Number of final documents to return.
        candidate_count : int
            Number of candidate matches to retrieve from each sub-retriever.
        filter_dict : dict | None
            Metadata filter block.
        """
        # 1. Query vector store
        vector_candidates = self.vector_store.similarity_search(
            query=query,
            k=candidate_count,
            filter=filter_dict,
        )

        # 2. Query BM25 retriever
        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            k=candidate_count,
            filter_dict=filter_dict,
        )

        # 3. Fuse lists using RRF
        fused_docs = reciprocal_rank_fusion(
            vector_docs=vector_candidates,
            bm25_docs=bm25_candidates,
            k=self.rrf_k,
        )

        return fused_docs[:k]
