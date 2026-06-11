"""
src/retrieval/reranker.py
=========================
Cross-Encoder reranker using ms-marco-MiniLM-L-6-v2.

Scores input candidate documents against the query to sort them by relevance,
improving precision for top-k output documents.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class EarningsReranker:
    """
    Reranks candidate chunks using a local Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        logger.info(f"Loading Cross-Encoder model: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Cross-Encoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = 5,
    ) -> list[Document]:
        """
        Rerank candidate documents against the query.

        Parameters
        ----------
        query : str
            The question or search query.
        docs : list[Document]
            List of candidate Document objects to rerank.
        top_n : int
            Number of documents to return.

        Returns
        -------
        list[Document]
            Sorted and sliced Document list with 'rerank_score' in their metadata.
        """
        if not docs:
            return []

        # Format input pairs for the cross encoder: [[query, text], [query, text], ...]
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        reranked: list[Document] = []
        for doc, score in zip(docs, scores):
            meta = doc.metadata.copy()
            meta["rerank_score"] = float(score)
            reranked.append(
                Document(
                    page_content=doc.page_content,
                    metadata=meta,
                )
            )

        # Sort descending by rerank score
        reranked.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)

        return reranked[:top_n]
