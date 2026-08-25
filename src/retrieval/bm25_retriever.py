"""
src/retrieval/bm25_retriever.py
===============================
BM25 exact keyword retriever for the Finance RAG pipeline.

Uses the rank_bm25 package to index raw text chunks, serialise/load the index
to/from disk, and retrieve candidate documents matching a keyword query.
Supports metadata filtering matching vector_store filters.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.logger import logger


def tokenize(text: str) -> list[str]:
    """Tokenize a string by lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


def matches_filter(metadata: dict, filter_dict: Optional[dict]) -> bool:
    """
    Check if a chunk's metadata matches the applied filter dictionary.
    Handles simple key-value filters and $and compound filters.
    """
    if not filter_dict:
        return True

    def match_item(meta_key: str, val) -> bool:
        actual_val = metadata.get(meta_key)
        if actual_val is None:
            return False
        # Normalize comparison to case-insensitive strings
        return str(actual_val).strip().lower() == str(val).strip().lower()

    if "$and" in filter_dict:
        for cond in filter_dict["$and"]:
            for k, v in cond.items():
                if not match_item(k, v):
                    return False
        return True
    else:
        for k, v in filter_dict.items():
            if not match_item(k, v):
                return False
        return True


class EarningsBM25Retriever:
    """
    In-memory BM25 index over earnings call transcript chunks.
    Can serialize itself to disk and reload.
    """

    def __init__(self, index_dir: str = "data/bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "bm25.pkl"
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list[dict] = []  # list of {"text": ..., "metadata": ...}

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.

        Parameters
        ----------
        chunks : list[dict]
            Each chunk has 'text' and 'metadata'.
        """
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            logger.warning("Empty chunk list provided to BM25. Index not created.")
            return

        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"Built BM25 index over {len(chunks)} chunks.")

    def save(self) -> None:
        """Serialize index and chunks to disk."""
        if not self.bm25:
            logger.warning("No BM25 index to save.")
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"Saved BM25 index to '{self.index_file}'")

    def load(self) -> bool:
        """
        Load BM25 index and chunks from disk.

        Returns
        -------
        bool
            True if load succeeded, False otherwise.
        """
        if not self.index_file.exists():
            logger.warning(f"BM25 index file '{self.index_file}' does not exist.")
            return False

        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
            logger.info(
                f"Successfully loaded BM25 index from '{self.index_file}' with {len(self.chunks)} chunks."
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to load BM25 index: {e}")
            return False

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """
        Retrieve top-k chunks matching the keyword query, applying metadata filters.

        Parameters
        ----------
        query : str
            Keyword search query.
        k : int
            Number of documents to return.
        filter_dict : dict | None
            ChromaDB-style metadata filter.

        Returns
        -------
        list[Document]
            LangChain Document objects.
        """
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty or not loaded. Returning empty list.")
            return []

        tokenized_query = tokenize(query)
        # Compute BM25 scores across the entire global corpus
        scores = self.bm25.get_scores(tokenized_query)

        # Apply metadata filters to locate valid matching chunks
        matched = []
        for idx, chunk in enumerate(self.chunks):
            if matches_filter(chunk["metadata"], filter_dict):
                matched.append((chunk, scores[idx]))

        # Sort matching chunks by BM25 score descending
        matched.sort(key=lambda item: item[1], reverse=True)

        # Return top-k as Document objects
        results = []
        for rank, (chunk, score) in enumerate(matched[:k], start=1):
            meta = chunk["metadata"].copy()
            meta["bm25_score"] = float(score)
            meta["bm25_rank"] = rank
            results.append(
                Document(
                    page_content=chunk["text"],
                    metadata=meta,
                )
            )

        return results

    def __repr__(self) -> str:
        count = len(self.chunks) if self.chunks else 0
        return f"EarningsBM25Retriever(chunks={count}, path='{self.index_file}')"
