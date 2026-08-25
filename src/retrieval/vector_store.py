"""
src/retrieval/vector_store.py
=============================
ChromaDB persistent vector store wrapper for the Finance RAG pipeline.

Collection: "earnings_transcripts"
Embedding:  all-MiniLM-L6-v2  (384 dimensions)
Persistence: data/chroma_db/   (file-based, no server required)

Key design decisions:
  - Collection is created once and reused across sessions.
  - All 8 metadata fields are stored and fully filterable via ChromaDB
    `where` clauses (company, ticker, year, quarter, period, section,
    source_file, chunk_id, chunk_index).
  - Deduplication: if a chunk_id already exists, it is skipped (safe
    to re-run the ingestion pipeline without duplicating data).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.ingestion.embedder import get_embedding_model
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "earnings_transcripts"
DEFAULT_DB_PATH = "data/chroma_db"


# ---------------------------------------------------------------------------
# Core vector store class
# ---------------------------------------------------------------------------
class EarningsVectorStore:
    """
    Persistent ChromaDB vector store for earnings call transcript chunks.

    Usage
    -----
    # Build / load the store
    store = EarningsVectorStore(db_path="data/chroma_db")

    # Add chunks (from pipeline output)
    store.add_chunks(chunks)           # list of {"text": ..., "metadata": ...}

    # Query
    results = store.similarity_search("What was gross margin?", k=5)
    results = store.similarity_search(
        "Azure growth", k=5,
        filter={"ticker": "MSFT", "year": 2024}
    )

    # Count
    print(store.count())
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.db_path         = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = get_embedding_model(model_name=model_name, device=device)

        # Ensure the persistence directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Build the LangChain Chroma wrapper
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.db_path),
        )

        logger.info(
            f"EarningsVectorStore initialised — collection='{collection_name}', "
            f"path='{db_path}', existing documents={self.count()}"
        )

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------
    def add_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """
        Add chunk records to the vector store.

        Parameters
        ----------
        chunks : list[dict]
            Each element must have keys 'text' and 'metadata'.
            The metadata dict must include 'chunk_id' (used as the
            ChromaDB document ID to prevent duplicates).
        batch_size : int
            Number of documents to upsert per batch.

        Returns
        -------
        int
            Number of new chunks actually added (skips existing chunk_ids).
        """
        if not chunks:
            return 0

        # Build LangChain Document objects
        docs: list[Document] = []
        ids:  list[str]      = []

        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id")
            if not chunk_id:
                logger.warning("Chunk missing chunk_id — skipping.")
                continue

            # ChromaDB metadata values must be str, int, or float
            safe_metadata = _sanitise_metadata(chunk["metadata"])

            docs.append(Document(
                page_content=chunk["text"],
                metadata=safe_metadata,
            ))
            ids.append(chunk_id)

        if not docs:
            return 0

        # Batch upsert
        added = 0
        total_batches = (len(docs) + batch_size - 1) // batch_size

        for batch_num, start in enumerate(range(0, len(docs), batch_size)):
            batch_docs = docs[start : start + batch_size]
            batch_ids  = ids[start : start + batch_size]

            self._store.add_documents(documents=batch_docs, ids=batch_ids)
            added += len(batch_docs)

            logger.info(
                f"Indexing batch {batch_num + 1}/{total_batches} "
                f"({added}/{len(docs)} chunks)"
            )

        logger.info(f"Added {added} chunks to collection '{self.collection_name}'.")
        return added

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Retrieve the top-k most similar chunks for a query.

        Parameters
        ----------
        query : str
            Natural language question or search string.
        k : int
            Number of results to return.
        filter : dict | None
            ChromaDB metadata filter. Single-key example:
              {"ticker": "AAPL"}
            Multi-key (AND) example:
              {"$and": [{"ticker": "AAPL"}, {"year": 2024}]}

        Returns
        -------
        list[Document]
            LangChain Document objects with .page_content and .metadata.
        """
        if filter:
            return self._store.similarity_search(query, k=k, filter=filter)
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Same as similarity_search but also returns cosine similarity scores."""
        if filter:
            return self._store.similarity_search_with_score(query, k=k, filter=filter)
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = 5, filter: Optional[dict] = None):
        """
        Return a LangChain-compatible retriever for use in chains.

        Parameters
        ----------
        k : int
            Number of documents to retrieve.
        filter : dict | None
            Optional ChromaDB metadata filter.
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self._store.as_retriever(search_kwargs=search_kwargs)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        try:
            # Works in both langchain-chroma 0.2.x and 1.x
            return self._store._collection.count()
        except AttributeError as e1:
            logger.warning(f"Chroma count direct failed (AttributeError): {e1}")
            try:
                # Fallback: use langchain-chroma's .get() API
                result = self._store.get()
                return len(result.get("ids", []))
            except Exception as e2:
                logger.exception(f"Chroma count fallback failed: {e2}")
                return -1
        except Exception as e:
            logger.exception(f"Unexpected error during Chroma count: {e}")
            return -1

    def get_all_metadata(self) -> list[dict]:
        """Return metadata for all documents (no text). For debugging."""
        try:
            result = self._store._collection.get(include=["metadatas"])
            return result.get("metadatas", [])
        except Exception as e:
            logger.exception(f"Failed to get metadata: {e}")
            return []

    def reset_collection(self) -> None:
        """
        Delete all documents from the collection.
        USE WITH CAUTION — requires full re-ingestion.
        """
        try:
            # langchain-chroma 1.x has a native reset_collection() method
            self._store.reset_collection()
        except AttributeError:
            # Fallback for older versions
            self._store._collection.delete(
                where={"chunk_id": {"$ne": "__nonexistent__"}}
            )
        logger.warning(f"Collection '{self.collection_name}' has been cleared.")

    def __repr__(self) -> str:
        return (
            f"EarningsVectorStore("
            f"collection='{self.collection_name}', "
            f"docs={self.count()}, "
            f"path='{self.db_path}')"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitise_metadata(metadata: dict) -> dict:
    """
    Ensure all metadata values are ChromaDB-compatible types:
    str, int, float, or bool. Converts anything else to str.
    """
    safe = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


def build_metadata_filter(
    company: Optional[str] = None,
    ticker: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    section: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB metadata filter from optional field values.

    Returns None if no filters are specified (→ search all documents).
    Returns a single condition dict if only one filter is set.
    Returns an $and compound filter for multiple conditions.

    Parameters
    ----------
    company : str | None
        e.g. "Apple", "Microsoft", "Nvidia"
    ticker : str | None
        e.g. "AAPL", "MSFT", "NVDA"
    year : int | None
        e.g. 2024
    quarter : str | None
        e.g. "Q3"
    section : str | None
        "summary" or "transcript"
    """
    conditions = []

    if company:
        conditions.append({"company": company})
    if ticker:
        conditions.append({"ticker": ticker.upper()})
    if year:
        conditions.append({"year": int(year)})
    if quarter:
        conditions.append({"quarter": quarter.upper()})
    if section:
        conditions.append({"section": section})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = EarningsVectorStore()
    logger.info(str(store))
    logger.info(f"Total chunks in store: {store.count()}")
