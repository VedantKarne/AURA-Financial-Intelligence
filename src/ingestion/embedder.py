"""
src/ingestion/embedder.py
=========================
HuggingFace embedding model wrapper for the Finance RAG pipeline.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384 dimensions
  - Runs fully on CPU (no GPU required)
  - ~80MB download, permanently cached by HuggingFace after first run
  - Cost: $0.00 — local inference, no API calls
  - MTEB retrieval score: strong baseline; upgradeable to BAAI/bge-small-en-v1.5
    in one line for Phase 2 if desired

This module provides a singleton embedding model and a batch embedding
function used by the ingestion pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model configuration (matches config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE     = "cpu"
DEFAULT_BATCH_SIZE = 64   # chunks per embedding batch


# ---------------------------------------------------------------------------
# Singleton pattern — reuse across multiple pipeline calls
# ---------------------------------------------------------------------------
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> HuggingFaceEmbeddings:
    """
    Return the shared HuggingFace embedding model instance.
    Initialises on first call, then returns the cached instance.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        'cpu' or 'cuda'. Use 'cpu' for the zero-dependency default.

    Returns
    -------
    HuggingFaceEmbeddings
        LangChain-compatible embedding model.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name} on {device}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded and cached.")

    return _embedding_model


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of text strings in batches.

    Parameters
    ----------
    texts : list[str]
        Text strings to embed.
    model_name : str
        HuggingFace model identifier.
    device : str
        'cpu' or 'cuda'.
    batch_size : int
        Number of texts to process per batch. Reduce if OOM on CPU.
    show_progress : bool
        Print progress to stdout.

    Returns
    -------
    list[list[float]]
        List of 384-dimensional embedding vectors (one per input text).
    """
    model = get_embedding_model(model_name=model_name, device=device)

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, start_idx in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start_idx : start_idx + batch_size]
        embeddings = model.embed_documents(batch)
        all_embeddings.extend(embeddings)

        if show_progress:
            print(
                f"\r  Embedding batch {batch_num + 1}/{total_batches} "
                f"({start_idx + len(batch)}/{len(texts)} chunks)",
                end="",
                flush=True,
            )

    if show_progress:
        print()  # newline after progress

    return all_embeddings


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "Apple reported revenue of $89.5 billion in Q3 2024.",
        "Nvidia's data center segment grew 154% year over year.",
        "Microsoft Azure revenue growth was 29% in constant currency.",
    ]

    print("Testing embedding model...")
    embeddings = embed_texts(test_sentences, show_progress=True)

    print(f"\nEmbedded {len(embeddings)} sentences.")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"First vector (first 5 dims): {embeddings[0][:5]}")
