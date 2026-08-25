"""
src/ingestion/pipeline.py
=========================
End-to-end ingestion orchestrator for the Finance RAG Platform.

Pipeline steps:
  1. Discover all 23 transcript files across Apple, Microsoft, Nvidia
  2. Parse each filename → metadata dict (file_parser)
  3. Read file content
  4. Split into sections (summary / transcript) and chunk with RCTS (chunker)
  5. Upsert all chunks into ChromaDB with their metadata (vector_store)

Run this script ONCE to build the vector store. Re-running is safe —
chunks are deduplicated by chunk_id so no duplicates are created.

Usage
-----
  # From project root:
  python -m src.ingestion.pipeline

  # Or with explicit args:
  python -m src.ingestion.pipeline --data-dir raw_data/... --db-path data/chroma_db
"""

from __future__ import annotations

# Monkeypatch importlib.metadata.version to bypass transformers strict range check for tokenizers on Python 3.14+
import importlib.metadata
_orig_metadata_version = importlib.metadata.version
importlib.metadata.version = lambda pkg: "0.23.0" if pkg == "tokenizers" else _orig_metadata_version(pkg)

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.file_parser import parse_filename
from src.ingestion.chunker import chunk_document
from src.retrieval.vector_store import EarningsVectorStore
from src.extraction.schema import get_engine, init_db, get_session_maker, EarningsKPI
from src.extraction.kpi_extractor import extract_kpis_from_text
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "Earning_Call_Transcripts"
    / "cleaned_ECTs_dataset"
)
DEFAULT_DB_PATH  = PROJECT_ROOT / "data" / "chroma_db"

# ---------------------------------------------------------------------------
# Chunking parameters (synced with config.yaml)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_transcript_files(data_dir: Path) -> list[Path]:
    """
    Recursively find all *_processed.txt files under data_dir.

    Returns a sorted list of absolute paths.
    """
    files = sorted(data_dir.rglob("*_processed.txt"))
    logger.info(f"Discovered {len(files)} transcript files under '{data_dir}'.")
    return files


# ---------------------------------------------------------------------------
# Single-file ingestion
# ---------------------------------------------------------------------------
def ingest_file(
    filepath: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Parse, chunk, and return all chunk records for a single transcript.

    Returns
    -------
    list[dict]
        Each element: {"text": <str>, "metadata": <dict>}
    """
    metadata = parse_filename(filepath.name)
    if metadata is None:
        logger.warning(f"Skipping '{filepath.name}' — filename doesn't match pattern.")
        return []

    raw_text = filepath.read_text(encoding="utf-8")
    chunks   = chunk_document(
        raw_text=raw_text,
        base_metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunks, metadata, raw_text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_ingestion_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    db_path: Path  = DEFAULT_DB_PATH,
    chunk_size: int    = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_reset: bool  = False,
) -> dict:
    """
    Full ingestion pipeline: discover → parse → chunk → embed → store.

    Parameters
    ----------
    data_dir : Path
        Root directory containing company subdirectories with transcripts.
    db_path : Path
        Path to the ChromaDB persistence directory.
    chunk_size : int
        RCTS chunk_size parameter.
    chunk_overlap : int
        RCTS chunk_overlap parameter.
    force_reset : bool
        If True, wipes the existing collection before ingesting.
        Use only when you want a completely fresh build.

    Returns
    -------
    dict
        Summary statistics: files_found, files_processed, total_chunks,
        chunks_added, elapsed_seconds.
    """
    t_start = time.time()

    logger.info("=" * 65)
    logger.info("Financial Earnings Intelligence — Ingestion Pipeline")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Initialise vector store
    # ------------------------------------------------------------------
    logger.info(f"[1/4] Initialising ChromaDB at '{db_path}' ...")
    store = EarningsVectorStore(db_path=str(db_path))

    sql_db_path = db_path.parent / "finance_kpis.db"
    logger.info(f"Initialising SQL DB at '{sql_db_path}' ...")
    engine = get_engine(str(sql_db_path))
    init_db(engine)
    Session = get_session_maker(engine)
    session = Session()

    if force_reset:
        logger.warning("force_reset=True - clearing existing collection and SQL DB.")
        store.reset_collection()
        session.query(EarningsKPI).delete()
        session.commit()

    existing_count = store.count()
    existing_kpis = session.query(EarningsKPI).count()
    logger.info(f"Existing chunks in collection: {existing_count}")
    logger.info(f"Existing KPIs in SQL DB: {existing_kpis}")

    # ------------------------------------------------------------------
    # 2. Discover files
    # ------------------------------------------------------------------
    logger.info(f"[2/4] Discovering transcript files in '{data_dir}' ...")
    files = discover_transcript_files(data_dir)
    logger.info(f"Found: {len(files)} files")
    for f in files:
        logger.debug(f"* {f.parent.name}/{f.name}")

    if not files:
        logger.error("No transcript files found. Check data_dir path.")
        return {"error": "No files found"}

    # ------------------------------------------------------------------
    # 3. Parse + chunk all files
    # ------------------------------------------------------------------
    logger.info(f"[3/4] Parsing and chunking {len(files)} files ...")
    all_chunks: list[dict] = []
    files_processed = 0
    chunk_counts_by_file = {}

    for i, filepath in enumerate(files):
        label = f"{filepath.parent.name}/{filepath.name}"
        logger.info(f"[{i+1:02d}/{len(files)}] {label}")

        result = ingest_file(
            filepath,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if result and len(result) == 3:
            chunks, metadata, raw_text = result
            files_processed += 1
            chunk_counts_by_file[label] = len(chunks)
            all_chunks.extend(chunks)
            n_summary = sum(1 for c in chunks if c["metadata"]["section"] == "summary")
            n_trans   = sum(1 for c in chunks if c["metadata"]["section"] == "transcript")
            logger.info(f"-> {len(chunks)} chunks (summary: {n_summary}, transcript: {n_trans})")
            
            # Extract KPIs if not already present
            existing = session.query(EarningsKPI).filter_by(period=metadata["period"]).first()
            if not existing:
                summary_text = raw_text.split("[ ")[0] if "[ " in raw_text else raw_text
                kpi_record = extract_kpis_from_text(summary_text, metadata)
                session.add(kpi_record)
                session.commit()
                logger.info(f"-> Extracted KPIs for {metadata['period']}")
            else:
                logger.info(f"-> KPIs already exist for {metadata['period']}, skipping extraction.")
        else:
            logger.warning(f"Skipped (no chunks produced).")

    logger.info(f"Total chunks to index: {len(all_chunks)}")

    # ------------------------------------------------------------------
    # 4. Add to vector store
    # ------------------------------------------------------------------
    logger.info(f"[4/4] Embedding and indexing chunks into ChromaDB ...")
    chunks_added = store.add_chunks(all_chunks)

    # ------------------------------------------------------------------
    # 5. Build and save BM25 index
    # ------------------------------------------------------------------
    logger.info(f"[BM25] Building and serialising BM25 index ...")
    try:
        from src.retrieval.bm25_retriever import EarningsBM25Retriever
        bm25_dir = PROJECT_ROOT / "data" / "bm25_index"
        bm25_retriever = EarningsBM25Retriever(index_dir=str(bm25_dir))
        bm25_retriever.build_index(all_chunks)
        bm25_retriever.save()
        logger.info("BM25 index successfully built and saved.")
    except Exception as e:
        logger.exception(f"Failed to build BM25 index: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = round(time.time() - t_start, 1)
    final_count = store.count()

    logger.info("=" * 65)
    logger.info("[OK] Ingestion Complete")
    logger.info("=" * 65)
    logger.info(f"Files found:       {len(files)}")
    logger.info(f"Files processed:   {files_processed}")
    logger.info(f"Total chunks:      {len(all_chunks)}")
    logger.info(f"Chunks added:      {chunks_added}")
    logger.info(f"Total in store:    {final_count}")
    logger.info(f"Elapsed:           {elapsed}s")
    logger.info("=" * 65)

    return {
        "files_found":      len(files),
        "files_processed":  files_processed,
        "total_chunks":     len(all_chunks),
        "chunks_added":     chunks_added,
        "final_store_count": final_count,
        "elapsed_seconds":  elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finance RAG — Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing company transcript folders.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="RCTS chunk_size (characters).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="RCTS chunk_overlap (characters).",
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        default=False,
        help="Wipe existing ChromaDB collection before ingesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stats = run_ingestion_pipeline(
        data_dir=args.data_dir,
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_reset=args.force_reset,
    )
    sys.exit(0 if "error" not in stats else 1)
