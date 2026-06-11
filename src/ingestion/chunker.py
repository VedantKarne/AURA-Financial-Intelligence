"""
src/ingestion/chunker.py
========================
Custom chunking pipeline for single-line earnings call transcripts.

THE KEY ENGINEERING CHALLENGE:
  Every transcript file is stored as a SINGLE LINE (~42–63 KB of text,
  zero newlines). Standard text splitters that rely on line breaks or
  paragraph delimiters will not split these files correctly.

SOLUTION — Two-step pipeline:
  Step 1: Split the raw text at the section marker "[ " into:
          - "summary"    : analytical prose (appears before the marker)
          - "transcript" : prepared remarks + Q&A (marker → end of file)
  Step 2: Apply LangChain's RecursiveCharacterTextSplitter (RCTS) with
          custom separators [". ", "? ", "! ", "; ", " ", ""] to each
          section independently. RCTS tries separators in order, so the
          vast majority of splits land on sentence boundaries (". "),
          preserving complete financial sentences in every chunk.

RCTS parameters (from config):
  chunk_size    = 1000  (~250 tokens)   — fits well within LLM context
  chunk_overlap = 200   (~50 tokens)    — preserves cross-chunk context
"""

from __future__ import annotations

import re
from typing import Optional
try:
    # LangChain 1.x — text splitters live in a dedicated package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # LangChain 0.3.x fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------
# The live transcript section starts with a "[" followed by a space.
# The summary section is everything before that first occurrence.
_SECTION_MARKER = "[ "


def split_into_sections(raw_text: str) -> dict[str, str]:
    """
    Split a raw transcript into 'summary' and 'transcript' sections.

    Parameters
    ----------
    raw_text : str
        The complete contents of a single transcript file (typically
        one very long line, ~42–63 KB).

    Returns
    -------
    dict with keys 'summary' and 'transcript'.
    'summary' may be an empty string if no content precedes the marker.
    """
    marker_pos = raw_text.find(_SECTION_MARKER)

    if marker_pos == -1:
        # No section marker found — treat the entire file as transcript
        return {
            "summary":    "",
            "transcript": raw_text.strip(),
        }

    summary_text    = raw_text[:marker_pos].strip()
    transcript_text = raw_text[marker_pos:].strip()

    return {
        "summary":    summary_text,
        "transcript": transcript_text,
    }


# ---------------------------------------------------------------------------
# RCTS splitter factory
# ---------------------------------------------------------------------------
def _make_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter tuned for financial transcripts.

    Separator priority order (highest → lowest):
      1. ". "  — most splits should land here (sentence boundaries)
      2. "? "  — question boundaries in Q&A sections
      3. "! "  — rare in finance; included for completeness
      4. "; "  — semicolon-joined clauses
      5. " "   — word boundary fallback
      6. ""    — character boundary (last resort, should almost never trigger)
    """
    return RecursiveCharacterTextSplitter(
        separators=[". ", "? ", "! ", "; ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------
def clean_transcript_text(text: str) -> str:
    """Remove low-value safe harbor, intro, operator, and transition sentences."""
    # Split by sentence boundaries (preserving the punctuation)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    cleaned_sentences = []
    exclude_phrases = [
        "operator:",
        "good afternoon",
        "welcome to",
        "investor relations",
        "forward-looking statements",
        "safe harbor",
        "our next question comes from",
        "our next question is from",
        "thank you and goodbye",
        "please go ahead",
        "turn the call over to",
        "questions from analysts",
        "take our first question",
        "concludes our question"
    ]
    
    for sent in sentences:
        sent_lower = sent.lower()
        if any(phrase in sent_lower for phrase in exclude_phrases):
            continue
        if "director of investor relations" in sent_lower:
            continue
        cleaned_sentences.append(sent)
        
    return " ".join(cleaned_sentences)


def chunk_document(
    raw_text: str,
    base_metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Chunk a single transcript document into annotated chunk records.

    Parameters
    ----------
    raw_text : str
        Full text of the transcript file.
    base_metadata : dict
        Document-level metadata from file_parser.parse_filename().
        Must contain at least: company, ticker, year, quarter, period,
        source_file.
    chunk_size : int
        RCTS chunk_size parameter (characters).
    chunk_overlap : int
        RCTS chunk_overlap parameter (characters).

    Returns
    -------
    list[dict]
        Each element is:
        {
            "text":        <str>   — the chunk text,
            "metadata":    <dict>  — full metadata for this chunk,
        }

    Chunk metadata schema
    ---------------------
    {
        # Document-level (from file_parser)
        "company":      "Apple",
        "ticker":       "AAPL",
        "year":         2024,
        "quarter":      "Q3",
        "period":       "2024-Q3",
        "section":      "transcript",   # or "summary"
        "source_file":  "2024_Q3_aapl_processed.txt",

        # Chunk-level
        "chunk_id":     "2024_Q3_aapl_transcript_042",
        "chunk_index":  42,
    }
    """
    sections = split_into_sections(raw_text)
    splitter = _make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks: list[dict] = []

    for section_name, section_text in sections.items():
        if not section_text:
            continue  # skip empty sections

        cleaned_text = clean_transcript_text(section_text)
        if not cleaned_text.strip():
            continue

        raw_chunks = splitter.split_text(cleaned_text)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue  # skip any whitespace-only chunks

            # Build unique chunk_id from period + section + index
            source_stem = base_metadata["source_file"].replace("_processed.txt", "")
            chunk_id = f"{source_stem}_{section_name}_{idx:03d}"

            chunk_metadata = {
                **base_metadata,
                "section":     section_name,
                "chunk_id":    chunk_id,
                "chunk_index": idx,
            }

            all_chunks.append({
                "text":     chunk_text,
                "metadata": chunk_metadata,
            })

    return all_chunks


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Quick smoke test on a real file if provided
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        raw = test_file.read_text(encoding="utf-8")

        # Fake base metadata for the test
        meta = {
            "company": "TestCo", "ticker": "TEST",
            "year": 2024, "quarter": "Q3",
            "period": "2024-Q3",
            "source_file": test_file.name,
        }

        chunks = chunk_document(raw, meta)
        print(f"\nTotal chunks: {len(chunks)}")

        # Show section distribution
        from collections import Counter
        sections = Counter(c["metadata"]["section"] for c in chunks)
        print(f"Section distribution: {dict(sections)}")

        # Show first and last chunk
        print(f"\n--- First chunk ---\n{chunks[0]['text'][:300]}")
        print(f"\n--- Last chunk ---\n{chunks[-1]['text'][:300]}")
    else:
        print("Usage: python chunker.py <path_to_transcript.txt>")
