"""
src/ingestion/file_parser.py
============================
Parses metadata from earnings call transcript filenames.

Filename convention:  {year}_Q{quarter}_{ticker}_processed.txt
Example:              2024_Q3_aapl_processed.txt

Produces a metadata dict per document for use as ChromaDB metadata
and chunk labelling throughout the pipeline.
"""

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ticker → Human-readable company name map
# ---------------------------------------------------------------------------
TICKER_MAP: dict[str, str] = {
    "aapl": "Apple",
    "msft": "Microsoft",
    "nvda": "Nvidia",
}

# ---------------------------------------------------------------------------
# Filename regex
# ---------------------------------------------------------------------------
_FILENAME_PATTERN = re.compile(
    r"^(?P<year>\d{4})_Q(?P<quarter>\d)_(?P<ticker>[a-z]+)_processed\.txt$",
    re.IGNORECASE,
)


def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse a transcript filename into a structured metadata dictionary.

    Parameters
    ----------
    filename : str
        Just the filename (basename), e.g. '2024_Q3_aapl_processed.txt'.
        Full paths are also accepted — only the basename is used.

    Returns
    -------
    dict | None
        Metadata dict on success, None if the filename doesn't match
        the expected pattern.

    Example
    -------
    >>> parse_filename("2024_Q3_aapl_processed.txt")
    {
        'company':     'Apple',
        'ticker':      'AAPL',
        'year':        2024,
        'quarter':     'Q3',
        'period':      '2024-Q3',
        'source_file': '2024_Q3_aapl_processed.txt',
    }
    """
    # Accept full paths — take only the basename
    basename = Path(filename).name

    match = _FILENAME_PATTERN.match(basename)
    if not match:
        return None

    year = int(match.group("year"))
    quarter_num = match.group("quarter")
    ticker_lower = match.group("ticker").lower()

    company = TICKER_MAP.get(ticker_lower, ticker_lower.upper())
    quarter = f"Q{quarter_num}"
    period = f"{year}-{quarter}"

    return {
        "company":     company,
        "ticker":      ticker_lower.upper(),
        "year":        year,
        "quarter":     quarter,
        "period":      period,
        "source_file": basename,
    }


def parse_filepath(filepath: str | Path) -> Optional[dict]:
    """
    Convenience wrapper that takes a full file path, reads the file's
    content, and returns (metadata_dict, raw_text).

    Parameters
    ----------
    filepath : str | Path
        Absolute or relative path to the transcript file.

    Returns
    -------
    tuple[dict, str] | None
        (metadata, raw_text) on success, None if the file doesn't match
        the naming pattern.
    """
    path = Path(filepath)
    metadata = parse_filename(path.name)
    if metadata is None:
        return None

    raw_text = path.read_text(encoding="utf-8")
    return metadata, raw_text


# ---------------------------------------------------------------------------
# CLI helper — run `python file_parser.py` to test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_cases = [
        "2024_Q3_aapl_processed.txt",
        "2023_Q1_msft_processed.txt",
        "2024_Q4_nvda_processed.txt",
        "bad_filename.txt",
    ]

    for name in test_cases:
        result = parse_filename(name)
        print(f"{name!r:45s} → {result}")
