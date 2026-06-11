"""Quick smoke test to verify all Phase 1 imports and core logic."""
import sys
sys.path.insert(0, ".")

print("Testing imports...")
from src.ingestion.file_parser import parse_filename
from src.ingestion.chunker import chunk_document, split_into_sections
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import EarningsVectorStore, build_metadata_filter
from src.generation.prompts import RAG_QA_PROMPT, format_source_citation
print("  All imports: OK")

print("\nTesting file_parser...")
meta = parse_filename("2024_Q3_aapl_processed.txt")
assert meta["company"] == "Apple", f"Expected Apple, got {meta['company']}"
assert meta["year"] == 2024
assert meta["quarter"] == "Q3"
assert meta["ticker"] == "AAPL"
print(f"  parse_filename: OK -> {meta}")

bad = parse_filename("bad_filename.txt")
assert bad is None
print("  bad filename returns None: OK")

print("\nTesting chunker...")
sample_text = (
    "This is a summary sentence. Another summary sentence here. "
    "[ Good afternoon and welcome to the Q3 earnings call. "
    "Revenue was 89.5 billion dollars. Gross margin was 46.3 percent. "
    "We are very pleased with these results."
)
sections = split_into_sections(sample_text)
assert "summary" in sections and "transcript" in sections
assert "summary" in sections["summary"].lower() or len(sections["summary"]) > 0
print(f"  sections: summary={len(sections['summary'])} chars, transcript={len(sections['transcript'])} chars")

chunks = chunk_document(sample_text, meta, chunk_size=200, chunk_overlap=50)
section_types = set(c["metadata"]["section"] for c in chunks)
print(f"  chunks: {len(chunks)} total, sections={section_types}")
assert len(chunks) > 0

# Verify chunk metadata schema
first_chunk = chunks[0]
required_keys = ["company", "ticker", "year", "quarter", "period", "section", "source_file", "chunk_id", "chunk_index"]
for key in required_keys:
    assert key in first_chunk["metadata"], f"Missing metadata key: {key}"
print("  chunk metadata schema: OK")

print("\nTesting metadata filter builder...")
f1 = build_metadata_filter(company="Apple", year=2024)
print(f"  two-field filter: {f1}")
assert "$and" in f1

f2 = build_metadata_filter(company="Apple")
print(f"  single-field filter: {f2}")
assert f2 == {"company": "Apple"}

f3 = build_metadata_filter()
assert f3 is None
print("  no-filter returns None: OK")

print("\n" + "="*50)
print("ALL SMOKE TESTS PASSED")
print("="*50)
