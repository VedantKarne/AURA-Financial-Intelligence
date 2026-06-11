"""Smoke test to verify Phase 2 retrievers and retrieval modes."""
import sys
sys.path.insert(0, ".")

print("Testing imports...")
from src.retrieval.bm25_retriever import EarningsBM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import EarningsReranker
from src.retrieval.vector_store import EarningsVectorStore
from src.generation.qa_chain import get_answer
print("  All Phase 2 imports: OK")

print("\nTesting BM25 Retriever loading and retrieval...")
bm25 = EarningsBM25Retriever(index_dir="data/bm25_index")
success = bm25.load()
assert success, "Failed to load BM25 index"
print(f"  BM25 index loaded: {len(bm25.chunks)} chunks")

# Retrieve
results_bm25 = bm25.retrieve("gross margin", k=3)
print(f"  BM25 retrieved: {len(results_bm25)} docs")
assert len(results_bm25) > 0, "BM25 retrieved empty results"
assert "bm25_score" in results_bm25[0].metadata
assert "bm25_rank" in results_bm25[0].metadata
print("  BM25 retrieval: OK")

print("\nTesting Hybrid Retriever...")
vector_store = EarningsVectorStore(db_path="data/chroma_db")
hybrid = HybridRetriever(vector_store, bm25)
results_hybrid = hybrid.retrieve("Azure growth", k=3)
print(f"  Hybrid retrieved: {len(results_hybrid)} docs")
assert len(results_hybrid) > 0, "Hybrid retrieved empty results"
assert "rrf_score" in results_hybrid[0].metadata
assert "rrf_rank" in results_hybrid[0].metadata
print("  Hybrid retrieval: OK")

print("\nTesting Reranker...")
reranker = EarningsReranker(device="cpu")
results_reranked = reranker.rerank("Azure growth", results_hybrid, top_n=2)
print(f"  Reranked: {len(results_reranked)} docs")
assert len(results_reranked) > 0, "Reranked empty results"
assert "rerank_score" in results_reranked[0].metadata
print("  Reranking: OK")

print("\nTesting get_answer with all retrieval modes...")
for mode in ["vector", "bm25", "hybrid", "rerank"]:
    print(f"  Running get_answer with mode: {mode}")
    out = get_answer(
        question="What was Apple's gross margin in Q3 2024?",
        vector_store=vector_store,
        k=3,
        retrieval_mode=mode,
        company="Apple",
        year=2024,
        quarter="Q3"
    )
    assert "answer" in out
    assert "source_documents" in out
    print(f"    Mode {mode} -> OK (sources={len(out['source_documents'])})")

print("\nTesting QueryRouter (Rule-based fallback)...")
from src.retrieval.router import QueryRouter
router = QueryRouter(llm=None)
route_info = router.route_query("Compare Apple and Microsoft gross margins")
print(f"  Rule-based Comparison Route: {route_info}")
assert route_info["strategy"] in ["comparison_query", "multi_entity_retrieval"]

route_info_summary = router.route_query("Summarize Nvidia Q4 call overview")
print(f"  Rule-based Summary Route: {route_info_summary}")
assert route_info_summary["strategy"] == "summary_section"

route_info_financial_summary = router.route_query("How has Apple's revenue generation been so far?")
print(f"  Rule-based Financial Summary Route: {route_info_financial_summary}")
assert route_info_financial_summary["strategy"] == "single_entity_financial_summary"

route_info_multi_risk = router.route_query("What are the risks faced by Apple, Nvidia, and Microsoft?")
print(f"  Rule-based Multi-Entity Risk Route: {route_info_multi_risk}")
assert route_info_multi_risk["strategy"] == "multi_entity_risk_analysis"

route_info_single_risk = router.route_query("What are the regulatory challenges Apple faces?")
print(f"  Rule-based Single-Entity Risk Route: {route_info_single_risk}")
assert route_info_single_risk["strategy"] == "single_entity_risk_analysis"

print("\nTesting get_answer with auto-routing and query transformer...")
# Test auto mode (loads the LLM and runs dynamic query routing)
out_auto = get_answer(
    question="What was Apple's gross margin in Q3 2024?",
    vector_store=vector_store,
    k=3,
    retrieval_mode="auto",
)
assert "answer" in out_auto
assert "routing_mode" in out_auto
assert "routing_strategy" in out_auto
assert "coverage_before_rerank" in out_auto
print(f"  Auto Route: {out_auto['routing_mode']} Strategy: {out_auto['routing_strategy']} Reason: {out_auto.get('routing_reason')}")

# Test with chat history for conversational query rewriting
chat_history = [
    {"role": "user", "content": "What was Apple's gross margin in Q3 2024?"},
    {"role": "assistant", "content": "Apple's gross margin in Q3 2024 was 46.3%."},
]
out_history = get_answer(
    question="What about the previous quarter?",
    vector_store=vector_store,
    k=3,
    retrieval_mode="auto",
    chat_history=chat_history,
)
assert "answer" in out_history
assert "source_documents" in out_history
if out_history.get("rewritten_query"):
    print(f"  Follow-up Rewritten Query: '{out_history.get('rewritten_query')}'")
else:
    print("  Follow-up Query did not require rewrite or rewrite failed.")

print("\nTesting Multi-Entity Query Detection and Retrieval...")
is_multi, detected = router.is_multi_entity("What are the key insights about all companies?")
assert is_multi
assert "Apple" in detected
assert "Microsoft" in detected
assert "Nvidia" in detected
print(f"  Multi-entity detection: OK ({detected})")

out_multi = get_answer(
    question="What are the key insights about Apple, Nvidia, and Microsoft?",
    vector_store=vector_store,
    k=12,
    retrieval_mode="auto",
)
assert "detected_entities" in out_multi
assert out_multi["detected_entities"] is not None
print(f"  Multi-entity Retrieval: OK (detected={out_multi['detected_entities']}, sources={len(out_multi['source_documents'])})")

print("\n" + "="*50)
print("ALL PHASE 2 SMOKE TESTS PASSED")
print("="*50)
