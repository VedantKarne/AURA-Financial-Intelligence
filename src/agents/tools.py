from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from src.retrieval.vector_store import EarningsVectorStore
from src.generation.qa_chain import get_answer
from src.extraction.schema import get_engine, get_session_maker, EarningsKPI
from pathlib import Path
import json
from langchain_core.runnables import RunnableConfig
from src.generation.prompts import format_source_citation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "finance_kpis.db"

# Lazy load vector store
_vector_store = None
GLOBAL_K = 6

# Companies for multi-entity detection
_KNOWN_COMPANIES = ["apple", "aapl", "microsoft", "msft", "azure", "nvidia", "nvda", "jensen"]
_ALL_COMPANY_PHRASES = [
    "all companies", "all the companies", "each company", "every company",
    "the companies", "all three", "each of the", "both companies",
    "across companies", "compare", "comparison", "versus", "vs"
]

def _count_entities_in_query(query: str) -> int:
    """Estimate how many companies are referenced in a query."""
    q = query.lower()
    if any(p in q for p in _ALL_COMPANY_PHRASES):
        return 3  # Assume all three companies
    count = 0
    if any(x in q for x in ["apple", "aapl"]):
        count += 1
    if any(x in q for x in ["microsoft", "msft", "azure"]):
        count += 1
    if any(x in q for x in ["nvidia", "nvda", "jensen"]):
        count += 1
    return max(1, count)

# Global cache for current active turn snippets
ACTIVE_RETRIEVED_DOCS = {}

def set_global_k(k: int):
    global GLOBAL_K
    GLOBAL_K = k

def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = EarningsVectorStore()
    return _vector_store

@tool
def rag_search(
    query: str, 
    company: Optional[str] = None, 
    year: Optional[int] = None, 
    quarter: Optional[str] = None,
    k_override: Optional[int] = None,
    config: Optional[RunnableConfig] = None
) -> str:
    """Search earnings call transcripts using hybrid RAG retrieval.
    Use this tool for qualitative questions, narrative summaries, and risk analysis.
    IMPORTANT: If the user asks about 'all companies', comparisons, or multiple entities, DO NOT specify the `company` argument. 
    CRITICAL: You MUST preserve words like 'compare', 'all companies', or list out the company names in the `query` argument itself so the retrieval system knows to fetch equal chunks for each company natively. Do not strip them out!
    RETRY LOGIC: If your first call to this tool fails to find the necessary information, do NOT give up. Call this tool again with `k_override=24` or `k_override=32` to cast a wider net and extract deeper transcript chunks.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    
    vs = _get_vector_store()
    
    # Auto-scale k for multi-entity queries: each company needs a fair share
    # A k of 6 for 3 companies = only 2 chunks each, which is far too sparse.
    n_entities = _count_entities_in_query(query)
    effective_k = GLOBAL_K
    
    if k_override is not None:
        effective_k = k_override
    elif n_entities > 1:
        # Minimum 6 chunks per entity, capped at 24 total
        effective_k = min(24, max(GLOBAL_K, n_entities * 6))
    
    result = get_answer(
        question=query,
        vector_store=vs,
        k=effective_k,
        retrieval_mode="auto",
        company=company,
        year=year,
        quarter=quarter
    )
    
    answer = result["answer"]
    citations = result.get("citations", [])
    source_docs = result.get("source_documents", [])
    
    # Cache retrieved documents and their snippets for frontend retrieval
    ACTIVE_RETRIEVED_DOCS[thread_id] = [
        {
            "citation": format_source_citation(doc.metadata),
            "snippet": doc.page_content.strip()
        }
        for doc in source_docs
    ]
    
    if citations:
        # Deduplicate citations while preserving order
        unique_citations = list(dict.fromkeys(citations))
        sources_str = "\n\n### Source Documents:\n" + "\n".join(f"- {c}" for c in unique_citations)
        return answer + sources_str
        
    return answer

@tool
def get_kpis(company: Optional[str] = None, year: Optional[int] = None, quarter: Optional[str] = None) -> str:
    """Retrieve structured financial KPIs from the database.
    Use this tool to get actual reported numbers like Revenue, EPS, Gross Margin, and Growth Rates.
    IMPORTANT: If you need KPIs for multiple companies, you can call this tool multiple times, or omit the `company` argument to get data for all companies if needed.
    """
    engine = get_engine(str(_DB_PATH))
    Session = get_session_maker(engine)
    session = Session()
    
    q = session.query(EarningsKPI)
    if company:
        q = q.filter_by(company=company)
    if year:
        q = q.filter_by(year=year)
    if quarter:
        q = q.filter_by(quarter=quarter)
        
    kpis = q.all()
    session.close()
    
    if not kpis:
        return f"No KPIs found for {company} (Year: {year}, Quarter: {quarter}) in the structured database. CRITICAL INSTRUCTION: Do not give up. You MUST immediately call the `rag_search` tool to find these financial metrics inside the transcript vector embeddings."
        
    results = []
    for k in kpis:
        results.append({
            "period": k.period,
            "revenue_b": k.revenue_b,
            "eps": k.eps_diluted,
            "gross_margin_pct": k.gross_margin_pct,
            "net_income_b": k.net_income_b,
            "revenue_growth_yoy_pct": k.revenue_growth_yoy_pct,
            "guidance_revenue_low_b": k.guidance_revenue_low_b,
            "guidance_revenue_high_b": k.guidance_revenue_high_b
        })
        
    return json.dumps(results, indent=2)

@tool
def generate_report_sections(company: str, year: int, quarter: str) -> str:
    """Retrieves all necessary information to generate an investment brief."""
    kpi_data = get_kpis.invoke({"company": company, "year": year, "quarter": quarter})
    narrative = rag_search.invoke({
        "query": f"Summarize {company}'s performance in {year} Q{quarter}. What were the key drivers, headwinds, and forward-looking statements?",
        "company": company,
        "year": year,
        "quarter": quarter
    })
    
    return f"--- KPIs ---\n{kpi_data}\n\n--- Narrative ---\n{narrative}"
