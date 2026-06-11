# Monkeypatch importlib.metadata.version to bypass transformers strict range check for tokenizers on Python 3.14+
import importlib.metadata
_orig_metadata_version = importlib.metadata.version
importlib.metadata.version = lambda pkg: "0.23.0" if pkg == "tokenizers" else _orig_metadata_version(pkg)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables before doing anything else
env_path = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path, override=True)
from pydantic import BaseModel
from typing import Optional
from src.agents.orchestrator import run_agent_query
from src.agents.tools import get_kpis, set_global_k
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Finance RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    company: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None
    section: Optional[str] = None
    k: Optional[int] = 6
    thread_id: Optional[str] = "default"

class ReportRequest(BaseModel):
    company: str
    year: int
    quarter: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Finance RAG API running"}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    # Pass to LangGraph orchestrator
    if req.k is not None:
        set_global_k(req.k)
        
    # If company/year/quarter are provided, we should hint them in the query
    context_hint = ""
    if req.company:
        context_hint += f" [Context: Company={req.company}"
        if req.year: context_hint += f", Year={req.year}"
        if req.quarter: context_hint += f", Quarter={req.quarter}"
        context_hint += "] "
        
    # NOTE: Detailed formatting rules (citations, comparison tables, multi-entity coverage)
    # are enforced inside the inner RAG SYSTEM_PROMPT (prompts.py) which is invoked on
    # every rag_search tool call. Keep this outer instruction lean to conserve TPM quota.
    full_query = (
        "You are a financial assistant. Answer in detail proportional to the context provided. "
        "Use emojis sparingly. "
        "CITATIONS: Copy all inline citations [Company | Quarter | Year | Section] from tool outputs into your final text verbatim. "
        "MULTI-ENTITY: Include a dedicated section for EACH company and a markdown comparison table for comparisons. "
        "TABLE CELLS: Write a concise description in every cell (e.g. '800 bps FX impact'), then append a numeric ref [1]. Never put full citation brackets inside table cells. "
        "SOURCES: Append the full 'Source Documents' list from tool output at the end of your response exactly as given.\n\n"
        f"USER QUERY: {context_hint}{req.query}"
    )
    response = run_agent_query(full_query, thread_id=req.thread_id or "default")
    from src.agents.tools import ACTIVE_RETRIEVED_DOCS
    snippets = ACTIVE_RETRIEVED_DOCS.pop(req.thread_id or "default", [])
    return {"message": response, "sources": snippets}

@app.get("/api/kpis")
async def get_kpis_endpoint(company: Optional[str] = None, year: Optional[int] = None, quarter: Optional[str] = None):
    # Use the tool directly for the KPI dashboard
    import json
    kpi_str = get_kpis.invoke({"company": company, "year": year, "quarter": quarter})
    if kpi_str.startswith("No KPIs"):
        return {"kpis": []}
    return {"kpis": json.loads(kpi_str)}

@app.post("/api/generate-report")
async def generate_report(req: ReportRequest):
    prompt = f"CRITICAL SYSTEM INSTRUCTION: DO NOT use ANY emojis in your response. This is a formal business document.\n\nGenerate a highly detailed and professional investment research brief for {req.company} in {req.year} {req.quarter}. Use the generate_report_sections tool to gather the facts, then synthesize a complete markdown report."
    response = run_agent_query(prompt)
    return {"report": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
