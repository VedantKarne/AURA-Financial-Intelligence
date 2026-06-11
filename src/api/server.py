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
        
    full_query = (
        "SYSTEM INSTRUCTION: You are a highly detailed financial assistant. "
        "You may use emojis in moderation. "
        "CRITICAL RULE ON LENGTH: The length and detail of your response MUST scale directly with the amount of source context provided. "
        "If you receive many source chunks, you must extract distinct insights from each chunk and write a much longer, comprehensive response. Do not aggressively compress or summarize away specific details. "
        "CRITICAL RULE ON CITATIONS: You MUST preserve and copy all inline citations (e.g., [Company | Quarter | Year | Section]) from the tool outputs and place them at the end of the corresponding sentences in your final text. Never remove or summarize away these inline brackets. "
        "CRITICAL RULE ON MULTI-ENTITY COVERAGE: If the query involves multiple companies or 'all companies', your final response MUST include a dedicated section for EACH company with equal depth. Do NOT skip or under-represent any company. "
        "CRITICAL RULE ON COMPARISON TABLES: If the query asks for a comparison, side-by-side analysis, or involves multiple companies, you MUST include a clean markdown comparison table summarising key metrics. "
        "TABLE CONTENT & CITATION SAFETY: Inside table cells, you MUST include a concise descriptive summary of the insight (e.g., '800 bps FX impact' or 'Cloud demand dropping'). Do NOT leave the cell blank or put only citation brackets. NEVER place full citations like [Apple | Q3 | 2023 | summary] inside table cells — the pipe characters break table formatting. Inside table cells write your concise description followed by [1], [2], [3] numeric refs only, then list the full citation key below the table. "
        "IMPORTANT: If any tool provides a 'Source Documents' or 'Sources' list, you MUST preserve it entirely and append it at the very end of your final response exactly as provided. Do not use phrases like '(Additional sources)'.\n\n"
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
