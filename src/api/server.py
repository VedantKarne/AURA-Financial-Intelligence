# Monkeypatch importlib.metadata.version to bypass transformers strict range check for tokenizers on Python 3.14+
import importlib.metadata
_orig_metadata_version = importlib.metadata.version
importlib.metadata.version = lambda pkg: "0.23.0" if pkg == "tokenizers" else _orig_metadata_version(pkg)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import asyncio
import json
import os
from pathlib import Path

# Load environment variables before doing anything else
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
from pydantic import BaseModel
from typing import Optional
from src.agents.orchestrator import run_agent_query
from src.agents.tools import get_kpis, set_global_k
from src.agents.events import subscribe, unsubscribe, register_loop
from pydantic import BaseModel
from typing import Optional
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="Finance RAG API", version="1.0.0")

@app.on_event("startup")
async def _store_event_loop():
    """Store the running event loop so emit() can bridge sync threadpool → async queues."""
    register_loop(asyncio.get_running_loop())

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


@app.get("/api/workflow-stream")
async def workflow_stream(request: Request):
    """
    Server-Sent Events endpoint for the Agent Workflow Monitor.

    The frontend opens EventSource('http://localhost:8000/api/workflow-stream').
    Every time run_agent_query() emits an event through events.py, it arrives
    here as a JSON-encoded SSE message.

    Sends a keepalive ping every 30 s so the browser doesn't close the connection.
    """
    queue = subscribe()

    async def generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive ping — prevents proxy / browser timeout
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )



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
    response = await run_in_threadpool(run_agent_query, full_query, thread_id=req.thread_id or "default")
    from src.agents.tools import ACTIVE_RETRIEVED_DOCS
    snippets = ACTIVE_RETRIEVED_DOCS.pop(req.thread_id or "default", [])
    return {"message": response, "sources": snippets}

@app.get("/api/kpis")
async def get_kpis_endpoint(company: Optional[str] = None, year: Optional[int] = None, quarter: Optional[str] = None):
    # Use the tool directly for the KPI dashboard
    import json
    kpi_str = await run_in_threadpool(get_kpis.invoke, {"company": company, "year": year, "quarter": quarter})
    if kpi_str.startswith("No KPIs"):
        return {"kpis": []}
    return {"kpis": json.loads(kpi_str)}

@app.post("/api/generate-report")
async def generate_report(req: ReportRequest):
    import datetime
    current_date = datetime.date.today().strftime("%B %d, %Y")
    prompt = (
        f"CRITICAL SYSTEM INSTRUCTION: DO NOT use ANY emojis in your response. This is a formal business document.\n\n"
        f"Generate a comprehensive, highly detailed professional investment research brief for {req.company} ({req.year} {req.quarter}).\n\n"
        f"STEP 1: Call the generate_report_sections tool with company='{req.company}', year={req.year}, quarter='{req.quarter}' to retrieve all data.\n\n"
        f"STEP 2: Using that retrieved data, synthesize a complete markdown investment brief that MUST contain ALL of the following sections in order:\n\n"
        f"# {req.company} {req.quarter} {req.year} Investment Research Brief\n\n"
        f"**Date:** {req.year} {req.quarter} | **Company:** {req.company}\n\n"
        f"---\n\n"
        f"## 1. Executive Summary\n"
        f"Write 2-3 paragraphs summarizing the quarter's key highlights — total revenue, revenue growth (YoY), operating income, operating margin, and EPS. "
        f"Include a bullet list of the 3-5 most important headline figures.\n\n"
        f"## 2. Key Financial Metrics\n"
        f"Create a detailed markdown table with columns: Metric | Value | Notes. "
        f"Include at minimum: Revenue, Revenue Growth (YoY), Operating Income, Operating Margin, EPS, Net Income, Gross Margin, and any other relevant KPIs from the data. "
        f"If structured KPIs are unavailable, derive metrics from the narrative and note this.\n\n"
        f"## 3. Segment Analysis\n"
        f"Break down performance by each major business segment or product line. Use sub-headers (###) for each segment. "
        f"Include revenue contribution, growth rate, and key drivers for each segment.\n\n"
        f"## 4. Strategic Drivers\n"
        f"### Growth Catalysts\n"
        f"List and explain 3-5 numbered strategic growth drivers (e.g. AI momentum, new products, geographic expansion, acquisitions).\n\n"
        f"### Key Headwinds\n"
        f"List and explain 2-3 notable headwinds or challenges management acknowledged.\n\n"
        f"## 5. Forward-Looking Guidance\n"
        f"Report management's official guidance for the next quarter and full year. Include revenue ranges, growth expectations, and margin outlook as sub-bullets.\n\n"
        f"## 6. Investment Implications\n"
        f"Write a 2-3 paragraph investment thesis. Cover: (a) the bull case based on the quarter's momentum, (b) key risks to monitor, "
        f"and (c) a one-sentence Recommendation (e.g. 'Maintain overweight...', 'Initiate at neutral...').\n\n"
        f"**Sources:** {req.company} {req.quarter} {req.year} Earnings Call Transcript, Narrative Analysis\n"
        f"**Prepared By:** Vedant Karne's AURA Intelligence Engine\n"
        f"**Date:** {current_date}\n\n"
        f"---\n\n"
        f"*Note: This document is for informational purposes only and does not constitute financial advice.*\n\n"
        f"CRITICAL RULES:\n"
        f"- Do NOT skip or abbreviate any section. Every section (1 through 6) must be present and substantive.\n"
        f"- Do NOT output raw tool data. Synthesize and narrate the findings professionally.\n"
        f"- Do NOT use emojis anywhere in the document.\n"
        f"- Back up every major claim with inline citations from the source data."
    )
    response = await run_in_threadpool(run_agent_query, prompt)
    return {"report": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
