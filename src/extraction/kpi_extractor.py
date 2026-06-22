"""
src/extraction/kpi_extractor.py
===============================
Extracts financial KPIs from transcripts using structured LLM output.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.extraction.schema import EarningsKPI

logger = logging.getLogger(__name__)

# .env paths searched in order (absolute path first for reliability)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATHS = [
    _PROJECT_ROOT / "config" / ".env",
    _PROJECT_ROOT / ".env",
    Path("config/.env"),
    Path(".env"),
]

def ensure_env_loaded() -> None:
    """Ensure environment variables are loaded from the configured .env paths."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        for p in _ENV_PATHS:
            if p.exists():
                load_dotenv(dotenv_path=p)
                logger.info(f"Loaded environment variables from {p}")
                return
        logger.warning("GROQ_API_KEY not found in environment and no .env file found.")

class CoreFinancials(BaseModel):
    """Pass 1: Core financial metrics."""
    revenue_b: Optional[float] = Field(default=None, description="Revenue in billions of dollars. E.g. for $94.9 billion, return 94.9")
    eps_diluted: Optional[float] = Field(default=None, description="Diluted EPS (Earnings Per Share) in dollars.")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin percentage. E.g. for 46.2%, return 46.2")
    net_income_b: Optional[float] = Field(default=None, description="Net income in billions of dollars.")
    op_cash_flow_b: Optional[float] = Field(default=None, description="Operating cash flow in billions of dollars.")

class GuidanceMetrics(BaseModel):
    """Pass 2: Forward-looking guidance."""
    guidance_revenue_low_b: Optional[float] = Field(default=None, description="Guidance for next quarter revenue low end in billions of dollars.")
    guidance_revenue_high_b: Optional[float] = Field(default=None, description="Guidance for next quarter revenue high end in billions of dollars.")
    guidance_gm_low_pct: Optional[float] = Field(default=None, description="Guidance for next quarter gross margin low end percentage.")
    guidance_gm_high_pct: Optional[float] = Field(default=None, description="Guidance for next quarter gross margin high end percentage.")

class GrowthAndSegments(BaseModel):
    """Pass 3: Growth percentages and qualitative segment notes."""
    revenue_growth_yoy_pct: Optional[float] = Field(default=None, description="Revenue growth year-over-year percentage.")
    eps_growth_yoy_pct: Optional[float] = Field(default=None, description="EPS growth year-over-year percentage.")
    segment_notes: Optional[str] = Field(default=None, description="Short text summarizing key segment metrics and performance.")

def _extract_with_retry(llm, prompt: str, schema_class, period: str, max_retries: int = 3, base_delay: int = 5):
    """Helper to run structured extraction with exponential backoff for Groq rate limits and API glitches."""
    structured_llm = llm.with_structured_output(schema_class)
    
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            wait_time = base_delay * (2 ** attempt)
            logger.warning(f"Groq API error ({e}) for {period}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
                
    logger.error(f"Exhausted retries for {period}. Returning empty {schema_class.__name__}.")
    return schema_class()

def extract_kpis_from_text(text: str, metadata: dict) -> EarningsKPI:
    """
    Extract structured financial KPIs from the provided text using a 3-pass LLM approach.

    Parameters
    ----------
    text : str
        The transcript text (usually the summary section) to extract from.
    metadata : dict
        The file metadata containing company, ticker, year, quarter, period.

    Returns
    -------
    EarningsKPI
        The SQLAlchemy model instance populated with the extracted values.
    """
    ensure_env_loaded()

    llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.0,
    )
    
    logger.info(f"Extracting KPIs via 3-pass pipeline for {metadata['period']} ...")
    
    # Truncating to 8000 characters (~2000 tokens) drastically reduces input token usage.
    truncated_text = text[:8000]

    base_prompt = (
        "You are an expert financial analyst. Extract the exact {metric_type} from the following "
        f"earnings call summary for {metadata['company']} ({metadata['ticker']}) "
        f"for {metadata['period']}.\n\n"
        "If a metric is not explicitly mentioned, leave it as null. Do not guess or calculate values yourself.\n\n"
        "--- TRANSCRIPT SUMMARY ---\n"
        f"{truncated_text}\n"
        "--------------------------\n"
    )

    core = _extract_with_retry(
        llm, 
        base_prompt.format(metric_type="core financial metrics (revenue, EPS, margins, net income, cash flow)"), 
        CoreFinancials, 
        metadata['period']
    )
    
    guidance = _extract_with_retry(
        llm, 
        base_prompt.format(metric_type="forward-looking guidance for the next quarter"), 
        GuidanceMetrics, 
        metadata['period']
    )
    
    growth = _extract_with_retry(
        llm, 
        base_prompt.format(metric_type="year-over-year growth percentages and segment performance notes"), 
        GrowthAndSegments, 
        metadata['period']
    )

    kpi_record = EarningsKPI(
        ticker=metadata.get("ticker"),
        company=metadata.get("company"),
        year=metadata.get("year"),
        quarter=metadata.get("quarter"),
        period=metadata.get("period"),

        revenue_b=core.revenue_b,
        eps_diluted=core.eps_diluted,
        gross_margin_pct=core.gross_margin_pct,
        net_income_b=core.net_income_b,
        op_cash_flow_b=core.op_cash_flow_b,

        guidance_revenue_low_b=guidance.guidance_revenue_low_b,
        guidance_revenue_high_b=guidance.guidance_revenue_high_b,
        guidance_gm_low_pct=guidance.guidance_gm_low_pct,
        guidance_gm_high_pct=guidance.guidance_gm_high_pct,

        revenue_growth_yoy_pct=growth.revenue_growth_yoy_pct,
        eps_growth_yoy_pct=growth.eps_growth_yoy_pct,

        segment_notes=growth.segment_notes,
    )

    return kpi_record
