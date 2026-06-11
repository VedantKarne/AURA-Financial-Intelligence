"""
src/extraction/kpi_extractor.py
===============================
Extracts financial KPIs from transcripts using structured LLM output.
"""

from __future__ import annotations

import logging
import os
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

class ExtractedKPIs(BaseModel):
    """Structured output for LLM to extract KPIs from text."""
    revenue_b: Optional[float] = Field(default=None, description="Revenue in billions of dollars. E.g. for $94.9 billion, return 94.9")
    eps_diluted: Optional[float] = Field(default=None, description="Diluted EPS (Earnings Per Share) in dollars.")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin percentage. E.g. for 46.2%, return 46.2")
    net_income_b: Optional[float] = Field(default=None, description="Net income in billions of dollars.")
    op_cash_flow_b: Optional[float] = Field(default=None, description="Operating cash flow in billions of dollars.")

    guidance_revenue_low_b: Optional[float] = Field(default=None, description="Guidance for next quarter revenue low end in billions of dollars.")
    guidance_revenue_high_b: Optional[float] = Field(default=None, description="Guidance for next quarter revenue high end in billions of dollars.")
    guidance_gm_low_pct: Optional[float] = Field(default=None, description="Guidance for next quarter gross margin low end percentage.")
    guidance_gm_high_pct: Optional[float] = Field(default=None, description="Guidance for next quarter gross margin high end percentage.")

    revenue_growth_yoy_pct: Optional[float] = Field(default=None, description="Revenue growth year-over-year percentage.")
    eps_growth_yoy_pct: Optional[float] = Field(default=None, description="EPS growth year-over-year percentage.")

    segment_notes: Optional[str] = Field(default=None, description="Short text summarizing key segment metrics and performance.")

def extract_kpis_from_text(text: str, metadata: dict) -> EarningsKPI:
    """
    Extract structured financial KPIs from the provided text using an LLM.

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

    # Using qwen/qwen3-32b as it is supported
    llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.0,
        max_tokens=1024,
    )
    
    # Bind structured output
    structured_llm = llm.with_structured_output(ExtractedKPIs)

    prompt = (
        f"You are an expert financial analyst. Extract the exact financial metrics from the following "
        f"earnings call summary for {metadata['company']} ({metadata['ticker']}) "
        f"for {metadata['period']}.\n\n"
        f"If a metric is not explicitly mentioned, leave it as null. Do not guess or calculate values yourself.\n\n"
        f"--- TRANSCRIPT SUMMARY ---\n"
        f"{text}\n"
        f"--------------------------\n"
    )

    logger.info(f"Extracting KPIs for {metadata['period']} ...")
    try:
        extracted: ExtractedKPIs = structured_llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Failed to extract KPIs for {metadata['period']}: {e}")
        # Return an empty KPI record if extraction fails
        extracted = ExtractedKPIs()

    # Create the SQLAlchemy model instance
    kpi_record = EarningsKPI(
        ticker=metadata.get("ticker"),
        company=metadata.get("company"),
        year=metadata.get("year"),
        quarter=metadata.get("quarter"),
        period=metadata.get("period"),

        revenue_b=extracted.revenue_b,
        eps_diluted=extracted.eps_diluted,
        gross_margin_pct=extracted.gross_margin_pct,
        net_income_b=extracted.net_income_b,
        op_cash_flow_b=extracted.op_cash_flow_b,

        guidance_revenue_low_b=extracted.guidance_revenue_low_b,
        guidance_revenue_high_b=extracted.guidance_revenue_high_b,
        guidance_gm_low_pct=extracted.guidance_gm_low_pct,
        guidance_gm_high_pct=extracted.guidance_gm_high_pct,

        revenue_growth_yoy_pct=extracted.revenue_growth_yoy_pct,
        eps_growth_yoy_pct=extracted.eps_growth_yoy_pct,

        segment_notes=extracted.segment_notes,
    )

    return kpi_record
