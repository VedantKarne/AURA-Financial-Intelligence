import sys
import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import discover_transcript_files, DEFAULT_DATA_DIR
from src.ingestion.file_parser import parse_filepath
from src.extraction.schema import get_engine, init_db, get_session_maker, EarningsKPI
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Single combined schema ─────────────────────────────────────────────────────
# Instead of 3 separate LLM calls (69 total), we extract everything in ONE call
# per file (23 total). This reduces peak memory by 3× and runs 3× faster.
class AllKPIs(BaseModel):
    """Combined KPI schema – extracted in a single LLM pass."""
    # Core financials
    revenue_b:              Optional[float] = Field(default=None, description="Revenue in billions USD. E.g. $94.9B → 94.9")
    eps_diluted:            Optional[float] = Field(default=None, description="Diluted EPS in dollars.")
    gross_margin_pct:       Optional[float] = Field(default=None, description="Gross margin %. E.g. 46.2% → 46.2")
    net_income_b:           Optional[float] = Field(default=None, description="Net income in billions USD.")
    op_cash_flow_b:         Optional[float] = Field(default=None, description="Operating cash flow in billions USD.")
    # Guidance
    guidance_revenue_low_b: Optional[float] = Field(default=None, description="Next-quarter revenue guidance low end in billions USD.")
    guidance_revenue_high_b:Optional[float] = Field(default=None, description="Next-quarter revenue guidance high end in billions USD.")
    guidance_gm_low_pct:    Optional[float] = Field(default=None, description="Next-quarter gross margin guidance low end %.")
    guidance_gm_high_pct:   Optional[float] = Field(default=None, description="Next-quarter gross margin guidance high end %.")
    # Growth & segments
    revenue_growth_yoy_pct: Optional[float] = Field(default=None, description="Revenue growth year-over-year %.")
    eps_growth_yoy_pct:     Optional[float] = Field(default=None, description="EPS growth year-over-year %.")
    segment_notes:          Optional[str]   = Field(default=None, description="Short text summarising key segment metrics and performance.")


# ── Core extraction helper ─────────────────────────────────────────────────────
async def extract_all_kpis(llm, text: str, metadata: dict) -> AllKPIs:
    """Single-pass async extraction. One LLM call, one context window, done."""
    structured_llm = llm.with_structured_output(AllKPIs)

    prompt = (
        "You are an expert financial analyst. Extract ALL of the following financial metrics "
        f"from the earnings call summary for {metadata['company']} ({metadata['ticker']}) "
        f"for {metadata['period']}.\n\n"
        "Return null for any metric that is not explicitly stated. Do NOT guess or calculate.\n\n"
        "Metrics to extract:\n"
        "  • revenue_b, eps_diluted, gross_margin_pct, net_income_b, op_cash_flow_b\n"
        "  • guidance_revenue_low_b, guidance_revenue_high_b, guidance_gm_low_pct, guidance_gm_high_pct\n"
        "  • revenue_growth_yoy_pct, eps_growth_yoy_pct, segment_notes\n\n"
        "--- TRANSCRIPT SUMMARY ---\n"
        f"{text[:6000]}\n"
        "--------------------------\n"
    )

    try:
        return await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.error(f"Ollama extraction failed for {metadata['period']}: {e}")
        return AllKPIs()


# ── Per-file worker ────────────────────────────────────────────────────────────
async def process_file(filepath: Path, semaphore: asyncio.Semaphore, results_map: dict):
    """Acquires the semaphore, processes one file, releases it."""
    async with semaphore:
        result = parse_filepath(filepath)
        if not result:
            logger.warning(f"Could not parse {filepath.name}. Skipping.")
            return

        metadata, raw_text = result
        period = metadata["period"]
        summary_text = raw_text.split("[ ")[0] if "[ " in raw_text else raw_text

        llm = ChatOllama(model="llama3", temperature=0.0, format="json")

        ticker = metadata.get("ticker", "UNK")
        map_key = f"{ticker}_{period}"  # composite key prevents AAPL/MSFT/NVDA collisions
        logger.info(f"[{map_key}] Starting single-pass extraction ...")
        kpis = await extract_all_kpis(llm, summary_text, metadata)
        logger.info(f"[{map_key}] Done!")

        results_map[map_key] = {
            "ticker":                   metadata.get("ticker"),
            "company":                  metadata.get("company"),
            "year":                     metadata.get("year"),
            "quarter":                  metadata.get("quarter"),
            "period":                   period,
            "revenue_b":                kpis.revenue_b,
            "eps_diluted":              kpis.eps_diluted,
            "gross_margin_pct":         kpis.gross_margin_pct,
            "net_income_b":             kpis.net_income_b,
            "op_cash_flow_b":           kpis.op_cash_flow_b,
            "guidance_revenue_low_b":   kpis.guidance_revenue_low_b,
            "guidance_revenue_high_b":  kpis.guidance_revenue_high_b,
            "guidance_gm_low_pct":      kpis.guidance_gm_low_pct,
            "guidance_gm_high_pct":     kpis.guidance_gm_high_pct,
            "revenue_growth_yoy_pct":   kpis.revenue_growth_yoy_pct,
            "eps_growth_yoy_pct":       kpis.eps_growth_yoy_pct,
            "segment_notes":            kpis.segment_notes,
        }


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def run_async_backfill():
    sql_db_path = PROJECT_ROOT / "data" / "finance_kpis.db"
    engine = get_engine(str(sql_db_path))

    logger.info("Dropping and recreating the earnings_kpis table ...")
    EarningsKPI.__table__.drop(engine, checkfirst=True)
    init_db(engine)
    Session = get_session_maker(engine)
    session = Session()

    files = discover_transcript_files(DEFAULT_DATA_DIR)
    logger.info(f"Found {len(files)} transcript files. Each will use ONE LLM call.")

    results_map: dict = {}

    # Semaphore(1) → strictly one file processed at a time.
    # One context window open at a time. Maximum memory safety.
    semaphore = asyncio.Semaphore(1)
    tasks = [process_file(f, semaphore, results_map) for f in files]

    logger.info("Starting sequential-safe async processing ...")
    await asyncio.gather(*tasks)

    df = pd.DataFrame.from_dict(results_map, orient="index")
    logger.info(f"\nDataFrame ({len(df)} rows):\n{df[['company', 'quarter', 'revenue_b']].to_string()}")

    kpi_records = [EarningsKPI(**row.to_dict()) for _, row in df.iterrows()]
    session.add_all(kpi_records)
    session.commit()
    logger.info(f"Successfully committed {len(kpi_records)} KPI records to SQLite!")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_async_backfill())
