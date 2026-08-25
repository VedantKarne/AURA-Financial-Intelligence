"""
normalize_kpis.py
=================
One-shot script to fix unit inconsistencies in the earnings_kpis SQLite table.
The backfill LLM (llama3) sometimes returned raw dollar amounts (117,200,000,000)
or raw millions (9,480,000,000) instead of the required billions (117.2 / 9.48).

Detection heuristic:
  value >= 1_000_000_000  → divide by 1_000_000_000  (stored as raw dollars)
  value >= 1_000_000      → divide by 1_000_000       (stored as raw millions — rare)
  value >= 1_000          → divide by 1_000            (stored as raw millions compact)
  value < 1_000           → keep as-is                 (already in billions)

EPS and gross_margin_pct are excluded — they are inherently small numbers.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.schema import get_engine, get_session_maker, EarningsKPI

# Fields that should be in BILLIONS (apply normalisation)
BILLION_FIELDS = [
    "revenue_b",
    "net_income_b",
    "op_cash_flow_b",
    "guidance_revenue_low_b",
    "guidance_revenue_high_b",
]

def normalise_to_billions(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 1_000_000_000:
        return round(value / 1_000_000_000, 4)
    if value >= 1_000_000:
        return round(value / 1_000_000, 4)
    if value >= 10_000:
        # e.g. 55500 → 55.5B  (stored as millions without trailing zeros)
        return round(value / 1_000, 4)
    return value  # Already in billions


def main():
    engine = get_engine(str(PROJECT_ROOT / "data" / "finance_kpis.db"))
    Session = get_session_maker(engine)
    session = Session()

    rows = session.query(EarningsKPI).all()
    print(f"Normalising {len(rows)} rows...\n")

    changed = 0
    for row in rows:
        updated = False
        for field in BILLION_FIELDS:
            raw = getattr(row, field)
            fixed = normalise_to_billions(raw)
            if fixed != raw:
                print(f"  {row.company:12} {row.period}  {field}: {raw} → {fixed}")
                setattr(row, field, fixed)
                updated = True
        if updated:
            changed += 1

    session.commit()
    session.close()
    print(f"\n✅ Done. Fixed {changed} rows out of {len(rows)}.")


if __name__ == "__main__":
    main()
