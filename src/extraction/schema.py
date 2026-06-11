"""
src/extraction/schema.py
========================
SQLAlchemy database models for extracted financial KPIs.
"""

from sqlalchemy import Column, Integer, String, Float, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EarningsKPI(Base):
    """
    SQLAlchemy model for storing extracted key performance indicators (KPIs)
    from earnings call transcripts.
    """
    __tablename__ = "earnings_kpis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)          # "AAPL"
    company = Column(String, index=True)         # "Apple"
    year = Column(Integer, index=True)           # 2024
    quarter = Column(String, index=True)         # "Q3"
    period = Column(String, index=True, unique=True) # "2024-Q3_AAPL" - to ensure no duplicates per company per quarter

    # Financial Actuals
    revenue_b = Column(Float, nullable=True)           # Revenue in $B
    eps_diluted = Column(Float, nullable=True)         # Diluted EPS
    gross_margin_pct = Column(Float, nullable=True)    # Gross margin %
    net_income_b = Column(Float, nullable=True)        # Net income in $B
    op_cash_flow_b = Column(Float, nullable=True)      # Operating cash flow in $B

    # Guidance (next quarter)
    guidance_revenue_low_b = Column(Float, nullable=True)
    guidance_revenue_high_b = Column(Float, nullable=True)
    guidance_gm_low_pct = Column(Float, nullable=True)
    guidance_gm_high_pct = Column(Float, nullable=True)

    # Growth Rates (YoY)
    revenue_growth_yoy_pct = Column(Float, nullable=True)
    eps_growth_yoy_pct = Column(Float, nullable=True)

    # Segment highlights (JSON string)
    segment_notes = Column(Text, nullable=True)        # JSON: key segment metrics


def get_engine(db_path: str):
    """Create and return a SQLAlchemy engine for the given SQLite database path."""
    engine = create_engine(f"sqlite:///{db_path}")
    return engine

def init_db(engine):
    """Create all tables in the engine."""
    Base.metadata.create_all(engine)

def get_session_maker(engine):
    """Return a sessionmaker bound to the given engine."""
    return sessionmaker(bind=engine)
