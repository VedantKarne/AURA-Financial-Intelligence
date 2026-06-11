"""
src/ui/app.py
=============
Streamlit UI for the Financial Earnings Intelligence Platform — Phase 1.

Features:
  • Dark finance-themed design with custom CSS
  • Company / Year / Quarter / Section sidebar filters
  • Chat interface with real-time streaming responses
  • Citation cards for every answer (source + excerpt)
  • Session chat history (last 10 exchanges)
  • Vector store status indicator
  • One-click ingestion trigger if store is empty

Run:
  streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import src.retrieval.vector_store
importlib.reload(src.retrieval.vector_store)
import src.retrieval.bm25_retriever
importlib.reload(src.retrieval.bm25_retriever)
import src.retrieval.hybrid_retriever
importlib.reload(src.retrieval.hybrid_retriever)
import src.retrieval.reranker
importlib.reload(src.retrieval.reranker)
import src.retrieval.router
importlib.reload(src.retrieval.router)
import src.retrieval.query_transformer
importlib.reload(src.retrieval.query_transformer)
import src.generation.qa_chain
importlib.reload(src.generation.qa_chain)

from src.retrieval.vector_store import EarningsVectorStore
from src.generation.qa_chain import get_answer, get_answer_streaming

def get_diagnostics_html(msg: dict) -> str:
    routing_mode = msg.get("routing_mode", "auto")
    routing_strategy = msg.get("routing_strategy", "fixed")
    routing_reason = msg.get("routing_reason", "")
    rewritten_query = msg.get("rewritten_query")
    multi_queries = msg.get("multi_queries")
    detected_entities = msg.get("detected_entities")
    coverage_before_rerank = msg.get("coverage_before_rerank")
    
    # Calculate coverage after rerank
    coverage_after = {}
    sources_list = msg.get("sources") or msg.get("source_documents") or []
    for doc in sources_list:
        co = doc.metadata.get("company", "Unknown")
        coverage_after[co] = coverage_after.get(co, 0) + 1
        
    mode_labels = {
        "auto": "Auto Router (Dynamic)",
        "vector": "Vector-only (Naive)",
        "bm25": "BM25-only (Keyword)",
        "hybrid": "Hybrid Search (RRF)",
        "rerank": "Hybrid + Reranking (Cross-Encoder)"
    }
    mode_display = mode_labels.get(routing_mode, str(routing_mode).upper())
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e2535;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 60px 8px 0;
        font-size: 0.82rem;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="color: #38bdf8; font-size: 1rem;">⚙️</span>
            <strong style="color: #f1f5f9; font-family: 'Inter', sans-serif; letter-spacing: 0.3px;">Analyst Query Routing Diagnostics</strong>
        </div>
        <div style="color: #94a3b8; font-family: 'Inter', sans-serif; line-height: 1.6;">
            <div><strong>Active Mode:</strong> <span style="color: #38bdf8; font-weight: 500;">{mode_display}</span> &middot; Strategy: <code style="color: #818cf8; font-family: 'JetBrains Mono', monospace;">{routing_strategy}</code></div>
    """
    if routing_reason:
        html += f"<div><strong>Reasoning:</strong> {routing_reason}</div>"
    if detected_entities:
        entities_str = ", ".join(detected_entities)
        html += f"<div style='margin-top: 4px;'><strong>🎯 Detected Entities:</strong> <span style='color: #38bdf8; font-weight: 500;'>{entities_str}</span></div>"
        
    # Append coverage display if present
    if coverage_before_rerank or coverage_after:
        all_cos = sorted(list(set(list(coverage_before_rerank.keys() if coverage_before_rerank else []) + list(coverage_after.keys()))))
        pills_before = []
        pills_after = []
        for co in all_cos:
            count_b = coverage_before_rerank.get(co, 0) if coverage_before_rerank else 0
            count_a = coverage_after.get(co, 0)
            icon = "🍎" if co == "Apple" else ("🪟" if co == "Microsoft" else ("🟢" if co == "Nvidia" else "🏢"))
            pills_before.append(f"<span style='color:#cbd5e1;'>{icon} {co}: {count_b}</span>")
            pills_after.append(f"<span style='color:#38bdf8;'>{icon} {co}: {count_a}</span>")
            
        html += f"""
        <div style="margin-top: 6px; border-top: 1px solid #1e2535; padding-top: 6px; font-size: 0.78rem;">
            <div><strong>📊 Retrieval Coverage (Before Rerank):</strong> {" &middot; ".join(pills_before)}</div>
            <div style="margin-top: 2px;"><strong>🎯 Reranker Selection (After Rerank):</strong> {" &middot; ".join(pills_after)}</div>
        </div>
        """
        
    if rewritten_query:
        html += f"<div style='margin-top: 4px;'><strong>🔄 Conversational Rewrite:</strong> <span style='color: #a78bfa; font-style: italic;'>\"{rewritten_query}\"</span></div>"
    if multi_queries:
        mq_list = "".join(f"<li>\"{q}\"</li>" for q in multi_queries)
        html += f"""
        <div style="margin-top: 6px;">
            <strong>🔍 Expanded Sub-Queries:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 20px; color: #cbd5e1; font-style: italic;">
                {mq_list}
            </ul>
        </div>
        """
    html += """
        </div>
    </div>
    """
    return html

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Earnings Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark finance aesthetic ─────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0d14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e2535;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Header gradient text */
.brand-header {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}

.brand-sub {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 400;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

/* Chat container */
.chat-container {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* User message bubble */
.user-bubble {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e2f4f 100%);
    border: 1px solid #2d4a6e;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 60px;
    color: #bfdbfe;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Assistant message bubble */
.assistant-bubble {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e2535;
    border-left: 3px solid #38bdf8;
    border-radius: 4px 12px 12px 12px;
    padding: 14px 16px;
    margin: 8px 60px 8px 0;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.7;
}

/* Citation card */
.citation-card {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-left: 3px solid #818cf8;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
}

.citation-tag {
    display: inline-block;
    background: #1e2535;
    color: #818cf8;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 6px;
}

.citation-excerpt {
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.5;
    font-style: italic;
    margin: 0;
    border-top: 1px solid #1e2535;
    padding-top: 6px;
    margin-top: 4px;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

.status-ok {
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
}

.status-warn {
    background: #431407;
    color: #fb923c;
    border: 1px solid #9a3412;
}

/* Filter section header */
.filter-header {
    color: #475569;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 16px;
}

/* Metric tiles */
.metric-row {
    display: flex;
    gap: 8px;
    margin: 12px 0;
}

.metric-tile {
    flex: 1;
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Input styling */
.stTextInput > div > div > input {
    background: #0f1420 !important;
    border: 1px solid #1e2535 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Dividers */
hr {
    border-color: #1e2535 !important;
    margin: 12px 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2d3748; }

/* Selectbox */
.stSelectbox > div > div {
    background: #0f1420 !important;
    border: 1px solid #1e2535 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"assistant", "content": ..., "meta": ...}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "store_count" not in st.session_state:
    st.session_state.store_count = 0


# ── Helper: load vector store (cached across reruns) ─────────────────────────
@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store() -> EarningsVectorStore:
    db_path = PROJECT_ROOT / "data" / "chroma_db"
    return EarningsVectorStore(db_path=str(db_path))


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-header">📈 FinIntel</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">EARNINGS INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
    st.divider()

    # -- Vector store status
    try:
        vs = load_vector_store()
        st.session_state.vector_store = vs
        count = vs.count()
        st.session_state.store_count = count

        if count > 0:
            st.markdown(
                f'<span class="status-badge status-ok">● Vector Store Ready — {count:,} chunks</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-warn">⚠ Vector Store Empty</span>',
                unsafe_allow_html=True,
            )
            st.warning(
                "Run the ingestion pipeline to populate the vector store:\n\n"
                "```\npython -m src.ingestion.pipeline\n```"
            )
    except Exception as e:
        st.error(f"Could not load vector store: {e}")
        st.session_state.vector_store = None

    st.divider()

    # -- Filters
    st.markdown('<p class="filter-header">🔍 Search Filters</p>', unsafe_allow_html=True)

    company_options = ["All Companies", "Apple", "Microsoft", "Nvidia"]
    selected_company = st.selectbox(
        "Company",
        options=company_options,
        index=0,
        key="filter_company",
    )

    year_options = ["All Years", 2023, 2024]
    selected_year = st.selectbox(
        "Year",
        options=year_options,
        index=0,
        key="filter_year",
    )

    quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4"]
    selected_quarter = st.selectbox(
        "Quarter",
        options=quarter_options,
        index=0,
        key="filter_quarter",
    )

    section_options = ["All Sections", "transcript", "summary"]
    selected_section = st.selectbox(
        "Section",
        options=section_options,
        index=0,
        key="filter_section",
    )

    st.divider()

    # -- Retrieval settings
    st.markdown('<p class="filter-header">⚙ Retrieval Settings</p>', unsafe_allow_html=True)
    selected_mode = st.selectbox(
        "Retrieval Mode",
        options=["auto", "vector", "bm25", "hybrid", "rerank"],
        format_func=lambda x: {
            "auto": "Auto Router (Dynamic)",
            "vector": "Vector-only (Naive)",
            "bm25": "BM25-only (Keyword)",
            "hybrid": "Hybrid Search (RRF)",
            "rerank": "Hybrid + Reranking (Cross-Encoder)"
        }[x],
        index=0,
        key="retrieval_mode",
    )
    top_k = st.slider("Top-K chunks", min_value=3, max_value=20, value=12, key="top_k")

    st.divider()

    # -- Clear chat
    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # -- Dataset info
    st.divider()
    st.markdown('<p class="filter-header">📂 Dataset Coverage</p>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.78rem; color:#64748b; line-height:1.8">
🍎 Apple · Q1–Q4 2023 · Q1–Q2 2024<br>
🪟 Microsoft · Q1–Q4 2023 · Q1–Q4 2024<br>
🟢 Nvidia · Q1–Q4 2023 · Q1–Q4 2024<br>
<br>
<span style="color:#475569">23 transcripts · ~1.15 MB · ~289K tokens</span>
</div>
""", unsafe_allow_html=True)


# ── Main layout ──────────────────────────────────────────────────────────────
tab_chat, tab_kpi = st.tabs(['💬 Intelligence Chat', '📊 KPI Dashboard'])

with tab_chat:
    col_main, col_sources = st.columns([3, 2])

    with col_main:
        st.markdown("## 💬 Earnings Q&A")

        # -- Active filter display
        active_filters = []
        if selected_company != "All Companies":
            active_filters.append(f"🏢 {selected_company}")
        if selected_year != "All Years":
            active_filters.append(f"📅 {selected_year}")
        if selected_quarter != "All Quarters":
            active_filters.append(f"Q {selected_quarter}")
        if selected_section != "All Sections":
            active_filters.append(f"📄 {selected_section}")

        if active_filters:
            st.markdown(
                f"<div style='font-size:0.8rem; color:#64748b; margin-bottom:8px'>"
                f"Active filters: {' · '.join(active_filters)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.8rem; color:#475569; margin-bottom:8px'>"
                "Searching across all companies, years, and quarters.</div>",
                unsafe_allow_html=True,
            )

        # -- Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Render diagnostics if present
                    diag_html = ""
                    if any(msg.get(k) for k in ["rewritten_query", "routing_strategy", "multi_queries"]):
                        diag_html = get_diagnostics_html(msg)
                    if diag_html:
                        st.markdown(diag_html, unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="assistant-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # -- Question input
        st.divider()

        with st.form(key="question_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_question = st.text_input(
                    label="question_input",
                    placeholder="Ask anything about the earnings calls... e.g. 'What was Apple's gross margin in Q3 2024?'",
                    label_visibility="collapsed",
                    key="question_input",
                )
            with col_btn:
                submitted = st.form_submit_button("Ask →", use_container_width=True)

        # -- Suggested questions
        st.markdown(
            "<div style='font-size:0.75rem; color:#475569; margin-top:4px'>"
            "💡 Try: <em>What was Apple's gross margin in Q3 2024?</em> · "
            "<em>What did Jensen Huang say about sovereign AI?</em> · "
            "<em>What was Microsoft's Azure revenue growth in Q4 2024?</em>"
            "</div>",
            unsafe_allow_html=True,
        )


    # ── Sources column ────────────────────────────────────────────────────────────
    with col_sources:
        st.markdown("## 📎 Source Citations")
        sources_placeholder = st.empty()

        # Show metrics
        st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-value">{st.session_state.store_count:,}</div>
            <div class="metric-label">Indexed Chunks</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{len(st.session_state.chat_history) // 2}</div>
            <div class="metric-label">Questions Asked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

        # Render citations for the last message in history if it exists
        last_assistant_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "assistant" and "sources" in msg:
                last_assistant_msg = msg
                break

        if last_assistant_msg:
            with sources_placeholder.container():
                # Count coverage of each company in the retrieved sources
                coverage = {}
                for doc in last_assistant_msg["sources"]:
                    co = doc.metadata.get("company", "Unknown")
                    coverage[co] = coverage.get(co, 0) + 1

                if coverage:
                    pills = []
                    for co in sorted(coverage.keys()):
                        count = coverage[co]
                        icon = "🏢"
                        if co == "Apple":
                            icon = "🍎"
                        elif co == "Microsoft":
                            icon = "🪟"
                        elif co == "Nvidia":
                            icon = "🟢"
                        pills.append(f"<span style='background:#111827; color:#38bdf8; border-radius:12px; padding:4px 10px; font-size:0.75rem; font-weight:600; border:1px solid #1e2535; margin-right:6px;'>{icon} {co}: {count}</span>")
                    pills_html = " ".join(pills)
                    st.markdown(f"""
                    <div style="margin-bottom:14px; margin-top:4px;">
                        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; font-weight:500;">Retrieved Coverage:</div>
                        <div style="display:flex; flex-wrap:wrap; gap:4px;">{pills_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**Answer cited {len(last_assistant_msg['sources'])} sources:**")
                for i, (doc, citation) in enumerate(
                    zip(last_assistant_msg["sources"], last_assistant_msg["citations"]), start=1
                ):
                    meta = doc.metadata
                    excerpt = doc.page_content[:280].strip()
                    if len(doc.page_content) > 280:
                        excerpt += "..."
                    # Build score details
                    score_details = []
                    if "rerank_score" in meta:
                        score_details.append(f"🎯 Rerank: {meta['rerank_score']:.3f}")
                    if "rrf_score" in meta:
                        score_details.append(f"🔀 RRF: {meta['rrf_score']:.4f} (Rank {meta.get('rrf_rank', '?')})")
                    if "vector_rank" in meta and meta["vector_rank"] != -1:
                        score_details.append(f"🔮 Vector Rank: {meta['vector_rank']}")
                    if "bm25_rank" in meta and meta["bm25_rank"] != -1:
                        bm25_score_str = f" ({meta['bm25_score']:.2f})" if "bm25_score" in meta else ""
                        score_details.append(f"📝 BM25 Rank: {meta['bm25_rank']}{bm25_score_str}")
                    elif "bm25_score" in meta:
                        score_details.append(f"📝 BM25 Score: {meta['bm25_score']:.2f}")

                    score_html = ""
                    if score_details:
                        score_pills = " · ".join(
                            f"<span style='background:#1e2535; color:#94a3b8; border-radius:4px; padding:2px 6px; font-family:\"JetBrains Mono\", monospace; font-size:0.7rem; margin-right:4px; border: 1px solid #2d3748;'>{detail}</span>"
                            for detail in score_details
                        )
                        score_html = f"<div style='margin-top:4px; margin-bottom:8px;'>{score_pills}</div>"

                    st.markdown(f"""
    <div class="citation-card">
        <div>
            <span style="color:#64748b;font-size:0.75rem;font-weight:600">#{i}</span>
            &nbsp;
            <span class="citation-tag">{citation}</span>
            &nbsp;
            <span style="color:#475569;font-size:0.72rem">chunk {meta.get('chunk_index', '?')}</span>
        </div>
        {score_html}
        <p class="citation-excerpt">{excerpt}</p>
    </div>
    """, unsafe_allow_html=True)


    # ── Process question ──────────────────────────────────────────────────────────
    if submitted and user_question.strip():
        vs = st.session_state.vector_store

        if vs is None or vs.count() == 0:
            st.error(
                "Vector store is empty or not loaded. "
                "Please run the ingestion pipeline first:\n\n"
                "```python -m src.ingestion.pipeline```"
            )
        else:
            # Resolve filter values
            company_filter  = None if selected_company == "All Companies" else selected_company
            year_filter     = None if selected_year == "All Years" else int(selected_year)
            quarter_filter  = None if selected_quarter == "All Quarters" else selected_quarter
            section_filter  = None if selected_section == "All Sections" else selected_section

            # Add user message to history
            st.session_state.chat_history.append({
                "role":    "user",
                "content": user_question,
            })

            # Render the user bubble manually since it won't be rendered by the loop until rerun
            with col_main:
                st.markdown(
                    f'<div class="user-bubble">🧑 {user_question}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.last_stream_metadata = None

            # Stream the answer
            with col_main:
                with st.spinner("Routing query and retrieving documents..."):
                    stream = get_answer_streaming(
                        question=user_question,
                        vector_store=vs,
                        k=top_k,
                        retrieval_mode=st.session_state.get("retrieval_mode", "auto"),
                        company=company_filter,
                        year=year_filter,
                        quarter=quarter_filter,
                        section=section_filter,
                        chat_history=st.session_state.chat_history,
                    )
                
                    # Fetch first token to trigger retrieval and metadata populating
                    try:
                        first_token = next(stream)
                    except StopIteration:
                        first_token = ""
            
                # Now retrieval is complete, clear the spinner and render diagnostics
                diag_placeholder = st.empty()
                bubble_placeholder = st.empty()
            
                metadata = st.session_state.get("last_stream_metadata", {})
                if metadata and any(metadata.get(k) for k in ["rewritten_query", "routing_strategy", "multi_queries"]):
                    diag_html = get_diagnostics_html(metadata)
                    diag_placeholder.markdown(diag_html, unsafe_allow_html=True)
                
                full_response = first_token
                bubble_placeholder.markdown(
                    f'<div class="assistant-bubble">{full_response}▌</div>',
                    unsafe_allow_html=True,
                )
            
                for token in stream:
                    full_response += token
                    bubble_placeholder.markdown(
                        f'<div class="assistant-bubble">{full_response}▌</div>',
                        unsafe_allow_html=True,
                    )
                
                bubble_placeholder.markdown(
                    f'<div class="assistant-bubble">{full_response}</div>',
                    unsafe_allow_html=True,
                )

            # Add assistant response to history
            metadata = st.session_state.get("last_stream_metadata", {})
            clean_content = full_response.replace("$", "USD ").replace("USD USD ", "USD ").replace("USD  USD ", "USD ")
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": clean_content,
                "sources": metadata.get("source_documents", []),
                "citations": metadata.get("citations", []),
                "routing_mode": metadata.get("routing_mode"),
                "routing_strategy": metadata.get("routing_strategy"),
                "routing_reason": metadata.get("routing_reason"),
                "rewritten_query": metadata.get("rewritten_query"),
                "multi_queries": metadata.get("multi_queries"),
                "coverage_before_rerank": metadata.get("coverage_before_rerank", {}),
            })

            # Trim history to last 10 exchanges
            max_msgs = 20  # 10 pairs
            if len(st.session_state.chat_history) > max_msgs:
                st.session_state.chat_history = st.session_state.chat_history[-max_msgs:]

            # Rerun to update chat display and citations column
            st.rerun()


with tab_kpi:
    st.markdown('## 📊 KPI Dashboard & Cross-Quarter Comparison')
    st.markdown('Select a company to view its key performance indicators extracted from earnings calls.')
    from src.extraction.schema import get_engine, get_session_maker, EarningsKPI
    import pandas as pd
    db_path = PROJECT_ROOT / 'data' / 'finance_kpis.db'
    try:
        engine = get_engine(str(db_path))
        Session = get_session_maker(engine)
        session = Session()
        kpis = session.query(EarningsKPI).all()
        session.close()
        if not kpis:
            st.warning('No KPI data found in the database. Please run the extraction pipeline.')
        else:
            data = []
            for k in kpis:
                data.append({'Company': k.company, 'Period': k.period, 'Year': k.year, 'Quarter': k.quarter, 'Revenue ()': k.revenue_b, 'EPS': k.eps_diluted, 'Gross Margin (%)': k.gross_margin_pct, 'Net Income ()': k.net_income_b, 'Op Cash Flow ()': k.op_cash_flow_b, 'Guidance Rev Low': k.guidance_revenue_low_b, 'Guidance Rev High': k.guidance_revenue_high_b, 'Guidance GM Low': k.guidance_gm_low_pct, 'Guidance GM High': k.guidance_gm_high_pct, 'Rev Growth YoY (%)': k.revenue_growth_yoy_pct, 'EPS Growth YoY (%)': k.eps_growth_yoy_pct})
            df = pd.DataFrame(data)
            df = df.sort_values(by=['Company', 'Year', 'Quarter'])
            companies = df['Company'].unique()
            selected_kpi_company = st.selectbox('Select Company for KPI Analysis', options=companies)
            company_df = df[df['Company'] == selected_kpi_company].copy()
            st.dataframe(company_df[['Period', 'Revenue ()', 'EPS', 'Gross Margin (%)', 'Rev Growth YoY (%)']], use_container_width=True, hide_index=True)
            st.markdown('### Revenue Trend')
            st.line_chart(company_df.set_index('Period')['Revenue ()'])
            st.markdown('### EPS Trend')
            st.bar_chart(company_df.set_index('Period')['EPS'])
    except Exception as e:
        st.error(f'Error loading KPI data: {e}')
