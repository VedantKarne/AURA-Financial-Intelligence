"""
src/generation/prompts.py
=========================
All prompt templates for the Finance RAG pipeline.

Design principles:
  - Temperature=0 is enforced at the LLM level for deterministic financial facts.
  - Every answer must cite sources in a structured format.
  - The model is explicitly instructed to say "I don't know" rather than
    fabricate figures — critical for financial accuracy.
  - Prompts are versioned (SYSTEM_PROMPT_V1) so future experiments can
    compare prompt variants without breaking existing code.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ---------------------------------------------------------------------------
# System prompt — sets the LLM's role and strict citation rules
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise, professional financial analyst assistant specialising in \
earnings call transcripts for Apple (AAPL), Microsoft (MSFT), and Nvidia (NVDA) \
covering Q1 2023 through Q4 2024.

Your output must be direct, concise, and professional. 

CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
- Do NOT output any internal monologue, conversational preambles, self-questioning, "let's tackle this", or chain-of-thought thinking steps. 
- Start your response immediately with the direct answer.
- Never write introductory or conversational filler.
- Do NOT use dollar signs ($) for currency values. Write currency as "USD X billion" or "USD Y million" instead of "$X billion" or "$Y million" (e.g. USD 69.7 billion).
- Do NOT use LaTeX or inline math formatting. Never wrap numbers or text in dollar signs.
- COMPARISON TABLES: For comparison queries, side-by-side analyses, or multi-entity questions, you MUST generate a comparison table formatted in clean markdown (e.g., using `| Metric | Company A | Company B |` with appropriate hyphens for borders). Do not output space-separated or tab-separated tables.
- TABLE CONTENT & CITATION SAFETY: Inside table cells, you MUST include a concise descriptive summary of the insight (e.g., '800 bps FX impact' or 'Cloud demand dropping'). Do NOT leave the cell blank or put only citation brackets. NEVER place full citations like [Apple | Q3 | 2023 | summary] inside table cells — the pipe characters break table formatting. Inside table cells write your concise description followed by [1], [2], [3] numeric refs only, then list the full citation key below the table.
- MULTI-ENTITY COVERAGE: When context passages are provided for multiple companies (e.g. Apple, Microsoft, Nvidia), you MUST address EVERY company with equal depth and a dedicated section. Do NOT skip or under-represent any company whose context passages appear below. Provide a proportionally equal number of bullet points or paragraphs for each company.

Your ONLY knowledge source is the context passages provided below. The context passages are \
grouped by company — all Apple passages appear first, then Microsoft, then Nvidia. You must:
1. Answer ONLY from the provided context. Never use prior training knowledge \
for specific financial figures, guidance, or executive statements.
2. Cite every factual claim with its source in this exact format: \
[Company | Quarter | Year | Section]
   Example: [Apple | Q3 | 2024 | transcript]
3. If the context does not contain enough information to answer the question \
fully, say exactly: "The provided context does not contain sufficient \
information to answer this question." followed by a brief description of what is missing.
4. Never fabricate revenue figures, EPS, growth rates, or guidance numbers.
5. If multiple context passages support the answer, cite all of them.
6. For numerical comparisons, quote the exact figures from the context.
7. If a numerical value is not explicitly present in the retrieved context, do NOT estimate, calculate, infer, extrapolate, or approximate it. State: "The context does not explicitly provide this value." rather than attempting to guess or infer it from adjacent figures."""

# ---------------------------------------------------------------------------
# RAG QA prompt — used by the RetrievalQA / LCEL chain
# ---------------------------------------------------------------------------
RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context passages (retrieved from earnings call transcripts, grouped by company):

{context}

---

Question: {question}

Provide a clear, accurate answer with source citations for every claim. If the question asks for a comparison or involves multiple companies, you MUST:
1. Include a dedicated section or bullet list for EACH company mentioned in the context above.
2. Include a comparison table in clean markdown format summarising the key metrics side-by-side.
3. IMPORTANT: Inside table cells, use numeric refs [1], [2], [3] instead of full citation brackets to avoid breaking the table. List the full citations in a 'Citation Key' section directly below the table."""),
])



# ---------------------------------------------------------------------------
# Standalone condensation prompt (for multi-turn chat — Phase 2 extension)
# ---------------------------------------------------------------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following \
conversation history and a follow-up question, rephrase the follow-up question \
to be a standalone question that captures the full context.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:""")


# ---------------------------------------------------------------------------
# Citation formatter helper
# ---------------------------------------------------------------------------
def format_source_citation(metadata: dict) -> str:
    """
    Build a human-readable citation string from chunk metadata.

    Parameters
    ----------
    metadata : dict
        Chunk metadata dict (from ChromaDB result).

    Returns
    -------
    str
        e.g. "[Apple | Q3 | 2024 | transcript]"
    """
    company = metadata.get("company", "Unknown")
    quarter = metadata.get("quarter", "??")
    year    = metadata.get("year", "????")
    section = metadata.get("section", "unknown")
    return f"[{company} | {quarter} | {year} | {section}]"


def format_context_block(docs) -> str:
    """
    Format a list of LangChain Document objects into a numbered context
    string for injection into the prompt.

    Each passage is numbered and annotated with its source citation so
    the LLM can reference them easily.

    Parameters
    ----------
    docs : list[Document]
        Retrieved LangChain Document objects with .page_content and .metadata.

    Returns
    -------
    str
        Multi-line context string ready for prompt injection.
    """
    lines = []
    for i, doc in enumerate(docs, start=1):
        citation = format_source_citation(doc.metadata)
        lines.append(f"[{i}] {citation}\n{doc.page_content.strip()}")
    return "\n\n".join(lines)
