"""
src/retrieval/router.py
=======================
Query Router deciding the best retrieval strategy based on query content.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# List of keywords indicating quantitative or exact phrase queries (optimal for BM25)
BM25_KEYWORDS = [
    "margin", "revenue", "eps", "guidance", "dollar", "percent", "%", "$",
    "forecast", "actual", "quarterly", "growth", "yoy", "sequential", "billion", "million"
]


class QueryRouter:
    """
    Decides the best retrieval mode: 'vector', 'bm25', 'hybrid', or 'rerank'.
    Also detects comparison queries that benefit from multi-query decomposition.
    """

    def __init__(self, llm: Optional[ChatGroq] = None):
        self.llm = llm

    def detect_entities(self, query: str) -> list[str]:
        """Detect known companies from the query."""
        query_lower = query.lower()
        known_companies = ["Apple", "Microsoft", "Nvidia"]
        
        # Check if user query refers to all companies
        all_companies_phrases = ["all companies", "across companies", "each company", "every company", "companies'"]
        if any(phrase in query_lower for phrase in all_companies_phrases):
            return known_companies
            
        # Check specific company mentions
        detected = []
        if any(x in query_lower for x in ["apple", "aapl"]):
            detected.append("Apple")
        if any(x in query_lower for x in ["microsoft", "msft", "azure"]):
            detected.append("Microsoft")
        if any(x in query_lower for x in ["nvidia", "nvda", "jensen"]):
            detected.append("Nvidia")
            
        return detected

    def is_multi_entity(self, query: str) -> tuple[bool, list[str]]:
        """Determine if query targets multiple entities."""
        detected = self.detect_entities(query)
        is_multi = len(detected) > 1 or any(
            phrase in query.lower() 
            for phrase in ["all companies", "compare", "across companies"]
        )
        return is_multi, detected

    def route_query_rule_based(self, query: str) -> dict:
        """Route query using simple keyword matching rules."""
        query_lower = query.lower()

        # Check for multi-entity
        is_multi, detected_cos = self.is_multi_entity(query)

        # Check for comparison terms
        comparison_terms = ["compare", "comparison", "trend", "change", "versus", "vs", "difference"]
        is_comparison = any(term in query_lower for term in comparison_terms)

        # Check for summary terms
        is_summary = any(term in query_lower for term in ["summarize", "summary", "overview", "synopsis"])

        # Check if financial / metrics terms are present
        financial_terms = ["revenue", "gross margin", "eps", "guidance", "sales", "income", "profit", "cogs", "growth", "financial"]
        is_financial = any(term in query_lower for term in financial_terms)

        # Check if exact fact term (like product name/executive) is present
        exact_terms = ["vision pro", "sovereign ai", "copilot", "jensen", "cook", "nadella", "drive orin"]
        is_exact = any(term in query_lower for term in exact_terms)

        # Check for risk/headwinds terms
        risk_terms = ["risk", "threat", "challenge", "headwind", "barrier", "regulatory", "competition", "shortage", "macroeconomic"]
        is_risk = any(term in query_lower for term in risk_terms)

        if is_multi:
            if is_risk:
                mode = "rerank"
                strategy = "multi_entity_risk_analysis"
            else:
                mode = "rerank"
                strategy = "multi_entity_retrieval"
        elif is_comparison:
            mode = "sql"
            strategy = "comparison_query"
        elif is_risk:
            mode = "rerank"
            strategy = "single_entity_risk_analysis"
        elif is_summary:
            if is_financial:
                mode = "rerank"
                strategy = "single_entity_financial_summary"
            else:
                mode = "vector"
                strategy = "summary_section"
        elif is_financial:
            # If broad historical request
            if any(term in query_lower for term in ["far", "history", "overall", "trend", "so far", "historically"]):
                mode = "rerank"
                strategy = "single_entity_financial_summary"
            else:
                mode = "rerank"
                strategy = "single_entity_financial_metric"
        elif is_exact:
            mode = "rerank"
            strategy = "exact_fact_query"
        else:
            mode = "rerank"
            strategy = "vector_only"

        return {
            "mode": mode,
            "strategy": strategy,
            "reason": "Rule-based keyword mapping",
            "is_comparison": is_comparison or strategy == "comparison_query"
        }

    def route_query(self, query: str) -> dict:
        """
        Route query, using LLM if available, falling back to rule-based.
        """
        if not self.llm:
            return self.route_query_rule_based(query)

        prompt = f"""You are a query routing assistant for a financial RAG system.
Given a user query, classify its primary retrieval intent into one of the following strategies:

1. "multi_entity_risk_analysis" (Use for queries asking about risks, challenges, headwinds, or competitors across multiple companies or "all companies")
2. "single_entity_risk_analysis" (Use for queries asking about risks, challenges, headwinds, or competitors for a single company)
3. "single_entity_financial_summary" (Use for queries asking for general financial summary, overall performance, or historical summaries of a single company, e.g. revenue generation, net income trends, segment trends over multiple quarters/years)
4. "single_entity_financial_metric" (Use for queries seeking a specific financial metric like gross margin, EPS, Azure revenue, segment growth for a single company in a specific quarter/year)
5. "multi_entity_retrieval" (Use for queries asking for generic insights, summaries, or retrieval across multiple companies or "all companies" that are not specific to risk or quantitative comparisons)
6. "comparison_query" (Use for queries comparing metrics, trends, or strategy between companies, quarters, or years)
7. "exact_fact_query" (Use for queries seeking a specific qualitative fact, quote, product name, or executive statement, e.g. "Vision Pro", "sovereign AI", "Copilot")
8. "summary_section" (Use for queries explicitly asking to summarize an entire call or a section of the call)
9. "vector_only" (Use for broad, subjective, qualitative, or strategic questions that do not focus on specific numerical metrics)

Output a JSON object with keys:
- "strategy": One of ["multi_entity_risk_analysis", "single_entity_risk_analysis", "single_entity_financial_summary", "single_entity_financial_metric", "multi_entity_retrieval", "comparison_query", "exact_fact_query", "summary_section", "vector_only"]
- "mode": One of ["rerank", "vector", "bm25", "hybrid", "sql"] (Map "comparison_query" to "sql", "summary_section" to "vector", and ALL other strategies to "rerank")
- "reason": A brief 1-sentence reason.

Make sure your response contains ONLY valid JSON inside code fences (e.g. ```json ... ```).

Query: {query}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            # Suppress think blocks if Qwen/reasoning is used
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = content.replace("<think>", "").replace("</think>", "").strip()

            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_str = match.group(1).strip() if match else content.strip()
            data = json.loads(json_str)

            if "mode" in data and "strategy" in data:
                logger.info(f"Router routing query to '{data['strategy']}' via mode '{data['mode']}'")
                return {
                    "mode": data["mode"],
                    "strategy": data["strategy"],
                    "reason": data.get("reason", ""),
                    "is_comparison": data["strategy"] in ["comparison_query", "multi_query"]
                }
        except Exception as e:
            logger.warning(f"LLM routing failed, falling back to rule-based: {e}")

        return self.route_query_rule_based(query)
