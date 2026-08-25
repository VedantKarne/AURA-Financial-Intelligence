"""
src/retrieval/query_transformer.py
==================================
Query transformation layer for Query Rewrite (chat history) and Multi-Query Expansion.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_groq import ChatGroq

from src.utils.logger import logger


class QueryTransformer:
    """
    Transforms, expands, and rewrites queries using LLM assistance.
    """

    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def rewrite_query(self, query: str, chat_history: list[dict]) -> str:
        """
        Rewrite vague/conversational follow-up query using chat history.
        """
        if not chat_history:
            return query

        # Format history string
        history_str = ""
        for turn in chat_history[-3:]:  # Last 3 turns
            role = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"]
            history_str += f"{role}: {content}\n"

        prompt = f"""You are a query rewriting assistant for a financial RAG system.
Given a user query and a brief conversation history, rewrite the user query to be a standalone, fully-explicit search query. 
Resolve pronouns (it, they, their, etc.) and implicit references to companies, tickers, metrics, years, or quarters.

CRITICAL INSTRUCTIONS:
- Do NOT restrict the temporal scope or narrow the user's intent. For example, if the user asks "so far", "over time", "historically", or "all quarters", do NOT rewrite it to target "the current period", "current quarter", or "a single quarter". Maintain the broad historical or multi-period scope of the query.
- Ensure that the output standalone query is fully independent. If no rewrite is necessary, return the original query.

Output a JSON object with a single key: "rewritten_query".
Make sure your response contains ONLY valid JSON inside code fences (e.g. ```json ... ```).

History:
{history_str}
Query to Rewrite: {query}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = content.replace("<think>", "").replace("</think>", "").strip()

            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_str = match.group(1).strip() if match else content.strip()
            data = json.loads(json_str)

            rewritten = data.get("rewritten_query", query).strip()
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.exception(f"Query rewrite failed, using original: {e}")
            return query

    def generate_multi_queries(self, query: str) -> list[str]:
        """
        Generate 3 related queries to expand retrieval recall for comparisons or complex queries.
        """
        prompt = f"""You are a financial retrieval assistant. Given a user search query, generate 3 search query variations targeting the financial data of the company mentioned. 
Each variation should focus on different aspects, synonyms, related financial metrics, or alternate wordings (e.g., "margins" vs "profitability" or "sales" vs "revenue").

Output a JSON object with a single key: "queries" containing a list of 3 strings.
Make sure your response contains ONLY valid JSON inside code fences (e.g. ```json ... ```).

Original Query: {query}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = content.replace("<think>", "").replace("</think>", "").strip()

            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_str = match.group(1).strip() if match else content.strip()
            data = json.loads(json_str)

            queries = data.get("queries", [])
            # Validate output list
            if isinstance(queries, list) and len(queries) > 0:
                logger.info(f"Multi-query generated: {queries}")
                return [q.strip() for q in queries[:3]]
        except Exception as e:
            logger.exception(f"Multi-query generation failed, returning original list: {e}")

        # Fallback to simple query expansion variations
        return [
            query,
            f"{query} financial guidance metric",
            f"{query} YoY sequential growth"
        ]
