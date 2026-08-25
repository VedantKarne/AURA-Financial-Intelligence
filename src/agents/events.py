"""
src/agents/events.py
====================
Global workflow event bus for the AURA Agent Workflow Monitor.

This module is a neutral intermediary:
  - src/agents/orchestrator.py  imports emit() to fire events
  - src/api/server.py           imports subscribe/unsubscribe to stream them

Neither module depends on the other — circular import avoided.

Event schema:
{
    "type":           "query_start | node_enter | tool_call | tool_result | node_exit | query_complete | error | ping",
    "node":           "chatbot | tools | null",
    "tool":           "rag_search | get_kpis | generate_report_sections | null",
    "args":           { ... },
    "output_preview": "first ~300 chars of output",
    "query":          "original user query",
    "thread_id":      "...",
    "timestamp":      "ISO8601",
    "log":            "human-readable log line"
}
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

# ── Subscriber registry ───────────────────────────────────────────────────────
# Each connected /monitor SSE client has one asyncio.Queue here.
_subscribers: list[asyncio.Queue] = []

# Reference to the main asyncio event loop (stored on server startup).
# run_agent_query() executes in FastAPI's threadpool — bridging to async
# queues requires loop.call_soon_threadsafe().
_loop: Optional[asyncio.AbstractEventLoop] = None


def register_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Called once from server.py on application startup.
    Stores the running event loop so thread-safe emission works.
    """
    global _loop
    _loop = loop


def subscribe() -> asyncio.Queue:
    """
    Register a new SSE subscriber.  Returns the queue the caller must drain.
    Call unsubscribe() when the client disconnects.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a subscriber queue when the SSE connection closes."""
    if q in _subscribers:
        _subscribers.remove(q)


# ── Core emit ─────────────────────────────────────────────────────────────────

def emit(event: dict) -> None:
    """
    Broadcast a workflow event to every connected SSE subscriber.

    Safe to call from any thread (including FastAPI threadpool workers that
    execute synchronous route handlers like run_agent_query).

    If no frontend is listening, returns immediately with zero overhead.
    """
    if not _subscribers:
        return  # Fast-exit — no one connected

    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    if _loop is not None and _loop.is_running():
        # Called from a background thread → bridge to the async event loop.
        for q in list(_subscribers):
            _loop.call_soon_threadsafe(_safe_put, q, event)
    else:
        # Called from an async context directly (edge case).
        for q in list(_subscribers):
            _safe_put(q, event)


def _safe_put(q: asyncio.Queue, event: dict) -> None:
    """Enqueue an event, silently dropping it if the subscriber is full."""
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass  # Slow consumer — drop rather than block
