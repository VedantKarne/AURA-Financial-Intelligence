# Phase 8: Live Agent Workflow Monitor — Features, Challenges & Solutions

> **AURA Financial Intelligence — Phase 8 Engineering Log**
> 
> Phase 8 introduces a completely new observability layer for the AURA platform: a real-time, n8n-style **Agent Workflow Monitor** that visualises the LangGraph agent's execution as it happens — with animated nodes, pulsating status borders, live event logs, and a dynamic side panel showing every tool call, argument, and output preview in real-time.

---

## 🚀 Key Features Added

### 1. **Live Agent Workflow Monitor Page (`/monitor`)**

A separate full-page React application at `frontend/src/app/monitor/page.tsx`, powered by `@xyflow/react` (React Flow), that renders the complete AURA agent execution graph:

- **7 nodes**: `START`, `Chatbot / LLM`, `Tool Router`, `rag_search`, `get_kpis`, `generate_report_sections`, `END`
- **9 edges** with the correct topology: START → Chatbot → Tool Router → each tool → back to Chatbot → END
- Each node has a **custom `WorkflowNode` component** with:
  - Status-driven colours: `idle` (dark grey), `running` (golden pulsating border), `completed` (green), `error` (red), `skipped` (dim)
  - Node-type-driven left accent bars: `start` (green), `llm` (purple), `router` (yellow), `tool` (teal), `end` (red)
  - Status badge text (uppercase, monospace)
  - Animated golden border glow for any node currently executing

### 2. **Golden Pulsating Border for Active Nodes**

When a node is in `running` state, its entire border switches from the node-type accent colour to `#ECC94B` (gold) and triggers a CSS `@keyframes borderPulse` animation:

```css
@keyframes borderPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(236, 201, 75, 0.5); }
  50%       { box-shadow: 0 0 0 7px rgba(236, 201, 75, 0); }
}
```

All four individual border properties (`borderTop`, `borderRight`, `borderBottom`, `borderLeft`) are set independently to avoid the React "shorthand + non-shorthand conflict" warning.

### 3. **Animated Edges for Active Execution Paths**

Active edges (the path currently being traversed by the agent) are switched to `animated: true` with a blue stroke (`#4299E1`). When execution completes (either successfully via `query_complete` or by error), all edge animations are stopped simultaneously.

### 4. **Dynamic Side Panel**

A right-hand panel showing real-time agent context:

- **Execution Context**: Current query text, active node name (as a pill badge), and active tool name
- **Tool Arguments**: Every key-value pair from the current tool call rendered as inline code blocks
- **Output Preview**: First 320 characters of the tool's return value
- **Event Log**: Rolling 150-entry timestamped log coloured by event type, with auto-scroll and a "clear" button

### 5. **Server-Sent Events (SSE) Backend Stream**

A new `GET /api/workflow-stream` endpoint in `src/api/server.py` that:

- Registers each browser connection as a subscriber via the event bus
- Streams events as `text/event-stream` with proper `Cache-Control: no-cache` headers
- Sends a keepalive `ping` event every 30 seconds to prevent proxy/browser timeouts
- Automatically unsubscribes and cleans up when the browser tab is closed

### 6. **Neutral Event Bus (`src/agents/events.py`)**

A new neutral intermediary module that neither `orchestrator.py` nor `server.py` depends on each other through. It provides:

- `subscribe()` → registers a new SSE client queue
- `unsubscribe()` → deregisters a client queue on disconnect
- `emit(event: dict)` → broadcasts an event to all connected clients
- `register_loop()` → stores the running asyncio event loop for thread-safe emission

### 7. **Orchestrator Instrumentation (`src/agents/orchestrator.py`)**

Six `emit()` call points added inside the `_execute()` stream loop with zero changes to the LangGraph graph structure:

| Event | Trigger |
|---|---|
| `query_start` | Top of `_execute()` before stream starts |
| `node_enter` | When `AIMessage` has `tool_calls` (agent chose to use a tool) |
| `tool_call` | For each individual tool call (with args preview) |
| `tool_result` | When a `ToolMessage` appears in the state |
| `node_exit` | When a final `AIMessage` has no `tool_calls` (final answer) |
| `query_complete` | After the stream loop exits normally |
| `error` | In all exception handlers — rate limit, quota, and system error |

### 8. **Improved Rate Limit Handling with Event Emission**

The `_parse_retry_after()` function was fully rewritten to:

- Correctly parse `m`, `s`, and `ms` units independently via separate regex matches
- Default to a 15-second wait if the error message contains no parseable wait time (instead of 0)
- Emit a frontend log event during auto-retry waits so the monitor shows the pause reason
- Emit an `error` event to the monitor when a rate limit exceeds the 45-second auto-retry ceiling

### 9. **Auto-Reconnect SSE Client**

The frontend `EventSource` client implements exponential-backoff-like auto-reconnect. If the connection drops (e.g., backend restart), it automatically retries after 3.5 seconds — showing "RECONNECTING…" in the header status indicator.

---

## 🛠️ Challenges Faced & Resolutions

---

### Challenge 1 — The Monitor Was Completely Static: Events Never Reached the Frontend

**The Problem:**
After the full implementation (events.py, orchestrator instrumentation, SSE endpoint, React monitor page), the monitor showed zero dynamic updates. Submitting a chat query completed successfully on the main page, but the monitor stayed frozen in its initial idle state throughout.

**Root Cause — Synchronous Blocking of the Async Event Loop:**

FastAPI is an `async` framework. Its SSE endpoint (`workflow_stream`) is an `async` generator that yields events as they arrive on the subscriber queue via `await queue.get()`. This requires the asyncio event loop to be **free and running** to process the `await`.

However, the `/api/chat` endpoint was calling `run_agent_query()` — a synchronous, blocking function — **directly inside the async handler** without offloading it to a threadpool:

```python
# Bug: Blocking call inside async context
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    response = run_agent_query(...)  # ❌ BLOCKS the entire event loop!
```

Because `run_agent_query()` runs the LangGraph `app_graph.stream()` loop (which takes 10–60 seconds for a full query), the **entire FastAPI event loop was frozen** for the duration of the query. The `await queue.get()` in the SSE generator could never execute. All `emit()` calls correctly placed events into subscriber queues, but the generator was never able to dequeue and stream them — not until the query finished.

This meant events arrived in batch only after the query completed, appearing as a sudden flood that the frontend couldn't animate sequentially.

**The Resolution:**

Replaced all synchronous blocking calls in FastAPI async endpoints with `run_in_threadpool`:

```python
from fastapi.concurrency import run_in_threadpool

# Fixed: Offload to threadpool so the event loop stays free
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    response = await run_in_threadpool(run_agent_query, full_query, thread_id=...)
```

`run_in_threadpool` runs the synchronous function in FastAPI's managed threadpool executor, returning an `awaitable` that suspends the coroutine without blocking the event loop. The event loop is now free to concurrently process `await queue.get()` in the SSE generator while `run_agent_query` runs in its background thread.

**The Learning:**

In any async Python web framework (FastAPI, Starlette, aiohttp), calling synchronous blocking I/O directly inside an `async def` handler freezes the **entire server** — all concurrent requests, all SSE streams, all background tasks. Long-running synchronous code (LangGraph streams, LLM calls, file I/O) must **always** be offloaded to a threadpool using `run_in_threadpool` (FastAPI) or `asyncio.to_thread` (stdlib).

---

### Challenge 2 — Thread Safety of the Event Queue: `emit()` Called from a Background Thread

**The Problem:**

After the threadpool fix, `run_agent_query` now runs in a background thread. But `emit()` calls `q.put_nowait()` on `asyncio.Queue` objects. Python's `asyncio.Queue` is **not thread-safe** — calling it from a non-async thread causes undefined behaviour and potential data corruption.

**Root Cause:**

`asyncio` queues and primitives assume single-threaded access from within the event loop. A background thread (the threadpool where LangGraph runs) calling `queue.put_nowait()` directly bypasses asyncio's internal locking.

**The Resolution:**

Stored a reference to the main event loop at startup via `register_loop()`, and used `loop.call_soon_threadsafe()` in `emit()` to safely schedule the put operation on the event loop thread:

```python
# events.py
_loop: Optional[asyncio.AbstractEventLoop] = None

def register_loop(loop):
    global _loop
    _loop = loop

def emit(event: dict):
    if _loop and _loop.is_running():
        # Thread-safe bridge: schedule the put on the event loop thread
        for q in list(_subscribers):
            _loop.call_soon_threadsafe(_safe_put, q, event)
    else:
        # Direct call if already in async context (edge case)
        for q in list(_subscribers):
            _safe_put(q, event)
```

`call_soon_threadsafe()` is the correct, thread-safe mechanism to schedule a callback from a non-asyncio thread onto the running event loop.

---

### Challenge 3 — Circular Import Between orchestrator.py and server.py

**The Problem:**

The natural design would have `server.py` import from `orchestrator.py` (to call `run_agent_query`) and `orchestrator.py` import from `server.py` (to emit events to SSE clients). This creates a **circular import** that Python cannot resolve.

**Root Cause:**
```
server.py → imports → orchestrator.py
orchestrator.py → imports → server.py  ← CIRCULAR
```

**The Resolution:**

Introduced a **neutral intermediary module** (`src/agents/events.py`) that holds no references to either `server.py` or `orchestrator.py`. Both modules independently import from `events.py`:

```
                  ┌──────────────┐
                  │  events.py   │  (neutral bus)
                  │  subscribe() │
                  │  emit()      │
                  └──────┬───────┘
                         │ imported by
               ┌─────────┼─────────┐
          server.py               orchestrator.py
    (subscribe/unsubscribe)         (emit events)
```

This completely decouples the two modules while enabling bidirectional communication through the shared queue registry.

---

### Challenge 4 — Rate Limit Regex Parsing Failure Causing Instant Failures

**The Problem:**

Users hitting the Groq API rate limit saw the instant "please wait ~30 seconds" error even when the actual required wait was only milliseconds (e.g., `"Please try again in 854ms"`). The auto-retry system was not activating at all for short waits.

**Root Cause:**

The original `_parse_retry_after()` regex was:

```python
match = re.search(r"Please try again in ([0-9.]+m)?([0-9.]+s)?", error_msg)
```

This pattern required the string to start with `"Please try again in"` exactly. Groq's actual error messages don't always follow this format. Also, the pattern used *optional* groups `?` — if neither group matched, `match.group(1)` and `match.group(2)` were both `None`, and the function returned `0.0`.

In the error handling code:
```python
MAX_AUTO_RETRY_SECS = 30
if 0 < wait_secs <= MAX_AUTO_RETRY_SECS:  # 0.0 < 0.0 → False!
```

A parsed time of `0.0` seconds skipped the `if` block, fell through to `elif wait_secs > MAX_AUTO_RETRY_SECS` (also `False`), then landed in the "Unknown format" catch-all that immediately returned the error message without retrying.

**The Resolution:**

Rewrote the parsing logic with independent regex matches for each time unit, and a safe default of 15 seconds when parsing fails:

```python
def _parse_retry_after(error_msg: str) -> float:
    import re
    m_match  = re.search(r"([0-9.]+)m\b", error_msg)
    s_match  = re.search(r"([0-9.]+)s\b", error_msg)
    ms_match = re.search(r"([0-9.]+)ms\b", error_msg)

    total = 0.0
    if m_match:  total += float(m_match.group(1)) * 60
    if ms_match: total += float(ms_match.group(1)) / 1000
    elif s_match: total += float(s_match.group(1))

    return total if total > 0.0 else 15.0  # Safe default
```

Also removed the `if 0 < wait_secs` lower bound check so that any wait time (including very short ones) triggers the auto-retry path.

---

### Challenge 5 — React Warning: Shorthand + Non-Shorthand Border Conflict

**The Problem:**

The browser console showed:

```
Warning: Updating a style property during rerender (border) when a conflicting 
property is set (borderLeft) can lead to styling bugs.
```

**Root Cause:**

The `WorkflowNode` component used both `border` (shorthand) and `borderLeft` (specific) on the same element:

```jsx
<div style={{
  border: `1.5px solid ${s.border}`,    // shorthand
  borderLeft: `3px solid ${t.accent}`,  // specific — conflicts!
}}>
```

React's reconciler processes style updates as object diffs. When a node transitions from `idle` → `running`, `s.border` changes colour — React diffs and patches only the `border` property. However, `borderLeft` was set *after* `border` in the style object and was already overriding the shorthand's left side. React's batched update ordering between shorthand and longhand properties is undefined, producing visual glitches.

**The Resolution:**

Replaced the shorthand `border` with four explicit properties:

```jsx
<div style={{
  borderTop:    `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
  borderRight:  `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
  borderBottom: `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
  borderLeft:   `3px solid ${isRunning ? '#ECC94B' : t.accent}`,
}}>
```

This also enabled the golden border feature — when `isRunning`, all four borders switch to gold `#ECC94B`.

---

### Challenge 6 — Animated Edges Not Stopping After Execution Completes

**The Problem:**

After a query finished, the edges that had been animated (showing active tool routing paths) stayed permanently animated — even after the agent had returned a final response and `query_complete` fired.

**Root Cause:**

The `query_complete` and `error` event handlers in the React component only updated node statuses but never reset the edge states:

```typescript
case 'query_complete':
  setNodeStatus('chatbot', 'completed');
  setNodeStatus('end', 'completed');
  setCurrentNode('');
  break;   // ← edges never touched!
```

**The Resolution:**

Added `setEdges` reset calls in both `query_complete` and `error` handlers:

```typescript
case 'query_complete':
  setNodeStatus('chatbot', 'completed');
  setNodeStatus('end', 'completed');
  setCurrentNode('');
  setEdges(es => es.map(e => applyEdge(e, false)));  // stop all animations
  break;

case 'error':
  setNodes(ns => ns.map(n =>
    n.data.status === 'running'
      ? { ...n, data: { ...n.data, status: 'error' } }
      : n
  ));
  setEdges(es => es.map(e => applyEdge(e, false)));  // stop all animations
  break;
```

---

### Challenge 7 — TypeScript Union Type Error on `INIT_NODES` Array

**The Problem:**

TypeScript raised a type error on the `INIT_NODES` constant because some node objects had an optional `subtitle` field and others did not. TypeScript inferred the array element type as a **union**:

```
{ data: { label: string; nodeType: string; status: NodeStatus; } }
| { data: { label: string; subtitle: string; nodeType: string; status: NodeStatus; } }
```

This union type was incompatible with the `useNodesState` generic parameter from `@xyflow/react`.

**Root Cause:**

TypeScript infers object literal types as narrowly as possible. Without an explicit type annotation, the array type becomes the union of all inferred element types.

**The Resolution:**

Added an explicit `NodeData` interface with `subtitle` as optional, and explicitly typed the `INIT_NODES` array:

```typescript
interface NodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  status: NodeStatus;
  subtitle?: string;  // optional
}

const INIT_NODES: {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: NodeData;
}[] = [ ... ];
```

The `extends Record<string, unknown>` on `NodeData` is required because `@xyflow/react`'s `Node` generic constraint requires data objects to be indexable by string keys.

---

## 📐 Architectural Learnings

### L1 — Synchronous LLM Calls Must Never Be Made Directly Inside Async Handlers

FastAPI's async architecture depends on the event loop being free to multiplex between coroutines. Any synchronous blocking call (LangGraph streams, LLM inference, file I/O) inside an `async def` handler freezes **the entire server process** — killing all concurrent SSE streams, background tasks, and HTTP requests. The rule: every synchronous function that takes more than a few milliseconds must be offloaded with `run_in_threadpool` or `asyncio.to_thread`.

### L2 — Thread Safety is Not Optional for Async Queue Communication

`asyncio` primitives are not thread-safe by design. Cross-thread communication from a threadpool worker (LangGraph) into an async context (SSE generator) must use `loop.call_soon_threadsafe()`. This is not an edge case — it is the standard pattern for mixing sync threads with an async event loop.

### L3 — Neutral Intermediary Modules Solve Circular Import Problems in Layered Architectures

In a layered system where the API layer (server.py) depends on the agent layer (orchestrator.py), adding reverse dependencies (agent → server) creates circular imports. The solution is to introduce a neutral, stateless bus module that holds shared state (subscriber queues, event loop reference) without depending on either layer.

### L4 — CSS Shorthand vs. Longhand Properties Are Undefined in React's Diff Algorithm

React's reconciler patches style objects incrementally. When a shorthand property (e.g., `border`) and a conflicting longhand (e.g., `borderLeft`) exist on the same element, React's update order during re-renders is undefined and can produce inconsistent visual results. Always use explicit longhand properties when mixing border sides.

### L5 — Event-Driven UI State Machines Require Explicit Terminal Transitions

A reactive event-driven component (like the workflow monitor) must handle every possible terminal event — including happy-path completion (`query_complete`), error (`error`), and unexpected disconnects — by explicitly resetting all animated state. UI state that is only updated via events will persist indefinitely if no terminal reset is written.

### L6 — Regex Parsing of Rate Limit Errors Must Be Defensive and Have Safe Defaults

External API error messages (like Groq's rate limit responses) do not follow a guaranteed format. A regex that fails silently (returning 0 or None) will cause the error handler to skip retry logic and fail the user immediately. Always design retry parsers with:
1. Independent pattern matches for each possible unit (m, s, ms)
2. A non-zero safe default when parsing produces nothing
3. A maximum retry ceiling to avoid indefinite blocking

---

## 📁 New Files Added in Phase 8

| File | Purpose |
|---|---|
| `src/agents/events.py` | **[NEW]** Neutral event bus — subscribe/emit/register_loop |
| `frontend/src/app/monitor/page.tsx` | **[NEW]** Full-page React Flow workflow monitor |

## 📝 Modified Files in Phase 8

| File | Change |
|---|---|
| `src/agents/orchestrator.py` | Added 7 `emit()` call points in `_execute()`; rewrote `_parse_retry_after()`; added `error` emits in exception handlers |
| `src/api/server.py` | Added SSE `/api/workflow-stream` endpoint; added `startup` event loop registration; changed `run_agent_query` calls to `await run_in_threadpool(...)` |

---

## ⚠️ Disclaimer

> *This document is part of an educational and research project. All financial outputs generated by AURA are for informational purposes only and do not constitute financial advice. Always consult a qualified financial professional before making investment decisions.*
