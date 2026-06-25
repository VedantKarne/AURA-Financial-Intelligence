# Phase 9: KPI Backfill — Features, Challenges & Solutions

> **AURA Financial Intelligence — Phase 9 Engineering Log**
>
> Phase 9 covers the complete lifecycle of populating the SQLite `earnings_kpis` database with
> historical financial KPI data for AAPL, MSFT, and NVDA across Q1 2023 – Q4 2024. The work
> involved designing, debugging, and iteratively improving an asynchronous multi-company backfill
> script, resolving cascading issues with model deprecations, API rate limits, memory constraints,
> extraction design, and dictionary key collisions.

---

## 🚀 New Functionality Added

### 1. `src/extraction/backfill_kpis.py` — Async KPI Backfill Script

A standalone utility script that discovers all 23 earnings call transcript files, extracts
structured financial KPIs via a local Ollama LLM, and bulk-inserts the results into the SQLite
`earnings_kpis` database. Key design features of the final version:

- **Single-pass extraction** using a combined `AllKPIs` Pydantic schema (12 fields in one LLM
  call per file — down from 3 separate calls in earlier iterations).
- **`asyncio.Semaphore(1)`** to safely queue all 23 files without triggering Out-of-Memory errors
  on the local Ollama server.
- **`asyncio.gather()`** to schedule all task coroutines concurrently — the Semaphore ensures
  only one runs at a time on the GPU.
- **Pandas DataFrame** as the intermediate in-memory result store, with a composite
  `ticker_period` key to prevent cross-company data collisions.
- **SQLAlchemy bulk `session.add_all()`** for atomic single-transaction database commit.
- **`langchain_ollama.ChatOllama`** for fully local, rate-limit-free LLM inference.

### 2. `AllKPIs` Combined Pydantic Schema

A new combined schema replacing the three separate schemas (`CoreFinancials`, `GuidanceMetrics`,
`GrowthAndSegments`) used in the online `kpi_extractor.py`. Because a local model has no token
quota, combining all fields into one prompt is strictly superior — fewer context windows, less
memory, faster execution.

### 3. Fixed `UniqueConstraint` in `schema.py`

The `EarningsKPI` SQLAlchemy model was changed from a column-level `unique=True` on `period` to a
table-level `UniqueConstraint('ticker', 'period')`. This allows multiple companies to share the
same period string (e.g., `"2023-Q1"`) as long as their tickers differ.

---

## 🛠️ Challenges Faced & Resolutions

---

### Challenge 1 — SQLite `UNIQUE constraint failed: earnings_kpis.period`

**The Problem:**

The first run of the backfill script crashed with:
```
sqlite3.IntegrityError: UNIQUE constraint failed: earnings_kpis.period
```

The script successfully committed Apple's `2023-Q1` record, then immediately crashed when it tried
to insert Microsoft's `2023-Q1`.

**Root Cause:**

The `EarningsKPI` model in `src/extraction/schema.py` had a column-level uniqueness constraint:
```python
period = Column(String, index=True, unique=True)
```
This means `"2023-Q1"` could only exist **once in the entire table**, permanently blocking all
companies after the first from inserting records for that period.

**Failed Approach:**

The first fix attempted was to delete all existing rows with `session.query(EarningsKPI).delete()`
before re-inserting. This did not work because the schema-level `UNIQUE` constraint was still
enforced on the column itself, not on a pair of columns.

**Resolution:**

Two coordinated changes were required:
1. **`schema.py`**: Removed `unique=True` from the column and added a table-level
   `UniqueConstraint('ticker', 'period')` — allowing `"2023-Q1"` to appear for AAPL, MSFT, and
   NVDA simultaneously, but blocking the same ticker from having duplicate quarters.
2. **`backfill_kpis.py`**: Added `EarningsKPI.__table__.drop(engine, checkfirst=True)` before
   `init_db(engine)` to physically recreate the table with the new constraint DDL. SQLite does not
   support `ALTER TABLE ADD CONSTRAINT`, so the only way to apply a new constraint to an existing
   table is to drop and recreate it.

```python
# schema.py fix
from sqlalchemy import Column, Integer, String, Float, Text, create_engine, UniqueConstraint

class EarningsKPI(Base):
    __tablename__ = "earnings_kpis"
    __table_args__ = (UniqueConstraint('ticker', 'period'),)  # composite constraint
    period = Column(String, index=True)  # unique=True removed
```

**Learning:**

SQLite's `UNIQUE` constraint is permanent once the table is created. `ALTER TABLE` cannot add or
remove constraints in SQLite. The only correct migration path is a drop-and-recreate pattern.

---

### Challenge 2 — Groq Model Decommissioned (`llama3-70b-8192`)

**The Problem:**

The second run of the backfill script crashed with hard `HTTP 400 Bad Request` errors across all
concurrent tasks:
```
Error code: 400 - {'error': {'message': 'The model `llama3-70b-8192` has been decommissioned...',
'code': 'model_decommissioned'}}
```

Unlike `429 Too Many Requests`, a `400` is a permanent client error. The exponential backoff retry
loop was correctly aborting after 4 attempts since retrying a decommissioned model will never
succeed.

**Resolution:**

Changed the model string in `backfill_kpis.py` to `"qwen/qwen3-32b"` — a currently active Groq
model. This revealed the next problem (Challenge 3).

**Learning:**

Groq silently decommissions models. Error code `model_decommissioned` (400) must be treated
differently from rate limits (429) in retry logic — it should abort immediately rather than retry.

---

### Challenge 3 — Groq Model Not Found (`qwen3-32b` vs `qwen/qwen3-32b`)

**The Problem:**

After changing to `qwen3-32b`, Groq returned:
```
Error code: 404 - {'error': {'message': 'The model `qwen3-32b` does not exist...',
'code': 'model_not_found'}}
```

**Root Cause:**

Groq's newer OSS model IDs include a provider prefix. The orchestrator (`orchestrator.py`) was
correctly using `"qwen/qwen3-32b"` — but the backfill script was using the bare `"qwen3-32b"`
without the `qwen/` namespace prefix.

This was confirmed by querying the Groq models API directly:
```python
from groq import Groq; client = Groq()
print([m.id for m in client.models.list().data])
# → ['qwen/qwen3-32b', 'llama-3.3-70b-versatile', ...]
```

**Resolution:**

Updated the model string to `"qwen/qwen3-32b"`. The Groq model list confirms the full namespace
is required for provider-hosted OSS models.

---

### Challenge 4 — Exhausted Retries Producing Empty Records (Groq TPM Limit)

**The Problem:**

With `qwen/qwen3-32b` correctly specified, the script ran but generated dozens of:
```
[ERROR] Exhausted retries for 2024-Q1. Returning empty GrowthAndSegments.
```

The final SQLite commit contained only 8 rows (all Nvidia) with many `NaN` values instead of 23
fully populated rows.

**Root Cause:**

The `qwen/qwen3-32b` model on Groq's free tier has a limit of **6,000 Tokens Per Minute (TPM)**.
Each extraction pass requests approximately 3,000 tokens. The script was firing all 69 requests
concurrently via `asyncio.gather()`, instantly flooding the 6,000 TPM bucket. All tasks were
waiting simultaneously for the same quota to reset.

The original retry logic used `max_retries=4` with a base delay of 5 seconds. The mathematical
maximum wait time was only ~75 seconds (`5 + 10 + 20 + 40 = 75s`). But draining a 69-request
backlog at 2 requests/minute requires approximately **35 minutes** of waiting. Tasks timed out
long before their turn came up.

**Failed Approach 1 — Multiple API Keys (Same Account):**

The user created 12 Groq API keys expecting 12× the rate limit. This did not work because Groq
enforces rate limits at the **organization/account level**, not at the API key level. All 12 keys
shared the exact same 6,000 TPM bucket.

**Failed Approach 2 — Increasing `max_retries` to 15:**

Setting `max_retries=15` with `base_delay=15` extended the maximum wait time to approximately
65 minutes mathematically. However, the exponential growth (`15 + 30 + 60 + 120...`) meant each
individual task was sleeping for extremely long periods between attempts, while all other tasks
were also sleeping. The concurrent sleeping tasks did not yield the rate limit bucket to each
other effectively.

**Resolution (Permanent):**

Switched entirely from Groq cloud API to **local Ollama inference**. Running a model locally
eliminates all rate limits because there is no external quota — the only constraint is local
hardware memory and compute.

---

### Challenge 5 — `ImportError: cannot import name 'ChatOllama' from langchain_community`

**The Problem:**

After switching to Ollama, the import statement:
```python
from langchain_community.chat_models import ChatOllama
```
raised an `ImportError`. `langchain-community` itself printed a deprecation warning explaining
it is being sunset.

**Root Cause:**

LangChain reorganised its integration packages. `ChatOllama` was moved from `langchain_community`
into its own standalone package `langchain-ollama` as part of LangChain's ecosystem refactoring.

**Resolution:**

```bash
pip install langchain-ollama
```
```python
from langchain_ollama import ChatOllama  # correct import
```

---

### Challenge 6 — Ollama `500 Internal Server Error`: Out of Memory

**The Problem:**

The first Ollama run immediately failed with:
```
HTTP 500 Internal Server Error
model requires more system memory (2.9 GiB) than is available (2.7 GiB)
```

**Root Cause:**

The script was using `asyncio.gather(core_pass, guidance_pass, growth_pass)` inside each file
worker. This sent **3 simultaneous requests** per file to the local Ollama server. With
`asyncio.Semaphore(2)` allowing 2 files at a time, Ollama was asked to open **6 concurrent
context windows** simultaneously. The model (llama3 at 4.7 GB) needed 2.9 GiB per context,
far exceeding available system RAM.

**Failed Approach — Semaphore Reduction Only:**

Lowering the Semaphore from `2` to `1` still left 3 concurrent requests per file (the 3 passes
ran together with `asyncio.gather()`). This reduced peak concurrency from 6 to 3 contexts, but
still caused OOM errors because even 1 file × 3 passes = 3 simultaneous context windows.

**Resolution — Architectural Redesign (Single-Pass Extraction):**

The 3-pass design was invented to work around Groq's small per-request token limits. Locally,
there are no such limits — so the 3 passes were consolidated into a single `AllKPIs` combined
schema and one prompt per file.

```python
# Before: 3 passes per file (69 total requests)
core, guidance, growth = await asyncio.gather(
    extract(CoreFinancials),
    extract(GuidanceMetrics),
    extract(GrowthAndSegments),
)

# After: 1 pass per file (23 total requests)
kpis = await extract(AllKPIs)  # all 12 fields in one call
```

Benefits of single-pass design for local inference:
- **3× fewer requests** (23 vs 69)
- **3× less peak memory** (one context window per file)
- **No concurrent context window accumulation**
- **Simpler code** — no pass result merging required

---

### Challenge 7 — Dictionary Key Collision: Only Nvidia Entries in Final DataFrame

**The Problem:**

The script processed all 23 files successfully (confirmed in logs: 3 groups × ~8 files), but the
final DataFrame only contained 8 rows — all Nvidia. Apple and Microsoft entries were completely
absent.

**Root Cause:**

The `results_map` dictionary was keyed by bare `period` string:
```python
results_map[period] = { ... }   # e.g. results_map["2023-Q1"] = {...}
```

Since all three companies share identical period strings (`"2023-Q1"`, `"2023-Q2"`, ...), each
company overwrote the previous company's entry for the same period. Processing order in the log
was AAPL → MSFT → NVDA, so Nvidia's entries were last and the only ones that survived.

The bug was invisible in the logs because the logger printed `[2023-Q1] Done!` for all three
companies without differentiating the ticker — making it look like 23 unique records were created.

**Resolution:**

Changed the dictionary key to a **composite `ticker_period` string**:
```python
ticker  = metadata.get("ticker", "UNK")
map_key = f"{ticker}_{period}"   # e.g. "AAPL_2023-Q1", "MSFT_2023-Q1", "NVDA_2023-Q1"
results_map[map_key] = { ... }
```

This guarantees 23 unique keys across 3 companies × 8 periods, with no collision possible even
if period strings are identical across companies. The `period` field stored inside the dict value
is still the bare string (e.g., `"2023-Q1"`), which is what the `EarningsKPI` model expects.

**Learning:**

When aggregating results from multiple entities into a shared dictionary, always use a composite
key that includes every dimension that can vary independently. Using only the period dimension when
multiple companies share that dimension is a silent, hard-to-debug data loss bug.

---

## 📐 Architectural Learnings

### L1 — SQLite Constraint Migrations Require Drop-and-Recreate

SQLite does not implement `ALTER TABLE ADD CONSTRAINT` or `ALTER TABLE DROP CONSTRAINT`. The only
way to change a table's constraints is to drop the table entirely and recreate it with the updated
schema DDL. This is a fundamental SQLite limitation compared to PostgreSQL/MySQL.

### L2 — External API Rate Limits Are Account-Level, Not Key-Level

Groq (and most API providers) enforce rate limits at the account or organisation level. Generating
multiple API keys from the same account provides zero additional quota. The only legitimate way to
multiply free-tier quota is to create separate accounts with separate email addresses and billing
identities — which has ToS implications.

### L3 — Model Decommissioning Requires Immediate Abort in Retry Logic

A `400 model_decommissioned` error is fundamentally different from a `429 rate_limit_exceeded`
error. A decommissioned model will never succeed regardless of how long you wait. Retry logic must
distinguish permanent errors (`400`, `404`) from transient ones (`429`, `500`) and abort
immediately for permanent failures.

### L4 — Local LLM Extraction Is Architecturally Superior When Hardware Allows

For batch offline workloads (like KPI backfill), routing through a local Ollama server is strictly
better than cloud APIs on free tiers:
- No rate limits (quota is your VRAM/RAM)
- No per-token cost
- No API key management
- No network latency per request
- Complete data privacy

The trade-off is inference speed — local llama3 at ~10 tokens/sec vs Groq's ~500 tokens/sec.
For a one-time 23-file backfill, the local approach is correct.

### L5 — Concurrent Context Windows Multiply Memory Requirements Linearly

Each simultaneous request to a local LLM server opens an independent context window, requiring
the full model + KV-cache allocation per concurrent request. With 2.9 GiB per context and 2.7 GiB
available, even a single concurrent request was impossible. The solution is always
`Semaphore(1)` for memory-constrained local inference environments.

### L6 — Multi-Dimensional Aggregation Always Needs Composite Keys

A Python dictionary used to aggregate results from multiple entities across a shared dimension
(e.g., period) must use a composite key across all varying dimensions. Using only a sub-dimension
key causes silent last-write-wins data loss that is invisible in logs unless the logger also
includes the full composite identity in every message.

---

### Challenge 8 — KPI Dashboard Frozen: Stale Zombie Uvicorn Process Blocking Port 8000

**The Problem:**

After backfilling the database, the KPI Analytics tab showed a perpetual
"EXECUTING QUANT DATA RETRIEVAL..." spinner that never resolved — even though the
`get_kpis` tool is a pure SQLite read with zero LLM involvement. The dashboard had
previously loaded in under a second.

A direct benchmark request:
```python
requests.get('http://localhost:8000/api/kpis?company=Apple Inc.')
```
Also hung indefinitely without returning a response.

**Root Cause — Two Simultaneous Uvicorn Processes:**

The IDE's terminal list showed two separate Uvicorn processes running at the same time:
- One that had been running continuously for **69+ hours** (the original session)
- One that had been running for **~3 minutes** (started recently after changes)

The new process attempted to bind to port 8000, failed (already occupied), and shut
down. But the old 69-hour process had accumulated significant memory fragmentation and
stale connection state over its lifetime. The `CLOSE_WAIT` connections visible in
`netstat` confirmed that the server's TCP stack had zombie connections it could not
clean up, causing new requests to queue and timeout.

**Diagnosis:**

Used `netstat` to identify the exact PID holding port 8000:
```powershell
netstat -ano | findstr :8000
```
Output showed PID `13864` in `LISTENING` state with three `CLOSE_WAIT` zombie
connections — a clear sign of a stale, memory-fragmented server process.

**Resolution:**

Killed the specific PID directly:
```powershell
Stop-Process -Id 13864 -Force
```
Then started a clean fresh Uvicorn process. The KPI tab responded in under 100ms
immediately after the restart.

**Learning:**

Do not let development servers run for multi-day periods without restart, especially
after schema migrations and database operations. `CLOSE_WAIT` connections in `netstat`
are a reliable indicator of a server process that needs a hard restart. Always use
`netstat -ano | findstr :<PORT>` to identify the owning PID before killing blindly.

---

### Challenge 9 — KPI Values Displaying with Wrong Scale (`$117200000000B` instead of `$117.2B`)

**The Problem:**

After the database backfill with Ollama's `llama3` model completed and the frontend
reloaded, the KPI Analytics dashboard displayed wildly incorrect revenue figures:

```
2023-Q1  Revenue: $117200000000B
2023-Q2  Revenue: $9480000000B
2023-Q4  Revenue: $89500000000B
2024-Q1  Revenue: $11960000000B
```

**Root Cause — LLM Ignored the "in billions" Unit Instruction:**

The `AllKPIs` Pydantic schema field description explicitly states:
```python
revenue_b: Optional[float] = Field(
    description="Revenue in billions USD. E.g. $94.9B → 94.9"
)
```

However, `llama3` (a smaller 4.7 GB model) inconsistently followed this instruction.
Across 23 files it returned values in three different scales with no pattern:

| Scale returned | Example | Correct value |
|---|---|---|
| Raw dollars | `117200000000` | `117.2` |
| Raw millions | `55500` | `55.5` |
| Correct billions | `90.53` | `90.53` |

Smaller models are significantly less reliable at following numeric unit conventions
in structured output compared to larger models like `qwen/qwen3-32b`.

**Resolution — Post-hoc Magnitude Normalisation Script:**

Created `src/extraction/normalize_kpis.py` — a one-shot utility that reads every
billion-denominated field from the database, detects its magnitude order, and divides
it down to the correct billions scale:

```python
def normalise(v):
    if v is None: return None
    if v >= 1_000_000_000: return round(v / 1_000_000_000, 4)  # raw dollars
    if v >= 1_000_000:     return round(v / 1_000_000, 4)       # raw millions
    if v >= 10_000:        return round(v / 1_000, 4)            # compact millions
    return v                                                      # already in billions
```

Fixed 20 bad values across 13 rows in a single pass. The dashboard immediately
displayed correct figures after the backend was restarted.

**Learning:**

When using smaller local LLMs for structured extraction, never trust numeric unit
conventions — always validate output magnitudes post-extraction. Design your backfill
pipeline to include a normalisation/validation stage before committing to the database,
or use model-level assertions (`validator` in Pydantic) to reject out-of-range values
at parse time.

---

### Challenge 10 — `normalize_kpis.py` `unable to open database file` Error

**The Problem:**

Running the normalisation script as a module:
```bash
python -m src.extraction.normalize_kpis
```
Failed with `sqlite3.OperationalError: unable to open database file`.

**Root Cause — Wrong `PROJECT_ROOT` Resolution Inside a Subpackage:**

The script computed the project root as:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
`__file__` resolves to `src/extraction/normalize_kpis.py`.  
`.parent` → `src/extraction/`  
`.parent.parent` → `src/`  

This pointed to `src/` instead of the true project root, so the database path
`src/data/finance_kpis.db` did not exist.

**Resolution:**

Ran the normalisation logic as an inline Python one-liner from the project root
directory where the relative path `data/finance_kpis.db` resolves correctly:
```bash
python -c "from pathlib import Path; from src.extraction.schema import ...; engine = get_engine('data/finance_kpis.db')..."
```
The standalone script was subsequently updated to use `.parent.parent.parent` for
correct resolution when invoked as a module from the project root.

**Learning:**

When a script lives inside a nested package (`src/extraction/`), `Path(__file__).resolve()`
resolves relative to the script's file location, not the working directory. Scripts
that must locate sibling directories (like `data/`) relative to the project root
should either:
1. Use `.parent` chain of appropriate depth (fragile — breaks if file moves)
2. Search upward for a sentinel file like `pyproject.toml` or `requirements.txt`
3. Accept the database path as a CLI argument

---

### L7 — Long-Running Dev Servers Accumulate Zombie TCP Connections

Uvicorn (and any WSGI/ASGI server) accumulates `CLOSE_WAIT` TCP connections over
long uptime periods. These zombie connections are not automatically cleaned up and
can block new requests from being accepted. Development servers should be restarted
after major operations (schema migrations, database resets, bulk imports). Use
`netstat -ano | findstr :<PORT>` to diagnose and `Stop-Process -Id <PID> -Force`
to terminate the exact offending process.

### L8 — Small Local LLMs Cannot Be Trusted for Numeric Unit Conventions

Models under ~7B parameters frequently ignore unit-formatting instructions in
structured output schemas (e.g., "return value in billions"). Always add a
post-extraction magnitude validation step before persisting LLM-generated numeric
data. Pydantic `@validator` decorators or a separate normalisation script can catch
unit violations before they propagate to the frontend.

---

## 📁 Files Added/Modified in Phase 9

| File | Change |
|---|---|
| `src/extraction/backfill_kpis.py` | **[NEW]** Async KPI backfill script — single-pass Ollama extraction, Semaphore concurrency control, Pandas DataFrame staging, SQLAlchemy bulk commit |
| `src/extraction/normalize_kpis.py` | **[NEW]** One-shot unit normalisation utility — detects magnitude of billion-denominated fields and divides to correct scale |
| `src/extraction/schema.py` | **[MODIFIED]** Replaced column-level `unique=True` on `period` with table-level `UniqueConstraint('ticker', 'period')` |

---

## ⚠️ Disclaimer

> *This document is part of an educational and research project. All financial outputs generated
> by AURA are for informational purposes only and do not constitute financial advice. Always
> consult a qualified financial professional before making investment decisions.*
