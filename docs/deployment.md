# Deployment Guide

> **[← Architecture](./architecture.md)** | **[← README](../README.md)**

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11–3.12 (avoid 3.14 — LangChain Pydantic v1 incompatibilities) |
| Node.js | 18 | 20 LTS |
| RAM | 4 GB | 8 GB (reranker model loaded in-memory) |
| Storage | 2 GB free | 4 GB (Docker images + model weights) |
| CPU | Any x86-64 | Multi-core (embedding speed scales with cores) |

### API Keys

| Service | Key Variable | Where to Get |
|---|---|---|
| Groq LPU Inference | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) — free tier available |

---

## Option A — Local Development Setup

### Step 1: Clone & Configure

```bash
git clone <your-repo-url>
cd Finance_RAG_Project
```

Create your environment file:
```bash
# config/.env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users**: Some packages (`chromadb`, `hnswlib`) require C++ build tools. If `pip install` fails, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### Step 3: Run Data Ingestion Pipeline

```bash
python -m src.ingestion.pipeline
```

**What this does:**
- Discovers all 23 `.txt` transcript files in `dataset_2/`
- Cleans boilerplate (operator instructions, Safe Harbor disclaimers)
- Chunks text into ~1,434 semantic passages
- Computes 384-dimensional embeddings using `all-MiniLM-L6-v2` locally
- Indexes chunks into ChromaDB (`data/chroma_db/`)
- Builds BM25 lexical index (`data/bm25_index/bm25.pkl`)
- Extracts structured KPIs into SQLite (`data/finance_kpis.db`)

**Expected duration:** 2–4 minutes on a modern CPU.

> **Skip ingestion** if `data/` directory already contains pre-built indices. The pipeline deduplicates automatically.

### Step 4: Start the Backend API Server

```bash
python -m src.api.server
```

Server boots on `http://localhost:8000`. Verify with:
```bash
curl http://localhost:8000/
# Expected: {"status": "ok", "message": "Finance RAG API running"}
```

### Step 5: Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Option B — Docker Compose (Full Stack)

Docker Compose orchestrates both services, handles networking, and mounts data volumes automatically.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Build & Launch

```bash
docker compose up --build
```

**First build takes 10–20 minutes** due to PyTorch (~1 GB) and Hugging Face model downloads. Subsequent starts are under 10 seconds (Docker layer cache).

### Run Ingestion Inside Container

If starting fresh with empty `data/` directory:

```bash
docker compose run --rm backend python -m src.ingestion.pipeline
```

### Access the Application

Open [http://localhost:3000](http://localhost:3000).

### Stopping

```bash
docker compose down          # Stop containers (data preserved)
docker compose down -v       # Stop + remove volumes (data wiped)
```

---

## Docker Architecture

```yaml
# docker-compose.yml
services:
  backend:
    build:
      context: .
      dockerfile: backend.Dockerfile
    ports: ["8000:8000"]
    env_file: ["config/.env"]
    volumes: ["./data:/app/data"]  # Persists ChromaDB + BM25 + SQLite

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports: ["3000:3000"]
    depends_on: [backend]
```

**Key design decisions:**
- `./data:/app/data` volume mount — database indices persist across container rebuilds
- `env_file: config/.env` — Groq API key injected at runtime, not baked into image
- `depends_on: [backend]` — frontend waits for backend service to be ready
- Internal DNS: frontend calls `http://backend:8000` (Docker service name resolution)

---

## Troubleshooting

### Port Already in Use (Error 10048)

```
ERROR: [Errno 10048] Only one usage of each socket address is permitted: 8000
```

Find and kill the process on port 8000:
```powershell
# Windows PowerShell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

```bash
# Linux/macOS
lsof -ti:8000 | xargs kill -9
```

---

### ChromaDB Compilation Error (Docker)

```
error: command 'gcc' failed: No such file or directory
```

This occurs if build tools are missing in the Docker image. Ensure `backend.Dockerfile` includes:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make python3-dev sqlite3 \
    && rm -rf /var/lib/apt/lists/*
```

---

### WSL2 Disk Space Not Freeing After Docker Prune (Windows)

Docker Desktop on Windows stores container layers in a WSL2 virtual disk (`.vhdx`) that **never automatically shrinks** even after `docker system prune`. To reclaim disk space:

1. Shut down WSL: `wsl --shutdown`
2. Open `diskpart` as Administrator
3. Run:
   ```
   select vdisk file="C:\Users\<Username>\AppData\Local\Docker\wsl\data\ext4.vhdx"
   compact vdisk
   ```

---

### Groq Rate Limit Errors

The UI displays a friendly message:
> 🛑 **Execution Paused: API Quota Reached** — Please try again in X seconds.

The free Groq tier has token-per-minute limits. For heavy usage, consider:
- Upgrading to a paid Groq plan
- Switching to `llama3-8b-8192` in `config/config.yaml` (faster, lower quality)

---

### Nvidia/Company Missing From Multi-Entity Response

If one company is consistently absent from comparison responses:

1. Check that the data ingestion pipeline has indexed transcripts for that company:
   ```bash
   python -c "from src.retrieval.vector_store import EarningsVectorStore; vs = EarningsVectorStore(); print(vs.count())"
   ```
   Expected: ~1,434 total chunks.

2. Increase the response richness slider to ≥12 references in the UI.

3. Use an explicit company mention in your query: *"Compare risks for Apple, Microsoft, and Nvidia"* rather than *"Compare risks for all companies"*.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq LPU API key for LLM inference |

Stored in: `config/.env` (gitignored — never commit this file).

---

## Performance Tuning

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `chunk_size` | `config/config.yaml` | 1000 chars | Larger = more context per chunk, slower embedding |
| `chunk_overlap` | `config/config.yaml` | 200 chars | Higher = better cross-boundary recall |
| `top_k` (UI slider) | Frontend | 6 | Higher = more sources, longer responses |
| `temperature` | `config/config.yaml` | 0.0 | Never increase — financial facts must be deterministic |
| `batch_size` | `config/config.yaml` | 64 chunks | Increase for faster ingestion on high-RAM machines |
