# Phase 6: Dockerization — Challenges & Solutions

This document outlines the engineering challenges, build configurations, and resolution procedures encountered while containerizing the **Financial Earnings Intelligence Platform** in Phase 6. It details how the codebase was optimized for shipping to other developers and how virtualization disk issues were handled.

---

## 🚀 Key Configurations Added

1. **Multi-Container Architecture (Docker Compose):**
   * Orchestrates two coordinated services: a `backend` FastAPI server (port `8000`) and a `frontend` Next.js production client (port `3000`).
2. **Optimized Multi-Stage Next.js Builder:**
   * Utilized a two-stage build pipeline (`Node:20-alpine`) for the frontend. 
   * Stage 1 compiles the application and dependencies, while Stage 2 copies only the optimized `.next` build files, `public` assets, and production node modules. This keeps the frontend container lightweight.
3. **Database Volume Persistence:**
   * Bound the local host database directory (`./data:/app/data`) via Docker Volumes. This ensures that the vector databases (ChromaDB), sparse lexical index files (BM25), and structured quantitative metrics databases (SQLite) are persisted across container rebuilds and restarts.
4. **Environment Isolation:**
   * Configured `docker-compose.yml` to inject the Groq API key dynamically at runtime using `env_file: config/.env`. This keeps secret keys separate from the build context and secures them from being baked into the Docker image itself.

---

## 🛠️ Challenges Faced & Resolutions

### 1. ChromaDB Native Compilation Failures
**The Challenge:**
The initial build of the backend container crashed with a compilation error during the execution of `pip install -r requirements.txt`.

**The Root Cause:**
The base image `python:3.11-slim` is optimized for size and does not contain C/C++ compiler tools. However, the vector store library **`chromadb`** depends on native C++ libraries (like `hnswlib` for vector indexing) which must be compiled from source during installation. Without compile tools, pip fails.

**The Resolution:**
We updated [backend.Dockerfile](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/backend.Dockerfile) to download and configure build dependencies at the system level before installing Python libraries:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*
```
This provided the necessary build toolchains, allowing `chromadb` to compile successfully during image assembly.

---

### 2. Slow Build Times & Large Transfer Payloads
**The Challenge:**
The first container compilation took over 15 minutes to run, occasionally freezing on step 5 of the backend build.

**The Root Cause:**
The python backend requires massive libraries for natural language processing and embeddings:
* **PyTorch (`torch`)** alone exceeds 1 GB in size.
* **Hugging Face (`transformers`, `sentence-transformers`)** takes another 300+ MB.
* Sending unnecessary directories (like local virtual environments `.venv/` or Next.js `node_modules/` folder containing hundreds of megabytes) into the Docker build daemon slowed down the transfer phase significantly.

**The Resolution:**
* We configured strict [.dockerignore](file:///c:/Users/ADMIN/Documents/3rd_Year_Projects/Finance_RAG_Project/.dockerignore) and `frontend/.dockerignore` files to prevent the Docker build daemon from copying local environments, git logs, database files, and package folders.
* We explained to developers that this build overhead is a **one-time cost**. Docker caches intermediate layers. Subsequent starts and rebuilds bypass this `pip install` step and launch in less than a second unless `requirements.txt` changes.

---

### 3. WSL2 VHDX Disk Inflation (Windows Explorer Storage Lockup)
**The Challenge:**
After building and subsequently pruning the Docker containers on Windows, the free disk space displayed in Windows Explorer C: drive did not increase.

**The Root Cause:**
Docker Desktop on Windows manages its virtual filesystem inside a WSL2 virtual disk file (`ext4.vhdx`).
While WSL2 automatically expands the `.vhdx` file as files are downloaded inside containers, **it never automatically shrinks the file when files are deleted inside containers.** The virtual hard disk keeps occupying its peak expanded space on the host machine.

**The Resolution:**
We provided a step-by-step WSL virtual disk compaction procedure using Windows `diskpart`:
1. Shut down the WSL VM (`wsl --shutdown`).
2. Run `diskpart` in an Administrator command prompt.
3. Select the virtual disk file:
   `select vdisk file="C:\Users\<Username>\AppData\Local\Docker\wsl\data\ext4.vhdx"`
4. Run `compact vdisk` to shrink the file down to only its active data size, freeing up gigabytes of host space on the C: drive.

---

### 4. Cross-Container API Routing
**The Challenge:**
The frontend web app inside the container failed to fetch data from the backend API, throwing connection refused errors when accessing `http://localhost:8000/api/chat`.

**The Root Cause:**
Within a containerized network, `localhost` refers to the container itself, not the host machine. The frontend container trying to contact `localhost:8000` was looking inside itself instead of communicating with the backend container.

**The Resolution:**
Docker Compose creates an internal network where containers can resolve each other by their service name. We mapped the service name of the backend FastAPI server to `backend` inside `docker-compose.yml`. The frontend can now securely communicate with the API server across the isolated network using the internal URI:
`http://backend:8000`
