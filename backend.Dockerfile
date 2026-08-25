FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies and C++ compiler tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend directories
COPY src/ /app/src/
COPY config/ /app/config/

# Create database volume mounting point
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Run API Server
CMD ["python", "-m", "src.api.server"]
