nano requirements.txt# ─────────────────────────────────────────────────────────────────────────────
# Mobile ECG AI Platform — Dockerfile
# Multi-stage build: builder installs deps, runtime is lean.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a local prefix
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ecg-ai-platform" \
      description="Mobile ECG AI Platform — FastAPI inference service" \
      version="1.0.0"

# Runtime system libs (OpenCV headless needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash ecgapp
USER ecgapp
WORKDIR /home/ecgapp/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=ecgapp:ecgapp app/ ./app/
COPY --chown=ecgapp:ecgapp models/ ./models/

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/home/ecgapp/app/models \
    PORT=8000 \
    WORKERS=2 \
    ORT_THREADS=4

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn app.main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers ${WORKERS} \
        --log-level info \
        --access-log"]
