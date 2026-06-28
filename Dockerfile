# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build dependencies needed by sentence-transformers / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source (preserve folder structure)
COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY .env.example .env.example

# Switch to non-root user
USER appuser

# ── Environment defaults (override via docker run -e or docker-compose) ─────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Health check — Streamlit responds on /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# Entrypoint
CMD ["python", "-m", "streamlit", "run", "backend/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
