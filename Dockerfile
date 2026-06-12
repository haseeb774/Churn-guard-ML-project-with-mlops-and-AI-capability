# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/           ./src/
COPY app.py         .
COPY streamlit_app.py .
COPY run_pipeline.py .

# Copy pre-trained model and analysis outputs if they exist
COPY outputs/       ./outputs/

# Create required directories
RUN mkdir -p data/raw data/processed logs

# Environment variables (override at runtime)
ENV GOOGLE_AI_STUDIO_API_KEY=""
ENV MLFLOW_TRACKING_URI="http://localhost:5000"
ENV MLFLOW_EXPERIMENT_NAME="churnguard-ai"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8000 8501

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]