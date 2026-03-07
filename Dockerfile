# ViziGenesis — Production Dockerfile
# Multi-stage build: slim runtime with GPU support option

# ── Stage 1: builder ──────────────────────────────────────────────────
FROM python:3.10.11-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.10.11-slim AS runtime

LABEL maintainer="ViziGenesis Team" \
      version="2.0.0" \
      description="AI-powered multi-horizon stock prediction platform"

WORKDIR /app

# Runtime system libs (libgomp for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Non-root user
RUN useradd -m -r vizigenesis && chown -R vizigenesis:vizigenesis /app
USER vizigenesis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v2/status || exit 1

# Expose port
EXPOSE 8000

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIZIGENESIS_ENV=production

# Run with uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
