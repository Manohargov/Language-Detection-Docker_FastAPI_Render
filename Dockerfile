# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Language Detection API — production image
#
# Notes:
#  * Built from `python:3.12-slim` instead of the deprecated
#    `tiangolo/uvicorn-gunicorn-fastapi` image. Uvicorn now manages its own
#    worker processes, so Gunicorn is no longer required.
#  * Dependencies are installed before the source is copied so that Docker
#    can cache the (slow) pip layer across code changes.
#  * Runs as a non-root user.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# curl is used by the HEALTHCHECK below
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── 1. Dependencies (cached layer) ───────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── 2. Application code ──────────────────────────────────────────────────────
COPY main.py ./
COPY model/ ./model/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ── 3. Non-root user ─────────────────────────────────────────────────────────
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
