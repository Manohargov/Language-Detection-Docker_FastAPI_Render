#!/bin/sh
# Container entrypoint — starts the API with uvicorn.
#
# `exec` replaces this shell so uvicorn becomes PID 1 and receives
# SIGTERM directly, letting the orchestrator shut the container down cleanly.
#
# PORT and WEB_CONCURRENCY are honoured so the same image works on
# Render (PORT=10000), Azure App Service and plain Docker.
set -e

PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WEB_CONCURRENCY}" \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --log-level info
