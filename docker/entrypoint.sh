#!/bin/sh
set -eu

cd /app/api_flask

PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${GUNICORN_WORKERS:-1}"
THREADS="${GUNICORN_THREADS:-4}"
TIMEOUT="${GUNICORN_TIMEOUT:-180}"

echo "Starting fraud-detection API on ${HOST}:${PORT}"
echo "workers=${WORKERS} threads=${THREADS} timeout=${TIMEOUT} ALLOW_CPU=${ALLOW_CPU:-1}"

# Preload models in the master process (BERT is large; keep workers=1).
exec gunicorn \
  --bind "${HOST}:${PORT}" \
  --workers "${WORKERS}" \
  --threads "${THREADS}" \
  --timeout "${TIMEOUT}" \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile - \
  --capture-output \
  "wsgi:app"
