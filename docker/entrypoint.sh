#!/bin/sh
set -eu

cd /app/model_service

PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${GUNICORN_WORKERS:-1}"
THREADS="${GUNICORN_THREADS:-4}"
TIMEOUT="${GUNICORN_TIMEOUT:-180}"

echo "Starting model_service API on ${HOST}:${PORT}"
echo "models=${MODEL_FILES_ROOT:-/app/model_service/models}"
echo "workers=${WORKERS} threads=${THREADS} timeout=${TIMEOUT} ALLOW_CPU=${ALLOW_CPU:-1}"

# Preload LR + BERT in the master process (keep workers=1 to avoid duplicate memory).
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
