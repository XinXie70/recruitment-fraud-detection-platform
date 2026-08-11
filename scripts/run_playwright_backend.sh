#!/bin/sh
set -eu

if [ -x ../.venv/bin/python ]; then
  exec ../.venv/bin/python ../scripts/run_playwright_backend.py
fi

exec python ../scripts/run_playwright_backend.py
