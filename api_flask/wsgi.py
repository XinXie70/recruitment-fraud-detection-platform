"""Gunicorn entrypoint: preload LR + BERT before serving traffic."""

from __future__ import annotations

import os

from app import app, bert_service, lr_service
from settings import load_runtime_config


def _bootstrap() -> None:
    allow_cpu = os.getenv("ALLOW_CPU", "1").strip().lower() not in {"0", "false", "no"}
    print("Loading runtime config...", flush=True)
    print(load_runtime_config(), flush=True)
    print("Loading LR...", flush=True)
    lr_service.load()
    print(f"Loading BERT (allow_cpu={allow_cpu})...", flush=True)
    bert_service.load(allow_cpu=allow_cpu)
    print("Models ready.", flush=True)


_bootstrap()

# Exported for gunicorn: `wsgi:app`
__all__ = ["app"]
