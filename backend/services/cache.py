"""
Lightweight in-process TTL cache for analysis results.

Avoids recomputing the full ensemble + XAI + Gentle AI pipeline when the
same job-ad text is submitted within the TTL window.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from config import settings


class TTLCache:
    """Thread-safe, in-memory cache with per-entry TTL."""

    def __init__(self, ttl_seconds: int | None = None):
        self._ttl = ttl_seconds or settings.analysis_cache_ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry[0] >= self._ttl:
                del self._store[key]
                return None
            return entry[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def text_key(text: str) -> str:
        """Deterministic cache key for a job-ad text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Module-level instance — one per process is sufficient.
analysis_cache = TTLCache()
