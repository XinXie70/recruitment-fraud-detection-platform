"""
Lightweight in-process TTL cache for analysis results.

Avoids recomputing the full ensemble + XAI + Gentle AI pipeline when the
same job-ad text is submitted within the TTL window.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from config import settings


class TTLCache:
    """Thread-safe, in-memory cache with per-entry TTL."""

    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
    ):
        self._ttl = (
            settings.analysis_cache_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )
        self._max_entries = (
            settings.analysis_cache_max_entries
            if max_entries is None
            else max_entries
        )
        if self._ttl <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if self._max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
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
            self._store.move_to_end(key)
            return entry[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._remove_expired(time.monotonic())
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            self._remove_expired(time.monotonic())
            return len(self._store)

    def _remove_expired(self, now: float) -> None:
        expired = [
            key
            for key, (created_at, _) in self._store.items()
            if now - created_at >= self._ttl
        ]
        for key in expired:
            del self._store[key]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def text_key(text: str) -> str:
        """Deterministic cache key for a job-ad text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Module-level instance — one per process is sufficient.
analysis_cache = TTLCache()
