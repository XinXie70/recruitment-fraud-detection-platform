"""Coalesce concurrent identical inference requests onto one computation."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")


class InferenceCoalescer:
    """Share in-flight work for the same key across concurrent callers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}

    def run(self, key: str, fn: Callable[[], T]) -> T:
        owner = False
        with self._lock:
            fut = self._inflight.get(key)
            if fut is None:
                fut = Future()
                self._inflight[key] = fut
                owner = True

        if not owner:
            return fut.result()

        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
            with self._lock:
                self._inflight.pop(key, None)
            raise
        else:
            fut.set_result(result)
            with self._lock:
                self._inflight.pop(key, None)
            return result
