"""Tests for the bounded in-process analysis cache."""

from __future__ import annotations

import threading
import time

import pytest

from backend.services import cache as cache_module
from backend.services.cache import TTLCache


def test_cache_evicts_least_recently_used_entry() -> None:
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    cache.set("first", 1)
    cache.set("second", 2)

    assert cache.get("first") == 1
    cache.set("third", 3)

    assert cache.get("first") == 1
    assert cache.get("second") is None
    assert cache.get("third") == 3


def test_cache_removes_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now)
    cache = TTLCache(ttl_seconds=5, max_entries=2)
    cache.set("old", 1)

    now = 106.0

    assert cache.size == 0
    assert cache.get("old") is None


@pytest.mark.parametrize(
    ("ttl_seconds", "max_entries"),
    [(0, 1), (1, 0)],
)
def test_cache_rejects_non_positive_limits(
    ttl_seconds: int,
    max_entries: int,
) -> None:
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=ttl_seconds, max_entries=max_entries)


def test_cache_coalesces_concurrent_misses() -> None:
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    calls = 0
    barrier = threading.Barrier(4)
    results: list[int] = []

    def factory() -> int:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return 42

    def worker() -> None:
        barrier.wait()
        results.append(cache.get_or_compute("same-key", factory))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert results == [42, 42, 42, 42]
