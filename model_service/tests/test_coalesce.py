from __future__ import annotations

import threading
import time

from services.coalesce import InferenceCoalescer


def test_coalesce_runs_function_once_for_concurrent_identical_keys() -> None:
    coalescer = InferenceCoalescer()
    call_count = 0
    lock = threading.Lock()

    def slow_fn() -> str:
        nonlocal call_count
        with lock:
            call_count += 1
        time.sleep(0.15)
        return "shared-result"

    results: list[str] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            results.append(coalescer.run("same-key", slow_fn))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert call_count == 1
    assert results == ["shared-result"] * 6


def test_coalesce_uses_separate_calls_for_different_keys() -> None:
    coalescer = InferenceCoalescer()
    call_count = 0

    def fn() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    assert coalescer.run("a", fn) == 1
    assert coalescer.run("b", fn) == 2
    assert call_count == 2


def test_coalesce_propagates_exceptions_to_waiters() -> None:
    coalescer = InferenceCoalescer()
    started = threading.Event()
    errors: list[RuntimeError] = []
    lock = threading.Lock()

    def failing_fn() -> None:
        started.set()
        time.sleep(0.15)
        raise RuntimeError("inference failed")

    def worker() -> None:
        try:
            coalescer.run("fail-key", failing_fn)
        except RuntimeError as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 3
    assert all(str(exc) == "inference failed" for exc in errors)
