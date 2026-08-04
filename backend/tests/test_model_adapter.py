"""Unit tests for model adapters and concurrent registry execution."""

from __future__ import annotations

import pytest
import httpx

from services.model_adapter import HttpModelAdapter, ModelAdapter, ModelRegistry, ModelSpec


class FakeAdapter:
    def __init__(self, key: str, score: float = 0.5, error: Exception | None = None):
        self.key = key
        self.display_name = key.upper()
        self.score = score
        self.error = error

    def predict_raw(self, text: str) -> float:
        if self.error:
            raise self.error
        return self.score

    def predict_raw_batch(self, texts):
        if self.error:
            raise self.error
        return [self.score for _ in texts]


def test_registry_reports_success_failure_and_missing_adapter() -> None:
    registry = ModelRegistry(
        [
            FakeAdapter("good", score=0.8),
            FakeAdapter("bad", error=RuntimeError("private details")),
        ],
        max_workers=2,
    )
    results = registry.predict_all(
        "listing",
        ["good", "bad", "missing"],
        timeout_seconds=1,
    )

    by_key = {result.key: result for result in results}
    assert by_key["good"].score == 0.8
    assert by_key["bad"].error_code == "inference_failed"
    assert "private details" not in by_key["bad"].error
    assert by_key["missing"].error_code == "not_registered"


def test_registry_scores_batches_and_handles_empty_input() -> None:
    registry = ModelRegistry([FakeAdapter("a", 0.25), FakeAdapter("b", 0.75)])
    assert registry.predict_all("text", [], timeout_seconds=1) == []
    assert registry.predict_raw_batches(["one", "two"], [], timeout_seconds=1) == {}
    assert registry.predict_raw_batches(
        ["one", "two"], ["a", "b"], timeout_seconds=1
    ) == {"a": [0.25, 0.25], "b": [0.75, 0.75]}


def test_registry_rejects_missing_batch_adapter() -> None:
    registry = ModelRegistry([])
    with pytest.raises(KeyError, match="not registered"):
        registry.predict_raw_batches(["text"], ["missing"], timeout_seconds=1)


def test_model_adapter_normalises_array_like_scores() -> None:
    class ArrayLike:
        def tolist(self):
            return [0.1, 0.9]

    assert ModelAdapter._normalise_scores(ArrayLike(), 2) == [0.1, 0.9]
    with pytest.raises(ValueError, match="Expected 1"):
        ModelAdapter._normalise_scores([0.1, 0.2], 1)
    with pytest.raises(ValueError, match="invalid probability"):
        ModelAdapter._normalise_scores([float("nan")], 1)


def test_model_adapter_checks_required_artifact(tmp_path) -> None:
    adapter = ModelAdapter(
        ModelSpec("model", "Model", "unused", artifact_relative_path="missing.bin"),
        project_root=tmp_path,
    )
    with pytest.raises(Exception, match="weight is missing"):
        adapter.predict_raw("text")


def test_http_model_adapter_uses_model_endpoint_and_fraud_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/lr"
        assert request.content == b'{"text":"listing"}'
        return httpx.Response(200, json={"fraud_score": 0.73})

    adapter = HttpModelAdapter("logistic_regression", "LR", "http://model")
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))

    assert adapter.predict_raw("listing") == 0.73


def test_http_model_adapter_rejects_invalid_response() -> None:
    adapter = HttpModelAdapter("bert", "BERT", "http://model")
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"prediction": 1})
        )
    )

    with pytest.raises(RuntimeError, match="no fraud_score"):
        adapter.predict_raw("listing")
