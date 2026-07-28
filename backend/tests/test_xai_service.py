"""Tests for bounded occlusion XAI and safe degradation."""

from __future__ import annotations

from xai_gentle.xai_service import XAIService, _build_segments


def test_segments_are_bounded_for_long_text() -> None:
    segments = _build_segments("one two three four five six", max_segments=2)
    assert len(segments) == 2


def test_occlusion_returns_ranked_model_derived_evidence() -> None:
    service = XAIService(prefer_shap=False, max_items=2, max_segments=10)

    def scorer(texts):
        baseline = 0.8
        return [baseline] + [0.2 + index * 0.1 for index in range(len(texts) - 1)]

    result = service.explain("gift card payment required", scorer, expected_output=0.8)
    assert result.status == "success"
    assert result.method == "occlusion_fallback"
    assert len(result.items) <= 2
    assert all(item.direction == "raises_risk" for item in result.items)


def test_empty_text_has_no_evidence() -> None:
    result = XAIService(prefer_shap=False).explain("...", lambda texts: [0.1], 0.1)
    assert result.status == "success"
    assert result.items == []


def test_scorer_contract_failure_is_safely_hidden() -> None:
    secret = "internal-model-secret"

    def broken_scorer(texts):
        raise RuntimeError(secret)

    result = XAIService(prefer_shap=False).explain("meaningful words", broken_scorer, 0.5)
    assert result.status == "unavailable"
    assert secret not in result.message


def test_shap_failure_falls_back_without_leaking_details(monkeypatch) -> None:
    service = XAIService(prefer_shap=True)
    monkeypatch.setattr(
        service,
        "_explain_with_shap",
        lambda *args: (_ for _ in ()).throw(RuntimeError("private-path")),
    )

    result = service.explain("gift card", lambda texts: [0.7] * len(texts), 0.7)
    assert result.method == "occlusion_fallback"
    assert "private-path" not in result.message
