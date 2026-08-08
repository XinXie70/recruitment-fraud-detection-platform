"""Tests for SHAP-only word and long-text phrase XAI."""

from __future__ import annotations

import math

import pytest

from xai_gentle import EvidenceSpan, XAIService
from xai_gentle.xai_service import (
    WORD_PATTERN,
    PartitionAttribution,
    TextSegment,
    _build_coarse_segments,
    _build_phrase_segments,
    _top_evidence,
)


def _span(text: str, value: str, contribution: float) -> EvidenceSpan:
    start = text.index(value)
    return EvidenceSpan(
        text=value,
        start=start,
        end=start + len(value),
        contribution=contribution,
        direction="raises_risk" if contribution > 0 else "lowers_risk",
    )


def test_evidence_selection_filters_noise_and_merges_adjacent_shap_tokens() -> None:
    text = "Monday to Friday. Pay a registration fee via cryptocurrency."
    items = [
        _span(text, "Monday", -0.25),
        _span(text, "to", -0.2),
        _span(text, "Friday", -0.15),
        _span(text, "Pay", 0.07),
        _span(text, "registration", 0.1),
        _span(text, "fee", 0.14),
        _span(text, "via", 0.11),
        _span(text, "cryptocurrency", 0.09),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert [item.text for item in selected] == [
        "Pay a registration fee",
        "cryptocurrency",
    ]
    phrase = selected[0]
    assert math.isclose(phrase.contribution, 0.31)
    assert text[phrase.start : phrase.end] == phrase.text


def test_evidence_selection_contextualizes_numbers_and_short_generic_words() -> None:
    text = "Pay a small $50 registration fee to get your training kit."
    items = [
        _span(text, "Pay", 0.178),
        _span(text, "small", -0.0002),
        _span(text, "50", -0.236),
        _span(text, "registration", 0.04),
        _span(text, "fee", 0.034),
        _span(text, "get", 0.015),
        _span(text, "training", 0.029),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert [item.text for item in selected] == [
        "Pay a small",
        "$50 registration fee",
        "get your training kit",
    ]
    assert math.isclose(selected[0].contribution, 0.1778)
    assert math.isclose(selected[1].contribution, -0.162)
    assert math.isclose(selected[2].contribution, 0.044)
    assert all(len(item.text.split()) <= 4 for item in selected)
    assert all(text[item.start : item.end] == item.text for item in selected)


def test_evidence_selection_never_erases_all_generic_attributions() -> None:
    text = "Monday to Friday. Respond to client enquiries by email."
    items = [
        _span(text, "Monday", -0.05),
        _span(text, "Friday", -0.04),
        _span(text, "Respond", -0.03),
        _span(text, "client", -0.02),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert selected
    assert all(len(item.text.split()) > 1 for item in selected)
    assert all(text[item.start : item.end] == item.text for item in selected)


def test_evidence_selection_falls_back_when_context_contributions_cancel() -> None:
    text = "Apply now for a service role."
    items = [
        _span(text, "Apply", 0.05),
        _span(text, "now", -0.05),
        _span(text, "service", 0.01),
        _span(text, "role", -0.01),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert selected
    assert all(text[item.start : item.end] == item.text for item in selected)


def test_evidence_selection_uses_relative_floor_for_small_model_scores() -> None:
    text = "Established employer with verified contact details."
    items = [
        _span(text, "Established", -0.00008),
        _span(text, "verified", -0.00004),
        _span(text, "contact", -0.00000001),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert [item.text for item in selected] == ["Established", "verified contact"]
    assert all(item.text != "contact" for item in selected)


def test_evidence_selection_retains_relative_signals_below_old_absolute_floor() -> None:
    text = "Established employer with verified details."
    items = [
        _span(text, "Established", -0.00000008),
        _span(text, "verified", -0.00000004),
        _span(text, "details", 0.00000000001),
    ]

    selected = _top_evidence(items, 10, text=text, merge_shap_tokens=True)

    assert selected
    assert any(item.text == "Established" for item in selected)
    assert all(text[item.start : item.end] == item.text for item in selected)


def test_shap_partition_path_preserves_offsets() -> None:
    pytest.importorskip("shap")
    text = "Urgent job requires an advance fee."

    def scorer(texts):
        return [
            0.2
            + (0.2 if "urgent" in value.lower() else 0)
            + (0.5 if "fee" in value.lower() else 0)
            for value in texts
        ]

    result = XAIService(max_evals=60).explain(text, scorer, scorer([text])[0])

    assert result.status == "success"
    assert result.method == "shap_partition"
    assert result.target == "ensemble_risk_score"
    assert result.version == "xai-v2"
    for item in result.items:
        assert text[item.start : item.end] == item.text


def test_coarse_segments_preserve_offsets_and_limit_words() -> None:
    text = (
        "Contact jobs@example.com for details.\n"
        "Earn money quickly with no interview and an immediate start."
    )

    segments = _build_coarse_segments(text, max_words=4)

    assert segments
    assert all(text[item.start : item.end].strip() for item in segments)
    assert all(
        len(text[item.start : item.end].split()) <= 4 for item in segments
    )
    assert any("jobs@example.com" in text[item.start : item.end] for item in segments)


def test_phrase_segments_are_short_and_preserve_offsets() -> None:
    text = (
        "Southern Cross University values potential as much as experience. "
        "Applicants must apply online and attach a CV and cover letter."
    )

    segments = _build_phrase_segments(text, max_words=6)

    assert segments
    assert all(
        len(WORD_PATTERN.findall(text[item.start : item.end])) <= 6
        for item in segments
    )
    assert all(text[item.start : item.end] for item in segments)
    assert all(". " not in text[item.start : item.end] for item in segments)


def test_hierarchical_shap_returns_original_word_offsets() -> None:
    pytest.importorskip("shap")
    text = (
        "Established company with an office and a normal recruitment process. "
        "The role includes ordinary customer support and administration duties. "
        "Pay a registration fee using cryptocurrency before work begins. "
        "Candidates receive training and work with an experienced local team."
    )

    def scorer(texts):
        outputs = []
        for value in texts:
            lowered = value.lower()
            score = 0.1
            score += 0.25 if "registration" in lowered else 0
            score += 0.3 if "fee" in lowered else 0
            score += 0.25 if "cryptocurrency" in lowered else 0
            outputs.append(score)
        return outputs

    result = XAIService(
        max_evals=60,
        hierarchical_min_words=20,
        hierarchical_top_segments=2,
        hierarchical_segment_words=20,
        hierarchical_max_evals=60,
    ).explain(text, scorer, scorer([text])[0])

    assert result.status == "success"
    assert result.method == "shap_partition"
    assert result.version == "xai-v2"
    assert (result.message or "").startswith("Long-text Partition SHAP")
    assert any(
        term in item.text.lower()
        for item in result.items
        for term in {"registration", "fee", "cryptocurrency"}
    )
    assert all(len(item.text.split()) <= 8 for item in result.items)
    for item in result.items:
        assert text[item.start : item.end] == item.text


def test_long_text_shap_does_not_present_sentences_as_phrase_evidence(
    monkeypatch,
) -> None:
    text = (
        "Established employer with formal interviews and verified contact details. "
        "The role provides ordinary duties, a salary, and a local office."
    )
    def fake_partition(value, _scorer, _builder, _max_evals):
        segments = _builder(value)
        return PartitionAttribution(
            segments=segments,
            contributions=[0.0] * len(segments),
            base_value=0.2,
        )

    service = XAIService(
        max_items=5,
        max_evals=40,
        hierarchical_min_words=1,
        hierarchical_top_segments=2,
        hierarchical_segment_words=20,
        hierarchical_max_evals=40,
    )
    monkeypatch.setattr(service, "_partition_attributions", fake_partition)

    result = service.explain(text, lambda values: [0.0] * len(values), 0.0)

    assert result.status == "success"
    assert result.method == "shap_partition"
    assert result.items == []
    assert "no reliable phrase-level evidence" in (result.message or "")


def test_shap_failure_returns_unavailable_without_leaking_details() -> None:
    secret = "private-model-path"

    def failing_scorer(_texts):
        raise RuntimeError(secret)

    result = XAIService(max_evals=20).explain(
        "Urgent job requires an advance fee.",
        failing_scorer,
        0.7,
    )

    assert result.status == "unavailable"
    assert result.method == "unavailable"
    assert result.output_value == pytest.approx(0.7)
    assert result.items == []
    assert secret not in (result.message or "")
    assert "temporarily unavailable" in (result.message or "")


def test_numeric_settings_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        XAIService(hierarchical_top_segments=-1)
