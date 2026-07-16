import math

import pytest

from backend.xai_gentle import EvidenceSpan, XAIService
from backend.xai_gentle.xai_service import _top_evidence


def _span(text: str, value: str, contribution: float) -> EvidenceSpan:
    start = text.index(value)
    return EvidenceSpan(
        text=value,
        start=start,
        end=start + len(value),
        contribution=contribution,
        direction="raises_risk" if contribution > 0 else "lowers_risk",
    )


def test_evidence_selection_filters_noise_and_merges_adjacent_shap_tokens():
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

    selected = _top_evidence(
        items,
        10,
        text=text,
        merge_shap_tokens=True,
    )

    assert [item.text for item in selected] == [
        "Pay",
        "registration fee",
        "cryptocurrency",
    ]
    phrase = selected[1]
    assert math.isclose(phrase.contribution, 0.24)
    assert text[phrase.start : phrase.end] == phrase.text


def test_occlusion_explains_the_supplied_ensemble_and_preserves_offsets():
    text = "Urgent role. Pay a registration fee before the interview."

    def scorer(texts):
        outputs = []
        for value in texts:
            lowered = value.lower()
            score = 0.2
            if "urgent" in lowered:
                score += 0.2
            if "fee" in lowered:
                score += 0.4
            if "interview" in lowered:
                score -= 0.1
            outputs.append(score)
        return outputs

    expected = scorer([text])[0]
    result = XAIService(prefer_shap=False, max_items=10).explain(text, scorer, expected)

    assert result.status == "success"
    assert result.method == "occlusion_fallback"
    assert math.isclose(result.output_value, expected)
    assert any(item.text.lower() == "fee" for item in result.items)
    assert all(item.text.lower() not in {"a", "the", "to"} for item in result.items)
    for item in result.items:
        assert text[item.start : item.end] == item.text


def test_shap_partition_path_preserves_offsets_when_shap_is_installed():
    pytest.importorskip("shap")
    text = "Urgent job requires an advance fee."

    def scorer(texts):
        return [
            0.2
            + (0.2 if "urgent" in value.lower() else 0)
            + (0.5 if "fee" in value.lower() else 0)
            for value in texts
        ]

    result = XAIService(prefer_shap=True, max_evals=60).explain(
        text,
        scorer,
        scorer([text])[0],
    )

    assert result.status == "success"
    assert result.method == "shap_partition"
    for item in result.items:
        assert text[item.start : item.end] == item.text
