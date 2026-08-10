from __future__ import annotations

import pytest

from services.ensemble_service import (
    EnsembleService,
    RiskService,
    apply_fp_gate,
    apply_risk,
)


@pytest.mark.parametrize(
    ("bert_score", "lr_score", "expected_label", "gate_triggered"),
    [
        (0.88, 0.72, "Fraudulent", False),
        (0.35, 0.02, "Legitimate", True),
        (0.10, 0.90, "Legitimate", False),
    ],
)
def test_apply_fp_gate_business_rules(
    bert_score: float,
    lr_score: float,
    expected_label: str,
    gate_triggered: bool,
) -> None:
    result = apply_fp_gate(
        bert_score=bert_score,
        lr_score=lr_score,
        bert_threshold=0.3,
        lr_gate=0.06,
    )

    assert result["predicted_label"] == expected_label
    assert result["gate_triggered"] is gate_triggered
    assert result["model"] == "ensemble_fp_gate"


@pytest.mark.parametrize(
    ("bert_score", "lr_score", "expected_level", "expected_source"),
    [
        (0.88, 0.72, "High", "bert"),
        (0.35, 0.02, "Suspicious", "lr_gate"),
        (0.001, 0.50, "Low", "bert"),
    ],
)
def test_apply_risk_levels(
    bert_score: float,
    lr_score: float,
    expected_level: str,
    expected_source: str,
) -> None:
    result = apply_risk(
        bert_score=bert_score,
        lr_score=lr_score,
        bert_high_threshold=0.3,
        lr_gate=0.06,
        bert_low_threshold=0.0024,
    )

    assert result["risk_level"] == expected_level
    assert result["risk_score_source"] == expected_source
    assert "decision_reason" in result
    assert result["thresholds"]["lr_gate"] == pytest.approx(0.06)


def test_ensemble_service_uses_precomputed_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    service = EnsembleService()

    def fail_score_pair(*_args: object, **_kwargs: object) -> tuple[dict, dict]:
        raise AssertionError("score_pair should not run when scores are supplied")

    monkeypatch.setattr("services.ensemble_service.score_pair", fail_score_pair)

    result = service.predict(
        "combined",
        "model",
        lr_score=0.4,
        bert_score=0.9,
    )

    assert result["lr_score"] == pytest.approx(0.4)
    assert result["bert_score"] == pytest.approx(0.9)
    assert result["predicted_label"] == "Fraudulent"


def test_risk_service_uses_precomputed_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    service = RiskService()

    def fail_score_pair(*_args: object, **_kwargs: object) -> tuple[dict, dict]:
        raise AssertionError("score_pair should not run when scores are supplied")

    monkeypatch.setattr("services.ensemble_service.score_pair", fail_score_pair)

    result = service.predict(
        "combined",
        "model",
        lr_score=0.4,
        bert_score=0.9,
    )

    assert result["risk_score"] == pytest.approx(0.9)
    assert result["risk_level"] == "High"
