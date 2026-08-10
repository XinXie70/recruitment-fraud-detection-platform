"""Unit tests for sprint3 BERT + LR FP-gate ensemble scoring."""

from __future__ import annotations

import pytest

from backend.services.ensemble_predictor import (
    CalibrationConfig,
    EnsembleConfig,
    EnsembleMemberConfig,
    EnsemblePredictor,
    EnsembleUnavailableError,
)
from backend.services.model_adapter import RawModelResult


def _fp_gate_config() -> EnsembleConfig:
    return EnsembleConfig(
        version="test-fp-gate",
        fitted=True,
        weight_source="unit-test",
        low_threshold=0.0024,
        high_threshold=0.3,
        models={
            "logistic_regression": EnsembleMemberConfig(
                0.0, CalibrationConfig("identity"), role="false_positive_gate"
            ),
            "bert": EnsembleMemberConfig(
                1.0, CalibrationConfig("identity"), role="primary_score"
            ),
        },
        method="bert_lr_fp_gate",
        bert_low_threshold=0.0024,
        bert_high_threshold=0.3,
        lr_gate=0.06,
    )


class FakeRegistry:
    def __init__(self, lr: float, bert: float):
        self.lr = lr
        self.bert = bert
        self.batch_timeout_seconds = "not-called"

    def predict_all(self, text, model_keys, timeout_seconds):
        scores = {
            "logistic_regression": self.lr,
            "bert": self.bert,
        }
        return [
            RawModelResult(key, key, "success", score=scores[key]) for key in model_keys
        ]

    def predict_raw_batches(self, texts, model_keys, timeout_seconds):
        self.batch_timeout_seconds = timeout_seconds
        mapping = {
            "logistic_regression": self.lr,
            "bert": self.bert,
        }
        return {key: [mapping[key] for _ in texts] for key in model_keys}

    def warm_up(self, sample, model_keys):
        return dict.fromkeys(model_keys)


def test_fp_gate_demotes_high_when_lr_is_low() -> None:
    predictor = EnsemblePredictor(FakeRegistry(lr=0.01, bert=0.5), _fp_gate_config())
    result = predictor.predict("job listing").ensemble
    assert result.method == "bert_lr_fp_gate"
    assert result.gate_triggered is True
    assert result.risk_level == "medium"
    assert result.risk_score_source == "lr_gate"
    assert result.risk_score == pytest.approx(0.01)


def test_fp_gate_keeps_high_when_lr_passes() -> None:
    predictor = EnsemblePredictor(FakeRegistry(lr=0.2, bert=0.5), _fp_gate_config())
    result = predictor.predict("job listing").ensemble
    assert result.gate_triggered is False
    assert result.risk_level == "high"
    assert result.risk_score_source == "bert"
    assert result.risk_score == pytest.approx(0.5)


def test_fp_gate_batch_uses_risk_scores() -> None:
    registry = FakeRegistry(lr=0.01, bert=0.5)
    computation = EnsemblePredictor(registry, _fp_gate_config()).predict("job listing")
    assert computation.score_batch(["a", "b"]) == pytest.approx([0.01, 0.01])
    assert registry.batch_timeout_seconds is None


def test_fp_gate_requires_both_models() -> None:
    class BrokenRegistry(FakeRegistry):
        def predict_all(self, text, model_keys, timeout_seconds):
            return [
                RawModelResult("logistic_regression", "LR", "success", score=0.2),
                RawModelResult("bert", "BERT", "error", error="missing"),
            ]

    with pytest.raises(EnsembleUnavailableError):
        EnsemblePredictor(BrokenRegistry(0.2, 0.5), _fp_gate_config()).predict("x")
