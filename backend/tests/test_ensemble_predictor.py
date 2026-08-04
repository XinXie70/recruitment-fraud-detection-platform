"""Unit tests for calibrated ensemble scoring and degradation."""

from __future__ import annotations

import pytest

from services.ensemble_predictor import (
    CalibrationConfig,
    EnsembleConfig,
    EnsembleConfigurationError,
    EnsembleMemberConfig,
    EnsemblePredictor,
    EnsembleUnavailableError,
    apply_calibration,
)
from services.model_adapter import RawModelResult


def _config() -> EnsembleConfig:
    return EnsembleConfig(
        version="test-v1",
        fitted=True,
        weight_source="unit-test",
        low_threshold=0.3,
        high_threshold=0.7,
        models={
            "a": EnsembleMemberConfig(0.6, CalibrationConfig("identity")),
            "b": EnsembleMemberConfig(0.4, CalibrationConfig("identity")),
        },
    )


class FakeRegistry:
    def __init__(self, results):
        self.results = results
        self.batch_timeout_seconds = "not-called"

    def predict_all(self, text, model_keys, timeout_seconds):
        return self.results

    def predict_raw_batches(self, texts, model_keys, timeout_seconds):
        self.batch_timeout_seconds = timeout_seconds
        return {key: [0.25 for _ in texts] for key in model_keys}

    def warm_up(self, sample, model_keys):
        return dict.fromkeys(model_keys)


def test_partial_failure_renormalizes_weights() -> None:
    registry = FakeRegistry(
        [
            RawModelResult("a", "A", "success", score=0.8),
            RawModelResult("b", "B", "error", error="failed", error_code="inference_failed"),
        ]
    )
    predictor = EnsemblePredictor(registry, _config())
    computation = predictor.predict("job listing")

    assert computation.ensemble.status == "degraded"
    assert computation.ensemble.risk_score == pytest.approx(0.8)
    assert computation.ensemble.risk_score_100 == pytest.approx(80.0)
    assert computation.members[0].effective_weight == 1
    assert computation.score_batch(["one", "two"]) == [0.25, 0.25]
    assert registry.batch_timeout_seconds is None


def test_all_model_failures_raise_unavailable() -> None:
    registry = FakeRegistry(
        [
            RawModelResult("a", "A", "error", error="failed"),
            RawModelResult("b", "B", "timeout", error="slow"),
        ]
    )
    with pytest.raises(EnsembleUnavailableError, match="No ensemble model succeeded"):
        EnsemblePredictor(registry, _config()).predict("job listing")


def test_calibration_is_bounded_and_monotonic() -> None:
    sigmoid = CalibrationConfig("sigmoid", coefficient=2, intercept=-0.5)
    assert apply_calibration(-1, CalibrationConfig("identity")) == 0
    assert apply_calibration(2, CalibrationConfig("identity")) == 1
    assert 0 < apply_calibration(0.2, sigmoid) < apply_calibration(0.8, sigmoid) < 1


@pytest.mark.parametrize(
    "config",
    [
        EnsembleConfig("v", False, "test", 0.3, 0.7, {}),
        EnsembleConfig("v", False, "test", 0.8, 0.7, {"a": EnsembleMemberConfig(1, CalibrationConfig("identity"))}),
        EnsembleConfig("v", False, "test", 0.3, 0.7, {"a": EnsembleMemberConfig(-1, CalibrationConfig("identity"))}),
        EnsembleConfig("v", False, "test", 0.3, 0.7, {"a": EnsembleMemberConfig(0.5, CalibrationConfig("identity"))}),
        EnsembleConfig("v", False, "test", 0.3, 0.7, {"a": EnsembleMemberConfig(1, CalibrationConfig("unknown"))}),
    ],
)
def test_invalid_ensemble_configuration_is_rejected(config) -> None:
    with pytest.raises(EnsembleConfigurationError):
        config.validate()
