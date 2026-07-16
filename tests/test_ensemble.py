import math
from pathlib import Path

from services.ensemble_predictor import (
    CalibrationConfig,
    EnsembleConfig,
    EnsembleMemberConfig,
    EnsemblePredictor,
    default_config_path,
)
from services.model_adapter import (
    DEFAULT_MODEL_SPECS,
    ModelAdapter,
    ModelArtifactUnavailableError,
    ModelRegistry,
    ModelSpec,
)

from helpers import FailingAdapter, FakeAdapter, SlowAdapter


def make_config() -> EnsembleConfig:
    identity = CalibrationConfig(kind="identity")
    return EnsembleConfig(
        version="ensemble-test",
        fitted=True,
        weight_source="unit_test",
        low_threshold=0.3,
        high_threshold=0.7,
        models={
            "first": EnsembleMemberConfig(0.2, identity),
            "second": EnsembleMemberConfig(0.3, identity),
            "third": EnsembleMemberConfig(0.5, identity),
        },
    )


def test_weighted_soft_voting_and_contributions_are_exact():
    registry = ModelRegistry(
        [
            FakeAdapter("first", "First", 0.2),
            FakeAdapter("second", "Second", 0.4),
            FakeAdapter("third", "Third", 0.8),
        ],
        max_workers=3,
    )
    computation = EnsemblePredictor(registry, make_config(), timeout_seconds=2).predict("text")

    assert math.isclose(computation.ensemble.risk_score, 0.56)
    assert computation.ensemble.classification_label == "Suspicious"
    assert math.isclose(
        sum(member.weighted_contribution for member in computation.members),
        computation.ensemble.risk_score,
    )


def test_failed_model_is_reported_and_remaining_weights_are_explicitly_normalised():
    registry = ModelRegistry(
        [
            FakeAdapter("first", "First", 0.2),
            FailingAdapter("second", "Second", 0.4),
            FakeAdapter("third", "Third", 0.8),
        ],
        max_workers=3,
    )
    computation = EnsemblePredictor(registry, make_config(), timeout_seconds=2).predict("text")

    assert computation.ensemble.status == "degraded"
    assert computation.ensemble.failed_model_count == 1
    assert math.isclose(
        sum(member.effective_weight for member in computation.members),
        1.0,
    )
    failed = next(member for member in computation.members if member.key == "second")
    assert failed.status == "error"
    assert failed.effective_weight == 0


def test_model_timeout_is_reported_without_waiting_for_the_slow_member():
    registry = ModelRegistry(
        [
            FakeAdapter("first", "First", 0.2),
            SlowAdapter("second", "Second", 0.4),
            FakeAdapter("third", "Third", 0.8),
        ],
        max_workers=3,
    )

    computation = EnsemblePredictor(
        registry,
        make_config(),
        timeout_seconds=0.01,
    ).predict("text")

    timed_out = next(member for member in computation.members if member.key == "second")
    assert computation.ensemble.status == "degraded"
    assert timed_out.status == "timeout"


def test_default_runtime_contract_registers_and_configures_all_eight_models():
    expected = {
        "logistic_regression",
        "svm",
        "xgboost",
        "dnn",
        "rnn",
        "bilstm",
        "bert",
        "roberta",
    }

    assert {spec.key for spec in DEFAULT_MODEL_SPECS} == expected
    config = EnsembleConfig.load(default_config_path())
    assert set(config.models) == expected
    assert math.isclose(sum(item.weight for item in config.models.values()), 1.0)


def test_transformer_lfs_pointer_is_reported_before_model_import(tmp_path: Path):
    artifact = tmp_path / "weight.safetensors"
    artifact.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\nsize 123456\n",
        encoding="utf-8",
    )
    adapter = ModelAdapter(
        ModelSpec(
            "transformer",
            "Transformer",
            "module.that.must.not.be.imported",
            artifact_relative_path="weight.safetensors",
        ),
        project_root=tmp_path,
    )

    try:
        adapter.predict_raw("text")
    except ModelArtifactUnavailableError as exc:
        assert "Git LFS pointer" in str(exc)
    else:  # pragma: no cover - makes the failure message clearer
        raise AssertionError("Expected a missing-artifact error")
