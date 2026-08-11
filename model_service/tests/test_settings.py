from __future__ import annotations
import importlib
import json
from pathlib import Path
import pytest
import settings
from settings import load_runtime_config
def test_batch_rate_limit_is_configured_independently(
    monkeypatch: pytest.MonkeyPatch,
    reload_settings,
) -> None:
    monkeypatch.setenv("RATE_LIMIT_PREDICT", "11/minute")
    monkeypatch.setenv("RATE_LIMIT_PREDICT_BATCH", "222/minute")
    reloaded = importlib.reload(settings)
    assert reloaded.RATE_LIMIT_PREDICT == "11/minute"
    assert reloaded.RATE_LIMIT_PREDICT_BATCH == "222/minute"

def test_production_requires_model_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        importlib.reload(settings)
    monkeypatch.setenv("APP_ENV", "test")
    importlib.reload(settings)
def test_load_runtime_config_reads_frozen_artifacts() -> None:
    cfg = load_runtime_config()

    assert Path(cfg["lr"]["artifact"]).name.endswith(".joblib")
    assert Path(cfg["bert"]["checkpoint"]).name == "best"
    assert cfg["ensemble"]["bert_threshold"] == pytest.approx(0.3)
    assert cfg["risk"]["bert_low_threshold"] == pytest.approx(0.0024)
def test_load_runtime_config_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    monkeypatch.setattr(settings, "LR_CONFIG", missing / "lr.json")
    monkeypatch.setattr(settings, "ENSEMBLE_CONFIG", missing / "ensemble.json")
    monkeypatch.setattr(settings, "RISK_BOUNDARY_CONFIG", missing / "risk.json")
    with pytest.raises(FileNotFoundError):
        load_runtime_config()

def test_load_runtime_config_from_custom_models_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lr_dir = tmp_path / "lr"
    ens_dir = tmp_path / "ensemble"
    risk_dir = tmp_path / "risk"
    bert_dir = tmp_path / "bert" / "best"
    for directory in (lr_dir, ens_dir, risk_dir, bert_dir):
        directory.mkdir(parents=True)
    (lr_dir / "config.json").write_text(
        json.dumps({"selected_threshold": 0.2, "experiment": "demo-lr"}),
        encoding="utf-8",
    )
    (ens_dir / "config.json").write_text(
        json.dumps({"bert_threshold": 0.31, "lr_gate": 0.05, "experiment": "demo-ens"}),
        encoding="utf-8",
    )
    (risk_dir / "risk_boundary_config.json").write_text(
        json.dumps(
            {
                "high_rule": {"bert_threshold": 0.31, "lr_gate": 0.05},
                "low_rule": {"bert_threshold": 0.01},
            }
        ),
        encoding="utf-8",
    )
    (bert_dir / "threshold.json").write_text(
        json.dumps({"threshold": 0.33}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MODELS_ROOT", tmp_path)
    monkeypatch.setattr(settings, "LR_ROOT", lr_dir)
    monkeypatch.setattr(settings, "LR_CONFIG", lr_dir / "config.json")
    monkeypatch.setattr(settings, "LR_ARTIFACT", lr_dir / "model.joblib")
    monkeypatch.setattr(settings, "BERT_ROOT", tmp_path / "bert")
    monkeypatch.setattr(settings, "BERT_CHECKPOINT", bert_dir)
    monkeypatch.setattr(settings, "ENSEMBLE_ROOT", ens_dir)
    monkeypatch.setattr(settings, "ENSEMBLE_CONFIG", ens_dir / "config.json")
    monkeypatch.setattr(settings, "RISK_ROOT", risk_dir)
    monkeypatch.setattr(settings, "RISK_BOUNDARY_CONFIG", risk_dir / "risk_boundary_config.json")
    cfg = load_runtime_config()
    assert cfg["lr"]["decision_threshold"] == pytest.approx(0.2)
    assert cfg["ensemble"]["lr_gate"] == pytest.approx(0.05)
    assert cfg["bert"]["model_threshold"] == pytest.approx(0.33)
