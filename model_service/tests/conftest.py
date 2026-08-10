"""Shared fixtures for model_service tests."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

MODEL_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVICE_ROOT))

# Stable defaults for fast, deterministic unit/integration runs.
os.environ.setdefault("MODEL_API_KEY", "")
os.environ.setdefault("RATE_LIMIT_PREDICT", "1000/minute")
os.environ.setdefault("RATE_LIMIT_PREDICT_BATCH", "1000/minute")
os.environ.setdefault("MAX_TEXT_CHARS", "50000")
os.environ.setdefault("MAX_BATCH_ITEMS", "100")
os.environ.setdefault("MAX_BATCH_TOTAL_CHARS", "500000")
os.environ.setdefault("ALLOW_CPU", "1")


@pytest.fixture
def sample_text() -> str:
    return (
        "Urgent work from home job. Send your bank details and passport "
        "copy to apply today. High salary guaranteed."
    )


@pytest.fixture
def sample_payload(sample_text: str) -> dict[str, str]:
    return {"text": sample_text, "record_id": "demo_001"}


@pytest.fixture
def mock_lr_result() -> dict[str, Any]:
    return {
        "model": "lr",
        "lr_score": 0.72,
        "threshold": 0.15,
        "predicted_label_id": 1,
        "predicted_label": "Fraudulent",
    }


@pytest.fixture
def mock_bert_result() -> dict[str, Any]:
    return {
        "model": "bert",
        "bert_score": 0.88,
        "threshold": 0.32,
        "max_length": 512,
        "predicted_label_id": 1,
        "predicted_label": "Fraudulent",
        "device": "cpu",
    }


@pytest.fixture
def mock_ensemble_result() -> dict[str, Any]:
    return {
        "model": "ensemble_fp_gate",
        "bert_threshold": 0.3,
        "lr_gate": 0.06,
        "bert_score": 0.88,
        "lr_score": 0.72,
        "gate_triggered": False,
        "bert_raw_prediction_id": 1,
        "predicted_label_id": 1,
        "predicted_label": "Fraudulent",
        "ranking_score": 0.88,
    }


@pytest.fixture
def mock_risk_result() -> dict[str, Any]:
    return {
        "model": "risk",
        "risk_score": 0.88,
        "risk_score_100": 88.0,
        "risk_level": "High",
        "bert_evidence_score": 0.88,
        "lr_score": 0.72,
        "risk_score_source": "bert",
        "gate_triggered": False,
        "high_rule_met": 1,
        "low_rule_met": 0,
        "decision_reason": "BERT High rule passed the LR gate",
        "thresholds": {
            "bert_high_threshold": 0.3,
            "lr_gate": 0.06,
            "bert_low_threshold": 0.0024,
        },
    }


@pytest.fixture
def mock_model_services(
    monkeypatch: pytest.MonkeyPatch,
    mock_lr_result: dict[str, Any],
    mock_bert_result: dict[str, Any],
    mock_ensemble_result: dict[str, Any],
    mock_risk_result: dict[str, Any],
) -> dict[str, MagicMock]:
    lr = MagicMock()
    lr.predict.return_value = mock_lr_result
    lr.load.return_value = None

    bert = MagicMock()
    bert.predict.return_value = mock_bert_result
    bert.load.return_value = None

    ensemble = MagicMock()
    ensemble.predict.return_value = mock_ensemble_result

    risk = MagicMock()
    risk.predict.return_value = mock_risk_result

    import api_http
    import prediction_routes

    monkeypatch.setattr(prediction_routes, "lr_service", lr)
    monkeypatch.setattr(prediction_routes, "bert_service", bert)
    monkeypatch.setattr(prediction_routes, "ensemble_service", ensemble)
    monkeypatch.setattr(prediction_routes, "risk_service", risk)
    monkeypatch.setattr(api_http, "MODEL_API_KEY", "")

    return {
        "lr": lr,
        "bert": bert,
        "ensemble": ensemble,
        "risk": risk,
    }


@pytest.fixture
def client(mock_model_services: dict[str, MagicMock]):
    from app import app

    return app.test_client()


@pytest.fixture
def authed_client(
    monkeypatch: pytest.MonkeyPatch,
    mock_model_services: dict[str, MagicMock],
):
    import api_http
    import app as app_module

    monkeypatch.setattr(api_http, "MODEL_API_KEY", "test-secret-key")
    return app_module.app.test_client()


@pytest.fixture
def reload_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reload settings after temporarily overriding environment variables."""
    import settings

    importlib.reload(settings)
    yield
    importlib.reload(settings)
