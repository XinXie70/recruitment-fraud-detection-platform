from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from services.lr_service import LRService


def test_lr_service_predict_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.28, 0.72]])

    monkeypatch.setattr("services.lr_service.joblib.load", lambda _path: mock_model)
    monkeypatch.setattr(
        "services.lr_service.load_runtime_config",
        lambda: {"lr": {"decision_threshold": 0.15}},
    )
    monkeypatch.setattr("services.lr_service.LR_ARTIFACT", MagicMock(exists=lambda: True))

    service = LRService()
    result = service.predict("sample job text")

    assert result["lr_score"] == pytest.approx(0.72)
    assert result["predicted_label"] == "Fraudulent"
    mock_model.predict_proba.assert_called_once()


def test_lr_service_missing_artifact_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = MagicMock()
    missing.exists.return_value = False
    monkeypatch.setattr("services.lr_service.LR_ARTIFACT", missing)

    service = LRService()
    with pytest.raises(FileNotFoundError, match="Required LR artifact"):
        service.load()


def test_lr_service_returns_consistent_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])

    monkeypatch.setattr("services.lr_service.joblib.load", lambda _path: mock_model)
    monkeypatch.setattr(
        "services.lr_service.load_runtime_config",
        lambda: {"lr": {"decision_threshold": 0.5}},
    )
    monkeypatch.setattr("services.lr_service.LR_ARTIFACT", MagicMock(exists=lambda: True))

    service = LRService()
    first = service.predict("same text")
    second = service.predict("same text")

    assert first["lr_score"] == second["lr_score"] == pytest.approx(0.6)
    assert first["predicted_label"] == "Fraudulent"
