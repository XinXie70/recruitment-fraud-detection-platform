"""Logistic Regression inference service."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

import joblib

from settings import LR_ARTIFACT, LR_TRAIN_SCRIPT, load_runtime_config


class LRService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._threshold: float | None = None

    def _ensure_artifact(self) -> None:
        if LR_ARTIFACT.exists():
            return
        if not LR_TRAIN_SCRIPT.exists():
            raise FileNotFoundError(
                f"LR artifact missing ({LR_ARTIFACT}) and train script not found"
            )
        result = subprocess.run(
            [sys.executable, str(LR_TRAIN_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not LR_ARTIFACT.exists():
            raise RuntimeError(
                "Failed to train LR artifact for API use.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._ensure_artifact()
            self._model = joblib.load(LR_ARTIFACT)
            self._threshold = float(
                load_runtime_config()["lr"]["decision_threshold"]
            )

    @property
    def threshold(self) -> float:
        self.load()
        assert self._threshold is not None
        return self._threshold

    def predict(self, combined_text: str) -> dict[str, Any]:
        self.load()
        assert self._model is not None and self._threshold is not None
        score = float(self._model.predict_proba([combined_text])[0, 1])
        pred = int(score >= self._threshold)
        return {
            "model": "lr",
            "lr_score": score,
            "threshold": self._threshold,
            "predicted_label_id": pred,
            "predicted_label": "Fraudulent" if pred == 1 else "Legitimate",
        }


lr_service = LRService()
