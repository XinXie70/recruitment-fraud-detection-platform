#Logistic Regression inference service for EMSCAD data
from __future__ import annotations
import threading
from typing import Any
import joblib
from services.coalesce import InferenceCoalescer
from settings import LR_ARTIFACT, load_runtime_config

class LRService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._threshold: float | None = None
        self._coalescer = InferenceCoalescer()
    def _ensure_artifact(self) -> None:
        if LR_ARTIFACT.exists():
            return
        raise FileNotFoundError(f"Required LR artifact is missing: {LR_ARTIFACT}")
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
        key = f"lr:{self._threshold}:{combined_text}"
        def _infer() -> dict[str, Any]:
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
        return self._coalescer.run(key, _infer)
lr_service = LRService()
