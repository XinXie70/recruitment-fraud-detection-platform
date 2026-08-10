"""Logistic Regression inference wrapping sprint3 LR Pipeline artifact."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from final_model_pipelines.paths import LR_ARTIFACT, LR_RESULTS_CONFIG, LR_THRESHOLD
from final_model_pipelines.text_prep import prepare_text_from_input

MODEL_DISPLAY_NAME = "Logistic Regression"
POSITIVE_LABEL = 1


@lru_cache(maxsize=1)
def _load_artifacts():
    if not LR_ARTIFACT.is_file():
        raise FileNotFoundError(
            f"Sprint3 LR artifact missing: {LR_ARTIFACT}. "
            "Train it with model_algorithm/sprint3/LR/code/train_lr_none_bigram_no_cv.py"
        )
    model = joblib.load(LR_ARTIFACT)
    threshold = 0.5
    if LR_THRESHOLD.is_file():
        threshold = float(json.loads(LR_THRESHOLD.read_text(encoding="utf-8"))["threshold"])
    elif LR_RESULTS_CONFIG.is_file():
        threshold = float(
            json.loads(LR_RESULTS_CONFIG.read_text(encoding="utf-8"))["selected_threshold"]
        )
    return model, threshold


def _predict_risk_score(texts: list[str]) -> np.ndarray:
    """Raw binary LR output: fraudulent-job probability."""
    model, _threshold = _load_artifacts()
    cleaned = [prepare_text_from_input(text) for text in texts]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(cleaned)[:, POSITIVE_LABEL]
    else:
        raise TypeError("Loaded LR artifact does not expose predict_proba")
    return np.asarray(scores, dtype=float)


def predict_job_posting(input_text: str) -> dict:
    """Single-text prediction used by ad-hoc scripts."""
    from final_model_pipelines.risk_mapping import build_structured_output
    from final_model_pipelines.validation_pipeline import (
        apply_validation_to_prediction,
        build_rejection_response,
        validate_job_input,
    )

    validation = validate_job_input(input_text)
    if not validation["is_valid"]:
        return build_rejection_response(MODEL_DISPLAY_NAME, validation)

    score = float(_predict_risk_score([input_text])[0])
    _, decision_threshold = _load_artifacts()
    # Display mapping uses ensemble boundaries; local single-model uses decision threshold.
    mapped = build_structured_output(
        MODEL_DISPLAY_NAME,
        score,
        low_threshold=min(0.15, decision_threshold),
        high_threshold=max(0.3, decision_threshold),
    )
    return apply_validation_to_prediction(validation, mapped)


def artifact_path() -> Path:
    return LR_ARTIFACT
