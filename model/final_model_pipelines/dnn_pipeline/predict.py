"""DNN inference: binary probability + risk mapping post-processing."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from scipy.sparse import issparse

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.dnn_pipeline.data_preprocessing import prepare_text_from_input  # noqa: E402
from final_model_pipelines.dnn_pipeline.model_config import (  # noqa: E402
    MODEL_DISPLAY_NAME,
    SAVED_MODEL_DIR,
)
from final_model_pipelines.evaluation_utils import load_thresholds  # noqa: E402
from final_model_pipelines.risk_mapping import build_structured_output  # noqa: E402
from final_model_pipelines.validation_pipeline import (  # noqa: E402
    apply_validation_to_prediction,
    build_rejection_response,
    validate_job_input,
)


def _to_dense(x) -> np.ndarray:
    if issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


@lru_cache(maxsize=1)
def _load_artifacts():
    model = tf.keras.models.load_model(SAVED_MODEL_DIR / "model.keras")
    vectorizer = joblib.load(SAVED_MODEL_DIR / "vectorizer.joblib")
    scaler = joblib.load(SAVED_MODEL_DIR / "scaler.joblib")
    low_t, high_t = load_thresholds(SAVED_MODEL_DIR)
    return model, vectorizer, scaler, low_t, high_t


def _predict_risk_score(texts: list[str]) -> np.ndarray:
    """Raw binary DNN output: fake job probability (Sigmoid) as risk_score."""
    model, vectorizer, scaler, _, _ = _load_artifacts()
    cleaned = [prepare_text_from_input(t) for t in texts]
    x = scaler.transform(_to_dense(vectorizer.transform(cleaned)))
    return model.predict(x, verbose=0).ravel()


def _apply_risk_mapping_layer(risk_score: float) -> dict:
    _, _, _, low_t, high_t = _load_artifacts()
    return build_structured_output(MODEL_DISPLAY_NAME, risk_score, low_t, high_t)


def predict_job_posting(input_text: str) -> dict:
    """Predict a single job posting text."""
    validation = validate_job_input(input_text)
    if not validation["is_valid"]:
        return build_rejection_response(MODEL_DISPLAY_NAME, validation)

    score = float(_predict_risk_score([input_text])[0])
    return apply_validation_to_prediction(validation, _apply_risk_mapping_layer(score))


def predict_batch_job_postings(input_texts: list[str]) -> list[dict]:
    """Batch prediction."""
    results: list[dict | None] = [None] * len(input_texts)
    valid_indices: list[int] = []
    valid_texts: list[str] = []
    validations: list[dict] = []

    for idx, text in enumerate(input_texts):
        validation = validate_job_input(text)
        if not validation["is_valid"]:
            results[idx] = build_rejection_response(MODEL_DISPLAY_NAME, validation)
            continue
        valid_indices.append(idx)
        valid_texts.append(text)
        validations.append(validation)

    if valid_texts:
        scores = _predict_risk_score(valid_texts)
        for i, score in enumerate(scores):
            results[valid_indices[i]] = apply_validation_to_prediction(
                validations[i],
                _apply_risk_mapping_layer(float(score)),
            )

    return results  # type: ignore[return-value]
