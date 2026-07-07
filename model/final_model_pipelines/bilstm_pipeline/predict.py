"""Bi-LSTM inference: binary probability + risk mapping post-processing."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.bilstm_pipeline.data_preprocessing import (  # noqa: E402
    load_sequence_config,
    prepare_text_from_input,
)
from final_model_pipelines.bilstm_pipeline.model_config import (  # noqa: E402
    MODEL_DISPLAY_NAME,
    SAVED_MODEL_DIR,
)
from final_model_pipelines.evaluation_utils import load_thresholds  # noqa: E402
from final_model_pipelines.risk_mapping import build_structured_output  # noqa: E402


@lru_cache(maxsize=1)
def _load_artifacts():
    model = tf.keras.models.load_model(SAVED_MODEL_DIR / "model.keras")
    tokenizer = joblib.load(SAVED_MODEL_DIR / "tokenizer.joblib")
    seq_cfg = load_sequence_config(SAVED_MODEL_DIR)
    max_len = int(seq_cfg["max_len"])
    low_t, high_t = load_thresholds(SAVED_MODEL_DIR)
    return model, tokenizer, max_len, low_t, high_t


def _predict_risk_score(texts: list[str]) -> np.ndarray:
    """Raw binary Bi-LSTM output: fake job probability (Sigmoid) as risk_score."""
    model, tokenizer, max_len, _, _ = _load_artifacts()
    cleaned = [prepare_text_from_input(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    x = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return model.predict(x, verbose=0).ravel()


def _apply_risk_mapping_layer(risk_score: float) -> dict:
    _, _, _, low_t, high_t = _load_artifacts()
    return build_structured_output(MODEL_DISPLAY_NAME, risk_score, low_t, high_t)


def predict_job_posting(input_text: str) -> dict:
    """Predict a single job posting text."""
    score = float(_predict_risk_score([input_text])[0])
    return _apply_risk_mapping_layer(score)


def predict_batch_job_postings(input_texts: list[str]) -> list[dict]:
    """Batch prediction."""
    scores = _predict_risk_score(input_texts)
    return [_apply_risk_mapping_layer(float(s)) for s in scores]
