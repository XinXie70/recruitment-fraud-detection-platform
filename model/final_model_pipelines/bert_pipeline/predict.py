"""BERT inference wrapping sprint3 paper-aligned maxlen=512 checkpoint."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from final_model_pipelines.paths import BERT_CHECKPOINT, BERT_MAX_LENGTH
from final_model_pipelines.text_prep import prepare_text_from_input

MODEL_DISPLAY_NAME = "BERT"
EVAL_BATCH_SIZE = 8


def _resolve_device() -> torch.device:
    allow_cpu = os.getenv("ALLOW_CPU", "1").strip().lower() not in {"0", "false", "no"}
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu:
        raise RuntimeError("CUDA is unavailable and ALLOW_CPU is disabled")
    return torch.device("cpu")


def _softmax_fraud_proba(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)[:, 1]


@lru_cache(maxsize=1)
def _load_artifacts():
    if not BERT_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Sprint3 BERT checkpoint missing: {BERT_CHECKPOINT}")
    weights = BERT_CHECKPOINT / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Sprint3 BERT weights missing: {weights}")

    device = _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_CHECKPOINT)
    model.to(device)
    model.eval()

    threshold = 0.5
    thr_path = BERT_CHECKPOINT / "threshold.json"
    if thr_path.is_file():
        threshold = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    max_length = BERT_MAX_LENGTH
    meta_path = BERT_CHECKPOINT / "train_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        max_length = int(meta.get("config", {}).get("max_length", max_length))

    return model, tokenizer, device, threshold, max_length


@torch.no_grad()
def _predict_risk_score(texts: list[str]) -> np.ndarray:
    """Raw binary BERT output: fraudulent-job probability."""
    model, tokenizer, device, _threshold, max_length = _load_artifacts()
    cleaned = [prepare_text_from_input(text) for text in texts]
    scores: list[float] = []
    for start in range(0, len(cleaned), EVAL_BATCH_SIZE):
        batch = cleaned[start : start + EVAL_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        batch_scores = _softmax_fraud_proba(logits).detach().cpu().numpy()
        scores.extend(float(value) for value in batch_scores)
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
    _model, _tokenizer, _device, _threshold, _max_length = _load_artifacts()
    mapped = build_structured_output(
        MODEL_DISPLAY_NAME,
        score,
        low_threshold=0.0024,
        high_threshold=0.3,
    )
    return apply_validation_to_prediction(validation, mapped)


def artifact_path() -> Path:
    return BERT_CHECKPOINT / "model.safetensors"
