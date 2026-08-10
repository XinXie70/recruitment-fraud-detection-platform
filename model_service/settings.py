"""Paths and frozen thresholds for the fraud detection API."""

from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_SERVICE_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = Path(
    os.environ.get("MODEL_FILES_ROOT", str(MODEL_SERVICE_ROOT / "models"))
).resolve()

LR_ROOT = MODELS_ROOT / "lr"
BERT_ROOT = MODELS_ROOT / "bert"
ENSEMBLE_ROOT = MODELS_ROOT / "ensemble"
RISK_ROOT = MODELS_ROOT / "risk"

LR_ARTIFACT = LR_ROOT / "lr_none_bigram_no_cv_paper_aligned_seed42.joblib"
LR_CONFIG = LR_ROOT / "config.json"

BERT_CHECKPOINT = BERT_ROOT / "best"
BERT_MAX_LENGTH = 512

ENSEMBLE_CONFIG = ENSEMBLE_ROOT / "config.json"
RISK_BOUNDARY_CONFIG = RISK_ROOT / "risk_boundary_config.json"

TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# Security / capacity limits (overridable via environment).
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(1 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "50000"))
MAX_BATCH_ITEMS = int(os.environ.get("MAX_BATCH_ITEMS", "100"))
MAX_BATCH_TOTAL_CHARS = int(os.environ.get("MAX_BATCH_TOTAL_CHARS", "500000"))
RATE_LIMIT_PREDICT = os.environ.get("RATE_LIMIT_PREDICT", "30/minute")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "").strip()
# When MODEL_API_KEY is empty, auth is skipped (local/dev). Set a key in deploy.


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_config() -> dict:
    lr_cfg = _load_json(LR_CONFIG)
    ens_cfg = _load_json(ENSEMBLE_CONFIG)
    risk_cfg = _load_json(RISK_BOUNDARY_CONFIG)

    bert_threshold_path = BERT_CHECKPOINT / "threshold.json"
    bert_model_threshold = None
    if bert_threshold_path.exists():
        bert_model_threshold = float(
            json.loads(bert_threshold_path.read_text(encoding="utf-8"))["threshold"]
        )

    return {
        "lr": {
            "artifact": str(LR_ARTIFACT),
            "decision_threshold": float(lr_cfg["selected_threshold"]),
            "experiment": lr_cfg.get("experiment"),
        },
        "bert": {
            "checkpoint": str(BERT_CHECKPOINT),
            "max_length": BERT_MAX_LENGTH,
            "model_threshold": bert_model_threshold,
            "experiment": "retrain_paper_aligned_seed42_maxlen512",
        },
        "ensemble": {
            "bert_threshold": float(ens_cfg["bert_threshold"]),
            "lr_gate": float(ens_cfg["lr_gate"]),
            "experiment": ens_cfg.get("experiment"),
        },
        "risk": {
            "bert_high_threshold": float(risk_cfg["high_rule"]["bert_threshold"]),
            "lr_gate": float(risk_cfg["high_rule"]["lr_gate"]),
            "bert_low_threshold": float(risk_cfg["low_rule"]["bert_threshold"]),
            "config_path": str(RISK_BOUNDARY_CONFIG),
        },
    }
