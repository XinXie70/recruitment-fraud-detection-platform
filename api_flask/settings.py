"""Paths and frozen thresholds for the fraud detection API."""

from __future__ import annotations

import json
import os
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent
CAPSTONE_ROOT = Path(
    os.environ.get("CAPSTONE_ROOT", str(API_ROOT.parent))
).resolve()

LR_ROOT = CAPSTONE_ROOT / "lr_none_bigram_no_cv_paper_aligned_seed42"
BERT_ROOT = CAPSTONE_ROOT / "retrain_paper_aligned_seed42_maxlen512"
ENSEMBLE_ROOT = CAPSTONE_ROOT / "ensemble_bert_fp_gate_lr_none_bigram_maxlen512"

LR_ARTIFACT = LR_ROOT / "artifacts" / "lr_none_bigram_no_cv_paper_aligned_seed42.joblib"
LR_CONFIG = LR_ROOT / "results" / "config.json"
LR_TRAIN_SCRIPT = LR_ROOT / "code" / "train_lr_none_bigram_no_cv.py"

BERT_CHECKPOINT = (
    BERT_ROOT / "weights" / "bert_paper_protocol_maxlen512" / "best"
)
BERT_CODE_DIR = BERT_ROOT / "code"
BERT_MAX_LENGTH = 512

ENSEMBLE_CONFIG = ENSEMBLE_ROOT / "results" / "config.json"
RISK_BOUNDARY_CONFIG = ENSEMBLE_ROOT / "risk_level" / "risk_boundary_config.json"

TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


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
