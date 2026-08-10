"""Resolve sprint3 artifact locations from the repository root."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = MODEL_ROOT.parent

SPRINT3_ROOT = PROJECT_ROOT / "model_algorithm" / "sprint3"
LR_WEIGHT_DIR = SPRINT3_ROOT / "LR" / "weight"
LR_ARTIFACT = LR_WEIGHT_DIR / "lr_none_bigram_no_cv_paper_aligned_seed42.joblib"
LR_THRESHOLD = LR_WEIGHT_DIR / "threshold.json"
LR_RESULTS_CONFIG = SPRINT3_ROOT / "LR" / "results" / "config.json"

BERT_CHECKPOINT = SPRINT3_ROOT / "BERT" / "weight" / "best"
BERT_MAX_LENGTH = 512

ENSEMBLE_CONFIG = (
    PACKAGE_ROOT / "ensemble_pipeline" / "saved_model" / "ensemble_config.json"
)
ENSEMBLE_SOURCE_CONFIG = SPRINT3_ROOT / "ensemble_BERT_FP" / "results" / "config.json"
RISK_BOUNDARY_CONFIG = SPRINT3_ROOT / "risk_level" / "risk_boundary_config.json"
