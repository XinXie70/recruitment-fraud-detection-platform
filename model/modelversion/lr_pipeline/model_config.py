"""Logistic Regression final pipeline config (best setup from lr_comparison experiments)."""

from pathlib import Path

from final_model_pipelines.shared_config import (
    CLEANED_DATA_PATH,
    DATA_DIR,
    LABEL_COL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    SPLIT_DIR,
    TEST_SIZE,
    TEXT_COL,
    TEXT_FIELDS,
    VAL_SIZE,
    resolve_raw_data_path,
)

PIPELINE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = PIPELINE_DIR / "saved_model"
OUTPUT_DIR = PIPELINE_DIR / "outputs"

# Selected from experiments: TF-IDF + Class Weighting (Test F1=0.791, Recall=0.786, Precision=0.795)
MODEL_DISPLAY_NAME = "Logistic Regression"
FEATURE_METHOD = "TF-IDF"
IMBALANCE_METHOD = "Class Weighting"
CLASS_WEIGHT = {0: 1, 1: 6}

MAX_FEATURES = 2000
NGRAM_RANGE = (1, 2)
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"

SAMPLE_TEXTS = [
    "Software Engineer at Google. Bachelor degree required. Competitive salary and benefits.",
    "URGENT HIRING! Work from home, earn $5000/week. No experience needed. Wire transfer required.",
    "Marketing manager needed. 3 years experience. Office in New York. Standard interview process.",
]

