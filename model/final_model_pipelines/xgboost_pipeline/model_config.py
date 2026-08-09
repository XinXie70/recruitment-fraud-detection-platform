"""XGBoost final pipeline config."""

from pathlib import Path

from final_model_pipelines.shared_config import (
    CLEANED_DATA_PATH as CLEANED_DATA_PATH,
    DATA_DIR as DATA_DIR,
    LABEL_COL as LABEL_COL,
    POSITIVE_LABEL as POSITIVE_LABEL,
    RANDOM_STATE as RANDOM_STATE,
    SPLIT_DIR as SPLIT_DIR,
    TEST_SIZE as TEST_SIZE,
    TEXT_COL as TEXT_COL,
    TEXT_FIELDS as TEXT_FIELDS,
    VAL_SIZE as VAL_SIZE,
    resolve_raw_data_path as resolve_raw_data_path,
)

PIPELINE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = PIPELINE_DIR / "saved_model"
OUTPUT_DIR = PIPELINE_DIR / "outputs"

MODEL_DISPLAY_NAME = "XGBoost"
FEATURE_METHOD = "TF-IDF"
IMBALANCE_METHOD = "scale_pos_weight"
CLASS_WEIGHT = {0: 1, 1: 6}
SCALE_POS_WEIGHT = 6

MAX_FEATURES = 2000
NGRAM_RANGE = (1, 2)
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

SAMPLE_TEXTS = [
    "Software Engineer at Google. Bachelor degree required. Competitive salary and benefits.",
    "URGENT HIRING! Work from home, earn $5000/week. No experience needed. Wire transfer required.",
    "Marketing manager needed. 3 years experience. Office in New York. Standard interview process.",
]
