"""DNN final pipeline config (best setup from dnn_comparison experiments)."""

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
)

PIPELINE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = PIPELINE_DIR / "saved_model"
OUTPUT_DIR = PIPELINE_DIR / "outputs"

MODEL_DISPLAY_NAME = "DNN"
FEATURE_METHOD = "BoW"
IMBALANCE_METHOD = "Class Weighting"
CLASS_WEIGHT = {0: 1, 1: 6}

MAX_FEATURES = 2000
NGRAM_RANGE = (1, 2)
HIDDEN_DIMS = (512, 256)
DROPOUT = 0.3
BATCH_SIZE = 128
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

SAMPLE_TEXTS = [
    "Software Engineer at Google. Bachelor degree required. Competitive salary and benefits.",
    "URGENT HIRING! Work from home, earn $5000/week. No experience needed. Wire transfer required.",
    "Marketing manager needed. 3 years experience. Office in New York. Standard interview process.",
]
