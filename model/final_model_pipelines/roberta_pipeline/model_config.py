"""RoBERTa final pipeline config."""

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

MODEL_DISPLAY_NAME = "RoBERTa"
FEATURE_METHOD = "RoBERTa BPE Embeddings"
IMBALANCE_METHOD = "Class Weighting"
CLASS_WEIGHT = {0: 1.0, 1: 6.0}

PRETRAINED_MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2
WARMUP_RATIO = 0.1

SAMPLE_TEXTS = [
    "Software Engineer at Google. Bachelor degree required. Competitive salary and benefits.",
    "URGENT HIRING! Work from home, earn $5000/week. No experience needed. Wire transfer required.",
    "Marketing manager needed. 3 years experience. Office in New York. Standard interview process.",
]
