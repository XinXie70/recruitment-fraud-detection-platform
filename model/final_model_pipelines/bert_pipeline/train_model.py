"""BERT training (bert-base-uncased fine-tune + class weighting)."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.bert_pipeline.data_preprocessing import load_or_create_splits  # noqa: E402
from final_model_pipelines.bert_pipeline.model_config import (  # noqa: E402
    BATCH_SIZE,
    CLASS_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    FEATURE_METHOD,
    IMBALANCE_METHOD,
    LABEL_COL,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_LEN,
    PRETRAINED_MODEL_NAME,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    TEXT_COL,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from final_model_pipelines.transformer_common import train_transformer_classifier  # noqa: E402


def train() -> Path:
    train_df, val_df, _ = load_or_create_splits()
    return train_transformer_classifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        train_texts=train_df[TEXT_COL],
        train_labels=train_df[LABEL_COL].values,
        val_texts=val_df[TEXT_COL],
        val_labels=val_df[LABEL_COL].values,
        saved_model_dir=SAVED_MODEL_DIR,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weight=CLASS_WEIGHT,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        seed=RANDOM_STATE,
        warmup_ratio=WARMUP_RATIO,
        meta_extra={
            "feature_method": FEATURE_METHOD,
            "imbalance_method": IMBALANCE_METHOD,
            "text_col": TEXT_COL,
            "label_col": LABEL_COL,
            "model_display_name": "BERT",
        },
    )


if __name__ == "__main__":
    train()
