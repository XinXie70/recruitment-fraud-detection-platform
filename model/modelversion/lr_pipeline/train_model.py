"""Logistic Regression training (TF-IDF + Class Weighting)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.lr_pipeline.data_preprocessing import (  # noqa: E402
    fit_tfidf_vectorizer,
    load_or_create_splits,
    transform_text,
)
from final_model_pipelines.lr_pipeline.model_config import (  # noqa: E402
    CLASS_WEIGHT,
    FEATURE_METHOD,
    IMBALANCE_METHOD,
    LABEL_COL,
    LR_MAX_ITER,
    LR_SOLVER,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    TEXT_COL,
)


def train() -> Path:
    train_df, _, _ = load_or_create_splits()

    vectorizer = fit_tfidf_vectorizer(train_df[TEXT_COL])
    x_train = transform_text(vectorizer, train_df[TEXT_COL])
    y_train = train_df[LABEL_COL].values

    model = LogisticRegression(
        class_weight=CLASS_WEIGHT,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
        solver=LR_SOLVER,
    )
    model.fit(x_train, y_train)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")

    meta = {
        "feature_method": FEATURE_METHOD,
        "imbalance_method": IMBALANCE_METHOD,
        "class_weight": CLASS_WEIGHT,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    (SAVED_MODEL_DIR / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved: {SAVED_MODEL_DIR}")
    return SAVED_MODEL_DIR


if __name__ == "__main__":
    train()
