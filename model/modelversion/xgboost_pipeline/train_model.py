"""XGBoost training (TF-IDF + scale_pos_weight)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from xgboost import XGBClassifier

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.xgboost_pipeline.data_preprocessing import (  # noqa: E402
    fit_tfidf_vectorizer,
    load_or_create_splits,
    transform_text,
)
from final_model_pipelines.xgboost_pipeline.model_config import (  # noqa: E402
    FEATURE_METHOD,
    IMBALANCE_METHOD,
    LABEL_COL,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    SCALE_POS_WEIGHT,
    TEXT_COL,
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


def train() -> Path:
    train_df, _, _ = load_or_create_splits()

    vectorizer = fit_tfidf_vectorizer(train_df[TEXT_COL])
    x_train = transform_text(vectorizer, train_df[TEXT_COL])
    y_train = train_df[LABEL_COL].values

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cuda",
    )
    try:
        model.fit(x_train, y_train)
        print("XGBoost training device: cuda")
    except Exception as exc:  # noqa: BLE001 - fall back if GPU build/runtime unavailable
        print(f"XGBoost CUDA unavailable ({exc}); falling back to CPU.")
        model.set_params(device="cpu")
        model.fit(x_train, y_train)
        print("XGBoost training device: cpu")

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")

    meta = {
        "feature_method": FEATURE_METHOD,
        "imbalance_method": IMBALANCE_METHOD,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    (SAVED_MODEL_DIR / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved: {SAVED_MODEL_DIR}")
    return SAVED_MODEL_DIR


if __name__ == "__main__":
    train()
