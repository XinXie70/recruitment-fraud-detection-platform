"""DNN evaluation: dual-threshold tuning, three-tier risk stats, visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from scipy.sparse import issparse

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.dnn_pipeline.data_preprocessing import load_or_create_splits, transform_text  # noqa: E402
from final_model_pipelines.dnn_pipeline.model_config import (  # noqa: E402
    LABEL_COL,
    MODEL_DISPLAY_NAME,
    OUTPUT_DIR,
    SAMPLE_TEXTS,
    SAVED_MODEL_DIR,
    TEXT_COL,
)
from final_model_pipelines.dnn_pipeline.predict import predict_job_posting  # noqa: E402
from final_model_pipelines.evaluation_utils import run_full_evaluation  # noqa: E402


def _to_dense(x):
    if issparse(x):
        return x.toarray().astype("float32")
    return x


def evaluate() -> dict:
    model_path = SAVED_MODEL_DIR / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError("DNN model not found. Run train_model.py first.")

    model = tf.keras.models.load_model(model_path)
    vectorizer = joblib.load(SAVED_MODEL_DIR / "vectorizer.joblib")
    scaler = joblib.load(SAVED_MODEL_DIR / "scaler.joblib")
    _, val_df, test_df = load_or_create_splits()

    val_x = scaler.transform(_to_dense(transform_text(vectorizer, val_df[TEXT_COL])))
    test_x = scaler.transform(_to_dense(transform_text(vectorizer, test_df[TEXT_COL])))
    val_prob = model.predict(val_x, verbose=0).ravel()
    test_prob = model.predict(test_x, verbose=0).ravel()

    summary = run_full_evaluation(
        model_display_name=MODEL_DISPLAY_NAME,
        y_val=val_df[LABEL_COL].values,
        prob_val=val_prob,
        y_test=test_df[LABEL_COL].values,
        prob_test=test_prob,
        sample_texts=SAMPLE_TEXTS,
        predict_fn=predict_job_posting,
        output_dir=OUTPUT_DIR,
        saved_model_dir=SAVED_MODEL_DIR,
    )

    print("\n=== DNN Binary Metrics ===")
    print(pd.DataFrame(summary["binary_metrics"]).to_string(index=False))
    print("\n=== Thresholds ===")
    print(json.dumps(summary["risk_mapping_thresholds"], indent=2))
    print("\n=== Tier Statistics (Test) ===")
    test_tiers = next(t for t in summary["tier_statistics_supplementary"] if t["split"] == "Test")
    print(json.dumps(test_tiers, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
