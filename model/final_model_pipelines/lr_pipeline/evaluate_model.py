"""Logistic Regression evaluation: dual-threshold tuning, three-tier risk stats, visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.evaluation_utils import run_full_evaluation  # noqa: E402
from final_model_pipelines.lr_pipeline.data_preprocessing import load_or_create_splits, transform_text  # noqa: E402
from final_model_pipelines.lr_pipeline.model_config import (  # noqa: E402
    LABEL_COL,
    MODEL_DISPLAY_NAME,
    OUTPUT_DIR,
    POSITIVE_LABEL,
    SAMPLE_TEXTS,
    SAVED_MODEL_DIR,
    TEXT_COL,
)
from final_model_pipelines.lr_pipeline.predict import predict_job_posting  # noqa: E402


def evaluate() -> dict:
    model_path = SAVED_MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("LR model not found. Run train_model.py first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(SAVED_MODEL_DIR / "vectorizer.joblib")
    _, val_df, test_df = load_or_create_splits()

    val_prob = model.predict_proba(transform_text(vectorizer, val_df[TEXT_COL]))[:, POSITIVE_LABEL]
    test_prob = model.predict_proba(transform_text(vectorizer, test_df[TEXT_COL]))[:, POSITIVE_LABEL]

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

    print("\n=== LR Binary Metrics ===")
    print(pd.DataFrame(summary["binary_metrics"]).to_string(index=False))
    print("\n=== Thresholds ===")
    print(json.dumps(summary["risk_mapping_thresholds"], indent=2))
    print("\n=== Tier Statistics (Test) ===")
    test_tiers = next(t for t in summary["tier_statistics_supplementary"] if t["split"] == "Test")
    print(json.dumps(test_tiers, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
