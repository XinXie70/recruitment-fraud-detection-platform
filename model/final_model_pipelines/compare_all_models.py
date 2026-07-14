"""
Aggregate evaluation results from all model pipelines for side-by-side comparison.

Reads each pipeline's outputs/evaluation_results.csv (Test split) and writes a combined report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PIPELINES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINES_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PIPELINES = [
    ("Logistic Regression", PIPELINES_ROOT / "lr_pipeline"),
    ("SVM", PIPELINES_ROOT / "svm_pipeline"),
    ("XGBoost", PIPELINES_ROOT / "xgboost_pipeline"),
    ("DNN", PIPELINES_ROOT / "dnn_pipeline"),
    ("RNN", PIPELINES_ROOT / "rnn_pipeline"),
    ("Bi-LSTM", PIPELINES_ROOT / "bilstm_pipeline"),
    ("BERT", PIPELINES_ROOT / "bert_pipeline"),
    ("RoBERTa", PIPELINES_ROOT / "roberta_pipeline"),
]

COMPARISON_OUTPUT_DIR = PIPELINES_ROOT / "comparison_outputs"


def collect_test_metrics() -> pd.DataFrame:
    rows = []
    for model_name, pipeline_dir in MODEL_PIPELINES:
        results_path = pipeline_dir / "outputs" / "evaluation_results.csv"
        if not results_path.exists():
            print(f"[SKIP] {model_name}: {results_path} not found (run evaluate_model.py first)")
            continue
        df = pd.read_csv(results_path)
        test_row = df[df["split"] == "Test"]
        if test_row.empty:
            print(f"[SKIP] {model_name}: no Test split in evaluation_results.csv")
            continue
        row = test_row.iloc[0].to_dict()
        row["model"] = model_name
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No evaluation results found. Train and evaluate models first.")

    comparison = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision", "recall", "f1", "decision_threshold", "high_threshold_risk_mapping"]
    existing_cols = [c for c in cols if c in comparison.columns]
    return comparison[existing_cols].sort_values("f1", ascending=False).reset_index(drop=True)


def compare_all_models() -> pd.DataFrame:
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = collect_test_metrics()

    csv_path = COMPARISON_OUTPUT_DIR / "model_comparison_test.csv"
    comparison.to_csv(csv_path, index=False)

    summary = {
        "comparison_split": "Test",
        "models_compared": comparison["model"].tolist(),
        "ranked_by_f1": comparison.to_dict(orient="records"),
    }
    json_path = COMPARISON_OUTPUT_DIR / "model_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Model Comparison (Test Set) ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    return comparison


if __name__ == "__main__":
    compare_all_models()
