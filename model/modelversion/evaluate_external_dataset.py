"""Evaluate all eight saved models on the held-out external dataset.

The external dataset is evaluation-only. This script never trains models or
changes thresholds; it only loads saved artifacts and records predictions.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PIPELINES_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PIPELINES_ROOT / "data" / "external_evaluation" / "external_evaluation.csv"
DEFAULT_OUTPUT_DIR = PIPELINES_ROOT / "external_evaluation_outputs"
DEFAULT_GIT_REF = (
    "dataset/external-dataset:"
    "model/final_model_pipelines/data/external_evaluation/external_evaluation.csv"
)
MODEL_ORDER = [
    "logistic_regression",
    "svm",
    "xgboost",
    "dnn",
    "rnn",
    "bilstm",
    "bert",
    "roberta",
]


def _register_legacy_package_alias() -> None:
    """Support the current folder name while internal imports use the old package name."""
    if "final_model_pipelines" in sys.modules:
        return
    package = types.ModuleType("final_model_pipelines")
    package.__path__ = [str(PIPELINES_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = "final_model_pipelines"
    sys.modules["final_model_pipelines"] = package


def _load_dataset(path: Path | None, git_ref: str) -> pd.DataFrame:
    if path is not None:
        return pd.read_csv(path)
    if DEFAULT_DATASET.exists():
        return pd.read_csv(DEFAULT_DATASET)

    try:
        csv_text = subprocess.check_output(
            ["git", "show", git_ref],
            cwd=PIPELINES_ROOT,
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        raise FileNotFoundError(
            f"External dataset was not found at {DEFAULT_DATASET} or Git ref {git_ref}: {message}"
        ) from exc
    return pd.read_csv(io.StringIO(csv_text))


def _validate_dataset(df: pd.DataFrame) -> None:
    required = {"sample_id", "label", "source_type", "scenario", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"External dataset is missing columns: {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        raise ValueError("External dataset contains duplicate sample_id values")
    labels = set(pd.to_numeric(df["label"], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"External dataset labels must be binary 0/1, found {sorted(labels)}")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("External dataset contains empty text")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _build_prediction_rows(
    dataset: pd.DataFrame, all_results: list[dict[str, dict]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (_, sample), model_results in zip(dataset.iterrows(), all_results):
        true_label = int(sample["label"])
        for model_name in MODEL_ORDER:
            result = model_results[model_name]
            prediction_text = result.get("prediction")
            predicted_label = {"real": 0, "fake": 1}.get(prediction_text)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "true_label": true_label,
                    "source_type": sample["source_type"],
                    "scenario": sample["scenario"],
                    "model": model_name,
                    "status": result.get("status"),
                    "risk_score": _safe_float(result.get("risk_score")),
                    "classification_label": result.get("classification_label"),
                    "prediction": prediction_text,
                    "predicted_label": predicted_label,
                    "is_correct": (
                        bool(predicted_label == true_label)
                        if predicted_label is not None
                        else False
                    ),
                    "message": result.get("message"),
                }
            )
    return pd.DataFrame(rows)


def _metric(value: float) -> float:
    return round(float(value), 4)


def _build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        group = predictions[predictions["model"] == model_name].copy()
        successful = group[group["predicted_label"].notna()].copy()
        y_true = successful["true_label"].astype(int).to_numpy()
        y_pred = successful["predicted_label"].astype(int).to_numpy()
        scores = successful["risk_score"].astype(float).to_numpy()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        score_mask = np.isfinite(scores)
        roc_auc = roc_auc_score(y_true[score_mask], scores[score_mask])
        pr_auc = average_precision_score(y_true[score_mask], scores[score_mask])

        summaries.append(
            {
                "model": model_name,
                "total_samples": len(group),
                "successful_predictions": len(successful),
                "rejected_inputs": int(len(group) - len(successful)),
                "accuracy": _metric(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": _metric(balanced_accuracy_score(y_true, y_pred)),
                "fake_precision": _metric(precision_score(y_true, y_pred, zero_division=0)),
                "fake_recall": _metric(recall_score(y_true, y_pred, zero_division=0)),
                "fake_f1": _metric(f1_score(y_true, y_pred, zero_division=0)),
                "real_specificity": _metric(tn / (tn + fp) if tn + fp else 0.0),
                "roc_auc": _metric(roc_auc),
                "pr_auc": _metric(pr_auc),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "mean_risk_real": _metric(group.loc[group["true_label"] == 0, "risk_score"].mean()),
                "mean_risk_fake": _metric(group.loc[group["true_label"] == 1, "risk_score"].mean()),
            }
        )
    return pd.DataFrame(summaries).sort_values(
        ["fake_f1", "balanced_accuracy", "pr_auc"], ascending=False
    )


def evaluate(dataset: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _register_legacy_package_alias()
    from final_model_pipelines.predict_all import predict_batch_with_all_models

    print(f"Running eight models on {len(dataset)} external samples...", flush=True)
    all_results = predict_batch_with_all_models(dataset["text"].astype(str).tolist())
    predictions = _build_prediction_rows(dataset, all_results)
    summary = _build_summary(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "external_predictions.csv", index=False)
    summary.to_csv(output_dir / "external_model_summary.csv", index=False)
    payload = {
        "dataset": {
            "samples": int(len(dataset)),
            "real_web": int((dataset["label"] == 0).sum()),
            "synthetic_fake": int((dataset["label"] == 1).sum()),
        },
        "models": summary.to_dict(orient="records"),
    }
    (output_dir / "external_model_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to external_evaluation.csv")
    parser.add_argument(
        "--git-ref",
        default=DEFAULT_GIT_REF,
        help="Fallback Git ref in '<revision>:<path>' form when --dataset is omitted",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset, args.git_ref)
    _validate_dataset(dataset)
    dataset["label"] = dataset["label"].astype(int)
    _, summary = evaluate(dataset, args.output_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"\nOutputs saved to: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
