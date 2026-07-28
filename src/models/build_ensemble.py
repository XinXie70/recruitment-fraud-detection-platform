"""Build LR+BERT ensemble from locked base-model predictions.

Supports two modes:
  --mode validation   Search weights + threshold on validation (default).
  --mode test         Apply locked config from validation to test predictions.

Uses continuous fraud scores. Does not retrain any base model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "reports" / "models"
ROOT_COMPARISON = REPORTS_DIR / "validation_comparison.csv"
ROOT_TEST_COMPARISON = REPORTS_DIR / "test_comparison.csv"

WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

CLASSIC_KEY = "logistic_regression_baseline"
CLASSIC_DIR = "logistic_regression"
ENSEMBLE_DIR_NAME = "ensemble_lr_bert"
MODEL_NAME = "ensemble_lr_bert_class_weighted"
CV_PR_AUC = 0.8507801078268361
BERT_KEY = "bert_class_weighted"


def load_predictions(path: Path, expected_model_name: str) -> pd.DataFrame:
    required = {
        "record_id",
        "model_name",
        "fraud_score",
        "threshold",
        "prediction",
        "true_label",
    }
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df["record_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate record_id values")
    if not df["model_name"].eq(expected_model_name).all():
        raise ValueError(
            f"{path.name} does not contain only model_name={expected_model_name}"
        )
    return df


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def select_best_blend(merged: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = merged["true_label_classic"].to_numpy(dtype=int)
    candidates = []
    best = None

    for weight_classic in WEIGHT_GRID:
        weight_bert = 1.0 - weight_classic
        scores = (
            weight_classic * merged["fraud_score_classic"].to_numpy(dtype=float)
            + weight_bert * merged["fraud_score_bert"].to_numpy(dtype=float)
        )
        for threshold in THRESHOLD_GRID:
            metrics = calculate_metrics(labels, scores, float(threshold))
            row = {
                "weight_classic": float(weight_classic),
                "weight_bert": float(weight_bert),
                **metrics,
            }
            candidates.append(row)
            if best is None:
                best = row
                continue
            current_key = (
                row["fraud_f1"],
                row["fraud_recall"],
                row["fraud_precision"],
                -row["threshold"],
            )
            best_key = (
                best["fraud_f1"],
                best["fraud_recall"],
                best["fraud_precision"],
                -best["threshold"],
            )
            if current_key > best_key:
                best = row

    return best, pd.DataFrame(candidates)


def load_classic_validation_metrics() -> dict:
    path = REPORTS_DIR / CLASSIC_DIR / "validation_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_threshold"]


def load_bert_validation_metrics() -> tuple[str, dict]:
    payload = json.loads(
        (REPORTS_DIR / "bert" / "validation_metrics.json").read_text(encoding="utf-8")
    )
    return payload["model_name"], payload["validation_metrics"]


def build_comparison_rows(ensemble_metrics: dict) -> pd.DataFrame:
    classic_metrics = load_classic_validation_metrics()
    bert_name, bert_metrics = load_bert_validation_metrics()

    rows = [
        {
            "model": CLASSIC_KEY,
            "cv_pr_auc": CV_PR_AUC,
            "validation_pr_auc": classic_metrics["pr_auc"],
            "validation_roc_auc": classic_metrics["roc_auc"],
            "threshold": classic_metrics["threshold"],
            "fraud_precision": classic_metrics["fraud_precision"],
            "fraud_recall": classic_metrics["fraud_recall"],
            "fraud_f1": classic_metrics["fraud_f1"],
            "tn": classic_metrics["confusion_matrix"]["tn"],
            "fp": classic_metrics["confusion_matrix"]["fp"],
            "fn": classic_metrics["confusion_matrix"]["fn"],
            "tp": classic_metrics["confusion_matrix"]["tp"],
        },
        {
            "model": bert_name,
            "cv_pr_auc": np.nan,
            "validation_pr_auc": bert_metrics["pr_auc"],
            "validation_roc_auc": bert_metrics["roc_auc"],
            "threshold": bert_metrics["threshold"],
            "fraud_precision": bert_metrics["fraud_precision"],
            "fraud_recall": bert_metrics["fraud_recall"],
            "fraud_f1": bert_metrics["fraud_f1"],
            "tn": bert_metrics["confusion_matrix"]["tn"],
            "fp": bert_metrics["confusion_matrix"]["fp"],
            "fn": bert_metrics["confusion_matrix"]["fn"],
            "tp": bert_metrics["confusion_matrix"]["tp"],
        },
        {
            "model": MODEL_NAME,
            "cv_pr_auc": np.nan,
            "validation_pr_auc": ensemble_metrics["pr_auc"],
            "validation_roc_auc": ensemble_metrics["roc_auc"],
            "threshold": ensemble_metrics["threshold"],
            "fraud_precision": ensemble_metrics["fraud_precision"],
            "fraud_recall": ensemble_metrics["fraud_recall"],
            "fraud_f1": ensemble_metrics["fraud_f1"],
            "tn": ensemble_metrics["confusion_matrix"]["tn"],
            "fp": ensemble_metrics["confusion_matrix"]["fp"],
            "fn": ensemble_metrics["confusion_matrix"]["fn"],
            "tp": ensemble_metrics["confusion_matrix"]["tp"],
        },
    ]
    return pd.DataFrame(rows)


def update_comparison_table(path: Path, rows_to_upsert: pd.DataFrame) -> None:
    if path.exists():
        current = pd.read_csv(path)
    else:
        current = pd.DataFrame(columns=rows_to_upsert.columns)
    remaining = current.loc[~current["model"].isin(rows_to_upsert["model"])]
    updated = pd.concat([remaining, rows_to_upsert], ignore_index=True)
    updated = updated.sort_values("model").reset_index(drop=True)
    updated.to_csv(path, index=False)


def build_test_comparison_row(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "threshold": metrics["threshold"],
                "fraud_precision": metrics["fraud_precision"],
                "fraud_recall": metrics["fraud_recall"],
                "fraud_f1": metrics["fraud_f1"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
        ]
    )


def run_validation() -> None:
    ensemble_dir = REPORTS_DIR / ENSEMBLE_DIR_NAME
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    classic_path = REPORTS_DIR / CLASSIC_DIR / "validation_predictions.csv"
    bert_path = REPORTS_DIR / "bert" / "validation_predictions.csv"

    classic = load_predictions(classic_path, CLASSIC_KEY)
    bert = load_predictions(bert_path, BERT_KEY)
    classic = classic.rename(
        columns={
            "fraud_score": "fraud_score_classic",
            "prediction": "prediction_classic",
            "true_label": "true_label_classic",
            "threshold": "threshold_classic",
            "model_name": "model_name_classic",
        }
    )
    bert = bert.rename(
        columns={
            "fraud_score": "fraud_score_bert",
            "prediction": "prediction_bert",
            "true_label": "true_label_bert",
            "threshold": "threshold_bert",
            "model_name": "model_name_bert",
        }
    )
    merged = classic.merge(bert, on="record_id", validate="one_to_one")

    if not merged["true_label_classic"].equals(merged["true_label_bert"]):
        raise ValueError("Classic and BERT true_label columns do not match after merge")

    best, candidates = select_best_blend(merged)
    ensemble_scores = (
        best["weight_classic"] * merged["fraud_score_classic"].to_numpy(dtype=float)
        + best["weight_bert"] * merged["fraud_score_bert"].to_numpy(dtype=float)
    )
    labels = merged["true_label_classic"].to_numpy(dtype=int)
    predictions = (ensemble_scores >= best["threshold"]).astype(int)

    prediction_frame = pd.DataFrame(
        {
            "record_id": merged["record_id"],
            "model_name": MODEL_NAME,
            "fraud_score": ensemble_scores,
            "threshold": best["threshold"],
            "prediction": predictions,
            "true_label": labels,
        }
    )
    prediction_frame.to_csv(ensemble_dir / "validation_predictions.csv", index=False)

    comparison = build_comparison_rows(best)
    comparison.to_csv(ensemble_dir / "validation_comparison.csv", index=False)
    update_comparison_table(ROOT_COMPARISON, comparison)

    config_payload = {
        "method": "weighted_average",
        "base_models": [CLASSIC_KEY, BERT_KEY],
        "weights": {
            CLASSIC_KEY: best["weight_classic"],
            BERT_KEY: best["weight_bert"],
        },
        "threshold_selection": "max_fraud_f1_on_validation",
        "selected_threshold": best["threshold"],
        "validation_input_files": {
            CLASSIC_KEY: str(classic_path),
            BERT_KEY: str(bert_path),
        },
    }
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    metrics_payload = {
        "model_name": MODEL_NAME,
        "method": "weighted_average",
        "weights": {
            CLASSIC_KEY: best["weight_classic"],
            BERT_KEY: best["weight_bert"],
        },
        "selected_threshold": best,
        "score_correlation": float(
            merged["fraud_score_classic"].corr(merged["fraud_score_bert"])
        ),
        "prediction_disagreement_count": int(
            (merged["prediction_classic"] != merged["prediction_bert"]).sum()
        ),
        "num_rows": int(len(prediction_frame)),
    }
    (ensemble_dir / "validation_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    candidates.sort_values(
        ["fraud_f1", "fraud_recall", "fraud_precision", "pr_auc"],
        ascending=[False, False, False, False],
    ).to_csv(ensemble_dir / "weight_threshold_search.csv", index=False)

    print(f"Wrote predictions to {ensemble_dir / 'validation_predictions.csv'}")
    print(
        f"Selected weights: lr={best['weight_classic']:.2f}, "
        f"bert={best['weight_bert']:.2f}"
    )
    print(f"Selected threshold: {best['threshold']:.2f}")
    print(f"Validation PR-AUC: {best['pr_auc']:.4f}")
    print(f"Validation fraud F1: {best['fraud_f1']:.4f}")


def run_test() -> None:
    ensemble_dir = REPORTS_DIR / ENSEMBLE_DIR_NAME
    config_path = ensemble_dir / "ensemble_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Run --mode validation first to produce ensemble_config.json"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    w_classic = config["weights"][CLASSIC_KEY]
    w_bert = config["weights"][BERT_KEY]
    threshold = config["selected_threshold"]

    classic_test = REPORTS_DIR / CLASSIC_DIR / "test_predictions.csv"
    bert_test = REPORTS_DIR / "bert" / "test_predictions.csv"

    if not classic_test.exists():
        raise FileNotFoundError(f"LR test predictions not found: {classic_test}")
    if not bert_test.exists():
        raise FileNotFoundError(f"BERT test predictions not found: {bert_test}")

    classic = load_predictions(classic_test, CLASSIC_KEY).rename(
        columns={
            "fraud_score": "fraud_score_classic",
            "true_label": "true_label_classic",
        }
    )
    bert = load_predictions(bert_test, BERT_KEY).rename(
        columns={
            "fraud_score": "fraud_score_bert",
            "true_label": "true_label_bert",
        }
    )
    merged = classic.merge(bert, on="record_id", validate="one_to_one")

    if not merged["true_label_classic"].equals(merged["true_label_bert"]):
        raise ValueError("LR and BERT true_label columns do not match after merge")

    ensemble_scores = (
        w_classic * merged["fraud_score_classic"].to_numpy(dtype=float)
        + w_bert * merged["fraud_score_bert"].to_numpy(dtype=float)
    )
    labels = merged["true_label_classic"].to_numpy(dtype=int)
    predictions = (ensemble_scores >= threshold).astype(int)

    prediction_frame = pd.DataFrame(
        {
            "record_id": merged["record_id"],
            "model_name": MODEL_NAME,
            "fraud_score": ensemble_scores,
            "threshold": threshold,
            "prediction": predictions,
            "true_label": labels,
        }
    )
    prediction_frame.to_csv(ensemble_dir / "test_predictions.csv", index=False)

    metrics = calculate_metrics(labels, ensemble_scores, threshold)
    metrics_payload = {
        "model_name": MODEL_NAME,
        "method": "weighted_average",
        "weights": {
            CLASSIC_KEY: w_classic,
            BERT_KEY: w_bert,
        },
        "threshold": threshold,
        "test_metrics": metrics,
        "num_rows": int(len(prediction_frame)),
    }
    (ensemble_dir / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    update_comparison_table(ROOT_TEST_COMPARISON, build_test_comparison_row(metrics))

    print(f"Wrote test predictions to {ensemble_dir / 'test_predictions.csv'}")
    print(f"Locked weights: lr={w_classic:.2f}, bert={w_bert:.2f}")
    print(f"Locked threshold: {threshold:.2f}")
    print(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Test fraud F1: {metrics['fraud_f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"Test CM: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LR+BERT ensemble")
    parser.add_argument(
        "--mode",
        type=str,
        default="validation",
        choices=["validation", "test"],
    )
    args = parser.parse_args()

    if args.mode == "validation":
        run_validation()
    else:
        run_test()


if __name__ == "__main__":
    main()
