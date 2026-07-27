"""Run the locked Logistic Regression and Linear SVM models on Test once.

This script does not train models or select thresholds. It loads the saved
models and the thresholds already selected on Validation.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "splits"
TEST_FILE = DATA_DIR / "test.csv"
EXPECTED_TEST_SHA256 = (
    "85457e3dc60c05a68aaea9e9f433b2fb8bc9cb775dbf0221d61efe2e1cbd6257"
)

MODELS = [
    {
        "name": "logistic_regression_baseline",
        "artifact": PROJECT_DIR
        / "artifacts/logistic_regression/logistic_regression_baseline.joblib",
        "report_dir": PROJECT_DIR / "reports/models/logistic_regression",
    },
    {
        "name": "linear_svm_baseline",
        "artifact": PROJECT_DIR
        / "artifacts/linear_svm/linear_svm_baseline.joblib",
        "report_dir": PROJECT_DIR / "reports/models/linear_svm",
    },
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(path):
    required = {"record_id", "combined_text", "label", "group_id"}
    data = pd.read_csv(path)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if data["record_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate record_id values")
    if data["combined_text"].isna().any():
        raise ValueError(f"{path.name} contains empty combined_text values")
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name} contains labels other than 0 and 1")
    return data


def check_group_isolation(test):
    for split_name in ["train", "validation"]:
        other = load_split(DATA_DIR / f"{split_name}.csv")
        overlap = set(test["group_id"]) & set(other["group_id"])
        if overlap:
            raise ValueError(
                f"{len(overlap)} groups overlap between {split_name} and Test"
            )


def calculate_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "test_rows": int(len(labels)),
        "test_fraud": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fraud_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main():
    existing_results = [
        item["report_dir"] / "test_metrics.json"
        for item in MODELS
        if (item["report_dir"] / "test_metrics.json").exists()
    ]
    if existing_results:
        names = ", ".join(str(path) for path in existing_results)
        raise FileExistsError(
            f"Test results already exist ({names}); evaluation will not rerun"
        )

    actual_hash = file_sha256(TEST_FILE)
    if actual_hash != EXPECTED_TEST_SHA256:
        raise ValueError(
            "Test SHA-256 does not match the locked Data Contract version"
        )

    test = load_split(TEST_FILE)
    check_group_isolation(test)
    evaluated_at = datetime.now(timezone.utc).isoformat()
    comparison_rows = []

    # Load every locked input before writing any Test result.
    locked_models = []
    for item in MODELS:
        config_path = item["report_dir"] / "selected_config.json"
        metrics_path = item["report_dir"] / "validation_metrics.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        validation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = validation_metrics["selected_threshold"]["threshold"]
        model = joblib.load(item["artifact"])
        locked_models.append((item, config_path, config, threshold, model))

    for item, config_path, config, threshold, model in locked_models:
        scores = model.predict_proba(test["combined_text"])[:, 1]
        predictions = (scores >= threshold).astype(int)
        metrics = calculate_metrics(
            test["label"].to_numpy(), scores, threshold
        )
        metrics["model_name"] = item["name"]
        metrics["test_sha256"] = actual_hash
        metrics["evaluated_at_utc"] = evaluated_at

        predictions_table = pd.DataFrame(
            {
                "record_id": test["record_id"],
                "model_name": item["name"],
                "fraud_score": scores,
                "threshold": threshold,
                "prediction": predictions,
                "true_label": test["label"],
            }
        )
        predictions_table.to_csv(
            item["report_dir"] / "test_predictions.csv", index=False
        )
        (item["report_dir"] / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        config["test_used"] = True
        config["test_evaluated_at_utc"] = evaluated_at
        config["test_sha256"] = actual_hash
        config_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        comparison_rows.append(
            {
                "model": item["name"],
                "threshold": threshold,
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "fraud_precision": metrics["fraud_precision"],
                "fraud_recall": metrics["fraud_recall"],
                "fraud_f1": metrics["fraud_f1"],
                "accuracy": metrics["accuracy"],
                **metrics["confusion_matrix"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        PROJECT_DIR / "reports/models/test_comparison.csv", index=False
    )
    print(comparison.to_string(index=False))
    print(f"\nTest SHA-256 verified: {actual_hash}")
    print("Test evaluation completed once. No training or threshold tuning was run.")


if __name__ == "__main__":
    main()
