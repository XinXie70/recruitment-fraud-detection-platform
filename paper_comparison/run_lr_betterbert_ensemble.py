"""Select and evaluate a simple LR + BetterBERT score ensemble.

Weights and the binary threshold are selected on Validation only. The selected
configuration is then frozen and evaluated once on the aligned Test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


LR_WEIGHTS = [0.25, 0.40, 0.50, 0.60, 0.75]
THRESHOLDS = np.arange(0.01, 1.00, 0.01)


def align_scores(lr_path: Path, bert_path: Path, split: str) -> pd.DataFrame:
    lr = pd.read_csv(lr_path).rename(columns={"fraud_score": "lr_score"})
    bert = pd.read_csv(bert_path).rename(
        columns={
            "original_id": "record_id",
            "true_label": "label",
            "fraud_probability": "bert_score",
            "fraud_score": "bert_score",
        }
    )
    lr = lr[["record_id", "label", "lr_score"]]
    bert = bert[["record_id", "label", "bert_score"]]
    aligned = lr.merge(
        bert,
        on=["record_id", "label"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(lr) or len(aligned) != len(bert):
        raise ValueError(f"{split}: LR and BERT rows are not fully aligned")
    if aligned["record_id"].duplicated().any():
        raise ValueError(f"{split}: duplicate record IDs found")
    return aligned


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "fraud_precision": float(precision[1]),
        "fraud_recall": float(recall[1]),
        "fraud_f1": float(f1[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-validation", type=Path, required=True)
    parser.add_argument("--bert-validation", type=Path, required=True)
    parser.add_argument("--lr-test", type=Path, required=True)
    parser.add_argument("--bert-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    validation = align_scores(args.lr_validation, args.bert_validation, "Validation")
    test = align_scores(args.lr_test, args.bert_test, "Test")
    if set(validation["record_id"]) & set(test["record_id"]):
        raise ValueError("Validation and Test IDs overlap")

    y_validation = validation["label"].to_numpy(dtype=int)
    sweep_rows = []
    for lr_weight in LR_WEIGHTS:
        bert_weight = 1.0 - lr_weight
        scores = (
            lr_weight * validation["lr_score"].to_numpy()
            + bert_weight * validation["bert_score"].to_numpy()
        )
        for threshold in THRESHOLDS:
            metrics = binary_metrics(y_validation, scores, float(threshold))
            sweep_rows.append(
                {
                    "lr_weight": lr_weight,
                    "bert_weight": bert_weight,
                    **metrics,
                }
            )

    sweep = pd.DataFrame(sweep_rows)
    selected = sweep.sort_values(
        ["fraud_f1", "fraud_recall", "fraud_precision", "pr_auc"],
        ascending=False,
    ).iloc[0]
    lr_weight = float(selected["lr_weight"])
    bert_weight = float(selected["bert_weight"])
    threshold = float(selected["threshold"])

    validation_scores = (
        lr_weight * validation["lr_score"].to_numpy()
        + bert_weight * validation["bert_score"].to_numpy()
    )
    validation_metrics = binary_metrics(y_validation, validation_scores, threshold)

    test_scores = (
        lr_weight * test["lr_score"].to_numpy()
        + bert_weight * test["bert_score"].to_numpy()
    )
    test_metrics = binary_metrics(test["label"].to_numpy(dtype=int), test_scores, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "validation_sweep.csv", index=False)
    validation.assign(
        ensemble_score=validation_scores,
        prediction=(validation_scores >= threshold).astype(int),
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    test.assign(
        ensemble_score=test_scores,
        prediction=(test_scores >= threshold).astype(int),
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    config = {
        "lr_weight": lr_weight,
        "bert_weight": bert_weight,
        "threshold": threshold,
        "candidate_lr_weights": LR_WEIGHTS,
        "candidate_thresholds": "0.01 to 0.99 in steps of 0.01",
        "selection_set": "validation",
        "selection_metric": "fraud_f1",
        "test_used_for_selection": False,
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    for filename, payload in (
        ("config.json", config),
        ("validation_metrics.json", validation_metrics),
        ("test_metrics.json", test_metrics),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print(json.dumps({"config": config, "validation": validation_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
