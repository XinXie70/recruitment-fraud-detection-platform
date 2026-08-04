"""Train LR with bigram, no class_weight, no CV on paper-aligned seed42 splits.

Protocol:
  - Fixed ngram_range=(1, 2) (unigram + bigram)
  - class_weight=None
  - No Train-internal 3-fold CV; fixed C=1.0
  - Validation selects Fraud F1 threshold; Test evaluated once

Reproducibility:
  - Fixed data splits under retrain_paper_aligned_seed42_maxlen512/data/splits
  - Fixed hyperparameters and random_state=42
  - Package versions recorded in results/config.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Make hash-based sklearn internals more stable across runs.
os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT.parent / "retrain_paper_aligned_seed42_maxlen512" / "data" / "splits"
)
OUTPUT_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "artifacts"

INNER_SEED = 42
FIXED_NGRAM_RANGE = (1, 2)
FIXED_C = 1.0
FIXED_CLASS_WEIGHT = None
EXPECTED = {
    "train": (12_873, 624),
    "validation": (1_431, 69),
    "test": (3_576, 173),
}


def set_reproducibility(seed: int = INNER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def package_versions() -> dict[str, str]:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv.gz"
    df = pd.read_csv(path, usecols=["record_id", "label", "combined_text"])
    if df["combined_text"].isna().any() or (df["combined_text"].astype(str).str.strip() == "").any():
        raise ValueError(f"{name}: empty combined_text rows found")
    actual = (len(df), int(df["label"].sum()))
    if actual != EXPECTED[name]:
        raise ValueError(f"{name} expected rows/fraud={EXPECTED[name]}, found {actual}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                    ngram_range=FIXED_NGRAM_RANGE,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1_000,
                    random_state=INNER_SEED,
                    C=FIXED_C,
                    class_weight=FIXED_CLASS_WEIGHT,
                ),
            ),
        ]
    )


def select_fraud_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_f1 = np.max(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    return float(thresholds[int(best_indices[0])])


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "fraud_rows": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def save_predictions(
    data: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    filename: str,
) -> None:
    output = data[["record_id", "label"]].copy()
    output["fraud_score"] = scores
    output["threshold"] = threshold
    output["prediction"] = (scores >= threshold).astype(int)
    output.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    set_reproducibility(INNER_SEED)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    model = build_pipeline()
    model.fit(train["combined_text"], train["label"])

    validation_scores = model.predict_proba(validation["combined_text"])[:, 1]
    threshold = select_fraud_f1_threshold(
        validation["label"].to_numpy(), validation_scores
    )
    validation_metrics = calculate_metrics(
        validation["label"].to_numpy(), validation_scores, threshold
    )

    test_scores = model.predict_proba(test["combined_text"])[:, 1]
    test_metrics = calculate_metrics(test["label"].to_numpy(), test_scores, threshold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    save_predictions(validation, validation_scores, threshold, "validation_predictions.csv")
    save_predictions(test, test_scores, threshold, "test_predictions.csv")

    fixed_parameters = {
        "tfidf__ngram_range": list(FIXED_NGRAM_RANGE),
        "model__C": FIXED_C,
        "model__class_weight": FIXED_CLASS_WEIGHT,
    }
    config = {
        "experiment": "lr_none_bigram_no_cv_paper_aligned_seed42",
        "class_weight": FIXED_CLASS_WEIGHT,
        "data_source": str(DATA_DIR),
        "input": "combined_text from paper-aligned seed42 splits",
        "reproducibility": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "random_seed": INNER_SEED,
            "logistic_regression_random_state": INNER_SEED,
            "fixed_hyperparameters": True,
            "train_internal_cv": False,
            "package_versions": package_versions(),
        },
        "split": {
            "seed": 42,
            "protocol": "80/20 outer split; 10% of train pool used as validation",
            "train_rows": int(len(train)),
            "train_fraud": int(train["label"].sum()),
            "validation_rows": int(len(validation)),
            "validation_fraud": int(validation["label"].sum()),
            "test_rows": int(len(test)),
            "test_fraud": int(test["label"].sum()),
        },
        "protocol": {
            "train_internal_3fold_cv_pr_auc": False,
            "fixed_ngram_range": list(FIXED_NGRAM_RANGE),
            "fixed_C": FIXED_C,
            "class_weight": FIXED_CLASS_WEIGHT,
        },
        "selection_metric": "none (fixed hyperparameters; no CV)",
        "threshold_rule": "maximum fraud F1 on Validation",
        "fixed_parameters": fixed_parameters,
        "selected_threshold": threshold,
        "validation_kept_separate_from_train": True,
        "test_used_for_selection": False,
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    joblib.dump(
        model,
        ARTIFACT_DIR / "lr_none_bigram_no_cv_paper_aligned_seed42.joblib",
    )

    cm = test_metrics["confusion_matrix"]
    vcm = validation_metrics["confusion_matrix"]
    results_md = f"""# LR (class_weight=None, bigram, no CV) on paper-aligned seed42

## Purpose

Fixed-protocol LR on paper-aligned seed42 splits:

- **Included**: unigram + bigram (`ngram_range=(1, 2)`)
- **Excluded**: `class_weight` (fixed to `None`)
- **Excluded**: Train-internal 3-fold CV (fixed `C=1.0`)

Same Train/Validation/Test splits as `retrain_paper_aligned_seed42_maxlen512`.

## Reproducibility

- Split seed: `42`
- Model `random_state`: `{INNER_SEED}`
- `PYTHONHASHSEED`: `{os.environ.get("PYTHONHASHSEED")}`
- Packages: `{package_versions()}`
- Threshold selected on Validation only; Test evaluated once

## Fixed configuration

- ngram_range: `{FIXED_NGRAM_RANGE}`
- C: `{FIXED_C}`
- class_weight: `{FIXED_CLASS_WEIGHT}`
- Validation threshold: `{threshold:.4f}`

## Validation

| Metric | Value |
|---|---:|
| Fraud Precision | {validation_metrics["fraud_precision"]:.4f} |
| Fraud Recall | {validation_metrics["fraud_recall"]:.4f} |
| Fraud F1 | {validation_metrics["fraud_f1"]:.4f} |
| Macro F1 | {validation_metrics["macro_f1"]:.4f} |
| PR-AUC | {validation_metrics["pr_auc"]:.4f} |
| ROC-AUC | {validation_metrics["roc_auc"]:.4f} |
| TP / FP / FN / TN | {vcm["tp"]} / {vcm["fp"]} / {vcm["fn"]} / {vcm["tn"]} |

## Test

| Metric | Value |
|---|---:|
| Fraud Precision | {test_metrics["fraud_precision"]:.4f} |
| Fraud Recall | {test_metrics["fraud_recall"]:.4f} |
| Fraud F1 | {test_metrics["fraud_f1"]:.4f} |
| Macro F1 | {test_metrics["macro_f1"]:.4f} |
| PR-AUC | {test_metrics["pr_auc"]:.4f} |
| ROC-AUC | {test_metrics["roc_auc"]:.4f} |
| TP / FP / FN / TN | {cm["tp"]} / {cm["fp"]} / {cm["fn"]} / {cm["tn"]} |
"""
    (OUTPUT_DIR / "RESULTS.md").write_text(results_md, encoding="utf-8")

    print(f"Fixed parameters: {fixed_parameters}")
    print(f"Validation threshold: {threshold:.4f}")
    print(
        f"Validation: PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"Fraud P={validation_metrics['fraud_precision']:.4f}, "
        f"Fraud R={validation_metrics['fraud_recall']:.4f}, "
        f"Fraud F1={validation_metrics['fraud_f1']:.4f}, "
        f"Macro F1={validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test: PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"Fraud P={test_metrics['fraud_precision']:.4f}, "
        f"Fraud R={test_metrics['fraud_recall']:.4f}, "
        f"Fraud F1={test_metrics['fraud_f1']:.4f}, "
        f"Macro F1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
