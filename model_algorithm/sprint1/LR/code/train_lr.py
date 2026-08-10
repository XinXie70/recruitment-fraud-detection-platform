"""Train Logistic Regression (TF-IDF + LR) on sprint1 70/15/15 splits.

- Fit TF-IDF on train only
- Choose decision threshold on validation by max Fraud F1
- Evaluate once on test
- Save: result/test_metrics.json (fraud F1 / Recall / Precision)
         weight/lr_tfidf.joblib (+ threshold.json)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "data" / "splits"
RESULT_DIR = ROOT / "result"
WEIGHT_DIR = ROOT / "weight"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, usecols=["record_id", "label", "combined_text"])
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    return df


def select_fraud_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best = float(np.max(f1_values))
    idx = int(np.flatnonzero(np.isclose(f1_values, best))[0])
    return float(thresholds[idx])


def fraud_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "fraud_precision": float(precision_score(labels, preds, zero_division=0)),
        "fraud_recall": float(recall_score(labels, preds, zero_division=0)),
        "fraud_f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def main() -> None:
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1_000,
                    random_state=SEED,
                    C=1.0,
                    class_weight=None,
                ),
            ),
        ]
    )
    model.fit(train["combined_text"], train["label"])

    val_scores = model.predict_proba(validation["combined_text"])[:, 1]
    threshold = select_fraud_f1_threshold(
        validation["label"].to_numpy(), val_scores
    )

    test_scores = model.predict_proba(test["combined_text"])[:, 1]
    metrics = fraud_metrics(test["label"].to_numpy(), test_scores, threshold)
    # Keep only the required test metrics in result/.
    result = {
        "fraud_f1": metrics["fraud_f1"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_precision": metrics["fraud_precision"],
    }

    (RESULT_DIR / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    weight_path = WEIGHT_DIR / "lr_tfidf.joblib"
    joblib.dump({"model": model, "threshold": threshold}, weight_path)

    print(json.dumps(result, indent=2))
    print(f"Saved weight: {weight_path}")
    print(f"Saved result: {RESULT_DIR / 'test_metrics.json'}")
    print(f"threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
