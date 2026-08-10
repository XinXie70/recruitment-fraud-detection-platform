"""Train Linear SVM (TF-IDF) on sprint1 70/15/15 splits."""

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
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent.parent / "sprint1" / "data" / "splits"
RESULT_DIR = ROOT / "result"
WEIGHT_DIR = ROOT / "weight"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv", usecols=["record_id", "label", "combined_text"])
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    return df


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = float(np.max(f1_values))
    idx = int(np.flatnonzero(np.isclose(f1_values, best))[0])
    return float(thresholds[idx])


def fraud_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "fraud_f1": float(f1_score(labels, preds, zero_division=0)),
        "fraud_recall": float(recall_score(labels, preds, zero_division=0)),
        "fraud_precision": float(precision_score(labels, preds, zero_division=0)),
    }


def main() -> None:
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    train, validation, test = load_split("train"), load_split("validation"), load_split("test")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.98,
        max_features=50_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.fit_transform(train["combined_text"])
    x_val = vectorizer.transform(validation["combined_text"])
    x_test = vectorizer.transform(test["combined_text"])

    model = LinearSVC(C=1.0, class_weight=None, max_iter=5_000, random_state=SEED, dual="auto")
    model.fit(x_train, train["label"])

    val_scores = model.decision_function(x_val)
    threshold = select_threshold(validation["label"].to_numpy(), val_scores)
    test_scores = model.decision_function(x_test)
    result = fraud_metrics(test["label"].to_numpy(), test_scores, threshold)

    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump(
        {"vectorizer": vectorizer, "model": model, "threshold": threshold},
        WEIGHT_DIR / "svm_tfidf.joblib",
    )
    print(json.dumps(result, indent=2))
    print(f"threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
