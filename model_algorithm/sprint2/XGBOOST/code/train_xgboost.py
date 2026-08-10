"""Train XGBoost (TF-IDF) on sprint1 70/15/15 splits."""

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
from xgboost import XGBClassifier

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
        return 0.5
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
        max_features=20_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    print("Fitting TF-IDF...", flush=True)
    x_train = vectorizer.fit_transform(train["combined_text"])
    x_val = vectorizer.transform(validation["combined_text"])
    x_test = vectorizer.transform(test["combined_text"])
    print(f"TF-IDF shape: {x_train.shape}", flush=True)

    y_train = train["label"].to_numpy()
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos

    # Sparse TF-IDF is more reliable on CPU with hist; still fast at this scale.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        device="cpu",
    )
    print("Fitting XGBoost...", flush=True)
    model.fit(x_train, y_train, eval_set=[(x_val, validation["label"].to_numpy())], verbose=False)
    print("Predicting...", flush=True)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(validation["label"].to_numpy(), val_scores)
    test_scores = model.predict_proba(x_test)[:, 1]
    result = fraud_metrics(test["label"].to_numpy(), test_scores, threshold)

    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump(
        {"vectorizer": vectorizer, "model": model, "threshold": threshold},
        WEIGHT_DIR / "xgboost_tfidf.joblib",
    )
    print(json.dumps(result, indent=2))
    print(f"threshold={threshold:.4f} scale_pos_weight={scale_pos_weight:.2f}")


if __name__ == "__main__":
    main()
