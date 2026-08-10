"""Shared helpers + entrypoint to train all four LR feature extractors.

Methods: TF-IDF, BoW, Word2Vec, Char TF-IDF.
Uses fixed sprint3/data/splits; threshold from Validation Fraud F1; Test once.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = EXPERIMENT_ROOT / "code"
RESULT_DIR = EXPERIMENT_ROOT / "result"
WEIGHT_DIR = EXPERIMENT_ROOT / "weight"
DATA_SPLITS = EXPERIMENT_ROOT.parent / "data" / "splits"
COMPARISON_CSV = RESULT_DIR / "comparison_test.csv"
COMPARISON_FIGURE = RESULT_DIR / "fraud_metrics_comparison.png"

INNER_SEED = 42
FIXED_C = 1.0
FIXED_CLASS_WEIGHT = None
TEXT_COLUMN = "combined_text"
LABEL_COLUMN = "label"

EXPECTED = {
    "train": (12_873, 624),
    "validation": (1_431, 69),
    "test": (3_576, 173),
}

METHODS = ("tfidf", "bow", "word2vec", "char_tfidf")
METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "bow": "BoW",
    "word2vec": "Word2Vec",
    "char_tfidf": "Char TF-IDF",
}
METRIC_COLUMNS = ("Fraud Precision", "Fraud Recall", "Fraud F1")


def set_reproducibility(seed: int = INNER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_SPLITS / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path, usecols=["record_id", LABEL_COLUMN, TEXT_COLUMN])
    if df[TEXT_COLUMN].isna().any() or (df[TEXT_COLUMN].astype(str).str.strip() == "").any():
        raise ValueError(f"{name}: empty {TEXT_COLUMN} rows found")
    actual = (len(df), int(df[LABEL_COLUMN].sum()))
    if actual != EXPECTED[name]:
        raise ValueError(f"{name} expected rows/fraud={EXPECTED[name]}, found {actual}")
    return df


def load_splits() -> Dict[str, pd.DataFrame]:
    return {
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    }


def make_lr() -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        max_iter=1_000,
        random_state=INNER_SEED,
        C=FIXED_C,
        class_weight=FIXED_CLASS_WEIGHT,
    )


def build_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                ),
            ),
            ("model", make_lr()),
        ]
    )


def build_bow_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                CountVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    ngram_range=(1, 2),
                ),
            ),
            ("model", make_lr()),
        ]
    )


def build_char_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                    ngram_range=(3, 5),
                ),
            ),
            ("model", make_lr()),
        ]
    )


class Word2VecDocumentEmbedder(BaseEstimator, TransformerMixin):
    """Train Word2Vec on fit corpus; transform docs to mean word vectors."""

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
        seed: int = INNER_SEED,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model_ = None

    @staticmethod
    def _tokenize(texts: List[str]) -> List[List[str]]:
        return [str(t).lower().split() for t in texts]

    def fit(self, X, y=None):
        from gensim.models import Word2Vec

        sentences = self._tokenize(list(X))
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=self.seed,
            sg=1,
            epochs=self.epochs,
        )
        return self

    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("Word2VecDocumentEmbedder must be fit before transform")
        key_to_index = self.model_.wv.key_to_index
        vectors = self.model_.wv.vectors
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            tokens = str(text).lower().split()
            idxs = [key_to_index[t] for t in tokens if t in key_to_index]
            if idxs:
                out[i] = vectors[idxs].mean(axis=0)
        return out


def build_word2vec_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                Word2VecDocumentEmbedder(
                    vector_size=200,
                    window=5,
                    min_count=2,
                    epochs=10,
                    seed=INNER_SEED,
                ),
            ),
            ("model", make_lr()),
        ]
    )


BUILDERS: Dict[str, Callable[[], Pipeline]] = {
    "tfidf": build_tfidf_pipeline,
    "bow": build_bow_pipeline,
    "word2vec": build_word2vec_pipeline,
    "char_tfidf": build_char_tfidf_pipeline,
}


def select_fraud_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_f1 = np.max(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    return float(thresholds[int(best_indices[0])])


def calculate_test_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "fraud_rows": int(labels.sum()),
        "Fraud Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Fraud Recall": float(recall_score(labels, predictions, zero_division=0)),
        "Fraud F1": float(f1_score(labels, predictions, zero_division=0)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def update_comparison(method: str, metrics: Dict[str, Any]) -> None:
    ensure_dirs(RESULT_DIR)
    cols = ["method", "Fraud Precision", "Fraud Recall", "Fraud F1"]
    slim = {
        "method": method,
        "Fraud Precision": metrics["Fraud Precision"],
        "Fraud Recall": metrics["Fraud Recall"],
        "Fraud F1": metrics["Fraud F1"],
    }
    if COMPARISON_CSV.exists():
        table = pd.read_csv(COMPARISON_CSV)
    else:
        table = pd.DataFrame(columns=cols)
    for col in cols:
        if col not in table.columns:
            table[col] = pd.NA
    mask = table["method"] == method
    if mask.any():
        for key, value in slim.items():
            table.loc[mask, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([slim])], ignore_index=True)
    # Keep a stable method order.
    order = {name: i for i, name in enumerate(METHODS)}
    table = table[cols]
    table["_ord"] = table["method"].map(lambda m: order.get(m, 99))
    table = table.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    table.to_csv(COMPARISON_CSV, index=False)
    print(f"Updated comparison: {COMPARISON_CSV}", flush=True)
    print(table.to_string(index=False), flush=True)


def plot_fraud_metrics_comparison(
    comparison_csv: Path = COMPARISON_CSV,
    output_path: Path = COMPARISON_FIGURE,
) -> Path:
    """Grouped bar chart of Fraud Precision / Recall / F1 on the test set."""
    import matplotlib.pyplot as plt

    if not comparison_csv.exists():
        raise FileNotFoundError(f"Missing comparison table: {comparison_csv}")

    table = pd.read_csv(comparison_csv)
    missing = [c for c in ("method", *METRIC_COLUMNS) if c not in table.columns]
    if missing:
        raise ValueError(f"comparison CSV missing columns: {missing}")

    order = {name: i for i, name in enumerate(METHODS)}
    table = table[table["method"].isin(METHODS)].copy()
    table["_ord"] = table["method"].map(order)
    table = table.sort_values("_ord").reset_index(drop=True)
    if table.empty:
        raise ValueError("No known methods found in comparison CSV")

    labels = [METHOD_LABELS.get(m, m) for m in table["method"].tolist()]
    x = np.arange(len(labels))
    width = 0.25
    offsets = (-width, 0.0, width)

    ensure_dirs(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, metric in zip(offsets, METRIC_COLUMNS):
        values = table[metric].astype(float).to_numpy()
        bars = ax.bar(x + offset, values, width=width, label=metric)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_title("Fraud Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.grid(True, linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure: {output_path}", flush=True)
    return output_path


def run_method(
    method: str,
    splits: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    if method not in BUILDERS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    set_reproducibility(INNER_SEED)
    ensure_dirs(RESULT_DIR, WEIGHT_DIR)
    if splits is None:
        splits = load_splits()

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    print(f"=== Feature method: {method} ===", flush=True)
    print(
        f"Splits: train={len(train)} val={len(validation)} test={len(test)}",
        flush=True,
    )

    model = BUILDERS[method]()
    model.fit(train[TEXT_COLUMN], train[LABEL_COLUMN])

    validation_scores = model.predict_proba(validation[TEXT_COLUMN])[:, 1]
    threshold = select_fraud_f1_threshold(
        validation[LABEL_COLUMN].to_numpy(), validation_scores
    )

    test_scores = model.predict_proba(test[TEXT_COLUMN])[:, 1]
    test_metrics = calculate_test_metrics(
        test[LABEL_COLUMN].to_numpy(), test_scores, threshold
    )
    test_metrics["method"] = method

    # Keep only test metrics under result/.
    metrics_path = RESULT_DIR / f"test_metrics_{method}.json"
    slim = {
        "method": method,
        "Fraud Precision": test_metrics["Fraud Precision"],
        "Fraud Recall": test_metrics["Fraud Recall"],
        "Fraud F1": test_metrics["Fraud F1"],
        "Accuracy": test_metrics["Accuracy"],
        "threshold": test_metrics["threshold"],
        "confusion_matrix": test_metrics["confusion_matrix"],
    }
    metrics_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    # Keep only weight artifact under weight/.
    weight_path = WEIGHT_DIR / f"lr_{method}.joblib"
    joblib.dump(
        {
            "method": method,
            "pipeline": model,
            "threshold": float(threshold),
            "text_column": TEXT_COLUMN,
        },
        weight_path,
    )

    update_comparison(method, test_metrics)
    print(json.dumps(slim, indent=2), flush=True)
    print(f"Saved weight: {weight_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    return slim


def run_all(methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    set_reproducibility(INNER_SEED)
    ensure_dirs(RESULT_DIR, WEIGHT_DIR)
    splits = load_splits()
    selected = list(methods) if methods else list(METHODS)
    results = []
    for method in selected:
        results.append(run_method(method, splits=splits))
    plot_fraud_metrics_comparison()
    print(f"\nAll done. Comparison -> {COMPARISON_CSV}", flush=True)
    print(f"Figure -> {COMPARISON_FIGURE}", flush=True)
    return results


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--plot-only"]:
        plot_fraud_metrics_comparison()
        return
    if args:
        unknown = [a for a in args if a not in METHODS]
        if unknown:
            raise SystemExit(
                f"Unknown method(s): {unknown}; choose from {METHODS} "
                "or pass --plot-only"
            )
        run_all(args)
    else:
        run_all()


if __name__ == "__main__":
    main()
