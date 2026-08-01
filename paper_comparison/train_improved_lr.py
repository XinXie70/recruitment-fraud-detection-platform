"""Train the improved Logistic Regression on the paper-aligned random split.

The outer 80/20 split follows Taneja et al. (2025), including random seed
12342. A validation set is created only inside the outer training portion.
The outer test set is not used for model or threshold selection.
"""

import html
import json
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data/raw/emscad_v1.csv"
OUTPUT_DIR = PROJECT_DIR / "paper_comparison/results/improved_lr"
ARTIFACT_DIR = PROJECT_DIR / "artifacts/paper_comparison"

PAPER_SEED = 12342
INNER_SEED = 42
OUTER_TEST_SIZE = 0.20
INNER_VALIDATION_SIZE = 0.20

TEXT_COLUMNS = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


def clean_text(value) -> str:
    """Apply the shared light cleaning without deleting useful fraud signals."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def convert_label(value) -> int:
    value = str(value).strip().lower()
    if value in {"1", "1.0", "t", "true"}:
        return 1
    if value in {"0", "0.0", "f", "false"}:
        return 0
    raise ValueError(f"Unexpected fraudulent label: {value!r}")


def load_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_FILE)
    required = set(TEXT_COLUMNS + ["fraudulent"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Raw EMSCAD is missing columns: {sorted(missing)}")

    output = pd.DataFrame()
    output["record_id"] = [f"emscad_{i:05d}" for i in range(1, len(data) + 1)]
    output["label"] = data["fraudulent"].map(convert_label)
    output["combined_text"] = data[TEXT_COLUMNS].apply(
        lambda row: "\n".join(
            text for text in (clean_text(value) for value in row) if text
        ),
        axis=1,
    )
    if output["combined_text"].eq("").any():
        raise ValueError("At least one row has no usable text across the 12 fields")
    return output


def make_splits(data: pd.DataFrame):
    all_indices = np.arange(len(data))
    outer_train, test = train_test_split(
        all_indices,
        test_size=OUTER_TEST_SIZE,
        random_state=PAPER_SEED,
        shuffle=True,
        stratify=data["label"],
    )
    inner_train, validation = train_test_split(
        outer_train,
        test_size=INNER_VALIDATION_SIZE,
        random_state=INNER_SEED,
        shuffle=True,
        stratify=data.iloc[outer_train]["label"],
    )
    return np.asarray(inner_train), np.asarray(validation), np.asarray(test), np.asarray(outer_train)


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                min_df=2,
                max_df=0.98,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "model",
            LogisticRegression(
                solver="liblinear",
                max_iter=1_000,
                random_state=INNER_SEED,
            ),
        ),
    ])


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
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    filename: str,
) -> None:
    rows = data.iloc[indices][["record_id", "label"]].copy()
    rows["fraud_score"] = scores
    rows["threshold"] = threshold
    rows["prediction"] = (scores >= threshold).astype(int)
    rows.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    data = load_data()
    inner_train_idx, validation_idx, test_idx, outer_train_idx = make_splits(data)

    inner_train = data.iloc[inner_train_idx]
    validation = data.iloc[validation_idx]
    outer_train = data.iloc[outer_train_idx]
    test = data.iloc[test_idx]

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "model__C": [0.5, 1.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=INNER_SEED),
        n_jobs=1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    search.fit(inner_train["combined_text"], inner_train["label"])

    validation_scores = search.best_estimator_.predict_proba(
        validation["combined_text"]
    )[:, 1]
    threshold = select_fraud_f1_threshold(
        validation["label"].to_numpy(), validation_scores
    )
    validation_metrics = calculate_metrics(
        validation["label"].to_numpy(), validation_scores, threshold
    )

    final_model = clone(search.best_estimator_)
    final_model.fit(outer_train["combined_text"], outer_train["label"])
    test_scores = final_model.predict_proba(test["combined_text"])[:, 1]
    test_metrics = calculate_metrics(test["label"].to_numpy(), test_scores, threshold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    assignments = data[["record_id", "label"]].copy()
    assignments["outer_split"] = "train"
    assignments.loc[test_idx, "outer_split"] = "test"
    assignments["development_split"] = "not_used"
    assignments.loc[inner_train_idx, "development_split"] = "train"
    assignments.loc[validation_idx, "development_split"] = "validation"
    assignments.loc[test_idx, "development_split"] = "test"
    assignments.to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results[[
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_tfidf__ngram_range",
        "param_model__C",
        "param_model__class_weight",
    ]].sort_values("rank_test_score").to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    save_predictions(
        data, validation_idx, validation_scores, threshold, "validation_predictions.csv"
    )
    save_predictions(data, test_idx, test_scores, threshold, "test_predictions.csv")

    config = {
        "experiment": "paper_aligned_improved_lr",
        "paper": "Taneja et al. (2025), Fraud-BERT",
        "dataset_rows": int(len(data)),
        "text_fields": TEXT_COLUMNS,
        "outer_split": {
            "train_ratio": 0.80,
            "test_ratio": 0.20,
            "random_seed": PAPER_SEED,
            "stratified": True,
        },
        "development_split": {
            "inner_train_ratio_of_full_data": 0.64,
            "validation_ratio_of_full_data": 0.16,
            "random_seed": INNER_SEED,
            "stratified": True,
        },
        "selection_metric": "mean 3-fold internal-train CV PR-AUC",
        "threshold_rule": "maximum fraud F1 on internal validation",
        "best_cv_pr_auc": float(search.best_score_),
        "best_parameters": search.best_params_,
        "counts": {
            "inner_train_rows": int(len(inner_train)),
            "inner_train_fraud": int(inner_train["label"].sum()),
            "validation_rows": int(len(validation)),
            "validation_fraud": int(validation["label"].sum()),
            "outer_train_rows": int(len(outer_train)),
            "outer_train_fraud": int(outer_train["label"].sum()),
            "test_rows": int(len(test)),
            "test_fraud": int(test["label"].sum()),
        },
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
    joblib.dump(final_model, ARTIFACT_DIR / "improved_lr_paper_split.joblib")

    print(f"Best parameters: {search.best_params_}")
    print(f"Validation threshold: {threshold:.4f}")
    print(
        f"Validation: PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"Fraud F1={validation_metrics['fraud_f1']:.4f}, "
        f"Macro F1={validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test: PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"Fraud F1={test_metrics['fraud_f1']:.4f}, "
        f"Macro F1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
