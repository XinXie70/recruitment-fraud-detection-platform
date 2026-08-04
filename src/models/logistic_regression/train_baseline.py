"""Train a simple TF-IDF Logistic Regression baseline.

Hyperparameters are selected using group-aware cross-validation inside Train.
The classification threshold is selected on Validation. Test is not read.
"""

import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[3]
TRAIN_FILE = PROJECT_DIR / "data/splits/train.csv"
VALIDATION_FILE = PROJECT_DIR / "data/splits/validation.csv"
REPORT_DIR = PROJECT_DIR / "reports/models/logistic_regression"
ARTIFACT_DIR = PROJECT_DIR / "model_weights/logistic_regression"

SEED = 42
MODEL_NAME = "logistic_regression_baseline"


def load_split(path):
    required_columns = {"record_id", "combined_text", "label", "group_id"}
    data = pd.read_csv(path)
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if data["record_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate record_id values")
    if data["combined_text"].isna().any():
        raise ValueError(f"{path.name} contains empty combined_text values")
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name} contains labels other than 0 and 1")
    return data


def select_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_f1 = np.max(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    # Thresholds are ascending. The lower threshold is used if F1 is tied,
    # which favours higher recall for the fraudulent class.
    best_index = int(best_indices[0])
    return float(thresholds[best_index])


def calculate_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fraud_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def save_cv_results(search):
    results = pd.DataFrame(search.cv_results_)
    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_tfidf__ngram_range",
        "param_model__C",
        "param_model__class_weight",
        "mean_fit_time",
    ]
    results[columns].sort_values("rank_test_score").to_csv(
        REPORT_DIR / "cv_results.csv", index=False
    )


def main():
    train = load_split(TRAIN_FILE)
    validation = load_split(VALIDATION_FILE)

    overlap = set(train["group_id"]) & set(validation["group_id"])
    if overlap:
        raise ValueError("A duplicate group appears in both Train and Validation")

    pipeline = Pipeline([
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
                random_state=SEED,
            ),
        ),
    ])

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.5, 1.0],
        "model__class_weight": [None, "balanced"],
    }
    cross_validation = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )

    search.fit(
        train["combined_text"],
        train["label"],
        groups=train["group_id"],
    )

    validation_scores = search.best_estimator_.predict_proba(
        validation["combined_text"]
    )[:, 1]
    selected_threshold = select_f1_threshold(
        validation["label"].to_numpy(), validation_scores
    )

    validation_metrics = {
        "selected_threshold": calculate_metrics(
            validation["label"].to_numpy(),
            validation_scores,
            selected_threshold,
        ),
        "default_threshold_0_5": calculate_metrics(
            validation["label"].to_numpy(),
            validation_scores,
            0.5,
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_cv_results(search)

    selected_config = {
        "model_name": MODEL_NAME,
        "random_seed": SEED,
        "selection_metric": "mean 3-fold Train CV PR-AUC",
        "threshold_rule": "maximum fraud F1 on Validation",
        "best_cross_validation_pr_auc": float(search.best_score_),
        "best_parameters": search.best_params_,
        "train_rows": int(len(train)),
        "train_fraud": int(train["label"].sum()),
        "validation_rows": int(len(validation)),
        "validation_fraud": int(validation["label"].sum()),
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "test_used": False,
    }

    (REPORT_DIR / "selected_config.json").write_text(
        json.dumps(selected_config, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )

    predictions = pd.DataFrame({
        "record_id": validation["record_id"],
        "model_name": MODEL_NAME,
        "fraud_score": validation_scores,
        "threshold": selected_threshold,
        "prediction": (validation_scores >= selected_threshold).astype(int),
        "true_label": validation["label"],
    })
    predictions.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    joblib.dump(
        search.best_estimator_,
        ARTIFACT_DIR / "logistic_regression_baseline.joblib",
    )

    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Validation threshold: {selected_threshold:.4f}")
    print(
        "Validation PR-AUC: "
        f"{validation_metrics['selected_threshold']['pr_auc']:.4f}"
    )
    print(
        "Validation fraud F1: "
        f"{validation_metrics['selected_threshold']['fraud_f1']:.4f}"
    )
    print(f"Reports saved to: {REPORT_DIR}")
    print("Test set used: No")


if __name__ == "__main__":
    main()
