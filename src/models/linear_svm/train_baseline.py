"""Train a group-aware TF-IDF Linear SVM baseline.

Hyperparameters are selected using group-aware cross-validation inside Train.
SVM decision scores are calibrated to 0–1 using Train-only group-aware folds.
The classification threshold is selected on Validation. Test is not read.
"""

import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parents[3]
TRAIN_FILE = PROJECT_DIR / "data/splits/train.csv"
VALIDATION_FILE = PROJECT_DIR / "data/splits/validation.csv"
REPORT_DIR = PROJECT_DIR / "reports/models/linear_svm"
ARTIFACT_DIR = PROJECT_DIR / "artifacts/linear_svm"

SEED = 42
MODEL_NAME = "linear_svm_baseline"


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


def build_pipeline(parameters=None):
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
            LinearSVC(
                max_iter=5_000,
                random_state=SEED,
            ),
        ),
    ])
    if parameters:
        pipeline.set_params(**parameters)
    return pipeline


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

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.5, 1.0],
        "model__class_weight": [None, "balanced"],
    }
    selection_splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=selection_splitter,
        n_jobs=1,
        refit=False,
        return_train_score=False,
        verbose=1,
    )
    search.fit(
        train["combined_text"],
        train["label"],
        groups=train["group_id"],
    )

    # These explicit indices make calibration group-aware. With ensemble=False,
    # scikit-learn uses out-of-fold decision scores to learn the sigmoid and then
    # fits the selected base estimator once on the complete Train set.
    calibration_splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )
    calibration_folds = list(
        calibration_splitter.split(
            train["combined_text"],
            train["label"],
            train["group_id"],
        )
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=build_pipeline(search.best_params_),
        method="sigmoid",
        cv=calibration_folds,
        n_jobs=1,
        ensemble=False,
    )
    calibrated_model.fit(train["combined_text"], train["label"])

    validation_scores = calibrated_model.predict_proba(
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
        "calibration": "Train-only group-aware 3-fold sigmoid calibration",
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
        calibrated_model,
        ARTIFACT_DIR / "linear_svm_baseline.joblib",
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
