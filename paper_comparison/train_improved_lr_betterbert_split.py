"""Train improved LR using the exact split assignments from betterBERT.

The saved assignment file is the source of truth. Hyperparameters are selected
with cross-validation inside Train, the threshold is selected on Validation,
and Test is used only after those choices are fixed.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from train_improved_lr import (
    INNER_SEED,
    build_pipeline,
    calculate_metrics,
    select_fraud_f1_threshold,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
BRANCH_EXPORT_FILE = (
    PROJECT_DIR
    / "exports/paper_aligned_standalone_seed42/data/source/"
    "emscad_condition_a_no_dedup_input_v1.csv.gz"
)
LOCAL_PROCESSED_FILE = PROJECT_DIR / "data/processed/emscad_processed_v1.csv"
ASSIGNMENT_FILE = (
    PROJECT_DIR
    / "paper_comparison/data/betterbert_seed42/assignments.csv.gz"
)
OUTPUT_DIR = (
    PROJECT_DIR
    / "paper_comparison/results/improved_lr_betterbert_split"
)
ARTIFACT_DIR = PROJECT_DIR / "artifacts/paper_comparison"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_file = (
        BRANCH_EXPORT_FILE if BRANCH_EXPORT_FILE.exists() else LOCAL_PROCESSED_FILE
    )
    if not data_file.exists():
        raise FileNotFoundError(
            "EMSCAD input not found in the betterBERT export or local processed data"
        )
    data = pd.read_csv(
        data_file,
        usecols=["record_id", "combined_text", "label"],
    )
    assignments = pd.read_csv(ASSIGNMENT_FILE)

    if assignments["record_id"].duplicated().any():
        raise ValueError("The betterBERT assignment file has duplicate record IDs")
    if set(assignments["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names in the betterBERT assignment file")

    merged = assignments.merge(data, on="record_id", how="left", validate="one_to_one")
    if merged[["combined_text", "label"]].isna().any().any():
        raise ValueError("At least one assigned record is missing from processed EMSCAD")
    if len(merged) != 17_880:
        raise ValueError(f"Expected 17,880 assigned rows, found {len(merged):,}")

    splits = {
        name: merged.loc[merged["split"] == name].copy()
        for name in ["train", "validation", "test"]
    }
    expected = {
        "train": (12_873, 624),
        "validation": (1_431, 69),
        "test": (3_576, 173),
    }
    for name, split in splits.items():
        actual = (len(split), int(split["label"].sum()))
        if actual != expected[name]:
            raise ValueError(
                f"{name} expected rows/fraud={expected[name]}, found {actual}"
            )
    return splits["train"], splits["validation"], splits["test"]


def save_predictions(
    data: pd.DataFrame,
    scores,
    threshold: float,
    filename: str,
) -> None:
    output = data[["record_id", "label"]].copy()
    output["fraud_score"] = scores
    output["threshold"] = threshold
    output["prediction"] = (scores >= threshold).astype(int)
    output.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    train, validation, test = load_splits()

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
    search.fit(train["combined_text"], train["label"])

    validation_scores = search.best_estimator_.predict_proba(
        validation["combined_text"]
    )[:, 1]
    threshold = select_fraud_f1_threshold(
        validation["label"].to_numpy(), validation_scores
    )
    validation_metrics = calculate_metrics(
        validation["label"].to_numpy(), validation_scores, threshold
    )

    # Keep Validation separate so its scores remain valid for later ensembling.
    test_scores = search.best_estimator_.predict_proba(test["combined_text"])[:, 1]
    test_metrics = calculate_metrics(
        test["label"].to_numpy(), test_scores, threshold
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results[[
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_tfidf__ngram_range",
        "param_model__C",
        "param_model__class_weight",
    ]].sort_values("rank_test_score").to_csv(
        OUTPUT_DIR / "cv_results.csv", index=False
    )

    save_predictions(
        validation,
        validation_scores,
        threshold,
        "validation_predictions.csv",
    )
    save_predictions(test, test_scores, threshold, "test_predictions.csv")

    config = {
        "experiment": "improved_lr_betterbert_split",
        "assignment_source": (
            "betterBERT@4c3b2b2:"
            "exports/paper_aligned_standalone_seed42/data/splits/"
            "assignments.csv.gz"
        ),
        "input": "shared five-field combined_text",
        "input_file": str(
            BRANCH_EXPORT_FILE.relative_to(PROJECT_DIR)
            if BRANCH_EXPORT_FILE.exists()
            else LOCAL_PROCESSED_FILE.relative_to(PROJECT_DIR)
        ),
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
        "selection_metric": "mean 3-fold Train CV PR-AUC",
        "threshold_rule": "maximum fraud F1 on Validation",
        "best_cv_pr_auc": float(search.best_score_),
        "best_parameters": search.best_params_,
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
        search.best_estimator_,
        ARTIFACT_DIR / "improved_lr_betterbert_split.joblib",
    )

    print(f"Best parameters: {search.best_params_}")
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
