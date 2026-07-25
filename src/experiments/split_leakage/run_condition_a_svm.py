"""Run the calibrated Linear SVM under split-leakage Condition A.

Condition A keeps all 17,880 processed rows and ignores duplicate groups during
the random 70/15/15 split and Train-only calibration. The binary threshold is
selected on the diagnostic Validation partition and evaluated on the diagnostic
Holdout. Official split files are not read.
"""

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from run_condition_a import (
    CONDITION_NAME,
    INPUT_FILE,
    REPORT_DIR,
    SEEDS,
    evaluate,
    exact_overlap_stats,
    normalise_for_exact_match,
    select_f1_threshold,
    split_indices,
)


RESULT_FILE = REPORT_DIR / "condition_a_svm_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_a_svm_summary.md"
COMPARISON_FILE = REPORT_DIR / "condition_a_model_comparison.csv"
LR_RESULT_FILE = REPORT_DIR / "condition_a_lr_results.csv"


def build_calibrated_model(train_labels, seed):
    """Use the fixed best SVM configuration and Train-only calibration."""
    base_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                min_df=2,
                max_df=0.98,
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "model",
            LinearSVC(
                C=1.0,
                class_weight=None,
                max_iter=5_000,
                random_state=seed,
            ),
        ),
    ])
    splitter = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )
    calibration_folds = list(
        splitter.split(
            range(len(train_labels)),
            train_labels,
        )
    )
    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_folds,
        n_jobs=1,
        ensemble=False,
    )


def create_summary(data, results, total_exact_groups, rows_in_exact_groups):
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "macro_f1",
    ]
    overlap_columns = [
        "cross_split_exact_groups",
        "cross_split_fraud_exact_groups",
        "cross_split_legitimate_exact_groups",
        "rows_in_cross_split_exact_groups",
        "train_validation_exact_groups",
        "train_holdout_exact_groups",
        "validation_holdout_exact_groups",
        "validation_rows_matching_train",
        "validation_fraud_rows_matching_train",
        "validation_legitimate_rows_matching_train",
        "holdout_rows_matching_train",
        "holdout_fraud_rows_matching_train",
        "holdout_legitimate_rows_matching_train",
    ]

    lines = [
        "# Split Leakage Experiment: Condition A — Linear SVM",
        "",
        "## Design",
        "",
        f"- Condition: **{CONDITION_NAME}**",
        f"- Input rows: **{len(data):,}**",
        f"- Legitimate rows: **{int((data['label'] == 0).sum()):,}**",
        f"- Fraudulent rows: **{int((data['label'] == 1).sum()):,}**",
        f"- Exact duplicate groups in the full input: **{total_exact_groups:,}**",
        f"- Rows belonging to exact duplicate groups: **{rows_in_exact_groups:,}**",
        "- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**",
        "- Seeds: **0 to 9**",
        "- Model: fixed TF-IDF Linear SVM baseline",
        "- Calibration: Train-only three-fold stratified sigmoid calibration, without group information",
        "- Threshold: selected independently on each diagnostic Validation partition by maximum fraud F1",
        "- Official Train, Validation and Test files used: **No**",
        "",
        "## Holdout performance across 10 seeds",
        "",
        "| Metric | Mean | Standard deviation | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in metric_columns:
        values = results[column]
        lines.append(
            f"| {column} | {values.mean():.4f} | {values.std(ddof=1):.4f} | "
            f"{values.min():.4f} | {values.max():.4f} |"
        )

    lines.extend([
        "",
        "## Exact-duplicate contamination across 10 seeds",
        "",
        "| Measure | Mean | Minimum | Maximum |",
        "|---|---:|---:|---:|",
    ])
    for column in overlap_columns:
        values = results[column]
        lines.append(
            f"| {column} | {values.mean():.1f} | "
            f"{int(values.min())} | {int(values.max())} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Condition A intentionally permits exact duplicates to cross all diagnostic partitions. These results require Conditions B and C before any performance inflation can be attributed to duplicate leakage.",
        "",
    ])
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_model_comparison(svm_results):
    if not LR_RESULT_FILE.exists():
        return
    lr_results = pd.read_csv(LR_RESULT_FILE)
    metrics = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "macro_f1",
    ]
    rows = []
    for model_name, results in [
        ("logistic_regression", lr_results),
        ("linear_svm", svm_results),
    ]:
        row = {"condition": "A", "model": model_name}
        for metric in metrics:
            row[f"{metric}_mean"] = results[metric].mean()
            row[f"{metric}_std"] = results[metric].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(COMPARISON_FILE, index=False)


def main():
    data = pd.read_csv(INPUT_FILE)
    required_columns = {"record_id", "combined_text", "label"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    if len(data) != 17_880:
        raise ValueError(f"Expected 17,880 rows, found {len(data):,}")

    normalised_text = data["combined_text"].map(normalise_for_exact_match)
    exact_group_ids = pd.Series(
        pd.factorize(normalised_text, sort=False)[0],
        index=data.index,
    )
    exact_group_sizes = exact_group_ids.value_counts()
    total_exact_groups = int((exact_group_sizes > 1).sum())
    rows_in_exact_groups = int(exact_group_sizes[exact_group_sizes > 1].sum())

    rows = []
    for seed in SEEDS:
        train_indices, validation_indices, holdout_indices = split_indices(
            data["label"], seed
        )
        train_labels = data.loc[train_indices, "label"].to_numpy()
        model = build_calibrated_model(train_labels, seed)
        model.fit(
            data.loc[train_indices, "combined_text"],
            train_labels,
        )

        validation_scores = model.predict_proba(
            data.loc[validation_indices, "combined_text"]
        )[:, 1]
        threshold = select_f1_threshold(
            data.loc[validation_indices, "label"].to_numpy(),
            validation_scores,
        )
        holdout_scores = model.predict_proba(
            data.loc[holdout_indices, "combined_text"]
        )[:, 1]

        result = {
            "condition": CONDITION_NAME,
            "model": "linear_svm",
            "seed": seed,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "holdout_rows": len(holdout_indices),
            "train_fraud": int(data.loc[train_indices, "label"].sum()),
            "validation_fraud": int(
                data.loc[validation_indices, "label"].sum()
            ),
            "holdout_fraud": int(data.loc[holdout_indices, "label"].sum()),
            "threshold": threshold,
        }
        result.update(
            exact_overlap_stats(
                exact_group_ids,
                data["label"],
                train_indices,
                validation_indices,
                holdout_indices,
            )
        )
        result.update(
            evaluate(
                data.loc[holdout_indices, "label"].to_numpy(),
                holdout_scores,
                threshold,
            )
        )
        rows.append(result)
        print(
            f"Seed {seed}: PR-AUC={result['pr_auc']:.4f}, "
            f"fraud F1={result['fraud_f1']:.4f}, "
            f"cross-split exact groups={result['cross_split_exact_groups']}"
        )

    results = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)
    create_summary(
        data,
        results,
        total_exact_groups,
        rows_in_exact_groups,
    )
    save_model_comparison(results)

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Model comparison saved to: {COMPARISON_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
