"""Condition A: no deduplication and ordinary stratified random splitting.

This diagnostic experiment uses the processed 17,880-row dataset. For each
seed, it creates a random 70/15/15 Train/Validation/Holdout allocation, fits the
same fixed Logistic Regression configuration, selects a binary threshold on the
diagnostic Validation partition, and evaluates the diagnostic Holdout.

The official data/splits files are not read or modified.
"""

from pathlib import Path
import re

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "emscad_condition_a_no_dedup_input_v1.csv.gz"
)
REPORT_DIR = PROJECT_DIR / "reports/experiments/split_leakage"
RESULT_FILE = REPORT_DIR / "condition_a_lr_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_a_lr_summary.md"

SEEDS = range(10)
CONDITION_NAME = "A: no dedup + stratified random split"


def normalise_for_exact_match(text):
    """Use the same exact-duplicate rule as group_duplicates.py."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def select_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_f1 = np.max(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    return float(thresholds[int(best_indices[0])])


def build_model(seed):
    """Use the fixed best LR configuration from the shared baseline."""
    return Pipeline([
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
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1_000,
                random_state=seed,
            ),
        ),
    ])


def evaluate(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "accuracy": accuracy_score(labels, predictions),
        "fraud_precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "fraud_recall": recall_score(labels, predictions, zero_division=0),
        "fraud_f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def exact_overlap_stats(
    exact_keys,
    labels,
    train_indices,
    validation_indices,
    holdout_indices,
):
    train_keys = set(exact_keys.iloc[train_indices])
    validation_keys = set(exact_keys.iloc[validation_indices])
    holdout_keys = set(exact_keys.iloc[holdout_indices])

    assignments = pd.DataFrame({
        "exact_key": exact_keys,
        "label": labels,
        "split": "",
    })
    assignments.loc[train_indices, "split"] = "train"
    assignments.loc[validation_indices, "split"] = "validation"
    assignments.loc[holdout_indices, "split"] = "holdout"

    grouped = assignments.groupby("exact_key").agg(
        size=("split", "size"),
        split_count=("split", "nunique"),
        label=("label", "first"),
        label_count=("label", "nunique"),
    )
    if grouped["label_count"].max() != 1:
        raise ValueError("An exact duplicate group contains conflicting labels")
    cross_split_groups = grouped[
        (grouped["size"] > 1) & (grouped["split_count"] > 1)
    ]

    validation_matches_train = exact_keys.iloc[validation_indices].isin(train_keys)
    holdout_matches_train = exact_keys.iloc[holdout_indices].isin(train_keys)
    validation_labels = labels.iloc[validation_indices]
    holdout_labels = labels.iloc[holdout_indices]

    return {
        "cross_split_exact_groups": int(len(cross_split_groups)),
        "cross_split_fraud_exact_groups": int(
            (cross_split_groups["label"] == 1).sum()
        ),
        "cross_split_legitimate_exact_groups": int(
            (cross_split_groups["label"] == 0).sum()
        ),
        "rows_in_cross_split_exact_groups": int(
            cross_split_groups["size"].sum()
        ),
        "train_validation_exact_groups": int(
            len(train_keys & validation_keys)
        ),
        "train_holdout_exact_groups": int(len(train_keys & holdout_keys)),
        "validation_holdout_exact_groups": int(
            len(validation_keys & holdout_keys)
        ),
        "validation_rows_matching_train": int(validation_matches_train.sum()),
        "validation_fraud_rows_matching_train": int(
            (validation_matches_train & (validation_labels == 1)).sum()
        ),
        "validation_legitimate_rows_matching_train": int(
            (validation_matches_train & (validation_labels == 0)).sum()
        ),
        "holdout_rows_matching_train": int(holdout_matches_train.sum()),
        "holdout_fraud_rows_matching_train": int(
            (holdout_matches_train & (holdout_labels == 1)).sum()
        ),
        "holdout_legitimate_rows_matching_train": int(
            (holdout_matches_train & (holdout_labels == 0)).sum()
        ),
    }


def split_indices(labels, seed):
    all_indices = np.arange(len(labels))
    train_indices, remaining_indices = train_test_split(
        all_indices,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    validation_indices, holdout_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        stratify=labels.iloc[remaining_indices],
        random_state=seed,
    )
    return train_indices, validation_indices, holdout_indices


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
        "# Split Leakage Experiment: Condition A — Logistic Regression",
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
        "- Model: fixed TF-IDF Logistic Regression baseline",
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
        "Condition A intentionally permits exact duplicate advertisements to cross diagnostic partitions. Its performance must not yet be interpreted as leakage inflation until it is compared with Conditions B and C under the controlled experiment.",
        "",
    ])
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    data = pd.read_csv(INPUT_FILE)
    required_columns = {"record_id", "combined_text", "label"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    if len(data) != 17_880:
        raise ValueError(f"Expected 17,880 rows, found {len(data):,}")
    if data["record_id"].duplicated().any():
        raise ValueError("record_id is not unique")
    if data["combined_text"].isna().any():
        raise ValueError("combined_text contains missing values")

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

        model = build_model(seed)
        model.fit(
            data.loc[train_indices, "combined_text"],
            data.loc[train_indices, "label"],
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

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
