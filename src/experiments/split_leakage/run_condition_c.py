"""Condition C: exact deduplication and group-aware splitting.

This diagnostic experiment uses the same 15,807-row input as Condition B.
Complete near-duplicate groups are allocated to a 70/15/15 split, so a group
cannot cross Train, Validation and Holdout. Official split files are not read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from run_condition_a import (
    REPORT_DIR,
    SEEDS,
    build_model,
    evaluate,
    select_f1_threshold,
)
from run_condition_b import load_data


PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src/data_pipeline"))
from analyse_split_options import allocate_groups, load_groups  # noqa: E402


RESULT_FILE = REPORT_DIR / "condition_c_lr_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_c_lr_summary.md"
CONDITION_NAME = "C: exact dedup + group-aware split"
RATIOS = [0.70, 0.15, 0.15]


def group_split_indices(data, group_data, seed):
    assignment = allocate_groups(group_data, RATIOS, seed)
    split_names = data["group_id"].map(assignment)
    if split_names.isna().any():
        raise ValueError("At least one group was not assigned to a split")

    train_indices = np.flatnonzero(split_names.eq("train"))
    validation_indices = np.flatnonzero(split_names.eq("validation"))
    holdout_indices = np.flatnonzero(split_names.eq("test"))
    return train_indices, validation_indices, holdout_indices


def split_stats(data, train_indices, validation_indices, holdout_indices):
    index_sets = {
        "train": train_indices,
        "validation": validation_indices,
        "holdout": holdout_indices,
    }
    group_sets = {
        name: set(data.loc[indices, "group_id"])
        for name, indices in index_sets.items()
    }
    cross_split_groups = (
        (group_sets["train"] & group_sets["validation"])
        | (group_sets["train"] & group_sets["holdout"])
        | (group_sets["validation"] & group_sets["holdout"])
    )
    if cross_split_groups:
        raise ValueError("A near-duplicate group crosses diagnostic splits")

    result = {"cross_split_near_groups": 0}
    for name, indices in index_sets.items():
        subset = data.loc[indices]
        fraud_groups = subset.loc[
            subset["label"] == 1, "group_id"
        ].nunique()
        result[f"{name}_groups"] = int(subset["group_id"].nunique())
        result[f"{name}_fraud_groups"] = int(fraud_groups)
        result[f"{name}_largest_group"] = int(
            subset["group_id"].value_counts().max()
        )
    return result


def create_summary(
    data,
    results,
    output_file=SUMMARY_FILE,
    model_title="Logistic Regression",
    model_description="fixed TF-IDF Logistic Regression baseline",
    calibration_note=None,
):
    metrics = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "macro_f1",
    ]
    split_measures = [
        "train_rows",
        "validation_rows",
        "holdout_rows",
        "train_fraud",
        "validation_fraud",
        "holdout_fraud",
        "train_groups",
        "validation_groups",
        "holdout_groups",
        "train_fraud_groups",
        "validation_fraud_groups",
        "holdout_fraud_groups",
    ]

    lines = [
        f"# Split Leakage Experiment: Condition C — {model_title}",
        "",
        "## Design",
        "",
        f"- Condition: **{CONDITION_NAME}**",
        f"- Input rows after exact deduplication: **{len(data):,}**",
        f"- Legitimate rows: **{int((data['label'] == 0).sum()):,}**",
        f"- Fraudulent rows: **{int((data['label'] == 1).sum()):,}**",
        "- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**",
        "- Split rule: complete `group_id` allocation using the shared data-pipeline allocator",
        "- Seeds: **0 to 9**",
        f"- Model: {model_description}",
    ]
    if calibration_note:
        lines.append(f"- Calibration: {calibration_note}")
    lines.extend([
        "- Threshold: selected on each diagnostic Validation partition by maximum fraud F1",
        "- Official Train, Validation and Test files used: **No**",
        "",
        "## Holdout performance across 10 seeds",
        "",
        "| Metric | Mean | Standard deviation | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ])
    for column in metrics:
        values = results[column]
        lines.append(
            f"| {column} | {values.mean():.4f} | {values.std(ddof=1):.4f} | "
            f"{values.min():.4f} | {values.max():.4f} |"
        )

    lines.extend([
        "",
        "## Split composition across 10 seeds",
        "",
        "| Measure | Mean | Minimum | Maximum |",
        "|---|---:|---:|---:|",
    ])
    for column in split_measures:
        values = results[column]
        lines.append(
            f"| {column} | {values.mean():.1f} | "
            f"{int(values.min())} | {int(values.max())} |"
        )

    lines.extend([
        "",
        "## Leakage check",
        "",
        "- Near-duplicate groups crossing splits: **0 for every seed**",
        "",
        "## Interpretation",
        "",
        "Condition C uses the same exact-deduplicated input as Condition B but prevents near-duplicate groups from crossing diagnostic partitions. For Logistic Regression, B-to-C isolates this outer split change. For Linear SVM, Condition C also makes Train-only calibration folds group-aware, so B-to-C represents the complete strict workflow rather than the outer split alone.",
        "",
    ])
    output_file.write_text("\n".join(lines), encoding="utf-8")


def run_model(data, group_data, model_name, model_builder):
    rows = []
    for seed in SEEDS:
        train_indices, validation_indices, holdout_indices = (
            group_split_indices(data, group_data, seed)
        )
        train_labels = data.loc[train_indices, "label"].to_numpy()
        train_groups = data.loc[train_indices, "group_id"].to_numpy()
        model = model_builder(train_labels, train_groups, seed)
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
            "model": model_name,
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
            split_stats(
                data,
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
            f"holdout fraud={result['holdout_fraud']}, "
            "cross-split groups=0"
        )
    return pd.DataFrame(rows)


def main():
    data = load_data()
    group_data, expected_rows = load_groups()
    if expected_rows != len(data):
        raise ValueError("Group allocator input does not match Condition C data")

    results = run_model(
        data,
        group_data,
        "logistic_regression",
        lambda _labels, _groups, seed: build_model(seed),
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)
    create_summary(data, results)

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
