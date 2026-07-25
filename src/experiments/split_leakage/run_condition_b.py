"""Condition B: exact deduplication and ordinary stratified random splitting.

This diagnostic experiment uses the 15,807-row exact-deduplicated dataset but
deliberately ignores near-duplicate group_id values during the 70/15/15 split.
The official split files are not read or modified.
"""

from pathlib import Path

import pandas as pd

from run_condition_a import (
    REPORT_DIR,
    SEEDS,
    build_model,
    evaluate,
    normalise_for_exact_match,
    select_f1_threshold,
    split_indices,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz"
)
RESULT_FILE = REPORT_DIR / "condition_b_lr_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_b_lr_summary.md"
CONDITION_NAME = "B: exact dedup + stratified random split"


def near_overlap_stats(
    group_ids,
    labels,
    train_indices,
    validation_indices,
    holdout_indices,
):
    """Measure near-duplicate group overlap caused by ignoring group_id."""
    train_groups = set(group_ids.iloc[train_indices])
    validation_groups = set(group_ids.iloc[validation_indices])
    holdout_groups = set(group_ids.iloc[holdout_indices])

    assignments = pd.DataFrame({
        "group_id": group_ids,
        "label": labels,
        "split": "",
    })
    assignments.loc[train_indices, "split"] = "train"
    assignments.loc[validation_indices, "split"] = "validation"
    assignments.loc[holdout_indices, "split"] = "holdout"

    grouped = assignments.groupby("group_id").agg(
        size=("split", "size"),
        split_count=("split", "nunique"),
        label=("label", "first"),
        label_count=("label", "nunique"),
    )
    if grouped["label_count"].max() != 1:
        raise ValueError("A near-duplicate group contains conflicting labels")

    crossing = grouped[
        (grouped["size"] > 1) & (grouped["split_count"] > 1)
    ]
    validation_matches_train = group_ids.iloc[validation_indices].isin(
        train_groups
    )
    holdout_matches_train = group_ids.iloc[holdout_indices].isin(train_groups)
    validation_labels = labels.iloc[validation_indices]
    holdout_labels = labels.iloc[holdout_indices]

    return {
        "cross_split_near_groups": int(len(crossing)),
        "cross_split_fraud_near_groups": int(
            (crossing["label"] == 1).sum()
        ),
        "cross_split_legitimate_near_groups": int(
            (crossing["label"] == 0).sum()
        ),
        "rows_in_cross_split_near_groups": int(crossing["size"].sum()),
        "train_validation_near_groups": int(
            len(train_groups & validation_groups)
        ),
        "train_holdout_near_groups": int(len(train_groups & holdout_groups)),
        "validation_holdout_near_groups": int(
            len(validation_groups & holdout_groups)
        ),
        "validation_rows_group_matching_train": int(
            validation_matches_train.sum()
        ),
        "validation_fraud_rows_group_matching_train": int(
            (validation_matches_train & (validation_labels == 1)).sum()
        ),
        "validation_legitimate_rows_group_matching_train": int(
            (validation_matches_train & (validation_labels == 0)).sum()
        ),
        "holdout_rows_group_matching_train": int(
            holdout_matches_train.sum()
        ),
        "holdout_fraud_rows_group_matching_train": int(
            (holdout_matches_train & (holdout_labels == 1)).sum()
        ),
        "holdout_legitimate_rows_group_matching_train": int(
            (holdout_matches_train & (holdout_labels == 0)).sum()
        ),
    }


def create_summary(
    data,
    results,
    multirow_groups,
    rows_in_multirow_groups,
    output_file=SUMMARY_FILE,
    model_title="Logistic Regression",
    model_description="fixed TF-IDF Logistic Regression baseline",
    calibration_note=None,
):
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
        "cross_split_near_groups",
        "cross_split_fraud_near_groups",
        "cross_split_legitimate_near_groups",
        "rows_in_cross_split_near_groups",
        "train_validation_near_groups",
        "train_holdout_near_groups",
        "validation_holdout_near_groups",
        "validation_rows_group_matching_train",
        "validation_fraud_rows_group_matching_train",
        "validation_legitimate_rows_group_matching_train",
        "holdout_rows_group_matching_train",
        "holdout_fraud_rows_group_matching_train",
        "holdout_legitimate_rows_group_matching_train",
    ]

    lines = [
        f"# Split Leakage Experiment: Condition B — {model_title}",
        "",
        "## Design",
        "",
        f"- Condition: **{CONDITION_NAME}**",
        f"- Input rows after exact deduplication: **{len(data):,}**",
        f"- Legitimate rows: **{int((data['label'] == 0).sum()):,}**",
        f"- Fraudulent rows: **{int((data['label'] == 1).sum()):,}**",
        f"- Multi-row near-duplicate groups: **{multirow_groups:,}**",
        f"- Rows in multi-row near-duplicate groups: **{rows_in_multirow_groups:,}**",
        "- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**",
        "- Split rule: ordinary label stratification; `group_id` deliberately ignored",
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
    for column in metric_columns:
        values = results[column]
        lines.append(
            f"| {column} | {values.mean():.4f} | {values.std(ddof=1):.4f} | "
            f"{values.min():.4f} | {values.max():.4f} |"
        )

    lines.extend([
        "",
        "## Near-duplicate contamination across 10 seeds",
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
        "Condition B removes exact duplicate rows, but intentionally permits near-duplicate groups to cross diagnostic partitions. Conditions A, B and C must be compared before drawing conclusions about leakage-related performance inflation.",
        "",
    ])
    output_file.write_text("\n".join(lines), encoding="utf-8")


def load_data():
    data = pd.read_csv(INPUT_FILE).reset_index(drop=True)
    required = {"record_id", "combined_text", "label", "group_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Grouped data is missing columns: {sorted(missing)}")
    if len(data) != 15_807:
        raise ValueError(f"Expected 15,807 rows, found {len(data):,}")
    if data["record_id"].duplicated().any():
        raise ValueError("record_id is not unique")
    if data["combined_text"].isna().any():
        raise ValueError("combined_text contains missing values")
    exact_keys = data["combined_text"].map(normalise_for_exact_match)
    if exact_keys.duplicated().any():
        raise ValueError("Condition B input still contains exact duplicates")
    return data


def run_model(data, model_name, model_builder):
    group_sizes = data["group_id"].value_counts()
    multirow_groups = int((group_sizes > 1).sum())
    rows_in_multirow_groups = int(group_sizes[group_sizes > 1].sum())
    rows = []

    for seed in SEEDS:
        train_indices, validation_indices, holdout_indices = split_indices(
            data["label"], seed
        )
        model = model_builder(data.loc[train_indices, "label"].to_numpy(), seed)
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
            near_overlap_stats(
                data["group_id"],
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
            f"cross-split near groups={result['cross_split_near_groups']}"
        )

    return (
        pd.DataFrame(rows),
        multirow_groups,
        rows_in_multirow_groups,
    )


def main():
    data = load_data()
    results, multirow_groups, rows_in_multirow_groups = run_model(
        data,
        "logistic_regression",
        lambda _labels, seed: build_model(seed),
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)
    create_summary(
        data,
        results,
        multirow_groups,
        rows_in_multirow_groups,
    )

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
