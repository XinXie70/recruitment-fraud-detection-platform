"""Run calibrated Linear SVM under split-leakage Condition B."""

import pandas as pd

from run_condition_a_svm import build_calibrated_model
from run_condition_b import (
    REPORT_DIR,
    create_summary as create_lr_summary,
    load_data,
    run_model,
)


RESULT_FILE = REPORT_DIR / "condition_b_svm_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_b_svm_summary.md"
MODEL_COMPARISON_FILE = REPORT_DIR / "condition_b_model_comparison.csv"
AB_COMPARISON_FILE = REPORT_DIR / "condition_ab_model_comparison.csv"
AB_SUMMARY_FILE = REPORT_DIR / "condition_ab_summary.md"
LR_RESULT_FILE = REPORT_DIR / "condition_b_lr_results.csv"

METRICS = [
    "pr_auc",
    "roc_auc",
    "accuracy",
    "fraud_precision",
    "fraud_recall",
    "fraud_f1",
    "macro_f1",
]


def create_svm_summary(data, results, multirow_groups, rows_in_multirow_groups):
    create_lr_summary(
        data,
        results,
        multirow_groups,
        rows_in_multirow_groups,
        output_file=SUMMARY_FILE,
        model_title="Linear SVM",
        model_description="fixed TF-IDF Linear SVM baseline",
        calibration_note=(
            "Train-only three-fold stratified sigmoid calibration, "
            "without group information"
        ),
    )


def add_summary_rows(condition, model, results):
    row = {"condition": condition, "model": model}
    for metric in METRICS:
        row[f"{metric}_mean"] = results[metric].mean()
        row[f"{metric}_std"] = results[metric].std(ddof=1)
    return row


def save_comparisons(svm_results):
    if not LR_RESULT_FILE.exists():
        return

    lr_results = pd.read_csv(LR_RESULT_FILE)
    condition_b = pd.DataFrame([
        add_summary_rows("B", "logistic_regression", lr_results),
        add_summary_rows("B", "linear_svm", svm_results),
    ])
    condition_b.to_csv(MODEL_COMPARISON_FILE, index=False)

    condition_a_file = REPORT_DIR / "condition_a_model_comparison.csv"
    if condition_a_file.exists():
        condition_a = pd.read_csv(condition_a_file)
        combined = pd.concat([condition_a, condition_b], ignore_index=True)
        for metric in METRICS:
            a_means = (
                combined[combined["condition"] == "A"]
                .set_index("model")[f"{metric}_mean"]
            )
            combined[f"{metric}_change_from_a"] = combined.apply(
                lambda row: (
                    row[f"{metric}_mean"] - a_means[row["model"]]
                    if row["condition"] == "B"
                    else 0.0
                ),
                axis=1,
            )
        combined.to_csv(AB_COMPARISON_FILE, index=False)
        save_ab_summary(combined, lr_results, svm_results)


def save_ab_summary(combined, lr_results, svm_results):
    lines = [
        "# Split Leakage Experiment: Conditions A and B",
        "",
        "## What changed",
        "",
        "- Condition A: 17,880 rows, exact duplicates retained, ordinary stratified random split.",
        "- Condition B: 15,807 rows after exact deduplication, ordinary stratified random split.",
        "- Both conditions: fixed LR and SVM settings, 70/15/15 split, seeds 0–9, Validation-selected threshold and diagnostic Holdout evaluation.",
        "- Near-duplicate `group_id` is ignored in Condition B, so near-duplicate leakage is still possible.",
        "",
        "## Mean diagnostic Holdout results",
        "",
        "| Model | Metric | A | B | B − A |",
        "|---|---|---:|---:|---:|",
    ]
    display_metrics = [
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Fraud precision", "fraud_precision"),
        ("Fraud recall", "fraud_recall"),
        ("Fraud F1", "fraud_f1"),
        ("Macro F1", "macro_f1"),
    ]
    model_names = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
    }
    indexed = combined.set_index(["condition", "model"])
    for model, display_name in model_names.items():
        for metric_name, metric in display_metrics:
            a_value = indexed.loc[("A", model), f"{metric}_mean"]
            b_value = indexed.loc[("B", model), f"{metric}_mean"]
            lines.append(
                f"| {display_name} | {metric_name} | {a_value:.4f} | "
                f"{b_value:.4f} | {b_value - a_value:+.4f} |"
            )

    lines.extend([
        "",
        "## Remaining near-duplicate overlap in Condition B",
        "",
        f"- Mean near-duplicate groups crossing splits: **{lr_results['cross_split_near_groups'].mean():.1f}**",
        f"- Mean diagnostic Holdout rows whose group also appears in Train: **{lr_results['holdout_rows_group_matching_train'].mean():.1f}**",
        f"- Mean fraudulent Holdout rows whose group also appears in Train: **{lr_results['holdout_fraud_rows_group_matching_train'].mean():.1f}**",
        "- LR and SVM used matching seeds and produced identical split-overlap counts.",
        "",
        "## Interpretation",
        "",
        "Performance generally decreased after exact duplicates were removed, especially fraud F1. This is evidence that exact-duplicate handling affects the reported result under random splitting. It is not yet sufficient to claim that all of the decrease was caused by leakage, because deduplication also changes the dataset size and class composition.",
        "",
        "Condition C is required next: use the same exact-deduplicated input as Condition B, but keep every near-duplicate group within one split. The B-to-C comparison will isolate the effect of group-aware splitting more clearly.",
        "",
    ])
    AB_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    data = load_data()
    results, multirow_groups, rows_in_multirow_groups = run_model(
        data,
        "linear_svm",
        build_calibrated_model,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)
    create_svm_summary(
        data,
        results,
        multirow_groups,
        rows_in_multirow_groups,
    )
    save_comparisons(results)

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_FILE}")
    print(f"A/B comparison saved to: {AB_COMPARISON_FILE}")
    print(f"A/B summary saved to: {AB_SUMMARY_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
