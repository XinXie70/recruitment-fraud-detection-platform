"""Run group-aware calibrated Linear SVM under Condition C."""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from run_condition_a import REPORT_DIR
from run_condition_b import load_data
from run_condition_c import build_group_data, create_summary, run_model


RESULT_FILE = REPORT_DIR / "condition_c_svm_results.csv"
SUMMARY_FILE = REPORT_DIR / "condition_c_svm_summary.md"
MODEL_COMPARISON_FILE = REPORT_DIR / "condition_c_model_comparison.csv"
ABC_COMPARISON_FILE = REPORT_DIR / "condition_abc_model_comparison.csv"
ABC_SUMMARY_FILE = REPORT_DIR / "condition_abc_summary.md"
LR_RESULT_FILE = REPORT_DIR / "condition_c_lr_results.csv"

METRICS = [
    "pr_auc",
    "roc_auc",
    "accuracy",
    "fraud_precision",
    "fraud_recall",
    "fraud_f1",
    "macro_f1",
]


def build_group_calibrated_model(train_labels, train_groups, seed):
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
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )
    calibration_folds = list(
        splitter.split(
            range(len(train_labels)),
            train_labels,
            train_groups,
        )
    )
    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_folds,
        n_jobs=1,
        ensemble=False,
    )


def summary_row(condition, model, results):
    row = {"condition": condition, "model": model}
    for metric in METRICS:
        row[f"{metric}_mean"] = results[metric].mean()
        row[f"{metric}_std"] = results[metric].std(ddof=1)
    return row


def save_comparisons(svm_results):
    if not LR_RESULT_FILE.exists():
        return
    lr_results = pd.read_csv(LR_RESULT_FILE)
    condition_c = pd.DataFrame([
        summary_row("C", "logistic_regression", lr_results),
        summary_row("C", "linear_svm", svm_results),
    ])
    condition_c.to_csv(MODEL_COMPARISON_FILE, index=False)

    rows = []
    for condition in ("A", "B", "C"):
        file = REPORT_DIR / f"condition_{condition.lower()}_model_comparison.csv"
        if file.exists():
            rows.append(pd.read_csv(file))
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(ABC_COMPARISON_FILE, index=False)
    save_abc_summary(combined, lr_results)


def save_abc_summary(combined, c_lr_results):
    indexed = combined.set_index(["condition", "model"])
    model_names = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
    }
    display_metrics = [
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Fraud precision", "fraud_precision"),
        ("Fraud recall", "fraud_recall"),
        ("Fraud F1", "fraud_f1"),
        ("Macro F1", "macro_f1"),
    ]

    lines = [
        "# Split Leakage Experiment: Conditions A, B and C",
        "",
        "## Conditions",
        "",
        "- A: no deduplication + ordinary stratified random split.",
        "- B: exact deduplication + ordinary stratified random split.",
        "- C: exact deduplication + group-aware split.",
        "- All results are means over seeds 0–9 on diagnostic Holdout partitions.",
        "",
        "## Mean diagnostic Holdout results",
        "",
        "| Model | Metric | A | B | C | B − A | C − B |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, model_name in model_names.items():
        for metric_name, metric in display_metrics:
            values = {
                condition: indexed.loc[
                    (condition, model), f"{metric}_mean"
                ]
                for condition in ("A", "B", "C")
            }
            lines.append(
                f"| {model_name} | {metric_name} | {values['A']:.4f} | "
                f"{values['B']:.4f} | {values['C']:.4f} | "
                f"{values['B'] - values['A']:+.4f} | "
                f"{values['C'] - values['B']:+.4f} |"
            )

    lines.extend([
        "",
        "## Leakage checks",
        "",
        "- Condition A deliberately allows exact duplicates to cross splits.",
        "- Condition B removes exact duplicates but still allows near-duplicate groups to cross splits.",
        "- Condition C had **0 groups crossing splits for every seed**.",
        "- In Condition C, Linear SVM calibration folds are also group-aware; therefore its B-to-C difference measures the complete strict workflow, not only the outer split.",
        f"- Condition C Holdout fraud count range: **{int(c_lr_results['holdout_fraud'].min())}–{int(c_lr_results['holdout_fraud'].max())}**.",
        "",
        "## Interpretation rule",
        "",
        "A-to-B shows the effect of exact deduplication under random splitting. For Logistic Regression, B-to-C isolates the outer group-aware split because both conditions use the same input and training procedure. For Linear SVM, B-to-C also includes group-aware calibration in Condition C and should be described as an end-to-end strict-workflow comparison. These results identify leakage risk, but they do not prove how any external paper performed its split.",
        "",
    ])
    ABC_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    data = load_data()
    group_data = build_group_data(data)

    results = run_model(
        data,
        group_data,
        "linear_svm",
        build_group_calibrated_model,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)
    create_summary(
        data,
        results,
        output_file=SUMMARY_FILE,
        model_title="Linear SVM",
        model_description="fixed TF-IDF Linear SVM baseline",
        calibration_note=(
            "Train-only three-fold StratifiedGroupKFold sigmoid calibration"
        ),
    )
    save_comparisons(results)

    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_FILE}")
    print(f"A/B/C comparison saved to: {ABC_COMPARISON_FILE}")
    print(f"A/B/C summary saved to: {ABC_SUMMARY_FILE}")
    print("Official split files used: No")


if __name__ == "__main__":
    main()
