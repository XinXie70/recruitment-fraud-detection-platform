"""Create paper-ready figures for the A/B/C split-leakage experiment."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "independent_ml_workflow_matplotlib"),
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_DIR / "reports/experiments/split_leakage"
FIGURE_DIR = REPORT_DIR / "figures"
CONDITIONS = ["A", "B", "C"]
MODELS = {
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
}
COLORS = {
    "Logistic Regression": "#0072B2",
    "Linear SVM": "#D55E00",
}


def load_results():
    results = {}
    for short_name, model_name in MODELS.items():
        for condition in CONDITIONS:
            file = REPORT_DIR / (
                f"condition_{condition.lower()}_{short_name}_results.csv"
            )
            results[(model_name, condition)] = pd.read_csv(file)
    return results


def save_figure(figure, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_mean_performance(results):
    metrics = [
        ("pr_auc", "PR-AUC"),
        ("fraud_f1", "Fraud F1"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(CONDITIONS))

    for axis, (column, label) in zip(axes, metrics):
        for model_name in MODELS.values():
            means = [
                results[(model_name, condition)][column].mean()
                for condition in CONDITIONS
            ]
            standard_deviations = [
                results[(model_name, condition)][column].std(ddof=1)
                for condition in CONDITIONS
            ]
            axis.errorbar(
                x,
                means,
                yerr=standard_deviations,
                marker="o",
                linewidth=2,
                capsize=4,
                label=model_name,
                color=COLORS[model_name],
            )
            for position, value in zip(x, means):
                axis.annotate(
                    f"{value:.3f}",
                    (position, value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        axis.set_xticks(x, CONDITIONS)
        axis.set_xlabel("Experimental condition")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0.70 if column == "fraud_f1" else 0.82, 0.97)

    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle("Mean holdout performance across 10 random seeds")
    figure.text(
        0.5,
        -0.01,
        "Error bars show ±1 standard deviation. A: no dedup/random; "
        "B: exact dedup/random; C: exact dedup/group-aware.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    save_figure(figure, "figure_1_mean_performance")


def plot_seed_variability(results):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(42)

    for axis, model_name in zip(axes, MODELS.values()):
        values = [
            results[(model_name, condition)]["fraud_f1"].to_numpy()
            for condition in CONDITIONS
        ]
        boxes = axis.boxplot(
            values,
            tick_labels=CONDITIONS,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(COLORS[model_name])
            box.set_alpha(0.28)
            box.set_edgecolor(COLORS[model_name])

        for position, condition_values in enumerate(values, start=1):
            jitter = rng.normal(0, 0.045, len(condition_values))
            axis.scatter(
                position + jitter,
                condition_values,
                s=24,
                color=COLORS[model_name],
                alpha=0.8,
                edgecolors="none",
            )

        axis.set_title(model_name)
        axis.set_xlabel("Experimental condition")
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Fraud F1")
    axes[0].set_ylim(0.65, 0.93)
    figure.suptitle("Fraud F1 variation across 10 random seeds")
    figure.text(
        0.5,
        -0.01,
        "Each point is one diagnostic Holdout result.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    save_figure(figure, "figure_2_seed_variability")


def plot_contamination(results):
    a = results[("Logistic Regression", "A")]
    b = results[("Logistic Regression", "B")]
    all_holdout = [
        100 * a["holdout_rows_matching_train"].mean()
        / a["holdout_rows"].mean(),
        100 * b["holdout_rows_group_matching_train"].mean()
        / b["holdout_rows"].mean(),
        0,
    ]
    fraud_holdout = [
        100 * a["holdout_fraud_rows_matching_train"].mean()
        / a["holdout_fraud"].mean(),
        100 * b["holdout_fraud_rows_group_matching_train"].mean()
        / b["holdout_fraud"].mean(),
        0,
    ]

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(CONDITIONS))
    width = 0.34
    bars_all = axis.bar(
        x - width / 2,
        all_holdout,
        width,
        label="All Holdout rows",
        color="#0072B2",
    )
    bars_fraud = axis.bar(
        x + width / 2,
        fraud_holdout,
        width,
        label="Fraud Holdout rows",
        color="#D55E00",
    )
    axis.bar_label(bars_all, fmt="%.1f%%", padding=3, fontsize=9)
    axis.bar_label(bars_fraud, fmt="%.1f%%", padding=3, fontsize=9)
    axis.set_xticks(x, CONDITIONS)
    axis.set_ylabel("Holdout rows with a matching Train group (%)")
    axis.set_xlabel("Experimental condition")
    axis.set_ylim(0, max(fraud_holdout) + 5)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Train–Holdout duplicate contamination")
    figure.text(
        0.5,
        -0.02,
        "A counts exact-text matches; B counts near-duplicate group matches; "
        "C prevents all group overlap.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    save_figure(figure, "figure_3_duplicate_contamination")


def save_summary_table(results):
    metrics = [
        "pr_auc",
        "roc_auc",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "macro_f1",
    ]
    rows = []
    for model_name in MODELS.values():
        for condition in CONDITIONS:
            data = results[(model_name, condition)]
            row = {"model": model_name, "condition": condition}
            for metric in metrics:
                row[f"{metric}_mean"] = data[metric].mean()
                row[f"{metric}_std"] = data[metric].std(ddof=1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        REPORT_DIR / "paper_results_table.csv",
        index=False,
    )


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })
    results = load_results()
    plot_mean_performance(results)
    plot_seed_variability(results)
    plot_contamination(results)
    save_summary_table(results)
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Table saved to: {REPORT_DIR / 'paper_results_table.csv'}")


if __name__ == "__main__":
    main()
