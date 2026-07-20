"""Shared evaluation and visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from final_model_pipelines.structured_output import (
    POSITIVE_LABEL,
    compute_binary_metrics,
    compute_tier_statistics,
    tune_dual_thresholds,
)


def plot_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve_with_thresholds(
    y_true,
    y_prob,
    low_threshold: float,
    high_threshold: float,
    title: str,
    out_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=POSITIVE_LABEL)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label="PR Curve (fraud class)")
    plt.axvline(
        x=_recall_at_threshold(y_true, y_prob, low_threshold),
        color="green",
        linestyle="--",
        label=f"LOW={low_threshold:.2f} (binary fake/real)",
    )
    plt.axvline(
        x=_recall_at_threshold(y_true, y_prob, high_threshold),
        color="red",
        linestyle="--",
        label=f"HIGH={high_threshold:.2f} (Likely Deceptive)",
    )
    plt.xlabel("Recall (Fraud Class)")
    plt.ylabel("Precision (Fraud Class)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _recall_at_threshold(y_true, y_prob, threshold: float) -> float:
    from sklearn.metrics import recall_score

    y_pred = (y_prob >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))


def run_full_evaluation(
    model_display_name: str,
    y_val: np.ndarray,
    prob_val: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
    sample_texts: list[str],
    predict_fn,
    output_dir: Path,
    saved_model_dir: Path,
) -> dict[str, Any]:
    """
    Full evaluation:
    1. Tune risk mapping dual thresholds on validation set
    2. Binary P/R/F1 (primary metric, decision boundary = LOW_THRESHOLD)
    3. Three-tier statistics (supplementary)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_cfg = tune_dual_thresholds(y_val, prob_val)
    low_t = threshold_cfg["low_threshold"]
    high_t = threshold_cfg["high_threshold"]

    (saved_model_dir / "threshold.json").write_text(json.dumps(threshold_cfg, indent=2), encoding="utf-8")
    (output_dir / "threshold_settings.json").write_text(json.dumps(threshold_cfg, indent=2), encoding="utf-8")

    split_results = []
    tier_results = []

    for split_name, y_true, y_prob in [
        ("Validation", y_val, prob_val),
        ("Test", y_test, prob_test),
    ]:
        # Primary metric: binary classification (prob >= LOW → fake)
        binary = compute_binary_metrics(y_true, y_prob, low_t)
        binary["split"] = split_name
        binary["decision_threshold"] = low_t
        binary["high_threshold_risk_mapping"] = high_t
        split_results.append(binary)

        # Supplementary: three-tier risk mapping distribution
        tiers = compute_tier_statistics(y_true, y_prob, low_t, high_t)
        tiers["split"] = split_name
        tier_results.append(tiers)

        y_pred = (y_prob >= low_t).astype(int)
        slug = split_name.lower()
        plot_confusion_matrix(
            y_true,
            y_pred,
            f"{model_display_name} Binary Confusion Matrix ({split_name})",
            output_dir / f"confusion_matrix_{slug}.png",
        )
        plot_pr_curve_with_thresholds(
            y_true,
            y_prob,
            low_t,
            high_t,
            f"{model_display_name} PR Curve ({split_name})",
            output_dir / f"pr_curve_{slug}.png",
        )
        report = classification_report(y_true, y_pred, target_names=["real", "fake"])
        (output_dir / f"classification_report_{slug}.txt").write_text(report, encoding="utf-8")

    pd.DataFrame(split_results).to_csv(output_dir / "evaluation_results.csv", index=False)
    (output_dir / "tier_statistics.json").write_text(
        json.dumps(tier_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    examples = [predict_fn(text) for text in sample_texts]
    (output_dir / "structured_prediction_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "model": model_display_name,
        "evaluation_type": "binary_classification_primary",
        "risk_mapping_thresholds": threshold_cfg,
        "binary_metrics": split_results,
        "tier_statistics_supplementary": tier_results,
        "structured_examples": examples,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def load_thresholds(saved_model_dir: Path) -> tuple[float, float]:
    path = saved_model_dir / "threshold.json"
    if not path.exists():
        return 0.35, 0.65
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return float(cfg["low_threshold"]), float(cfg["high_threshold"])
