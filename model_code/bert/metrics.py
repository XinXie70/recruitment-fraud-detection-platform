"""Evaluation metrics, threshold search, plots, and result tables.

Primary selection metric: validation fraud-class F1 (not accuracy alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute a full metric suite focused on the fraud class (label=1)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "legitimate_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
            output_dict=True,
        ),
    }

    # ROC / PR need both classes present in y_true for a meaningful score.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Evaluate fraud metrics for thresholds in [0.05, 0.95]."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for thr in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append(
            {
                "threshold": float(thr),
                "fraud_precision": m["fraud_precision"],
                "fraud_recall": m["fraud_recall"],
                "fraud_f1": m["fraud_f1"],
                "false_positive": m["confusion_matrix"]["fp"],
                "false_negative": m["confusion_matrix"]["fn"],
                "tn": m["confusion_matrix"]["tn"],
                "tp": m["confusion_matrix"]["tp"],
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    mode: str = "max_fraud_f1",
    min_fraud_recall: Optional[float] = None,
) -> Tuple[float, pd.DataFrame]:
    """Choose a classification threshold on the VALIDATION set only.

    Modes:
      - max_fraud_f1: threshold with highest fraud F1
      - min_recall_then_precision: among thresholds meeting min fraud recall,
        pick highest fraud precision (falls back to max fraud F1 if none qualify)
    """
    sweep = threshold_sweep(y_true, y_prob)
    if mode == "min_recall_then_precision" and min_fraud_recall is not None:
        eligible = sweep[sweep["fraud_recall"] >= float(min_fraud_recall)]
        if len(eligible) > 0:
            best = eligible.sort_values(
                ["fraud_precision", "fraud_f1", "fraud_recall"],
                ascending=False,
            ).iloc[0]
            return float(best["threshold"]), sweep

    best = sweep.sort_values(
        ["fraud_f1", "fraud_recall", "fraud_precision"],
        ascending=False,
    ).iloc[0]
    return float(best["threshold"]), sweep


def build_predictions_frame(
    *,
    record_ids: Sequence[str],
    titles: Sequence[str],
    companies: Sequence[str],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    model_name: str,
    imbalance_strategy: str,
    row_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    if row_indices is None:
        row_indices = list(range(n))
    return pd.DataFrame(
        {
            "sample_index": list(row_indices),
            "original_id": list(record_ids),
            "company": list(companies),
            "title": list(titles),
            "true_label": y_true,
            "predicted_label": y_pred,
            "legitimate_probability": 1.0 - y_prob,
            "fraud_probability": y_prob,
            "threshold": threshold,
            "correct": (y_true == y_pred).astype(int),
            "model_name": model_name,
            "imbalance_strategy": imbalance_strategy,
        }
    )


def build_error_analysis(
    preds: pd.DataFrame,
    texts: Sequence[str],
    *,
    high_conf: float = 0.85,
    near_margin: float = 0.05,
) -> pd.DataFrame:
    """Flag FP / FN / high-confidence errors / near-threshold samples."""
    df = preds.copy()
    df["text_preview"] = [str(t)[:300].replace("\n", " ") for t in texts]
    thr = float(df["threshold"].iloc[0]) if len(df) else 0.5

    def error_type(row) -> str:
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            base = "false_positive"
        elif row["true_label"] == 1 and row["predicted_label"] == 0:
            base = "false_negative"
        else:
            base = "correct"
        extras = []
        if base != "correct" and (
            (row["predicted_label"] == 1 and row["fraud_probability"] >= high_conf)
            or (row["predicted_label"] == 0 and row["fraud_probability"] <= 1 - high_conf)
        ):
            extras.append("high_confidence_error")
        if abs(row["fraud_probability"] - thr) <= near_margin:
            extras.append("near_threshold")
        if extras:
            return base + "|" + "|".join(extras)
        return base

    df["error_type"] = df.apply(error_type, axis=1)
    df = df.rename(
        columns={
            "original_id": "sample_id",
            "true_label": "original_label",
        }
    )
    cols = [
        "sample_id",
        "title",
        "company",
        "original_label",
        "predicted_label",
        "fraud_probability",
        "threshold",
        "error_type",
        "text_preview",
    ]
    return df[cols]


def save_comparison_row(
    path: Path,
    row: Dict[str, Any],
) -> pd.DataFrame:
    """Append / upsert one model row into model_comparison.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Model",
        "Imbalance method",
        "Accuracy",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1",
        "Macro F1",
        "ROC-AUC",
        "PR-AUC",
        "Inference time (s)",
        "Notes",
    ]
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = pd.DataFrame(columns=cols)

    key_model = row.get("Model")
    key_imb = row.get("Imbalance method")
    mask = (table["Model"] == key_model) & (table["Imbalance method"] == key_imb)
    for c in cols:
        if c not in table.columns:
            table[c] = np.nan
    if mask.any():
        for k, v in row.items():
            table.loc[mask, k] = v
    else:
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def plot_training_curves(history: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "train_loss" in history.columns:
        ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: Dict[str, Any], out_path: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = np.asarray(cm["matrix"] if isinstance(cm, dict) and "matrix" in cm else cm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    roc_path: Path,
    pr_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_true, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_prob):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)


def plot_class_distribution(splits: Dict[str, pd.DataFrame], out_path: Path, label_col: str = "label") -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names, legit, fraud = [], [], []
    for name, df in splits.items():
        names.append(name)
        legit.append(int((df[label_col] == 0).sum()))
        fraud.append(int((df[label_col] == 1).sum()))
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, legit, width, label="Legitimate")
    ax.bar(x + width / 2, fraud, width, label="Fraudulent")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if df.empty:
        return
    # Drop unavailable rows from bar chart values but keep labels.
    plot_df = df.copy()
    for col in ["Fraud Recall", "Fraud F1", "PR-AUC"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    labels = [
        f"{m}\n({i})" for m, i in zip(plot_df["Model"], plot_df["Imbalance method"])
    ]
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 5))
    ax.bar(x - width, plot_df["Fraud Recall"], width, label="Fraud Recall")
    ax.bar(x, plot_df["Fraud F1"], width, label="Fraud F1")
    ax.bar(x + width, plot_df["PR-AUC"], width, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (fraud-focused)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
