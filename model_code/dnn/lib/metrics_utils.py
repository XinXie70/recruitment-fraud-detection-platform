"""
评估指标、阈值搜索与结果汇总。

AUC 指标必须基于预测概率计算；Precision/Recall/F1 基于阈值二分类预测。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as cfg


# 与评估报告要求一致的核心指标（含混淆矩阵）
CORE_METRIC_SPECS: List[Tuple[str, str]] = [
    ("fraud_precision", "Fraud Precision"),
    ("fraud_recall", "Fraud Recall"),
    ("fraud_f1", "Fraud F1"),
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("accuracy", "Accuracy"),
]

CORE_METRIC_KEYS: List[str] = [key for key, _ in CORE_METRIC_SPECS]


def confusion_matrix_to_dataframe(
    metrics: Dict[str, float],
    context: str = "",
) -> pd.DataFrame:
    """将 tn/fp/fn/tp 转为标准混淆矩阵 CSV 行。"""
    row: Dict[str, object] = {
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }
    if context:
        row = {"context": context, **row}
    return pd.DataFrame(
        [
            {
                **row,
                "matrix": f"[[{int(metrics['tn'])} {int(metrics['fp'])}] "
                f"[{int(metrics['fn'])} {int(metrics['tp'])}]]",
            }
        ]
    )


def core_metrics_to_dataframe(
    metrics: Dict[str, float],
    context: str = "",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """导出 7 项标量指标（Precision/Recall/F1/PR-AUC/ROC-AUC/BalAcc/Accuracy）。"""
    row: Dict[str, object] = {}
    if context:
        row["context"] = context
    if threshold is not None:
        row["threshold"] = float(threshold)
    for key in CORE_METRIC_KEYS:
        row[key] = float(metrics[key])
    return pd.DataFrame([row])


def print_evaluation_metrics(
    title: str,
    metrics: Dict[str, float],
    threshold: Optional[float] = None,
) -> None:
    """统一打印 7 项指标 + 混淆矩阵。"""
    print(f"\n{title}")
    if threshold is not None:
        print(f"  分类阈值: {threshold:.4f}")
    for key, label in CORE_METRIC_SPECS:
        print(f"  {label}: {metrics[key]:.4f}")
    print(
        "  混淆矩阵 Confusion Matrix "
        f"[[TN FP] [FN TP]] = "
        f"[[{int(metrics['tn'])} {int(metrics['fp'])}] "
        f"[{int(metrics['fn'])} {int(metrics['tp'])}]]"
    )


def save_evaluation_artifacts(
    output_dir: Path,
    prefix: str,
    metrics: Dict[str, float],
    threshold: Optional[float] = None,
    context: str = "",
) -> None:
    """保存 metrics CSV 与 confusion matrix CSV。"""
    out = Path(output_dir)
    core_metrics_to_dataframe(metrics, context=context, threshold=threshold).to_csv(
        out / f"{prefix}_metrics.csv", index=False
    )
    confusion_matrix_to_dataframe(metrics, context=context).to_csv(
        out / f"{prefix}_confusion_matrix.csv", index=False
    )


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = cfg.DEFAULT_THRESHOLD,
    fold_label: str = "",
) -> Dict[str, float]:
    """
    在未经采样的原始验证/测试集上计算核心指标。
    欺诈类为正类 1。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    fraud_rate = float(n_pos / len(y_true)) if len(y_true) else float("nan")

    if n_pos == 0 or n_neg == 0:
        msg = (
            f"{fold_label} 仅含单一类别 (pos={n_pos}, neg={n_neg})，"
            "ROC-AUC / PR-AUC 无法可靠计算。"
        )
        raise RuntimeError(msg)

    # AUC 指标必须基于预测概率计算
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    if y_pred.sum() == 0:
        print(
            f"[警告] {fold_label} 在阈值 {threshold:.3f} 下未预测出任何欺诈样本，"
            "Precision 将按 zero_division=0 处理。"
        )

    metrics = {
        "fraud_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fraud_rate": fraud_rate,
        "threshold": float(threshold),
        "n_samples": float(len(y_true)),
        "n_fraud": float(n_pos),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = float(cm[0, 0])
    metrics["fp"] = float(cm[0, 1])
    metrics["fn"] = float(cm[1, 0])
    metrics["tp"] = float(cm[1, 1])
    return metrics


def summarize_cv_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """对每折指标汇总：均值、标准差、最小、最大。"""
    rows = []
    for col in CORE_METRIC_KEYS:
        if col not in fold_results.columns:
            continue
        values = fold_results[col].astype(float)
        rows.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                **{f"fold_{int(r.fold)}": float(r[col]) for _, r in fold_results.iterrows()},
            }
        )
    return pd.DataFrame(rows)


def search_threshold_oof(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = cfg.THRESHOLD_SEARCH_START,
    end: float = cfg.THRESHOLD_SEARCH_END,
    step: float = cfg.THRESHOLD_SEARCH_STEP,
    objective: str = cfg.THRESHOLD_OBJECTIVE,
    min_precision: Optional[float] = cfg.THRESHOLD_MIN_PRECISION,
) -> Tuple[float, pd.DataFrame]:
    """
    基于全部折的 out-of-fold 预测统一选择最终阈值。
    严禁根据 test.csv 指标反向选择阈值。

    搜索范围: [start, end]，步长 step。
    优化指标默认最大化欺诈类 F1。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    # AUC 指标必须基于预测概率计算，与阈值无关
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    thresholds = np.arange(start, end + 1e-12, step)
    records: List[Dict[str, float]] = []

    best_t = cfg.DEFAULT_THRESHOLD
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_true, y_pred)
        # Youden's J = TPR + TNR - 1 = sensitivity + specificity - 1
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        youden = tpr + tnr - 1.0

        if objective == "fraud_f1":
            score = f1
        elif objective == "balanced_accuracy":
            score = bal
        elif objective == "youden_j":
            score = youden
        elif objective == "recall_at_precision":
            if min_precision is None:
                raise ValueError("recall_at_precision 需要设置 THRESHOLD_MIN_PRECISION")
            score = rec if prec >= min_precision else -1.0
        else:
            raise ValueError(f"未知阈值优化目标: {objective}")

        records.append(
            {
                "threshold": float(t),
                "fraud_precision": float(prec),
                "fraud_recall": float(rec),
                "fraud_f1": float(f1),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "balanced_accuracy": float(bal),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "youden_j": float(youden),
                "objective_score": float(score),
            }
        )
        if score > best_score:
            best_score = score
            best_t = float(t)

    result_df = pd.DataFrame(records)
    print(
        f"\n阈值搜索（仅 OOF，目标={objective}）: "
        f"最佳阈值={best_t:.3f}, score={best_score:.4f}"
    )
    print(
        f"  搜索范围=[{start}, {end}], 步长={step}, "
        f"min_precision={min_precision}"
    )
    return best_t, result_df
