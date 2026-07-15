"""
Evaluation utilities: binary metrics (primary) + dual-threshold tuning (for risk mapping) + tier stats (supplementary).

Notes:
- LR / DNN are trained as real/fake binary classifiers; evaluation focuses on fraud class P/R/F1.
- LOW / HIGH thresholds are for the risk mapping layer's three-tier frontend display only, tuned on the validation set.
- Binary prediction follows risk mapping: prob >= LOW → fake, prob < LOW → real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from final_model_pipelines.risk_mapping import (
    LABEL_LIKELY_DECEPTIVE,
    LABEL_LIKELY_LEGITIMATE,
    LABEL_SUSPICIOUS,
)

POSITIVE_LABEL = 1


def tune_dual_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: int = POSITIVE_LABEL,
    min_gap: float = 0.08,
    step: float = 0.02,
) -> dict[str, float]:
    """
    Search for (LOW, HIGH) dual thresholds for risk mapping on the validation set.

    Optimization goals (aligned with PR curve business objectives):
      - LOW: control missed fraud in Likely Legitimate while balancing binary F1/Recall
      - HIGH: improve precision for the Likely Deceptive tier
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_fraud = int((y_true == pos_label).sum())

    best_score = -1.0
    best_low, best_high = 0.35, 0.65

    for low in np.arange(0.15, 0.76, step):
        high_start = low + min_gap
        binary_f1 = f1_score(
            y_true, (y_prob >= low).astype(int), pos_label=pos_label, zero_division=0
        )
        binary_recall = recall_score(
            y_true, (y_prob >= low).astype(int), pos_label=pos_label, zero_division=0
        )

        for high in np.arange(high_start, 0.96, step):
            legit = y_prob < low
            deceptive = y_prob >= high

            fraud_in_legit = float(((y_true == pos_label) & legit).sum()) / max(n_fraud, 1)
            fraud_capture = 1.0 - fraud_in_legit

            n_deceptive = int(deceptive.sum())
            deceptive_prec = (
                float(((y_true == pos_label) & deceptive).sum()) / n_deceptive
                if n_deceptive
                else 0.0
            )

            n_legit = int(legit.sum())
            legit_prec = (
                float(((y_true != pos_label) & legit).sum()) / n_legit if n_legit else 0.0
            )

            score = (
                0.30 * fraud_capture
                + 0.25 * deceptive_prec
                + 0.15 * legit_prec
                + 0.20 * binary_f1
                + 0.10 * binary_recall
            )
            if score > best_score:
                best_score = score
                best_low, best_high = float(low), float(high)

    return {
        "low_threshold": round(best_low, 4),
        "high_threshold": round(best_high, 4),
        "tuned_on": "validation",
        "optimization_score": round(best_score, 4),
        "binary_f1_at_low_threshold": round(
            float(
                f1_score(
                    y_true,
                    (y_prob >= best_low).astype(int),
                    pos_label=pos_label,
                    zero_division=0,
                )
            ),
            4,
        ),
        "binary_recall_at_low_threshold": round(
            float(
                recall_score(
                    y_true,
                    (y_prob >= best_low).astype(int),
                    pos_label=pos_label,
                    zero_division=0,
                )
            ),
            4,
        ),
        "note": (
            "Thresholds are for risk_mapping layer only. "
            "Binary classification metrics use LOW as fake/real decision boundary."
        ),
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    pos_label: int = POSITIVE_LABEL,
) -> dict[str, float]:
    """Binary evaluation (fraud class Accuracy / Precision / Recall / F1)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }


def compute_tier_statistics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    pos_label: int = POSITIVE_LABEL,
) -> dict[str, Any]:
    """Three-tier risk mapping statistics (supplementary analysis, not training metrics)."""
    tiers = {
        LABEL_LIKELY_LEGITIMATE: y_prob < low_threshold,
        LABEL_SUSPICIOUS: (y_prob >= low_threshold) & (y_prob < high_threshold),
        LABEL_LIKELY_DECEPTIVE: y_prob >= high_threshold,
    }

    total_fraud = int((y_true == pos_label).sum())
    tier_stats = {}
    fraud_coverage = {}

    for name, mask in tiers.items():
        count = int(mask.sum())
        fraud_in_tier = int(((y_true == pos_label) & mask).sum())
        real_in_tier = int(((y_true != pos_label) & mask).sum())
        tier_stats[name] = {
            "total_count": count,
            "fake_job_count": fraud_in_tier,
            "real_job_count": real_in_tier,
            "fake_job_ratio_in_tier": round(fraud_in_tier / count, 4) if count else 0.0,
        }
        fraud_coverage[name] = round(fraud_in_tier / total_fraud, 4) if total_fraud else 0.0

    not_in_legitimate = y_prob >= low_threshold
    fraud_capture_rate = (
        float(((y_true == pos_label) & not_in_legitimate).sum()) / total_fraud
        if total_fraud
        else 0.0
    )

    return {
        "tier_statistics": tier_stats,
        "fraud_coverage_by_tier": fraud_coverage,
        "overall_fraud_capture_rate": round(fraud_capture_rate, 4),
        "total_fraud_jobs": total_fraud,
    }
