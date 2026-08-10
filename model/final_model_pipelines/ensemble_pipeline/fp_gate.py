"""Sprint3 FP-gate and risk-level helpers (BERT primary + LR gate)."""

from __future__ import annotations

from typing import Any


def _decision_reason(*, risk_level: str, gate_triggered: bool) -> str:
    if gate_triggered:
        return "BERT High candidate was demoted by the LR gate"
    if risk_level == "High":
        return "BERT High rule passed the LR gate"
    if risk_level == "Low":
        return "BERT score is below the Low boundary"
    return "BERT score is between the Low and High boundaries"


def apply_fp_gate(
    bert_score: float,
    lr_score: float,
    bert_threshold: float,
    lr_gate: float,
) -> dict[str, Any]:
    bert_pred = int(bert_score >= bert_threshold)
    gated = bool(bert_pred == 1 and lr_score < lr_gate)
    final_pred = 0 if gated else bert_pred
    ranking_score = min(bert_score, lr_score) if gated else bert_score
    return {
        "model": "ensemble_fp_gate",
        "bert_threshold": bert_threshold,
        "lr_gate": lr_gate,
        "bert_score": bert_score,
        "lr_score": lr_score,
        "gate_triggered": gated,
        "bert_raw_prediction_id": bert_pred,
        "predicted_label_id": final_pred,
        "predicted_label": "Fraudulent" if final_pred == 1 else "Legitimate",
        "ranking_score": ranking_score,
    }


def apply_risk(
    bert_score: float,
    lr_score: float,
    bert_high_threshold: float,
    lr_gate: float,
    bert_low_threshold: float,
) -> dict[str, Any]:
    high_candidate = bert_score >= bert_high_threshold
    gate_triggered = bool(high_candidate and lr_score < lr_gate)
    high = bool(high_candidate and not gate_triggered)
    low = bool(bert_score < bert_low_threshold)

    risk_score = lr_score if gate_triggered else bert_score
    if high:
        risk_level = "High"
    elif low:
        risk_level = "Low"
    else:
        risk_level = "Suspicious"

    return {
        "model": "risk",
        "risk_score": float(risk_score),
        "risk_score_100": float(risk_score * 100.0),
        "risk_level": risk_level,
        "bert_evidence_score": float(bert_score),
        "lr_score": float(lr_score),
        "risk_score_source": "lr_gate" if gate_triggered else "bert",
        "gate_triggered": gate_triggered,
        "high_rule_met": int(high),
        "low_rule_met": int(low),
        "decision_reason": _decision_reason(
            risk_level=risk_level, gate_triggered=gate_triggered
        ),
        "thresholds": {
            "bert_high_threshold": bert_high_threshold,
            "lr_gate": lr_gate,
            "bert_low_threshold": bert_low_threshold,
        },
    }
