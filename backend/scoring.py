"""
Ensemble Scoring Module — combines predictions from 8 ML models into a unified risk score.

================================================================================
FORMULA & RATIONALE
================================================================================

The combined score uses a two-mode strategy driven by model agreement:

  MODE 1 — Models AGREE (all fake or all real):
    Final Score = Average Score

  MODE 2 — Models DISAGREE (mixed predictions):
    Final Score = α × Average Score + (1 − α) × Highest Risk Score

    where α = 0.60 (consensus weight), (1 − α) = 0.40 (caution weight).

-------------------------------------------------------------------------------
WHY α = 0.60 ? (Average / Consensus Weight)
-------------------------------------------------------------------------------

1. ENSEMBLE WISDOM PRINCIPLE
   With 8 architecturally diverse models (LR, SVM, XGBoost, DNN, RNN, BiLSTM,
   BERT, RoBERTa), the average captures complementary perspectives:
   - Linear models (LR, SVM) excel at clear-cut text patterns
   - Tree-based (XGBoost) captures non-linear feature interactions
   - Deep models (DNN, RNN, BiLSTM) learn complex sequential dependencies
   - Transformers (BERT, RoBERTa) leverage pre-trained language understanding

   Averaging across these reduces variance and guards against individual model
   overfitting. Weight 0.60 ensures the majority voice prevails.

2. MODEL PERFORMANCE ANALYSIS
   Based on test-set F1 scores (see model_comparison_summary.json):
   - RoBERTa: 0.824 | BERT: 0.803 | BiLSTM: 0.747 | LR: 0.690 | XGBoost: 0.680
   - Top models cluster around 0.70–0.82 F1, so no single model dominates.
   - A uniform-weight average is a robust prior when no model is overwhelmingly
     superior.

3. STABILITY vs. SENSITIVITY TRADE-OFF
   - α = 1.0 → pure average: stable but ignores worst-case signals
   - α = 0.0 → pure max: overly sensitive to a single outlier model
   - α = 0.6 → balanced: consensus dominates while outliers are audible

-------------------------------------------------------------------------------
WHY 1 − α = 0.40 ? (Highest Risk / Caution Weight)
-------------------------------------------------------------------------------

1. ASYMMETRIC COST OF ERRORS (FRAUD DETECTION CONTEXT)
   - FALSE NEGATIVE (missed fake job): User applies, shares personal data,
     potentially loses money. Cost = HIGH.
   - FALSE POSITIVE (flagged real job): User investigates further or skips
     one opportunity. Cost = LOW.

   The 0.40 caution weight operationalizes the precautionary principle:
   when models disagree, the most pessimistic model may have detected subtle
   fraud signals others missed.

2. MODEL DIVERSITY AS A SAFETY NET
   Different models are sensitive to different fraud patterns:
   - LR/SVM: keyword-based red flags (e.g., "no experience required")
   - XGBoost: feature interaction patterns (e.g., "remote" + "urgent")
   - DNN/RNN: semantic and sequential cues
   - BERT/RoBERTa: contextual nuance and linguistic manipulation

   When ONE model fires strongly while others don't, it could be noise OR
   it could be that only that model's architecture catches the specific
   fraud pattern. The 0.40 weight errs on the side of caution.

3. EMPIRICAL JUSTIFICATION
   - At α = 0.60, a single high-risk model (e.g., 0.90) with others at 0.30
     yields: 0.6 × 0.48 + 0.4 × 0.90 = 0.648 → elevated to medium risk
   - At α = 0.80, same scenario: 0.8 × 0.48 + 0.2 × 0.90 = 0.564 → still medium
   - At α = 0.40, same scenario: 0.4 × 0.48 + 0.6 × 0.90 = 0.732 → high risk

   α = 0.60 provides the right sensitivity: a single high score elevates
   the result meaningfully without triggering false alarms for every outlier.

-------------------------------------------------------------------------------
RISK TIER THRESHOLDS
-------------------------------------------------------------------------------

  Score < 30   → Low Risk (Likely Legitimate)
  30 ≤ Score < 60 → Medium Risk (Suspicious — Review Required)
  Score ≥ 60   → High Risk (Likely Deceptive — Action Required)

These thresholds are aligned with the model-level LOW/HIGH thresholds from
`structured_output.tune_dual_thresholds()`, which were tuned on the
validation set using PR-curve optimization.

-------------------------------------------------------------------------------
BINARY PREDICTION THRESHOLD
-------------------------------------------------------------------------------

  combined_prob ≥ 0.40 → "fake"
  combined_prob < 0.40 → "legitimate"

This is slightly above the typical 0.50 to err on the side of flagging,
consistent with the asymmetric cost of fraud detection.

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Configurable weights
# ---------------------------------------------------------------------------
CONSENSUS_WEIGHT: float = 0.60  # α — weight for average score
CAUTION_WEIGHT: float = 0.40  # 1 − α — weight for maximum (most cautious) score
BINARY_THRESHOLD: float = 0.40  # Prediction threshold (fake if prob ≥ this)

# Risk tier boundaries
LOW_RISK_THRESHOLD: float = 30.0  # score < 30 → low
HIGH_RISK_THRESHOLD: float = 60.0  # score ≥ 60 → high; in-between → medium


@dataclass
class CombinedResult:
    """Structured output from ensemble scoring."""

    risk_score: float  # 0-100 integer
    risk_level: Literal["low", "medium", "high"]
    prediction: Literal["fake", "legitimate"]
    combined_prob: float  # raw 0-1 probability
    model_count: int
    fake_count: int
    real_count: int
    all_agree_fake: bool
    all_agree_real: bool
    models_disagree: bool
    average_score: float  # raw mean of all model risk_scores
    max_score: float  # highest individual model risk_score
    formula_used: str  # human-readable description of which formula was applied


def compute_combined_score(model_results: list[dict[str, Any]]) -> CombinedResult:
    """
    Compute an ensemble risk score from individual model predictions.

    Parameters
    ----------
    model_results : list[dict]
        Each dict should have at minimum:
          - ``risk_score`` (float, 0–1): model's fraud probability
          - ``prediction`` (str): "fake" or "real"

    Returns
    -------
    CombinedResult
        Structured combined score with metadata.
    """
    # --- Filter valid models -------------------------------------------------
    valid = [
        m for m in model_results
        if isinstance(m.get("risk_score"), (int, float)) and not (
            isinstance(m.get("risk_score"), float) and
            (m["risk_score"] != m["risk_score"])  # NaN check
        )
    ]

    if not valid:
        return CombinedResult(
            risk_score=0,
            risk_level="low",
            prediction="legitimate",
            combined_prob=0.0,
            model_count=0,
            fake_count=0,
            real_count=0,
            all_agree_fake=False,
            all_agree_real=False,
            models_disagree=False,
            average_score=0.0,
            max_score=0.0,
            formula_used="no valid models",
        )

    # --- Extract scores and predictions --------------------------------------
    scores = [m["risk_score"] for m in valid]
    predictions = [m.get("prediction", "") for m in valid]
    fake_count = sum(1 for p in predictions if p == "fake")
    real_count = sum(1 for p in predictions if p == "real")
    n = len(valid)

    average = sum(scores) / n
    maximum = max(scores)

    all_fake = fake_count == n
    all_real = real_count == n
    disagree = not all_fake and not all_real

    # --- Compute combined probability ----------------------------------------
    if disagree:
        combined_prob = CONSENSUS_WEIGHT * average + CAUTION_WEIGHT * maximum
        formula = (
            f"{CONSENSUS_WEIGHT} × avg({average:.4f}) "
            f"+ {CAUTION_WEIGHT} × max({maximum:.4f}) "
            f"= {combined_prob:.4f} (models disagree)"
        )
    else:
        combined_prob = average
        formula = f"avg({average:.4f}) = {combined_prob:.4f} (models agree)"

    # --- Risk score (0–100) --------------------------------------------------
    risk_score = round(min(100.0, max(0.0, combined_prob * 100)))

    # --- Risk tier -----------------------------------------------------------
    if risk_score >= HIGH_RISK_THRESHOLD:
        risk_level = "high"
    elif risk_score >= LOW_RISK_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    # --- Binary prediction ---------------------------------------------------
    if all_real and combined_prob < BINARY_THRESHOLD:
        prediction = "legitimate"
    elif all_fake:
        prediction = "fake"
    else:
        prediction = "fake" if combined_prob >= BINARY_THRESHOLD else "legitimate"

    return CombinedResult(
        risk_score=risk_score,
        risk_level=risk_level,
        prediction=prediction,
        combined_prob=round(combined_prob, 4),
        model_count=n,
        fake_count=fake_count,
        real_count=real_count,
        all_agree_fake=all_fake,
        all_agree_real=all_real,
        models_disagree=disagree,
        average_score=round(average, 4),
        max_score=round(maximum, 4),
        formula_used=formula,
    )
