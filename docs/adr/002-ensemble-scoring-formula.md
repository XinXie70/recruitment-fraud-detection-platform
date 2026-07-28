# ADR-002: Ensemble Scoring Formula

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

We have 8 architecturally diverse models (LR, SVM, XGBoost, DNN, RNN, BiLSTM, BERT, RoBERTa). We need a formula to combine their individual risk scores into a single ensemble score.

## Decision

Use a **two-mode weighted strategy** (`backend/scoring.py`):

**Mode 1 — Models Agree** (all fake or all legitimate):
```
Final Score = Average of all model scores
```

**Mode 2 — Models Disagree** (mixed predictions):
```
Final Score = α × Average + (1 − α) × Max(Highest Risk)
```

Where `α = 0.60` (consensus weight), `(1 − α) = 0.40` (caution weight).

Risk tiers:
- Score < 30 → Low Risk
- 30 ≤ Score < 60 → Medium Risk
- Score ≥ 60 → High Risk

## Rationale

1. **Ensemble wisdom**: Averaging across diverse architectures (linear, tree, deep, transformer) reduces variance and guards against individual overfitting.

2. **Asymmetric cost**: In fraud detection, false negatives (missed fake) are far costlier than false positives. The 0.40 caution weight ensures that a single model detecting fraud signals elevates the overall score.

3. **Empirical tuning**: Thresholds were derived from validation-set PR-curve optimization (see `structured_output.tune_dual_thresholds()`).

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Pure average (α=1.0) | Too conservative: ignores outlier model signals |
| Pure max (α=0.0) | Too sensitive: single model noise triggers false alarms |
| Learned meta-model | Requires re-training when adding/removing members; harder to interpret |

## Consequences

- Ensemble is interpretable: the formula used is reported in every response
- Adding a 9th model requires no code changes (just update `ensemble_config.json`)
- The 0.40 caution weight means ~40% of borderline cases err toward "suspicious" — acceptable given the domain
