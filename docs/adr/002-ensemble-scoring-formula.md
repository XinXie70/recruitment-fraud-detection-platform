# ADR-002: Ensemble Scoring Formula

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

We have 8 architecturally diverse models (LR, SVM, XGBoost, DNN, RNN, BiLSTM, BERT, RoBERTa). We need a formula to combine their individual risk scores into a single ensemble score.

## Decision

Use the **configuration-driven calibrated weighted ensemble** implemented by
`backend/services/ensemble_predictor.py`.

For every successful model `i`:

```text
calibrated_i = calibrator_i(raw_probability_i)
effective_weight_i = configured_weight_i / sum(weights of successful models)
final_risk = sum(calibrated_i * effective_weight_i)
```

The final probability is mapped to low, medium, or high risk using the low/high
thresholds stored in the same versioned configuration. If a model fails, its
weight is redistributed proportionally across successful models. If all models
fail, the API returns 503 rather than inventing a score.

## Rationale

1. **Ensemble wisdom**: Averaging across diverse architectures (linear, tree, deep, transformer) reduces variance and guards against individual overfitting.

2. **Asymmetric cost**: Validation-selected thresholds can reflect that false
   negatives are more costly than false positives without hard-coding UI logic.

3. **Operational resilience**: Renormalising successful weights allows an
   explicitly degraded result when individual models fail.

4. **Separation of code and evidence**: Weights, calibrators, thresholds, fit
   status, and provenance are stored in a versioned JSON artifact.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Pure average | Assumes equal model quality and calibration |
| Maximum model score | Too sensitive to a single noisy member |
| Learned meta-model | Requires re-training when adding/removing members; harder to interpret |

## Consequences

- Ensemble contributions are inspectable in each member output.
- Models, calibration, weights, and thresholds can change through a reviewed
  configuration artifact rather than application-code edits.
- More models increase latency and operational complexity.
- The checked-in development fallback is equal-weight and marked `fitted:
  false`; it must be replaced with a validation-fitted locked artifact before
  claiming empirically optimised production weights.
