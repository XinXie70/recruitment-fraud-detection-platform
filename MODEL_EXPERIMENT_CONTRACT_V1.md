# Model Experiment Contract v1

This document records the shared rules used when developing the six models.

## Data

- Use the fixed `train.csv`, `validation.csv`, and `test.csv`.
- Use `combined_text` as the model input and `label` as the target.
- Do not use `record_id` or `group_id` as model features.
- Internal cross-validation is allowed only inside Train.
- When cross-validation is used, keep each complete `group_id` in one fold.
- Fit TF-IDF, vocabulary, scalers, resampling, and other learned preprocessing
  using the relevant training fold only.

## Model selection

- Primary model-selection metric: PR-AUC (average precision).
- Validation may be used for early stopping and threshold selection.
- Select the classification threshold on Validation by maximising fraud-class F1.
- Do not use Test for model selection, hyperparameter tuning, early stopping, or
  threshold selection.

## Reproducibility

- Default random seed: `42`.
- Record the selected hyperparameters and software versions.
- Keep model-specific preprocessing documented.

## Shared prediction output

Each model must produce:

| Field | Meaning |
|---|---|
| `record_id` | Advertisement ID |
| `model_name` | Model name |
| `fraud_score` | Score from 0 to 1 |
| `threshold` | Threshold selected on Validation |
| `prediction` | Predicted label, `0` or `1` |
| `true_label` | Actual label |

The binary threshold is used to report Precision, Recall, F1 and the confusion
matrix for each base model. It is not the final product risk-level boundary.
Future ensemble development must use the continuous `fraud_score`, not the
binary `prediction`.

The final product will map the ensemble risk score into Low, Medium and High
risk levels. Those two risk-level thresholds will be selected later using
Validation after the ensemble method is fixed.

## Final evaluation

- Lock the model configuration and threshold before using Test.
- Report fraud Precision, Recall and F1, together with PR-AUC, ROC-AUC and a
  confusion matrix.
- Test is reserved for final evaluation.
