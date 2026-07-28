# Test Performance Summary v1

The fixed test split contains 2,372 advertisements, including 107 fraudulent
advertisements. SHA-256:

`85457e3dc60c05a68aaea9e9f433b2fb8bc9cb775dbf0221d61efe2e1cbd6257`

## Three-model test results

| Model | PR-AUC | ROC-AUC | Fraud Precision | Fraud Recall | Fraud F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8735 | 0.9852 | 0.7642 | 0.7570 | 0.7606 | 2240 | 25 | 26 | 81 |
| BERT (class weighted) | 0.7942 | 0.9684 | 0.8302 | 0.8224 | 0.8263 | 2247 | 18 | 19 | 88 |
| LR + BERT Ensemble | 0.9058 | 0.9905 | 0.8817 | 0.7664 | 0.8200 | 2254 | 11 | 25 | 82 |

## Result sources

- `reports/models/logistic_regression/test_metrics.json`
- `reports/models/bert/test_metrics.json`
- `reports/models/ensemble_lr_bert/test_metrics.json`

The test split is used only for final evaluation. It must not be used to tune
hyperparameters, classification thresholds, or risk-band boundaries.
