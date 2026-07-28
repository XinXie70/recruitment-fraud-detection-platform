# LR + BERT Ensemble v1

## Configuration

This ensemble combines the Logistic Regression baseline with the
class-weighted BERT model. It uses a weighted average of the two fraud scores:

```text
ensemble_score = 0.60 * lr_score + 0.40 * bert_score
```

An advertisement is classified as fraudulent when the ensemble score is at
least `0.62`.

The model configuration is:

- Logistic Regression weight: `0.60`
- Class-weighted BERT weight: `0.40`
- Binary classification threshold: `0.62`
- Selection set: Validation
- Selection objective: maximum fraud-class F1
- Source implementation commit: `4dc0300`

The weights were selected from 21 combinations, with the Logistic Regression
weight ranging from 0 to 1 in steps of 0.05. For each weight, thresholds from
0.01 to 0.99 were checked on Validation.

The final weight was not the only combination with the highest Validation F1.
Logistic Regression weights from 0.45 to 0.60 produced the same precision,
recall, F1 and confusion matrix when paired with their selected thresholds.
The implementation selected 0.60/0.40 through its tie-breaking rule.

## Results

| Split | Model | PR-AUC | ROC-AUC | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Logistic Regression | 0.8108 | 0.9560 | 0.7879 | 0.7290 | 0.7573 |
| Validation | Class-weighted BERT | 0.8279 | 0.9641 | 0.8571 | 0.7850 | 0.8195 |
| Validation | LR + BERT ensemble | 0.8508 | 0.9580 | 0.9318 | 0.7664 | 0.8410 |
| Test | Logistic Regression | 0.8735 | 0.9852 | 0.7642 | 0.7570 | 0.7606 |
| Test | Class-weighted BERT | 0.7942 | 0.9684 | 0.8302 | 0.8224 | 0.8263 |
| Test | LR + BERT ensemble | 0.9058 | 0.9905 | 0.8817 | 0.7664 | 0.8200 |

On Validation, the ensemble improved PR-AUC and fraud F1 over both base
models. On Test, it had the highest PR-AUC of the three models. Its Test fraud
F1 was much higher than Logistic Regression but slightly lower than the
class-weighted BERT result.

The ensemble was selected as a balance between performance and transparency.
Logistic Regression provides feature coefficients that can be inspected, and
the weighted-average formula shows the numerical contribution from each base
model. The BERT component is still not fully interpretable.

## Freeze decision

This configuration is frozen as Ensemble v1. The Test set has already been
evaluated, so the weights and binary threshold must not be changed using Test
performance.

The ensemble score is treated as a risk score, not as a calibrated probability
of fraud. The `0.62` threshold is only for the binary legitimate/fraudulent
evaluation. Risk Band v1 uses `0.1567` as the Low/Suspicious boundary and
reuses `0.62` as the Suspicious/High boundary; its selection is documented in
`risk_band_v1_summary.md`.
