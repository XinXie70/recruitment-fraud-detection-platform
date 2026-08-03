# Three-Level Risk Boundary Report

## Selection protocol

- Selection data: Validation only
- Test used for selection: No
- Labels: 0 = legitimate, 1 = fraudulent
- Suspicious is an operational review band, not a ground-truth class
- BERT remains the primary risk-scoring model
- LR remains only the existing false-positive gate for High candidates

## High rule

The existing FP-gate search selected the pair with the highest Validation
Fraud F1:

```text
High if BERT score >= 0.30 and LR score >= 0.06
```

Validation Fraud Precision: 0.9508  
Validation Fraud Recall: 0.8406  
Validation Fraud F1: 0.8923

## Low boundary

The Low boundary uses only the official BERT risk score. This preserves the
frozen ensemble design: LR is not introduced as a second Low-risk decision
mechanism.

Candidate BERT thresholds were compared on Validation. The selected boundary
is the highest threshold that keeps at least 90% of known fraud
advertisements outside Low.

```text
Low if BERT score < 0.0024
```

Selected Validation result:

- Fraud kept outside Low: 63/69 (91.30%)
- Fraud left in Low: 6
- Suspicious advertisements: 39
- Legitimate advertisements in Suspicious: 34

## Validation trade-off

| Minimum target | BERT Low boundary | Fraud kept out of Low | Fraud in Low | Suspicious | Legitimate in Suspicious |
|---:|---:|---:|---:|---:|---:|
| 90.0% | 0.00240 | 63/69 (91.30%) | 6 | 39 | 34 |
| 92.5% | 0.00060 | 64/69 (92.75%) | 5 | 125 | 119 |
| 95.0% | 0.00019 | 66/69 (95.65%) | 3 | 311 | 303 |
| 97.5% | 0.00011 | 68/69 (98.55%) | 1 | 569 | 559 |

## Final three-level rule

```text
if BERT score >= 0.30 and LR score >= 0.06:
    High
elif BERT score < 0.0024:
    Low
else:
    Suspicious
```

The thresholds must be frozen before Test evaluation. The model scores are
operational scores and should not be interpreted as calibrated probabilities.
