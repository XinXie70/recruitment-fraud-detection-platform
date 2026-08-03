# Risk Score and Three-Level Risk Output

This document defines the three-level output for the frozen BERT + LR FP-gate
ensemble.

## 1. Model roles

The ensemble is intentionally asymmetric:

- **BERT** is the primary fraud detector and provides the official risk score.
- **Logistic Regression** is used only as a false-positive gate when BERT
  reaches the High candidate threshold.

```text
risk_score = bert_score
```

The risk score is an operational model score. It should not be interpreted as
a calibrated probability of fraud.

## 2. High rule

The BERT threshold and LR gate were jointly selected on Validation by
maximising Fraud F1.

```text
High if BERT score >= 0.30 and LR score >= 0.06
```

When BERT reaches `0.30` but LR is below `0.06`, the FP-gate sends the
advertisement to Suspicious rather than High. LR does not promote BERT
negative predictions and does not participate in the Low boundary.

## 3. Low boundary

The Low boundary uses only the official BERT risk score:

```text
Low if BERT score < 0.0024
```

Candidate BERT thresholds were compared on Validation. The selected value is
the highest threshold that keeps at least 90% of known fraud advertisements
outside Low.

Validation result:

| Measure | Result |
|---|---:|
| Fraud kept outside Low | 63/69 (91.30%) |
| Fraud left in Low | 6 |
| Suspicious advertisements | 39 |
| Legitimate in Suspicious | 34 |

## 4. Complete rule

```text
if BERT score >= 0.30 and LR score >= 0.06:
    risk_level = High
elif BERT score < 0.0024:
    risk_level = Low
else:
    risk_level = Suspicious
```

This means:

- **High:** BERT provides high-risk evidence and LR does not trigger the
  false-positive gate.
- **Low:** the BERT primary risk score is below the Validation-selected Low
  boundary.
- **Suspicious:** all remaining cases, including High candidates demoted by
  the LR gate.

Suspicious is an operational review band, not a ground-truth class in EMSCAD.
In binary terms, High maps to fraudulent, while Low and Suspicious both map to
legitimate.

## 5. Recommended output fields

| Field | Type | Description |
|---|---|---|
| `risk_score` | float | Equal to `bert_score` |
| `risk_level` | `Low`, `Suspicious`, `High` | Final three-level output |
| `bert_score` | float | BERT fraud score |
| `lr_score` | float | LR score used by the High FP-gate |
| `high_rule_met` | bool | Whether the High rule was satisfied |
| `low_rule_met` | bool | Whether the BERT Low rule was satisfied |

## 6. Data-use rule

- High and Low parameters are selected on Validation only.
- The selected configuration must be frozen before Test evaluation.
- Test is used once for final reporting and must not be used to adjust any
  threshold.

Selection code and detailed results:

- `code/select_risk_boundaries.py`
- `results/risk_boundary_config.json`
- `results/RISK_BOUNDARY_REPORT.md`
- `results/low_boundary_tradeoff.csv`
