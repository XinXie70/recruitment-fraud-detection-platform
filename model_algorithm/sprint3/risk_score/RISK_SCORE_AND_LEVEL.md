# Ensemble Risk Score and Three-Level Output

This document defines the final risk score and three-level output for the
frozen BERT + LR FP-gate ensemble.

## 1. Model roles

- **BERT** is the primary fraud detector and normally supplies the risk score.
- **Logistic Regression** is the false-positive gate for BERT High candidates.
- When the gate triggers, LR supplies the final operational risk score.

Both raw model scores are retained for explanation.

## 2. Ensemble risk score

```text
gate_triggered = BERT score >= 0.30 and LR score < 0.06

if gate_triggered:
    risk_score = LR score
else:
    risk_score = BERT score
```

Risk level is determined separately by the original BERT-LR gate rule. It is
not reconstructed by applying the BERT boundaries to this mixed-source score
alone.

For display:

```text
risk_score_100 = 100 * risk_score
```

This is an operational mixed-source ensemble score, not a calibrated
probability of fraud. A gated sample may have a score below the BERT Low
boundary while remaining Suspicious because the level uses the two-model gate
rule rather than the final score alone.

## 3. High rule

The BERT threshold and LR gate were jointly selected on Validation by
maximising Fraud F1.

```text
High if BERT score >= 0.30 and LR score >= 0.06
```

When BERT reaches `0.30` but LR is below `0.06`, the advertisement is sent to
Suspicious and its risk score is supplied directly by LR.

## 4. Low boundary

The Low boundary was selected from the BERT score on Validation:

```text
Low if BERT score < 0.0024
```

It is the highest tested BERT threshold that kept at least 90% of known
Validation fraud advertisements outside Low.

Validation result:

| Measure | Result |
|---|---:|
| Fraud kept outside Low | 63/69 (91.30%) |
| Fraud left in Low | 6 |
| Suspicious advertisements | 39 |
| Legitimate in Suspicious | 34 |

LR does not participate in selecting the Low boundary. Its only role remains
the High false-positive gate.

## 5. Complete output rule

```text
gate_triggered = BERT score >= 0.30 and LR score < 0.06

if gate_triggered:
    risk_score = LR score
else:
    risk_score = BERT score

if BERT score >= 0.30 and LR score >= 0.06:
    risk_level = High
elif BERT score < 0.0024:
    risk_level = Low
else:
    risk_level = Suspicious
```

The level meanings are:

- **High:** BERT provides High-risk evidence and LR does not trigger the gate.
- **Low:** the BERT primary score is below the Validation-selected Low
  boundary.
- **Suspicious:** all remaining cases, including High candidates demoted by
  the LR gate.

Suspicious is an operational review band, not a ground-truth EMSCAD class.
