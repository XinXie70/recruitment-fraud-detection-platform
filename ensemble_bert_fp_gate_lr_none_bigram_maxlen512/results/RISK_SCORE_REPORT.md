# Ensemble Risk Score Report

## Definition

```text
Normally:       risk_score = BERT score
If gate fires:  risk_score = max(LR score, 0.0024)
Display score:  risk_score_100 = 100 * risk_score
```

The Low-boundary floor keeps every gated High candidate in Suspicious. Raw
`bert_evidence_score`, `lr_score`, `risk_score_source`, and `gate_triggered`
are retained for explanation.

This is an operational ensemble decision score, not a calibrated probability.

## Metrics

| Split | BERT PR-AUC | Ensemble score PR-AUC | BERT ROC-AUC | Ensemble score ROC-AUC | Gate scores from LR |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8561 | 0.8596 | 0.9755 | 0.9756 | 1 |
| Test | 0.9405 | 0.9478 | 0.9931 | 0.9933 | 3 |

No risk-score parameter or threshold was selected on Test. Test only applies
the frozen High rule, Low boundary, and score formula.
