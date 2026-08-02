# BERT + LR FP-gate ensemble

## Method

1. Predict with BERT using `bert_threshold`
2. If BERT predicts fraud **and** `LR_score < lr_gate`, flip to legitimate
3. Select `(bert_threshold, lr_gate)` on Validation by Fraud F1
4. Freeze and evaluate once on Test

## Selected configuration

- BERT threshold: `0.30`
- LR gate: `0.06`
- Validation gate flips: `1/1431`
- Test gate flips: `3/3576`
- Validation Fraud F1: `0.8923`
- Validation pure-BERT best Fraud F1: `0.8855` (thr 0.30)

## Test comparison

| Model | Fraud P | Fraud R | Fraud F1 | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved LR (`None`) | 0.9338 | 0.8150 | 0.8704 | 0.9622 | 0.9060 | 0.9321 | 0.9318 | 0.9900 |
| BERT max_length=512 (paper thr) | 0.9273 | 0.8844 | 0.9053 | 0.9607 | 0.9404 | 0.9503 | 0.9405 | 0.9931 |
| BERT at selected thr `0.30` (no gate) | 0.9217 | 0.8844 | 0.9027 | 0.9579 | 0.9403 | 0.9489 | 0.9405 | 0.9931 |
| BERT + LR FP-gate | 0.9387 | 0.8844 | 0.9107 | 0.9664 | 0.9407 | 0.9532 | 0.9478 | 0.9933 |

Fraud F1 vs paper BERT: higher.

Confusion matrix on Test: TN 3393, FP 10, FN 20, TP 153.
