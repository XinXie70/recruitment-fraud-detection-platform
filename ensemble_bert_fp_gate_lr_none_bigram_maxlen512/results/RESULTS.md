# BERT + LR(None/bigram/no-CV) FP-gate ensemble

## Method

1. Predict with BERT (`max_length=512`) using `bert_threshold`
2. If BERT predicts fraud **and** `LR_score < lr_gate`, flip to legitimate
3. Select `(bert_threshold, lr_gate)` on Validation by Fraud F1
4. Freeze and evaluate once on Test

LR branch: `lr_none_bigram_no_cv_paper_aligned_seed42`
(`class_weight=None`, unigram+bigram, no Train CV).

BERT branch: `retrain_paper_aligned_seed42_maxlen512`.

## Reproducibility

- Selection seed / hash seed: `42` / `42`
- Frozen LR predictions: `F:\better-BERT\capstone-project-26t2-9900-h09c-almond\lr_none_bigram_no_cv_paper_aligned_seed42\results\test_predictions.csv`
- Frozen BERT predictions: `F:\better-BERT\capstone-project-26t2-9900-h09c-almond\retrain_paper_aligned_seed42_maxlen512\results\predictions_bert_paper_protocol_maxlen512.csv`
- Packages: `{'python': '3.13.7', 'numpy': '2.4.4', 'pandas': '3.0.3', 'scikit_learn': '1.9.0'}`
- Test report keeps only: LR / BERT / this ensemble

## Selected configuration

- BERT threshold: `0.30`
- LR gate: `0.06`
- Validation gate flips: `1/1431`
- Test gate flips: `3/3576`
- Validation Fraud F1: `0.8923`
- Validation pure-BERT best Fraud F1: `0.8855` (thr 0.30)

## Test comparison (Macro metrics)

The ensemble PR-AUC and ROC-AUC use the formal operational risk score: BERT
score normally, and the LR score with the Low-boundary floor when the FP-gate
triggers. The score is not a calibrated probability.

| Model | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LR (`None`, bigram, no CV) | 0.9622 | 0.9060 | 0.9321 | 0.9318 | 0.9900 |
| BERT max_length=512 | 0.9607 | 0.9404 | 0.9503 | 0.9405 | 0.9931 |
| **BERT + LR(None/bigram) FP-gate** | 0.9664 | 0.9407 | 0.9532 | 0.9478 | 0.9933 |

## Test comparison (Fraud metrics)

| Model | Fraud P | Fraud R | Fraud F1 |
|---|---:|---:|---:|
| LR (`None`, bigram, no CV) | 0.9338 | 0.8150 | 0.8704 |
| BERT max_length=512 | 0.9273 | 0.8844 | 0.9053 |
| **BERT + LR(None/bigram) FP-gate** | 0.9387 | 0.8844 | 0.9107 |

Macro F1 vs BERT-512 alone: higher (0.9503 → 0.9532).
Fraud F1 vs BERT-512 alone: higher (0.9053 → 0.9107).

Confusion matrix on Test: TN 3393, FP 10, FN 20, TP 153.
