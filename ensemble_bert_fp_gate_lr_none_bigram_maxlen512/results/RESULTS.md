# BERT + LR(None/bigram/no-CV) FP-gate ensemble

## Method

1. Predict with BERT (`max_length=512`) using `bert_threshold`
2. If BERT predicts fraud **and** `LR_score < lr_gate`, flip to legitimate
3. Select `(bert_threshold, lr_gate)` on Validation by Fraud F1
4. Freeze and evaluate once on Test

LR branch: `lr_none_bigram_no_cv_paper_aligned_seed42`
(`class_weight=None`, unigram+bigram, no Train CV).

## Selected configuration

- BERT threshold: `0.30`
- LR gate: `0.06`
- Validation gate flips: `1/1431`
- Test gate flips: `3/3576`
- Validation Fraud F1: `0.8923`
- Validation pure-BERT best Fraud F1: `0.8855` (thr 0.30)

## Test comparison (Macro metrics)

| Model | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LR (`None`, bigram, no CV) | 0.9622 | 0.9060 | 0.9321 | 0.9318 | 0.9900 |
| BERT max_length=512 (paper thr) | 0.9607 | 0.9404 | 0.9503 | 0.9405 | 0.9931 |
| BERT at selected thr `0.30` (no gate) | 0.9579 | 0.9403 | 0.9489 | 0.9405 | 0.9931 |
| BERT + LR(`None`/CV) FP-gate (thr=0.30, gate=0.06) | 0.9664 | 0.9407 | 0.9532 | 0.9478 | 0.9933 |
| BERT + LR(cw/unigram) FP-gate (thr=0.30, gate=0.29) | 0.9635 | 0.9406 | 0.9517 | 0.9408 | 0.9931 |
| BERT + LR(cw/bigram) FP-gate (thr=0.29, gate=0.33) | 0.9693 | 0.9409 | 0.9546 | 0.9476 | 0.9933 |
| **BERT + LR(None/bigram) FP-gate** | 0.9664 | 0.9407 | 0.9532 | 0.9478 | 0.9933 |

## Test comparison (Fraud metrics)

| Model | Fraud P | Fraud R | Fraud F1 |
|---|---:|---:|---:|
| LR (`None`, bigram, no CV) | 0.9338 | 0.8150 | 0.8704 |
| BERT max_length=512 (paper thr) | 0.9273 | 0.8844 | 0.9053 |
| **BERT + LR(None/bigram) FP-gate** | 0.9387 | 0.8844 | 0.9107 |

Macro F1 vs BERT-512 alone: higher (0.9503 → 0.9532).
Fraud F1 vs BERT-512 alone: higher (0.9053 → 0.9107).

Confusion matrix on Test: TN 3393, FP 10, FN 20, TP 153.
