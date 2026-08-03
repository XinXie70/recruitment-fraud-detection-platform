# LR (class_weight=None, bigram, no CV) on paper-aligned seed42

## Purpose

Fixed-protocol LR on paper-aligned seed42 splits:

- **Included**: unigram + bigram (`ngram_range=(1, 2)`)
- **Excluded**: `class_weight` (fixed to `None`)
- **Excluded**: Train-internal 3-fold CV (fixed `C=1.0`)

Same Train/Validation/Test splits as `retrain_paper_aligned_seed42_maxlen512`.

## Fixed configuration

- ngram_range: `(1, 2)`
- C: `1.0`
- class_weight: `None`
- Validation threshold: `0.1518`

## Validation

| Metric | Value |
|---|---:|
| Fraud Precision | 0.8710 |
| Fraud Recall | 0.7826 |
| Fraud F1 | 0.8244 |
| Macro F1 | 0.9080 |
| PR-AUC | 0.8911 |
| ROC-AUC | 0.9890 |
| TP / FP / FN / TN | 54 / 8 / 15 / 1354 |

## Test

| Metric | Value |
|---|---:|
| Fraud Precision | 0.9338 |
| Fraud Recall | 0.8150 |
| Fraud F1 | 0.8704 |
| Macro F1 | 0.9321 |
| PR-AUC | 0.9318 |
| ROC-AUC | 0.9900 |
| TP / FP / FN / TN | 141 / 10 / 32 / 3393 |
