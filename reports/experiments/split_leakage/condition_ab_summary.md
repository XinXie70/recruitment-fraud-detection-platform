# Split Leakage Experiment: Conditions A and B

## What changed

- Condition A: 17,880 rows, exact duplicates retained, ordinary stratified random split.
- Condition B: 15,807 rows after exact deduplication, ordinary stratified random split.
- Both conditions: fixed LR and SVM settings, 70/15/15 split, seeds 0–9, Validation-selected threshold and diagnostic Holdout evaluation.
- Near-duplicate `group_id` is ignored in Condition B, so near-duplicate leakage is still possible.

## Mean diagnostic Holdout results

| Model | Metric | A | B | B − A |
|---|---|---:|---:|---:|
| Logistic Regression | PR-AUC | 0.9071 | 0.8856 | -0.0215 |
| Logistic Regression | ROC-AUC | 0.9843 | 0.9834 | -0.0009 |
| Logistic Regression | Fraud precision | 0.9285 | 0.8886 | -0.0399 |
| Logistic Regression | Fraud recall | 0.7777 | 0.7551 | -0.0226 |
| Logistic Regression | Fraud F1 | 0.8452 | 0.8137 | -0.0315 |
| Logistic Regression | Macro F1 | 0.9190 | 0.9028 | -0.0162 |
| Linear SVM | PR-AUC | 0.9276 | 0.9142 | -0.0134 |
| Linear SVM | ROC-AUC | 0.9846 | 0.9848 | +0.0003 |
| Linear SVM | Fraud precision | 0.9411 | 0.9153 | -0.0258 |
| Linear SVM | Fraud recall | 0.8392 | 0.7916 | -0.0476 |
| Linear SVM | Fraud F1 | 0.8860 | 0.8468 | -0.0392 |
| Linear SVM | Macro F1 | 0.9403 | 0.9200 | -0.0202 |

## Remaining near-duplicate overlap in Condition B

- Mean near-duplicate groups crossing splits: **348.3**
- Mean diagnostic Holdout rows whose group also appears in Train: **221.5**
- Mean fraudulent Holdout rows whose group also appears in Train: **10.7**
- LR and SVM used matching seeds and produced identical split-overlap counts.

## Interpretation

Performance generally decreased after exact duplicates were removed, especially fraud F1. This is evidence that exact-duplicate handling affects the reported result under random splitting. It is not yet sufficient to claim that all of the decrease was caused by leakage, because deduplication also changes the dataset size and class composition.

Condition C is required next: use the same exact-deduplicated input as Condition B, but keep every near-duplicate group within one split. The B-to-C comparison will isolate the effect of group-aware splitting more clearly.
