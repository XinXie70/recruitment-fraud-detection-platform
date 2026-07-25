# Split Leakage Experiment: Condition C — Logistic Regression

## Design

- Condition: **C: exact dedup + group-aware split**
- Input rows after exact deduplication: **15,807**
- Legitimate rows: **15,095**
- Fraudulent rows: **712**
- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**
- Split rule: complete `group_id` allocation using the shared data-pipeline allocator
- Seeds: **0 to 9**
- Model: fixed TF-IDF Logistic Regression baseline
- Threshold: selected on each diagnostic Validation partition by maximum fraud F1
- Official Train, Validation and Test files used: **No**

## Holdout performance across 10 seeds

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| pr_auc | 0.8731 | 0.0308 | 0.8049 | 0.9256 |
| roc_auc | 0.9818 | 0.0060 | 0.9732 | 0.9918 |
| accuracy | 0.9825 | 0.0048 | 0.9713 | 0.9895 |
| fraud_precision | 0.8527 | 0.0965 | 0.6696 | 0.9730 |
| fraud_recall | 0.7548 | 0.0475 | 0.6792 | 0.8224 |
| fraud_f1 | 0.7970 | 0.0449 | 0.6937 | 0.8718 |
| macro_f1 | 0.8939 | 0.0237 | 0.8393 | 0.9331 |

## Split composition across 10 seeds

| Measure | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| train_rows | 11063.1 | 11046 | 11066 |
| validation_rows | 2374.8 | 2371 | 2409 |
| holdout_rows | 2369.1 | 2352 | 2372 |
| train_fraud | 498.1 | 498 | 499 |
| validation_fraud | 107.0 | 107 | 107 |
| holdout_fraud | 106.9 | 106 | 107 |
| train_groups | 10312.2 | 10241 | 10370 |
| validation_groups | 2177.4 | 2115 | 2242 |
| holdout_groups | 2183.4 | 2119 | 2231 |
| train_fraud_groups | 462.5 | 458 | 470 |
| validation_fraud_groups | 98.3 | 92 | 103 |
| holdout_fraud_groups | 100.2 | 98 | 104 |

## Leakage check

- Near-duplicate groups crossing splits: **0 for every seed**

## Interpretation

Condition C uses the same exact-deduplicated input as Condition B but prevents near-duplicate groups from crossing diagnostic partitions. For Logistic Regression, B-to-C isolates this outer split change. For Linear SVM, Condition C also makes Train-only calibration folds group-aware, so B-to-C represents the complete strict workflow rather than the outer split alone.
