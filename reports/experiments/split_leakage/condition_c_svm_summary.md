# Split Leakage Experiment: Condition C — Linear SVM

## Design

- Condition: **C: exact dedup + group-aware split**
- Input rows after exact deduplication: **15,807**
- Legitimate rows: **15,095**
- Fraudulent rows: **712**
- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**
- Split rule: complete `group_id` allocation using the shared data-pipeline allocator
- Seeds: **0 to 9**
- Model: fixed TF-IDF Linear SVM baseline
- Calibration: Train-only three-fold StratifiedGroupKFold sigmoid calibration
- Threshold: selected on each diagnostic Validation partition by maximum fraud F1
- Official Train, Validation and Test files used: **No**

## Holdout performance across 10 seeds

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| pr_auc | 0.8997 | 0.0228 | 0.8556 | 0.9367 |
| roc_auc | 0.9809 | 0.0086 | 0.9670 | 0.9951 |
| accuracy | 0.9867 | 0.0026 | 0.9823 | 0.9916 |
| fraud_precision | 0.8840 | 0.0442 | 0.8095 | 0.9485 |
| fraud_recall | 0.8129 | 0.0264 | 0.7757 | 0.8598 |
| fraud_f1 | 0.8464 | 0.0282 | 0.8019 | 0.9020 |
| macro_f1 | 0.9197 | 0.0148 | 0.8963 | 0.9488 |

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
