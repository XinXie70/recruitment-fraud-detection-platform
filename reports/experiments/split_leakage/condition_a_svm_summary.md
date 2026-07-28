# Split Leakage Experiment: Condition A — Linear SVM

## Design

- Condition: **A: no dedup + stratified random split**
- Input rows: **17,880**
- Legitimate rows: **17,014**
- Fraudulent rows: **866**
- Exact duplicate groups in the full input: **726**
- Rows belonging to exact duplicate groups: **2,799**
- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**
- Seeds: **0 to 9**
- Model: fixed TF-IDF Linear SVM baseline
- Calibration: Train-only three-fold stratified sigmoid calibration, without group information
- Threshold: selected independently on each diagnostic Validation partition by maximum fraud F1
- Official Train, Validation and Test files used: **No**

## Holdout performance across 10 seeds

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| pr_auc | 0.9276 | 0.0222 | 0.8976 | 0.9524 |
| roc_auc | 0.9846 | 0.0084 | 0.9733 | 0.9945 |
| accuracy | 0.9896 | 0.0022 | 0.9866 | 0.9925 |
| fraud_precision | 0.9411 | 0.0283 | 0.8881 | 0.9741 |
| fraud_recall | 0.8392 | 0.0522 | 0.7538 | 0.9154 |
| fraud_f1 | 0.8860 | 0.0271 | 0.8448 | 0.9187 |
| macro_f1 | 0.9403 | 0.0141 | 0.9189 | 0.9574 |

## Exact-duplicate contamination across 10 seeds

| Measure | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| cross_split_exact_groups | 419.2 | 400 | 437 |
| cross_split_fraud_exact_groups | 25.8 | 19 | 29 |
| cross_split_legitimate_exact_groups | 393.4 | 376 | 418 |
| rows_in_cross_split_exact_groups | 2082.4 | 2052 | 2125 |
| train_validation_exact_groups | 229.4 | 215 | 241 |
| train_holdout_exact_groups | 233.1 | 216 | 252 |
| validation_holdout_exact_groups | 93.7 | 86 | 106 |
| validation_rows_matching_train | 375.0 | 338 | 396 |
| validation_fraud_rows_matching_train | 26.2 | 21 | 36 |
| validation_legitimate_rows_matching_train | 348.8 | 314 | 364 |
| holdout_rows_matching_train | 369.8 | 334 | 401 |
| holdout_fraud_rows_matching_train | 27.3 | 21 | 32 |
| holdout_legitimate_rows_matching_train | 342.5 | 313 | 373 |

## Interpretation

Condition A intentionally permits exact duplicates to cross all diagnostic partitions. These results require Conditions B and C before any performance inflation can be attributed to duplicate leakage.
