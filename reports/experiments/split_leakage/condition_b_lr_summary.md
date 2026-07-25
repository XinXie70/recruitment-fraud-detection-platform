# Split Leakage Experiment: Condition B — Logistic Regression

## Design

- Condition: **B: exact dedup + stratified random split**
- Input rows after exact deduplication: **15,807**
- Legitimate rows: **15,095**
- Fraudulent rows: **712**
- Multi-row near-duplicate groups: **651**
- Rows in multi-row near-duplicate groups: **1,785**
- Split: **70% diagnostic Train / 15% diagnostic Validation / 15% diagnostic Holdout**
- Split rule: ordinary label stratification; `group_id` deliberately ignored
- Seeds: **0 to 9**
- Model: fixed TF-IDF Logistic Regression baseline
- Threshold: selected on each diagnostic Validation partition by maximum fraud F1
- Official Train, Validation and Test files used: **No**

## Holdout performance across 10 seeds

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| pr_auc | 0.8856 | 0.0228 | 0.8588 | 0.9259 |
| roc_auc | 0.9834 | 0.0049 | 0.9759 | 0.9895 |
| accuracy | 0.9844 | 0.0024 | 0.9819 | 0.9903 |
| fraud_precision | 0.8886 | 0.0579 | 0.8091 | 0.9605 |
| fraud_recall | 0.7551 | 0.0569 | 0.6729 | 0.8411 |
| fraud_f1 | 0.8137 | 0.0297 | 0.7882 | 0.8867 |
| macro_f1 | 0.9028 | 0.0154 | 0.8894 | 0.9408 |

## Near-duplicate contamination across 10 seeds

| Measure | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| cross_split_near_groups | 348.3 | 326 | 367 |
| cross_split_fraud_near_groups | 21.4 | 16 | 26 |
| cross_split_legitimate_near_groups | 326.9 | 309 | 346 |
| rows_in_cross_split_near_groups | 1116.2 | 1065 | 1165 |
| train_validation_near_groups | 173.3 | 154 | 188 |
| train_holdout_near_groups | 185.0 | 170 | 200 |
| validation_holdout_near_groups | 57.8 | 50 | 70 |
| validation_rows_group_matching_train | 218.1 | 193 | 238 |
| validation_fraud_rows_group_matching_train | 9.8 | 6 | 13 |
| validation_legitimate_rows_group_matching_train | 208.3 | 184 | 229 |
| holdout_rows_group_matching_train | 221.5 | 206 | 235 |
| holdout_fraud_rows_group_matching_train | 10.7 | 5 | 15 |
| holdout_legitimate_rows_group_matching_train | 210.8 | 195 | 221 |

## Interpretation

Condition B removes exact duplicate rows, but intentionally permits near-duplicate groups to cross diagnostic partitions. Conditions A, B and C must be compared before drawing conclusions about leakage-related performance inflation.
