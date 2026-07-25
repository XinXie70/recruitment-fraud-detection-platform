# Split Leakage Experiment: Condition B — Linear SVM

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
- Model: fixed TF-IDF Linear SVM baseline
- Calibration: Train-only three-fold stratified sigmoid calibration, without group information
- Threshold: selected on each diagnostic Validation partition by maximum fraud F1
- Official Train, Validation and Test files used: **No**

## Holdout performance across 10 seeds

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| pr_auc | 0.9142 | 0.0219 | 0.8872 | 0.9456 |
| roc_auc | 0.9848 | 0.0048 | 0.9787 | 0.9922 |
| accuracy | 0.9871 | 0.0017 | 0.9852 | 0.9899 |
| fraud_precision | 0.9153 | 0.0428 | 0.8396 | 1.0000 |
| fraud_recall | 0.7916 | 0.0550 | 0.7196 | 0.8692 |
| fraud_f1 | 0.8468 | 0.0233 | 0.8211 | 0.8835 |
| macro_f1 | 0.9200 | 0.0121 | 0.9068 | 0.9391 |

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
