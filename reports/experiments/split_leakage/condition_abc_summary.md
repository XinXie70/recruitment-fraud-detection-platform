# Split Leakage Experiment: Conditions A, B and C

## Conditions

- A: no deduplication + ordinary stratified random split.
- B: exact deduplication + ordinary stratified random split.
- C: exact deduplication + group-aware split.
- All results are means over seeds 0–9 on diagnostic Holdout partitions.

## Mean diagnostic Holdout results

| Model | Metric | A | B | C | B − A | C − B |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | PR-AUC | 0.9071 | 0.8856 | 0.8731 | -0.0215 | -0.0125 |
| Logistic Regression | ROC-AUC | 0.9843 | 0.9834 | 0.9818 | -0.0009 | -0.0015 |
| Logistic Regression | Fraud precision | 0.9285 | 0.8886 | 0.8527 | -0.0399 | -0.0358 |
| Logistic Regression | Fraud recall | 0.7777 | 0.7551 | 0.7548 | -0.0226 | -0.0003 |
| Logistic Regression | Fraud F1 | 0.8452 | 0.8137 | 0.7970 | -0.0315 | -0.0167 |
| Logistic Regression | Macro F1 | 0.9190 | 0.9028 | 0.8939 | -0.0162 | -0.0089 |
| Linear SVM | PR-AUC | 0.9276 | 0.9142 | 0.8997 | -0.0134 | -0.0145 |
| Linear SVM | ROC-AUC | 0.9846 | 0.9848 | 0.9809 | +0.0003 | -0.0040 |
| Linear SVM | Fraud precision | 0.9411 | 0.9153 | 0.8840 | -0.0258 | -0.0314 |
| Linear SVM | Fraud recall | 0.8392 | 0.7916 | 0.8129 | -0.0476 | +0.0213 |
| Linear SVM | Fraud F1 | 0.8860 | 0.8468 | 0.8464 | -0.0392 | -0.0004 |
| Linear SVM | Macro F1 | 0.9403 | 0.9200 | 0.9197 | -0.0202 | -0.0003 |

## Leakage checks

- Condition A deliberately allows exact duplicates to cross splits.
- Condition B removes exact duplicates but still allows near-duplicate groups to cross splits.
- Condition C had **0 groups crossing splits for every seed**.
- In Condition C, Linear SVM calibration folds are also group-aware; therefore its B-to-C difference measures the complete strict workflow, not only the outer split.
- Condition C Holdout fraud count range: **106–107**.

## Interpretation rule

A-to-B shows the effect of exact deduplication under random splitting. For Logistic Regression, B-to-C isolates the outer group-aware split because both conditions use the same input and training procedure. For Linear SVM, B-to-C also includes group-aware calibration in Condition C and should be described as an end-to-end strict-workflow comparison. These results identify leakage risk, but they do not prove how any external paper performed its split.
