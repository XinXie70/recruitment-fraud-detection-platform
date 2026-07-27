# Test Performance Summary v1

This report brings the currently available model results into one table. The
fixed Test set contains 2,372 advertisements, including 107 fraudulent
advertisements. Its SHA-256 is:

`85457e3dc60c05a68aaea9e9f433b2fb8bc9cb775dbf0221d61efe2e1cbd6257`

## Results

| Model | Status | PR-AUC | ROC-AUC | Fraud Precision | Fraud Recall | Fraud F1 | Accuracy | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Primary | 0.8735 | 0.9852 | 0.7642 | 0.7570 | 0.7606 | 0.9785 | 2240 | 25 | 26 | 81 |
| Linear SVM | Primary | 0.9113 | 0.9885 | 0.9326 | 0.7757 | 0.8469 | 0.9874 | 2259 | 6 | 24 | 83 |
| DNN | Current result | 0.8045 | 0.9759 | 0.6560 | 0.7664 | 0.7069 | 0.9713 | 2222 | 43 | 25 | 82 |
| BERT (no class weight) | Primary BERT run | 0.8708 | 0.9780 | 0.8617 | 0.7570 | 0.8060 | 0.9836 | 2252 | 13 | 26 | 81 |
| BERT (class weighted) | Ablation | 0.7942 | 0.9684 | 0.8302 | 0.8224 | 0.8263 | 0.9844 | 2247 | 18 | 19 | 88 |

## Interpretation

- Linear SVM has the highest Test PR-AUC and Fraud F1 among the current primary
  runs.
- BERT without class weights remains the primary BERT run because it has the
  higher Validation PR-AUC. The class-weighted run is retained as an ablation
  and must not be selected because its Test Fraud F1 is higher.
- Logistic Regression and Linear SVM follow the shared protocol: model
  development uses Train, and the threshold is selected on Validation.
- The current DNN result is not directly comparable with the other primary
  results because its final model was trained on Train + Validation and its
  threshold was selected from Train out-of-fold predictions. It is included
  transparently as the currently available result.
- Test results must not be used for further hyperparameter, threshold, model
  variant, ensemble, or risk-boundary selection.

## Result sources

- `reports/models/logistic_regression/test_metrics.json`
- `reports/models/linear_svm/test_metrics.json`
- `model_results/dnn/final_test_metrics.csv`
- `model_results/bert/bert_test_metrics_bert_no_class_weight.json`
- `model_results/bert/bert_test_metrics_bert_class_weighted.json`
