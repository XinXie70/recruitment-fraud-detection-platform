# Test Performance Summary v1

固定 Test 集包含 2,372 条广告（107 条欺诈）。SHA-256：

`85457e3dc60c05a68aaea9e9f433b2fb8bc9cb775dbf0221d61efe2e1cbd6257`

## 三模型 Test 结果

| Model | PR-AUC | ROC-AUC | Fraud Precision | Fraud Recall | Fraud F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8735 | 0.9852 | 0.7642 | 0.7570 | 0.7606 | 2240 | 25 | 26 | 81 |
| BERT (class weighted) | 0.7942 | 0.9684 | 0.8302 | 0.8224 | 0.8263 | 2247 | 18 | 19 | 88 |
| LR + BERT Ensemble | 0.9058 | 0.9905 | 0.8817 | 0.7664 | 0.8200 | 2254 | 11 | 25 | 82 |

## 结果来源

- `reports/models/logistic_regression/test_metrics.json`
- `reports/models/bert/test_metrics.json`
- `reports/models/ensemble_lr_bert/test_metrics.json`

Test 集仅用于最终评估，不得用于超参、阈值或风险边界调优。
