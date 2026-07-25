# model_results

存放 BERT / DNN 的评估结果、预测与图表（不含大权重文件）。

```text
model_results/
├── bert/
│   ├── *.csv / *.json                 # 指标、预测、训练历史
│   ├── figures/
│   └── logs/
└── dnn/
    ├── final_test_metrics.csv
    ├── final_test_predictions.csv
    ├── cv_*.csv / oof_*.csv
    └── external_v2/                   # 可选外部评估
```
