# DNN train-only CV（正式保留实验）

仅在 `train.csv` 内执行 StratifiedGroupKFold、OOF 预测与阈值搜索；`validation.csv`
不参与交叉验证。最终在 train+validation 上重训，并对 test 评估一次。

## 运行

```powershell
. E:\ml\activate.ps1
cd F:\final-version\independent_ml_workflow
python src/models/dnn_fraud_train_cv/main.py
```

## 输出

`outputs/dnn_fraud_train_cv/`

共享训练实现位于 `src/models/dnn_fraud/`（库代码，非独立实验入口）。
