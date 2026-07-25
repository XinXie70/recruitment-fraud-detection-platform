# DNN + TF-IDF 共享训练库

本目录是 **DNN (train_cv)** 使用的共享实现（数据加载、TF-IDF/SVD、DNN、CV 训练循环）。

请不要直接以本目录的 `main.py` 作为正式实验入口；正式保留实验为：

```powershell
. E:\ml\activate.ps1
cd F:\final-version\independent_ml_workflow
python src/models/dnn_fraud_train_cv/main.py
```

说明见 `src/models/dnn_fraud_train_cv/README.md`，产物在 `outputs/dnn_fraud_train_cv/`。
