# BERT 可运行代码包

本目录与同级的 `model_weights/`、`model_results/`、以及 `data/splits/` 一起使用。

## 依赖安装

```powershell
. E:\ml\activate.ps1
pip install -r model_code/requirements-bert.txt
pip install -r requirements.txt
```

## 运行 BERT（class-weighted）

```powershell
cd model_code/bert

# 训练 class-weighted 模型，权重写入 ../../model_weights/bert/bert_class_weighted
python train_bert.py

# 用已保存权重在 test 上评估
python evaluate_bert.py --checkpoint_dir ..\..\model_weights\bert\bert_class_weighted\best

# 单条推理
python predict_bert.py --text "Urgent work-from-home role. Send bank details."
```

## Git LFS（权重文件）

BERT `model.safetensors` 约 418MB，超过 GitHub 普通文件 100MB 限制，必须用 Git LFS：

```powershell
git lfs install
git lfs pull
```
