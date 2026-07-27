# BERT / DNN 可运行代码包

本目录与同级的 `model_weights/`、`model_results/`、以及 `data/splits/` 一起使用。
从 GitHub 克隆后，三者齐全即可训练或加载已有权重评估。

## 目录

```text
independent_ml_workflow/
├── data/splits/          # 固定 train/validation/test（必填）
├── model_code/           # 本目录：BERT + DNN 源码
├── model_weights/        # 训练好的参数
└── model_results/        # 指标、预测、图表
```

## 依赖安装

BERT（PyTorch + CUDA 环境，例如本机 `E:\ml\venv`）：

```powershell
. E:\ml\activate.ps1
pip install -r model_code/requirements-bert.txt
```

DNN（TensorFlow）：

```powershell
. E:\ml\activate.ps1
pip install -r model_code/requirements-dnn.txt
```

## 运行 BERT

```powershell
cd model_code/bert

# 训练两组对照（None / Class weight），权重写入 ../../model_weights/bert
python run_bert_experiments.py

# 用已保存权重在 test 上评估
python evaluate_bert.py --checkpoint_dir ..\..\model_weights\bert\bert_class_weighted\best

# 单条推理
python predict_bert.py --text "Urgent work-from-home role. Send bank details."

# Split-leakage A/B/C（class-weighted BERT，使用 data/experiment_splits）
# 权重 → model_weights/bert/split_leakage/；结果 → model_results/bert/split_leakage/
python run_bert_split_leakage.py
python run_bert_split_leakage.py --conditions A B C --seeds 0 1 2 --skip_existing
```

## 运行 DNN（train-only CV）

```powershell
cd <repo>/independent_ml_workflow

# 完整训练（CV + 最终重训）。结果→model_results/dnn，权重→model_weights/dnn
python model_code/dnn/train_cv/main.py

# 仅加载已有权重评估 test
python model_code/dnn/evaluate_saved.py
```

## Git LFS（权重文件）

BERT `model.safetensors` 约 418MB/个，超过 GitHub 普通文件 100MB 限制，必须用 Git LFS：

```powershell
git lfs install
git lfs pull
```

若克隆后权重缺失，先执行 `git lfs pull`。
