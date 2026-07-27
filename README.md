# Fake Job Advertisement Detection — LR + BERT + Ensemble

虚假招聘广告检测项目，保留三个生产模型：

| 模型 | 说明 |
|---|---|
| **Logistic Regression** | TF-IDF + LR baseline |
| **BERT (class-weighted)** | 带类别权重的 BERT 微调 |
| **LR + BERT Ensemble** | 加权融合 + 风险分级 |

## 目录结构

```text
├── data/splits/              # 固定 train / validation / test（唯一训练数据组）
├── model_code/bert/          # BERT 训练与推理代码
├── model_weights/            # LR joblib + BERT safetensors
├── model_results/bert/       # BERT 评估结果
├── reports/models/           # 三模型指标、预测、ensemble 与 risk band 配置
├── src/
│   ├── data_pipeline/        # 共享数据处理
│   ├── models/               # LR 训练、ensemble、risk band 脚本
│   └── api/                  # FastAPI 推理服务
└── scripts/smoke_test_api.py # E 盘冒烟测试
```

## 快速开始

### 1. 环境（E 盘 CUDA）

```powershell
git lfs install
git lfs pull
. E:\ml\activate.ps1
cd f:\final-version2\capstone-project-26t2-9900-h09c-almond
pip install -r requirements.txt
pip install -r model_code/requirements-bert.txt
```

### 2. 训练 LR（若权重不存在）

```powershell
python src/models/logistic_regression/train_baseline.py
```

权重输出：`model_weights/logistic_regression/logistic_regression_baseline.joblib`

### 3. 启动 FastAPI

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger 文档：http://127.0.0.1:8000/docs

### 4. 冒烟测试（E 盘）

```powershell
python scripts/smoke_test_api.py
```

结果写入 `E:\ml\smoke-test-results\fake-job-api-smoke.json`

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/predict/lr` | Logistic Regression 预测 |
| POST | `/predict/bert` | BERT (class-weighted) 预测 |
| POST | `/predict/ensemble` | LR+BERT 融合预测 |
| POST | `/predict/ensemble/risk` | 融合预测 + 风险分级 (Low/Suspicious/High) |
| POST | `/predict/lr/batch` | LR 批量预测（最多 100 条） |

### 请求示例

```json
POST /predict/ensemble/risk
{
  "text": "Urgent work-from-home job. Send bank details to apply."
}
```

### 响应示例

```json
{
  "model": "ensemble_lr_bert_class_weighted",
  "fraud_score": 0.82,
  "threshold": 0.62,
  "prediction": 1,
  "predicted_label": "Fraudulent",
  "weights": {
    "logistic_regression_baseline": 0.6,
    "bert_class_weighted": 0.4
  },
  "lr_fraud_score": 0.75,
  "bert_fraud_score": 0.91,
  "risk_score": 82.0,
  "risk_level": "High",
  "binary_threshold": 0.62,
  "low_suspicious_threshold": 0.1567,
  "suspicious_high_threshold": 0.62
}
```

## 后端集成示例（Python）

```python
import httpx

resp = httpx.post(
    "http://127.0.0.1:8000/predict/ensemble/risk",
    json={"text": job_ad_text},
    timeout=30.0,
)
result = resp.json()
risk_level = result["risk_level"]      # Low | Suspicious | High
risk_score = result["risk_score"]      # 0–100
```

## 数据集

固定划分位于 `data/splits/`（70% Train / 15% Validation / 15% Test，seed 42）。

原始数据需自行获取 EMSCAD 并重命名为 `data/raw/emscad_v1.csv`，详见 `DATA_CONTRACT_V1.md`。

## 模型训练与评估流程

```powershell
# LR baseline
python src/models/logistic_regression/train_baseline.py

# BERT 评估（已有权重）
cd model_code/bert
python evaluate_bert.py --checkpoint_dir ..\..\model_weights\bert\bert_class_weighted\best

# Ensemble（validation 搜权重，test 应用锁定配置）
cd ..\..
python src/models/build_ensemble.py --mode validation
python src/models/build_ensemble.py --mode test

# 风险分级
python src/models/build_risk_bands.py --mode validation
python src/models/build_risk_bands.py --mode test
```

## 风险分级配置

冻结配置：`reports/models/risk_band_v1_config.json`

| 等级 | 规则 |
|---|---|
| **Low** | fraud_score < 0.1567 |
| **Suspicious** | 0.1567 ≤ fraud_score < 0.62 |
| **High** | fraud_score ≥ 0.62 |

`risk_score = fraud_score × 100`

## 保留的结果文件

- `reports/models/logistic_regression/` — LR 指标与预测
- `reports/models/bert/` — BERT 指标与预测
- `reports/models/ensemble_lr_bert/` — 融合配置与预测
- `reports/models/risk_band_v1_*` — 风险分级配置与结果

更多实验约定见 `MODEL_EXPERIMENT_CONTRACT_V1.md`。
