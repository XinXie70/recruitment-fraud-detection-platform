# Fraud Detection Flask API

为当前冻结的 **LR / BERT / Ensemble / Risk score / Risk level** 提供可被前后端调用的 HTTP 接口，并附带 Swagger 文档。

Docker / 云端部署文件已单独放在：[`../docker/`](../docker/)

## 目录

```text
api_flask/
  app.py                 # Flask 入口 + Swagger
  wsgi.py                # Gunicorn 入口（Docker 使用）
  settings.py            # 模型路径与冻结阈值
  requirements.txt
  services/
    text_utils.py
    lr_service.py
    bert_service.py
    ensemble_service.py
```

## 依赖模型

| 组件 | 来源 |
|---|---|
| LR | `../lr_none_bigram_no_cv_paper_aligned_seed42` |
| BERT | `../retrain_paper_aligned_seed42_maxlen512` |
| Ensemble / Risk | `../ensemble_bert_fp_gate_lr_none_bigram_maxlen512/results/config.json` + `../ensemble_bert_fp_gate_lr_none_bigram_maxlen512/risk_level/risk_boundary_config.json` |

## 本地启动

```powershell
. E:\ml\activate.ps1
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond\api_flask
pip install -r requirements.txt
python app.py
```

- API：`http://127.0.0.1:5000`
- Swagger：`http://127.0.0.1:5000/apidocs/`
- OpenAPI：`http://127.0.0.1:5000/apispec.json`

已启用 CORS，前端可直接跨域调用。

## Docker / 云端

详见 [`../docker/README.md`](../docker/README.md)。

```powershell
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond\docker
docker compose up --build -d
```

## 主要接口

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/config` | 冻结阈值与模型路径 |
| POST | `/predict/lr` | 仅 LR |
| POST | `/predict/bert` | 仅 BERT |
| POST | `/predict/ensemble` | BERT + LR FP-gate |
| POST | `/predict/risk` | `risk_score` + `risk_level` |
| POST | `/predict/all` | 一次返回全部结果（推荐前端） |
| POST | `/predict/batch` | 批量 risk（最多 100 条） |

## 请求体示例

自由文本：

```json
{
  "record_id": "demo_001",
  "text": "Urgent work from home job. Send your bank details to apply."
}
```

结构化字段（可选）：

```json
{
  "record_id": "demo_002",
  "title": "Account Executive",
  "company_profile": "We help companies grow.",
  "description": "Full-time remote role. Send passport and bank details.",
  "requirements": "No experience required",
  "benefits": "High salary guaranteed"
}
```

## `/predict/all` 响应字段要点

- `lr.lr_score` / `lr.predicted_label`
- `bert.bert_score` / `bert.predicted_label`
- `ensemble.predicted_label` / `ensemble.gate_triggered`
- `risk.risk_score` / `risk.risk_score_100` / `risk.risk_level`
- `risk.risk_score_source`：`bert` 或 `lr_gate`
- `risk.decision_reason`

风险规则与实验一致：

```text
High:        BERT >= 0.30 and LR >= 0.06
Low:         BERT < 0.0024
Suspicious:  otherwise (含被 LR gate 降级的 High 候选)

risk_score = BERT score
if gate fires: risk_score = LR score
risk_score_100 = risk_score * 100
```

## curl 示例

```powershell
curl -X POST http://127.0.0.1:5000/predict/all `
  -H "Content-Type: application/json" `
  -d "{\"record_id\":\"demo_001\",\"text\":\"Urgent work from home. Send bank details now.\"}"
```
