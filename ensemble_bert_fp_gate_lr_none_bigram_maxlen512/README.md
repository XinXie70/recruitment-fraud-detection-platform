# BERT + LR(None/bigram/no-CV) FP-gate

BERT（`max_length=512`）为主；当 BERT 判欺诈且 LR 分数低于门控阈值时，改判正常（压 FP）。

## 组成

| 分支 | 路径 |
|---|---|
| LR | `../lr_none_bigram_no_cv_paper_aligned_seed42`（`class_weight=None`，bigram，无 CV） |
| BERT | `../retrain_paper_aligned_seed42_maxlen512` |
| Ensemble | 本目录（FP-gate） |

Test 结果只报告上述三支：LR / BERT / Ensemble。

## 复现步骤

```powershell
. E:\ml\activate.ps1

# 1) 复现 LR（写 predictions + metrics）
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\lr_none_bigram_no_cv_paper_aligned_seed42\code
python train_lr_none_bigram_no_cv.py

# 2) 复现 FP-gate（依赖冻结的 BERT 预测与上一步 LR 预测）
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\ensemble_bert_fp_gate_lr_none_bigram_maxlen512\code
python run_fp_gate_ensemble.py
```

需要已存在：

- `results/bert_validation_predictions.csv`（本目录）
- `../retrain_paper_aligned_seed42_maxlen512/results/predictions_bert_paper_protocol_maxlen512.csv`

对外风险输出口径见：[RISK_SCORE_AND_LEVEL.md](./RISK_SCORE_AND_LEVEL.md)
