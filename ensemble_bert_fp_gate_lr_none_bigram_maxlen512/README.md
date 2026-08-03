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

## 三档风险边界

High 参数由原有 FP-gate Validation 搜索确定。Low 参数通过 Validation
trade-off 搜索确定，Test 不参与选参。

```powershell
cd ensemble_bert_fp_gate_lr_none_bigram_maxlen512

# 只使用 Validation 选择并冻结边界
python code/select_risk_boundaries.py --mode select

# 冻结后才可将配置应用到 Test；此模式不会重新搜索阈值
python code/select_risk_boundaries.py --mode apply-test
```

当前冻结规则：

```text
High: BERT score >= 0.30 and LR score >= 0.06
Low:  BERT score < 0.0024
Otherwise: Suspicious
```

BERT 是主风险评分模型。LR 只作为 High 候选的 false-positive gate，
不参与 Low boundary。

正式 risk score：

```text
Normally:       risk_score = BERT score
If gate fires:  risk_score = LR score
Display:        risk_score_100 = risk_score * 100
```

Risk level仍由原始BERT-LR gate规则决定，不能只根据这个混合来源的
risk score反推。这是操作分数，不是经过校准的欺诈概率。原始BERT和LR
分数会同时保留。

Risk score输出：

- `results/RISK_SCORE_REPORT.md`
- `results/risk_score_metrics.csv`
- `results/test_risk_levels.csv`
- `results/test_risk_level_summary.json`

主要输出：

- `results/risk_boundary_config.json`
- `results/RISK_BOUNDARY_REPORT.md`
- `results/low_boundary_tradeoff.csv`
- `results/low_boundary_target_comparison.csv`
- `results/validation_risk_levels.csv`
