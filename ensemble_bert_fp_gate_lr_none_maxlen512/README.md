# BERT + LR FP-gate

BERT 为主；当 BERT 判欺诈且 LR 分数低于门控阈值时，改判正常（压 FP）。

对外风险输出口径见：[RISK_SCORE_AND_LEVEL.md](./RISK_SCORE_AND_LEVEL.md)

- **Risk score（0–1）** = `bert_score`（不改写）
- **Risk level** = 无 / 低 / 高（分数阈值 + FP-gate 降档）

```powershell
. E:\ml\activate.ps1
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\ensemble_bert_fp_gate_lr_none_maxlen512\code
python run_fp_gate_ensemble.py
```
