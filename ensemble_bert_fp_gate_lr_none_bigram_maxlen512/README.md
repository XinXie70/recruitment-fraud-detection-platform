# BERT + LR(None/bigram/no-CV) FP-gate

BERT（`max_length=512`）为主；当 BERT 判欺诈且 LR 分数低于门控阈值时，改判正常（压 FP）。

LR 来源：`lr_none_bigram_no_cv_paper_aligned_seed42`
（`class_weight=None`，unigram+bigram，无 Train 内 CV）。

对外风险输出口径见：[RISK_SCORE_AND_LEVEL.md](./RISK_SCORE_AND_LEVEL.md)

```powershell
. E:\ml\activate.ps1
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\ensemble_bert_fp_gate_lr_none_bigram_maxlen512\code
python run_fp_gate_ensemble.py
```
