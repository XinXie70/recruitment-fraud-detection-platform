# 重训：paper-aligned seed42，仅改 max_length=512

相对 `retrain_paper_aligned_seed42`，**只改序列长度 256→512**。

其余保持一致：
- EMSCAD 17,880，seed 42
- class weight：原始逆频率
- lr=5e-5，epochs=3
- 阈值：validation 上最大化 Fraud F0.5
- 有效 batch 仍为 16（RTX 3060 上用 batch=8 × accum=2，避免 OOM）

## 运行

```powershell
. E:\ml\activate.ps1
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond\retrain_paper_aligned_seed42_maxlen512\code
python run_bert_paper_vs_project.py
```

权重 → `weights/`；结果 → `results/`

## Ensemble 所需 Validation 分数

FP-gate ensemble（`../ensemble_bert_fp_gate_lr_none_bigram_maxlen512`）需要
Validation 上的 BERT 分数文件。用已训练好的 checkpoint 导出：

```powershell
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond\retrain_paper_aligned_seed42_maxlen512\code
python export_validation_predictions.py
```

输出：

- `results/bert_validation_predictions.csv`（规范路径）
- 同步刷新 ensemble 目录下同名文件（若该目录存在）
