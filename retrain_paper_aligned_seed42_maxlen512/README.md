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
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\retrain_paper_aligned_seed42_maxlen512\code
python run_bert_paper_vs_project.py
```

权重 → `weights/`；结果 → `results/`
