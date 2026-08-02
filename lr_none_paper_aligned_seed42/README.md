# LR（无 class_weight）— paper-aligned seed42

使用 `retrain_paper_aligned_seed42_maxlen512` 的同一套划分，训练 **不带类别权重** 的 Improved LR（TF-IDF + LogisticRegression）。

## 数据

- 来源：`../retrain_paper_aligned_seed42_maxlen512/data/splits/`
- seed：**42**
- 规模：Train 12,873 / Validation 1,431 / Test 3,576
- 文本字段：`combined_text`

## 协议

- `class_weight=None`（不重平衡少数类）
- Train 内 3-fold CV（PR-AUC）选择 `ngram_range` / `C`
- Validation 上最大化 Fraud F1 选阈值
- Test 仅做一次冻结评估

## 运行

```powershell
. E:\ml\activate.ps1
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\lr_none_paper_aligned_seed42\code
python train_lr_no_class_weight.py
```

模型 → `artifacts/`；指标与预测 → `results/`
