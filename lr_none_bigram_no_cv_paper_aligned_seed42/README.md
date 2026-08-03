# LR（无 class_weight + bigram + 无 CV）— paper-aligned seed42

| 设置 | 本实验 |
|---|---|
| bigram | 固定 `ngram_range=(1, 2)` |
| class_weight | **`None`（不使用）** |
| Train 内 3-fold CV | **不使用**（固定 `C=1.0`） |
| 阈值 | Validation 最大化 Fraud F1；Test 只评一次 |

同一套 seed42 划分、Train-only TF-IDF。

## 数据

- 来源：`../retrain_paper_aligned_seed42_maxlen512/data/splits/`
- seed：**42**
- 规模：Train 12,873 / Validation 1,431 / Test 3,576
- 文本字段：`combined_text`

## 复现

固定项：

- 划分与输入文本文件
- `random_state=42`、`PYTHONHASHSEED=42`
- 超参：`ngram_range=(1,2)`、`C=1.0`、`class_weight=None`
- Validation 选阈值，Test 只评一次

包版本写入 `results/config.json` 的 `reproducibility.package_versions`。

对应 ensemble：`../ensemble_bert_fp_gate_lr_none_bigram_maxlen512`

## 运行

```powershell
. E:\ml\activate.ps1
cd F:\better-BERT\capstone-project-26t2-9900-h09c-almond\lr_none_bigram_no_cv_paper_aligned_seed42\code
python train_lr_none_bigram_no_cv.py
```

模型 → `artifacts/`；指标与预测 → `results/`
