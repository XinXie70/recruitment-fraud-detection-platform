# Paper-aligned 独立包（seed=42）

包含 Paper-aligned 实验所需的**模型代码、原始数据、切分脚本、参考切分、权重、结果**。

## 目录结构

| 路径 | 说明 |
|---|---|
| data/raw/emscad_v1.csv | **原始 EMSCAD**（18 列） |
| data/processed/ | Condition A 处理后的输入（5 字段 combined_text） |
| data/splits/ | 参考划分：ssignments + 	rain/alidation/	est（含 model_text） |
| code/ | Paper-aligned 训练相关代码 + **可复现切分脚本** |
| code/split_paper_aligned_data.py | 从原始数据按实验协议切分 |
| weights/best/ | 最佳 checkpoint（含 tokenizer、	hreshold.json） |
| 
esults/ | 指标、预测、曲线图、run config |
| SEED.txt | 42 |
| PROTOCOL.json | 协议摘要 |

## 种子与切分协议

- **Seed:** 42
- **外层:** 分层 **80% train_pool / 20% test**
- **内层:** 从 train_pool 再划 **10% → validation**（其余为 train）
- **规模:** train=12873 (fraud=624), val=1431 (fraud=69), test=3576 (fraud=173)

## 从原始数据重新切分

在本目录下：

`powershell
. E:\ml\activate.ps1
cd F:\final-version-1\capstone-project-26t2-9900-h09c-almond\exports\paper_aligned_standalone_seed42
python code/split_paper_aligned_data.py
# 或指定输出目录
python code/split_paper_aligned_data.py --raw data/raw/emscad_v1.csv --out data/splits
`

该脚本会：

1. 读取 data/raw/emscad_v1.csv
2. 生成 5 字段 combined_text + Paper-aligned model_text（分节标签）
3. 按 seed=42 做 80/20 + 10% val
4. 写出 ssignments.csv.gz 与 	rain/validation/test.csv.gz

已验证：从 raw 重切分与包内参考 data/splits/assignments.csv.gz **100% 一致**。

## 训练配方（Paper-aligned）

- ert-base-uncased
- max_length=256, lr=5e-5, epochs=3, atch=16
- raw inverse-frequency class weight
- 文本：[TITLE]/[COMPANY PROFILE]/... + 按节词上限

> 注意：这是项目 **Paper-aligned** 协议，不是 Fraud-BERT 论文 Table 4 的忠实复现（论文为 lr=1e-5 / maxlen=510 / 无 CW）。

## 结果（本包 weights 对应 run）

见 
esults/model_comparison.csv / test metrics JSON。参考 test：

- Fraud F1 ≈ **0.9003**
- macro-F1 ≈ **0.9477**
- threshold ≈ **0.99**

## 代码说明

code/ 内为原仓库 model_code/bert 中 Paper-aligned 相关文件（含 
un_bert_paper_vs_project.py、	rain_bert.py 等）。  
完整重训仍依赖 CUDA 环境（如 E:\ml）与 	ransformers / 	orch；切分脚本仅需 pandas / scikit-learn。
