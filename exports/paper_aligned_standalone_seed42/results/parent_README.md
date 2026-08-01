# BERT 对照实验：论文协议 vs 项目协议

| Arm | 说明 |
|---|---|
| `paper_protocol_emscad_seed42/` | 对齐论文：EMSCAD 全量、80/20+Val、seed42、lr=5e-5、max_len=256、3 epoch、原始 class weight |
| （已有）`../bert_cw_improved_condition_b_seed42/` | 项目改进：Condition B、70/15/15、seed42、improved CW |

对比表生成后见：
- `comparison_summary.md`
- `comparison_summary.csv`
