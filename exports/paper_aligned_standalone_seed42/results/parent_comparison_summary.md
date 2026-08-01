# BERT Paper Protocol vs Project Protocol

## Design

| Arm | Data | Split | Model recipe |
|---|---|---|---|
| **Paper-aligned** | EMSCAD full (17,880) | 80/20 + 10% of train as Val, seed 42 | max_len=256, lr=5e-5, batch=16, epochs=3, raw class weight |
| **Project improved** | Condition B (~15,807) | 70/15/15, seed 42 | max_len=384, lr=2e-5, oversample×3 + sqrt_clip, min_recall≥0.85 |
| **Paper reported** | EMSCAD (authors) | 80/20, seed 42 | numbers from the paper (reference) |

## Results

| Protocol | Eval | Fraud F1 | Precision | Recall | PR-AUC | ROC-AUC | Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Paper-aligned | validation | 0.8571 | 0.9474 | 0.7826 | 0.9151 | 0.9878 | 0.9874 |
| Paper-aligned | test | 0.9003 | 0.9430 | 0.8613 | 0.9341 | 0.9921 | 0.9908 |
| Paper reported (reference) | test | 0.8802 | 0.9130 | 0.8497 | — | 0.9909 | 0.9888 |
| Project improved CW | validation | 0.8922 | 0.9381 | 0.8505 | 0.9004 | 0.9755 | 0.9907 |
| Project improved CW | holdout/test | 0.8473 | 0.8958 | 0.8037 | 0.8700 | 0.9635 | 0.9869 |

## Interpretation notes

- Paper-aligned and Project arms use **different data/split protocols**; gaps are expected.
- If Paper-aligned Test F1 approaches the paper's 0.8802, the remaining gap to Project Holdout is largely **evaluation design**, not a broken BERT implementation.
- Project Val F1 can exceed Paper Test F1 while Holdout is slightly lower — that is consistent with a stricter/different holdout.

- Paper artifacts: `F:/final-version-1/capstone-project-26t2-9900-h09c-almond/experiments/bert_paper_vs_project/paper_protocol_emscad_seed42`
- Project artifacts: `F:/final-version-1/capstone-project-26t2-9900-h09c-almond/experiments/bert_cw_improved_condition_b_seed42`
