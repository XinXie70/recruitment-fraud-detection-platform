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
| Paper-aligned | validation | 0.8855 | 0.9355 | 0.8406 | 0.8895 | 0.9757 | 0.9895 |
| Paper-aligned | test | 0.9053 | 0.9273 | 0.8844 | 0.9405 | 0.9931 | 0.9911 |
| Paper reported (reference) | test | 0.8802 | 0.9130 | 0.8497 | — | 0.9909 | 0.9888 |

## Interpretation notes

- Paper-aligned and Project arms use **different data/split protocols**; gaps are expected.
- If Paper-aligned Test F1 approaches the paper's 0.8802, the remaining gap to Project Holdout is largely **evaluation design**, not a broken BERT implementation.
- Project Val F1 can exceed Paper Test F1 while Holdout is slightly lower — that is consistent with a stricter/different holdout.

- Paper artifacts: `F:/better-BERT/retrain_paper_aligned_seed42_maxlen512`
- Project artifacts: `F:/better-BERT/retrain_paper_aligned_seed42_maxlen512/experiments/bert_cw_improved_condition_b_seed42`
