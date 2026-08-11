# Optimized BERT vs Paper Reported

## Design

| Arm | Data | Split | Model recipe |
|---|---|---|---|
| **Optimized BERT** | EMSCAD full (17,880) | 80/20 + 10% of train as Val, seed 42 | max_len=512, lr=5e-5, batch=8×accum2, epochs=3, raw class weight |
| **Paper reported** | EMSCAD (authors) | 80/20, seed 42 | numbers from the paper (reference) |

## Results

| Protocol | Eval | Fraud F1 | Fraud P | Fraud R | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC | Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized BERT | validation | 0.8855 | 0.9355 | 0.8406 | 0.9637 | 0.9188 | 0.9400 | 0.8895 | 0.9757 | 0.9895 |
| Optimized BERT | test | 0.9053 | 0.9273 | 0.8844 | 0.9607 | 0.9404 | 0.9503 | 0.9405 | 0.9931 | 0.9911 |
| Paper reported (reference) | test | 0.8802 | 0.9130 | 0.8497 | 0.94 | 0.92 | 0.93 | — | 0.9909 | 0.9888 |

Source table: `comparison_summary.csv`. Detailed evaluate artifacts: `results/metrics_summary.json`.

## Interpretation notes

- Optimized BERT follows the paper-aligned split/recipe and slightly exceeds the paper's reported test Fraud F1 (0.9053 vs 0.8802).
- Macro P/R/F1 for Paper reported come from the paper table; Optimized BERT macros are computed on the same threshold (0.32) used for fraud metrics.
- Threshold is selected on validation only, then frozen for test.
