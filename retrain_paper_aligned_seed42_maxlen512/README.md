# BERT retraining — paper-aligned seed 42 with `max_length=512`

Compared with `retrain_paper_aligned_seed42`, this experiment changes only the
sequence length from 256 to 512.

All other settings remain fixed:

- 17,880 EMSCAD records with seed 42
- Original inverse-frequency class weights
- Learning rate `5e-5` and 3 epochs
- Threshold selected by maximising Fraud F0.5 on validation data
- Effective batch size 16; batch size 8 with two accumulation steps on an
  RTX 3060 to avoid out-of-memory errors

## Run

From the repository root, activate the project environment and run:

```bash
python retrain_paper_aligned_seed42_maxlen512/code/run_bert_paper_vs_project.py
```

The script writes weights to `weights/` and results to `results/` within this
experiment directory.

## Validation scores required by the ensemble

The FP-gate ensemble in `../ensemble_bert_fp_gate_lr_none_bigram_maxlen512`
requires BERT scores for the validation split. Export them from the trained
checkpoint with:

```bash
python retrain_paper_aligned_seed42_maxlen512/code/export_validation_predictions.py
```

The script writes the canonical output to
`results/bert_validation_predictions.csv` and, when the ensemble directory is
available, refreshes its copy of the same file.
