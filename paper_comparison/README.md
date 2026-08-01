# Paper comparison experiment

This folder contains the random-split comparison with Taneja et al. (2025),
*Fraud-BERT: transformer based context aware online recruitment fraud
detection*.

The experiment uses the paper's outer split:

- original EMSCAD data with 17,880 advertisements;
- 80% Train and 20% Test;
- stratified random split with seed `12342`;
- the same 12 input fields used by the paper.

For a fair development process, 20% of the outer Train portion is used as an
internal Validation set. This gives 64% internal Train, 16% Validation and 20%
final Test. Validation is used for model and threshold selection. The final
model is refitted on the complete outer 80% Train before one Test evaluation.

The improved Logistic Regression retains the settings used in our existing LR
workflow: train-only TF-IDF, unigram/bigram selection, class-weight selection,
three-fold cross-validation and validation-based threshold selection.

Run from the project root:

```bash
python paper_comparison/train_improved_lr.py
```

Outputs are stored in `paper_comparison/results/improved_lr/`. The split
assignment and prediction files should be reused by BERT so that the two models
can be combined safely later.

## LR aligned to the betterBERT branch

The second LR experiment uses the exact saved assignments from the team
`betterBERT` branch (commit `4c3b2b2`). It keeps the branch's 12,873/1,431/3,576
Train/Validation/Test allocation and trains the existing improved LR on the
shared five-field `combined_text`.

```bash
python paper_comparison/train_improved_lr_betterbert_split.py
```

Its outputs are stored in
`paper_comparison/results/improved_lr_betterbert_split/`. These Validation and
Test prediction files are the LR inputs for an ensemble with betterBERT.

Important metric note: the paper's reported F1 values are macro averages. This
experiment reports both Macro F1 and Fraud F1.

## BetterBERT Validation predictions

BetterBERT Validation probabilities are required before selecting ensemble
weights. They can be regenerated from the checkpoint already stored in the
`betterBERT` branch:

```bash
python paper_comparison/generate_betterbert_validation_predictions.py \
  --checkpoint exports/paper_aligned_standalone_seed42/weights/best \
  --validation exports/paper_aligned_standalone_seed42/data/splits/validation.csv.gz \
  --output paper_comparison/results/improved_lr_betterbert_split/betterbert_validation_predictions.csv \
  --batch-size 32 \
  --max-length 256
```

The saved predictions were checked against the branch's existing Test
predictions before the ensemble experiment.

## LR + BetterBERT ensemble v1

The ensemble uses aligned LR and BetterBERT probabilities. Five simple weight
combinations and thresholds from 0.01 to 0.99 were compared on Validation.
Test was evaluated only after the configuration was selected.

```bash
python paper_comparison/run_lr_betterbert_ensemble.py \
  --lr-validation paper_comparison/results/improved_lr_betterbert_split/validation_predictions.csv \
  --bert-validation paper_comparison/results/improved_lr_betterbert_split/betterbert_validation_predictions.csv \
  --lr-test paper_comparison/results/improved_lr_betterbert_split/test_predictions.csv \
  --bert-test exports/paper_aligned_standalone_seed42/results/predictions_bert_paper_protocol.csv \
  --output-dir paper_comparison/results/lr_betterbert_ensemble_v1
```

The selected configuration is 25% LR and 75% BetterBERT with threshold 0.87.
Full metrics and predictions are in
`paper_comparison/results/lr_betterbert_ensemble_v1/`.

## Folder contents

- `data/`: the fixed BetterBERT split assignments used by LR;
- `train_improved_lr.py`: the seed-12342 paper comparison;
- `train_improved_lr_betterbert_split.py`: LR trained on the BetterBERT split;
- `generate_betterbert_validation_predictions.py`: saved-checkpoint inference;
- `run_lr_betterbert_ensemble.py`: Validation selection and frozen Test run;
- `results/improved_lr/`: the closer paper-seed LR comparison;
- `results/improved_lr_betterbert_split/`: aligned LR and BERT scores;
- `results/lr_betterbert_ensemble_v1/`: ensemble configuration and results.

Trained model binaries and local Python environments are not duplicated in
this folder. The BetterBERT weights remain in the existing Git LFS checkpoint.
