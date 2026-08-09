# Model weights

This directory stores the locked artifacts used by the standalone LR/BERT
reference API in `src/api/`. Large binary artifacts are tracked with Git LFS.

## Directory layout

```text
model_weights/
├── logistic_regression/
│   └── logistic_regression_baseline.joblib
└── bert/bert_class_weighted/best/
    ├── model.safetensors
    ├── config.json
    ├── threshold.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── train_meta.json
```

These artifacts are separate from weights produced by experiments under
`retrain_paper_aligned_seed42_maxlen512/` and model-specific saved artifacts
under `model/final_model_pipelines/`.

## Retrieve artifacts

Install Git LFS before cloning, or retrieve the binary objects after cloning:

```bash
git lfs install
git lfs pull
```

If a model file starts with `version https://git-lfs.github.com/spec/v1`, only
the pointer is present. Run `git lfs pull` before starting inference.

## Regenerate the LR artifact

From the repository root:

```bash
python src/models/logistic_regression/train_baseline.py
```

This writes
`model_weights/logistic_regression/logistic_regression_baseline.joblib`.

The BERT artifact is produced by the training workflow documented in
[`model_code/README.md`](../model_code/README.md). Do not replace locked weights
without also recording the training configuration, threshold, evaluation
results, and artifact provenance.
