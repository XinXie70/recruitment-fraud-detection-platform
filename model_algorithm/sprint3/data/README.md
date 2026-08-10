# Dataset handling

The complete EMSCAD dataset and generated train/validation/test files are not
stored in Git. Obtain the team-approved source URL, then run:

```bash
EMSCAD_DATASET_URL="https://approved-source/emscad_v1.csv" ./download_data.sh
```

The script downloads to a temporary file, verifies the pinned SHA-256 digest,
and only then generates the processed dataset and deterministic splits. The
known digests for the current data bundle are recorded in `checksums.sha256`.
Verify an existing bundle from this directory with:

```bash
shasum -a 256 -c checksums.sha256
```

`samples/emscad_tiny.csv` is a synthetic, non-production fixture for smoke tests.
It is intentionally too small for training or evaluation.

The BERT `model.safetensors` file is managed by Git LFS. After cloning, use
`git lfs pull` when the full inference model is required.

## Existing Git history

Removing files from the current tree does not shrink old clones. A repository
administrator should schedule a coordinated migration after all branches are
backed up and collaborators are notified:

```bash
git lfs migrate import --everything --include="*.safetensors"
git filter-repo --path model_algorithm/sprint3/data/raw/emscad_v1.csv \
  --path model_algorithm/sprint3/data/processed \
  --path model_algorithm/sprint3/data/splits \
  --path-glob 'model_algorithm/sprint3/**/results/*predictions*.csv' \
  --path-glob 'model_algorithm/sprint3/**/results/*error_analysis*.csv' \
  --path model_algorithm/sprint3/ensemble_BERT_FP/results/validation_sweep.csv \
  --invert-paths
```

Review the rewritten repository, force-push all rewritten refs during a freeze
window, and require every collaborator and CI cache to make a fresh clone.
