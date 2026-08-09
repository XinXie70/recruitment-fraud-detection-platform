# Runnable BERT Package

Use this directory together with the repository-level `model_weights/`,
`model_results/`, and `data/splits/` directories.

## Install dependencies

From the repository root, activate the project virtual environment and install
the dependencies:

```bash
source .venv/bin/activate
pip install -r model_code/requirements-bert.txt
pip install -r requirements.txt
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` for activation.

## Run class-weighted BERT

```bash
# Train the class-weighted model and write weights to ../../model_weights/bert/bert_class_weighted
python model_code/bert/train_bert.py

# Evaluate saved weights on the test split
python model_code/bert/evaluate_bert.py \
  --checkpoint_dir model_weights/bert/bert_class_weighted/best

# Run inference on one record
python model_code/bert/predict_bert.py \
  --text "Urgent work-from-home role. Send bank details."
```

## Git LFS for model weights

The BERT `model.safetensors` file is approximately 418 MB, which exceeds
GitHub's 100 MB regular-file limit. Use Git LFS:

```bash
git lfs install
git lfs pull
```
