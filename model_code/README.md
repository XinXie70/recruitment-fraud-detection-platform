# Runnable BERT Package

Use this directory together with the repository-level `model_weights/`,
`model_results/`, and `data/splits/` directories.

## Install dependencies

```powershell
. E:\ml\activate.ps1
pip install -r model_code/requirements-bert.txt
pip install -r requirements.txt
```

## Run class-weighted BERT

```powershell
cd model_code/bert

# Train the class-weighted model and write weights to ../../model_weights/bert/bert_class_weighted
python train_bert.py

# Evaluate saved weights on the test split
python evaluate_bert.py --checkpoint_dir ..\..\model_weights\bert\bert_class_weighted\best

# Run inference on one record
python predict_bert.py --text "Urgent work-from-home role. Send bank details."
```

## Git LFS for model weights

The BERT `model.safetensors` file is approximately 418 MB, which exceeds
GitHub's 100 MB regular-file limit. Use Git LFS:

```powershell
git lfs install
git lfs pull
```
