# Data Directory

Place the raw job posting dataset `DataSet.csv` in this directory, or at the repository root.

Running `python final_model_pipelines/prepare_data.py` will automatically generate:

- `cleaned_data.csv` — cleaned data (includes `combined_text`)
- `splits/train.csv`, `val.csv`, `test.csv` — stratified splits

If `DataSet.csv` is large, you may add it to `.gitignore` and have team members place it locally.
