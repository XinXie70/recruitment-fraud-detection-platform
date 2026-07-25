# Fake Job Advertisement Detection — Independent ML Workflow

This is a Master of IT project for detecting fraudulent English job
advertisements.

The final system will combine models developed by two team members. My planned
models are:

- Logistic Regression
- Linear SVM
- DistilRoBERTa

At the current stage, no models are trained. The goal is to prepare one shared
data pipeline so that all models use the same data and evaluation rules.

## Dataset

The main dataset is EMSCAD:

```text
data/raw/emscad_v1.csv
```

It contains 17,880 job advertisements:

- 17,014 legitimate advertisements
- 866 fraudulent advertisements

The source file uses `f` and `t` for the label. The pipeline converts them to:

- `0` = legitimate
- `1` = fraudulent

The file SHA-256 is:

```text
25e52f6d34939510f3f2ca5afc55ccfe7259ccc01283319bd484c466ea88baaf
```

The dataset CSV is not stored in GitHub because of its size. Each team member
must obtain the same file, rename it to `emscad_v1.csv`, place it in `data/raw/`,
and confirm that its SHA-256 matches the value above.

Dataset background: Vidros et al. (2017), *Automatic Detection of Online
Recruitment Frauds: Characteristics, Methods, and a Public Dataset*.
https://doi.org/10.3390/fi9010006

## Frontend development

The frontend requires Node.js 20.19 or a compatible newer release. With `nvm`:

```bash
nvm use
cd frontend
npm ci
npm run dev
```

Before opening a pull request, run:

```bash
npm run lint
npm test
npm run build
```

Husky and lint-staged automatically format and lint staged frontend files before
each commit. GitHub Actions also runs linting, tests, and a production build.

## Shared rules

- Use the same basic text cleaning for every model.
- Use the same train, validation, and test sets for all models.
- Duplicate and near-duplicate advertisements must stay in the same split.
- Fit TF-IDF, vocabulary, scalers, and other learned preprocessing on train only.
- Choose the final threshold using validation only; never use test for tuning.
- Do not use test or external data for tuning.

More details are in `DATA_CONTRACT_V1.md`.

## Raw data audit

Run:

```bash
python3 src/data_pipeline/audit_raw.py
```

The script only reads the raw CSV and saves a simple report to:

```text
data/diagnostics/raw_audit_v1.md
```

## Prepare the shared model dataset

Run:

```bash
python3 src/data_pipeline/prepare_data.py
```

This cleans and combines the five agreed text fields, converts the label to
`0/1`, and creates a stable ID for each advertisement. The result is:

```text
data/processed/emscad_processed_v1.csv
```

The processed file contains three columns: `record_id`, `combined_text`, and
`label`.

## Group duplicate advertisements

Run:

```bash
python3 src/data_pipeline/group_duplicates.py
```

This keeps the first copy of each exact duplicate, retains different
near-duplicate texts, and gives near duplicates the same `group_id`. It creates:

```text
data/processed/emscad_grouped_v1.csv
data/diagnostics/duplicate_report_v1.md
```

The groups will be used in the next stage so that similar advertisements cannot
appear in different data splits.

## Compare split ratios

Before creating the final splits, run:

```bash
python3 src/data_pipeline/analyse_split_options.py
```

This compares several group-aware split ratios using sample counts, class balance,
and sensitivity across random seeds. It does not train models or create the final
split files. The result is saved to:

```text
reports/split_feasibility_v1.md
```

The selected ratio is **70% Train / 15% Validation / 15% Test**, using seed 42.

## Create the fixed splits

Run:

```bash
python3 src/data_pipeline/create_splits.py
```

This creates the three shared files in `data/splits/`. Every model must use
these same files. The integrity check is saved to:

```text
data/diagnostics/split_report_v1.md
```

The fixed `train.csv`, `validation.csv`, and `test.csv` files are stored in
GitHub so team members can use the same data directly. The additional file
`data/splits/split_assignments_v1.csv` records the agreed assignment of every
`record_id` and makes the split reproducible.

Expected SHA-256 values for the raw, processed, and split files are recorded in
`data/checksums_v1.txt` for team verification.

## Files stored in GitHub

GitHub contains the pipeline code, documentation, reports, the fixed split
assignment file, and the three final split CSV files. Raw data, intermediate
processed data, trained model files, and local Python environments are excluded
by `.gitignore`.

To rebuild the shared data after placing `emscad_v1.csv` in `data/raw/`, run:

```bash
python3 src/data_pipeline/audit_raw.py
python3 src/data_pipeline/prepare_data.py
python3 src/data_pipeline/group_duplicates.py
python3 src/data_pipeline/create_splits.py
```

## Logistic Regression baseline

Create the local environment and install the project dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Train the baseline:

```bash
.venv/bin/python src/models/logistic_regression/train_baseline.py
```

The baseline uses group-aware cross-validation inside Train and selects its
threshold on Validation. It does not read Test. Results are saved under
`reports/models/logistic_regression/`.

## Linear SVM baseline

Train the baseline:

```bash
.venv/bin/python src/models/linear_svm/train_baseline.py
```

The SVM uses the same group-aware Train cross-validation and Validation
threshold rule as Logistic Regression. Its decision scores are converted to
0–1 fraud scores using group-aware sigmoid calibration fitted on Train only.
Results are saved under `reports/models/linear_svm/`.
