# Split Leakage Experiment

This experiment studies whether duplicate handling and split strategy change
the measured performance of the same model.

The required experiment inputs are included in compressed form:

- `data/processed/emscad_condition_a_with_exact_duplicates_v1.csv.gz`
- `data/processed/emscad_conditions_b_c_exact_deduplicated_grouped_v1.csv.gz`

Pandas reads these files directly; no manual extraction is required.

Condition A uses all 17,880 processed EMSCAD rows, keeps exact duplicates, and
applies an ordinary stratified random 70/15/15 split. It is a diagnostic
experiment and does not read or modify the official Train, Validation, or Test
files.

Condition B uses the 15,807 rows remaining after exact deduplication. It still
uses an ordinary stratified random 70/15/15 split, deliberately ignoring the
near-duplicate `group_id`. This separates the effect of removing exact
duplicates from the effect of enforcing group-aware splitting.

Condition C uses the same 15,807-row input as Condition B, but allocates whole
near-duplicate groups to the 70/15/15 partitions. Linear SVM calibration is
also group-aware within diagnostic Train.

Run Condition A from the project root:

```bash
.venv/bin/python src/experiments/split_leakage/run_condition_a.py
.venv/bin/python src/experiments/split_leakage/run_condition_a_svm.py
```

Run Condition B:

```bash
.venv/bin/python src/experiments/split_leakage/run_condition_b.py
.venv/bin/python src/experiments/split_leakage/run_condition_b_svm.py
```

Run Condition C:

```bash
.venv/bin/python src/experiments/split_leakage/run_condition_c.py
.venv/bin/python src/experiments/split_leakage/run_condition_c_svm.py
```

Create the experiment figures and paper summary table:

```bash
.venv/bin/python src/experiments/split_leakage/create_experiment_figures.py
```

Results are written to `reports/experiments/split_leakage/`.
