# Split Leakage Experiment

This experiment studies whether duplicate handling and split strategy change
the measured performance of the same model.

The required experiment inputs are included in compressed form:

- `data/processed/emscad_condition_a_no_dedup_input_v1.csv.gz`
- `data/processed/emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz`

Pandas reads these files directly; no manual extraction is required.

## A = No dedup + Random split

Condition A uses all 17,880 processed EMSCAD rows, keeps exact duplicates, and
applies an ordinary stratified random 70/15/15 split.

## B = Exact dedup + Random split

Condition B uses the 15,807 rows remaining after exact deduplication and applies
an ordinary stratified random 70/15/15 split. It deliberately ignores the
near-duplicate `group_id`.

## C = Exact dedup + Group-aware split

Condition C uses the same 15,807-row input as Condition B, but allocates whole
near-duplicate groups to the 70/15/15 partitions. Linear SVM calibration is also
group-aware within diagnostic Train.

All three conditions are diagnostic experiments and do not read or modify the
official Train, Validation, or Test files.

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
