# Processed experiment data

These compressed CSV files are the shared inputs for the A/B/C split-leakage
experiment. Pandas reads them directly; they do not need to be extracted.

| File | Rows | Use |
|---|---:|---|
| `emscad_condition_a_no_dedup_input_v1.csv.gz` | 17,880 | A = No dedup + Random split |
| `emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz` | 15,807 | Shared input for B = Exact dedup + Random split and C = Exact dedup + Group-aware split |

SHA-256 of the compressed files:

```text
d06943fc079d31eb34c10fc13c96b9c39c6228be84bb9ba67aa7e84dc6db9b73  emscad_condition_a_no_dedup_input_v1.csv.gz
943d402ffd80725e49acaf93bc61192b04972a317476c839ccff3bc1a500b7ca  emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz
```

SHA-256 after decompression:

```text
c394f8e3a31014ad4f2faea06a9e3ac82deb4335706a94554eb469df29afc057  emscad_processed_v1.csv
d259b9ffc717abfa78296c3509e87f9bd99b3900b1132e9174137e9c9ccfd8f6  emscad_grouped_v1.csv
```
