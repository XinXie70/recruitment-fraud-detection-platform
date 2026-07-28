# A/B/C experiment split assignments

These files record the exact Train, Validation and Holdout assignment used by
every model in the split-leakage experiment.

| File | Experiment |
|---|---|
| `condition_a_no_dedup_random_split_assignments_v1.csv.gz` | A = No dedup + Random split |
| `condition_b_exact_dedup_random_split_assignments_v1.csv.gz` | B = Exact dedup + Random split |
| `condition_c_exact_dedup_group_aware_split_assignments_v1.csv.gz` | C = Exact dedup + Group-aware split |

Each file contains:

```text
record_id,seed,split
```

The files include seeds 0–9. For each seed, every input record appears exactly
once and `split` is one of `train`, `validation`, or `holdout`.

| Condition | Input records per seed | Total assignment rows |
|---|---:|---:|
| A | 17,880 | 178,800 |
| B | 15,807 | 158,070 |
| C | 15,807 | 158,070 |

SHA-256:

```text
ef6cb9f677e696e017d5f54e796a8e1379fdde530ced5fb2275080d192e15cea  condition_a_no_dedup_random_split_assignments_v1.csv.gz
b4d61e7cd5892e2dc46537a9067d528f06d0493b69a55f636a85c250f0a0653a  condition_b_exact_dedup_random_split_assignments_v1.csv.gz
e053ba74612d0bd6960b4caae71b91769c66f9ddd5211d8f5704e0a27e1c2054  condition_c_exact_dedup_group_aware_split_assignments_v1.csv.gz
```

These are diagnostic experiment assignments. The official fixed project split
remains in `data/splits/`.
