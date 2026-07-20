# Split Feasibility Analysis v1

This analysis compares split ratios before model training. It uses only group size and label; no text features or model results are used.

## Data

- Rows after exact deduplication: **15,807**
- Groups: **14,673**
- Fraudulent rows: **712** (4.50%)
- Main reproducible seed: **42**
- Sensitivity seeds: **0 to 99**

## Pre-specified minority reference

A calculated reference of **97 fraudulent rows** per holdout was obtained from `ceil(1.96^2 × 0.25 / 0.10^2)`. This corresponds to an approximate worst-case 95% half-width of 10 percentage points for an independent binary proportion. The practical planning target is approximately **100 fraudulent rows** in Validation and Test.

This is a practical design reference, not a universal rule. Near-duplicate rows within a group are correlated, so the effective independent sample size may be smaller. Group counts and confidence intervals should also be reported in the final study.

## Summary for seed 42

| Option | Train rows | Val fraud | Test fraud | Minimum holdout fraud | Meets 97 reference? | Max size deviation |
|---|---:|---:|---:|---:|---|---:|
| A: 80/10/10 | 12,646 | 71 | 71 | 71 | No | 0.004% |
| B: 75/10/15 | 11,855 | 71 | 107 | 71 | No | 0.006% |
| C: 72/14/14 | 11,380 | 100 | 100 | 100 | Yes | 0.007% |
| D: 70/15/15 | 11,064 | 107 | 107 | 107 | Yes | 0.006% |
| E: 60/20/20 | 9,485 | 142 | 143 | 142 | Yes | 0.009% |

## Sensitivity across 100 seeds

| Option | Seeds meeting row reference | Range of minimum holdout fraud | Range of minimum fraud groups | Range of max size deviation |
|---|---:|---:|---:|---:|
| A: 80/10/10 | 0/100 | 70-71 | 60-69 | 0.004%-0.243% |
| B: 75/10/15 | 0/100 | 71-73 | 58-70 | 0.002%-0.239% |
| C: 72/14/14 | 100/100 | 99-100 | 85-95 | 0.000%-0.241% |
| D: 70/15/15 | 100/100 | 106-107 | 92-101 | 0.001%-0.240% |
| E: 60/20/20 | 100/100 | 142-143 | 126-136 | 0.004%-0.238% |

## Seed 42 details

### A: 80/10/10

| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 12,646 | 80.00% | 570 | 529 | 4.51% | 11,707 | 82 | 0.65% |
| Validation | 1,580 | 10.00% | 71 | 65 | 4.49% | 1,476 | 31 | 1.96% |
| Test | 1,581 | 10.00% | 71 | 67 | 4.49% | 1,490 | 14 | 0.89% |

### B: 75/10/15

| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 11,855 | 75.00% | 534 | 493 | 4.50% | 11,041 | 19 | 0.16% |
| Validation | 1,580 | 10.00% | 71 | 68 | 4.49% | 1,476 | 31 | 1.96% |
| Test | 2,372 | 15.01% | 107 | 100 | 4.51% | 2,156 | 82 | 3.46% |

### C: 72/14/14

| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 11,380 | 71.99% | 512 | 479 | 4.50% | 10,587 | 31 | 0.27% |
| Validation | 2,213 | 14.00% | 100 | 91 | 4.52% | 2,085 | 14 | 0.63% |
| Test | 2,214 | 14.01% | 100 | 91 | 4.52% | 2,001 | 82 | 3.70% |

### D: 70/15/15

| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 11,064 | 69.99% | 498 | 465 | 4.50% | 10,286 | 31 | 0.28% |
| Validation | 2,371 | 15.00% | 107 | 97 | 4.51% | 2,235 | 14 | 0.59% |
| Test | 2,372 | 15.01% | 107 | 99 | 4.51% | 2,152 | 82 | 3.46% |

### E: 60/20/20

| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 9,485 | 60.01% | 427 | 396 | 4.50% | 8,783 | 82 | 0.86% |
| Validation | 3,160 | 19.99% | 142 | 135 | 4.49% | 2,953 | 31 | 0.98% |
| Test | 3,162 | 20.00% | 143 | 130 | 4.52% | 2,937 | 19 | 0.60% |

## Integrity result

All **15,807 records** and all **14,673 groups** were assigned exactly once in every analysed allocation. Because assignment is performed at group level, a group cannot cross splits.

## Interpretation rule

1. Reject any allocation with missing/duplicated records or groups crossing splits.
2. Prefer options where Validation and Test each contain at least 97 fraudulent rows and are close to the operational target of 100.
3. Prefer options that meet the reference consistently across sensitivity seeds.
4. Check that row shares and fraud rates remain close to their targets.
5. Among options satisfying the above conditions, retain the larger Train partition.
6. Do not use model performance to choose the ratio or seed.

## Feasibility result and final decision

- Options A: 80/10/10, B: 75/10/15 do not meet the minority reference consistently.
- Stable options: C: 72/14/14, D: 70/15/15, E: 60/20/20.
- The initial row-count rule favoured C: 72/14/14, because it retained the largest Train partition among stable options meeting the reference.
- The team selected **D: 70/15/15** after also considering that near-duplicate rows within a group are correlated. For seed 42, D has 107 fraud rows in each holdout and 97/99 fraud groups, compared with 100 fraud rows and 91/91 fraud groups for C.
- D reduces Train by 316 rows (2.8%) compared with C, while giving both holdouts a larger buffer above the 97-row planning reference and greater fraud-group diversity.
- This decision was recorded before model training and did not use model performance.

## Method references

- Vidros et al. (2017), EMSCAD: https://doi.org/10.3390/fi9010006
- Collins et al. (2006), effective sample size for validation: https://doi.org/10.1016/j.jclinepi.2005.05.014
- Riley et al. (2021), sample size for binary prediction-model validation: https://doi.org/10.1002/sim.9025
