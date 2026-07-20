# Raw Data Audit v1

- File: `data/raw/emscad_v1.csv`
- SHA-256: `25e52f6d34939510f3f2ca5afc55ccfe7259ccc01283319bd484c466ea88baaf`
- Rows: **17,880**
- Columns: **18**

## Label distribution

- Legitimate (0): 17,014 (95.16%)
- Fraudulent (1): 866 (4.84%)

## Missing values

| Column | Missing | Percentage |
|---|---:|---:|
| title | 0 | 0.00% |
| location | 346 | 1.94% |
| department | 11,553 | 64.61% |
| salary_range | 15,012 | 83.96% |
| company_profile | 3,308 | 18.50% |
| description | 0 | 0.00% |
| requirements | 2,689 | 15.04% |
| benefits | 7,196 | 40.25% |
| telecommuting | 0 | 0.00% |
| has_company_logo | 0 | 0.00% |
| has_questions | 0 | 0.00% |
| employment_type | 3,471 | 19.41% |
| required_experience | 7,050 | 39.43% |
| required_education | 8,105 | 45.33% |
| industry | 4,903 | 27.42% |
| function | 6,455 | 36.10% |
| fraudulent | 0 | 0.00% |
| in_balanced_dataset | 0 | 0.00% |

## Combined text length

| Measure | Characters | Words |
|---|---:|---:|
| Mean | 2,703.7 | 393.8 |
| Median | 2,561.0 | 368.0 |
| 95th percentile | 5,370.1 | 794.0 |
| Maximum | 14,927 | 2,118 |

## Duplicate check

- Exact duplicate groups: **726**
- Rows inside exact duplicate groups: **2,799**
- Preliminary near-duplicate groups: **654**
- Near-duplicate pairs found: **7,724** from 62,343 checked pairs
- Large same-title groups skipped: **3**

Near-duplicate rule: ads must have the same cleaned title and at least 0.90 Jaccard similarity between their word sets. This is a preliminary audit, not the final split rule.
