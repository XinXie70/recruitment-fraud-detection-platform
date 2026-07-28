# Risk Band v1

## Boundaries

Risk Band v1 uses the frozen LR + BERT ensemble score.

```text
score < 0.1567                 Low
0.1567 <= score < 0.62        Suspicious
score >= 0.62                 High
```

The stored Low/Suspicious boundary is `0.1567`. The original candidate score
was `0.1567337285519143`; rounding it to four decimal places does not change
any Validation assignment. If the score is shown on a 0–100 scale, the
boundaries are 15.67 and 62.

These values are score boundaries, not estimated probabilities of fraud.

## How the boundaries were selected

Only Validation predictions were used.

The High boundary remains at `0.62`, which is the frozen binary threshold for
Ensemble v1. At this boundary, 82 of the 88 High-risk advertisements were
fraudulent, giving a Validation precision of 93.18%.

The High threshold also sits at a useful trade-off point. Lowering it to about
0.56 gives one additional true positive but adds three false positives.
Raising it to about 0.81 increases precision to 95.45%, but reduces the number
of detected fraudulent advertisements from 82 to 63.

The Low boundary was chosen as the highest Validation threshold that covers at
least 90% of fraudulent advertisements in the combined Suspicious and High
bands. The selected threshold covers 97 of 107 fraudulent advertisements
(90.65%) while placing 206 of 2,371 advertisements (8.69%) in the Suspicious
band.

## Validation distribution

| Band | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| Low | 2,077 | 10 | 0.48% |
| Suspicious | 206 | 15 | 7.28% |
| High | 88 | 82 | 93.18% |

The Suspicious band catches 15 fraudulent advertisements that do not reach the
High threshold. Ten fraudulent advertisements remain in the Low band.

## Other Low-boundary options

| Low threshold | Fraud covered by Suspicious + High | Suspicious rows | Review rate | Fraud left in Low |
|---:|---:|---:|---:|---:|
| 0.2263 | 87.85% | 90 | 3.80% | 13 |
| 0.1567 | 90.65% | 206 | 8.69% | 10 |
| 0.1214 | 92.52% | 324 | 13.67% | 8 |
| 0.0871 | 95.33% | 506 | 21.34% | 5 |

Moving from the selected option to 92.52% coverage catches two more fraudulent
advertisements but adds 118 advertisements to the Suspicious band. Moving to
95.33% coverage catches five more fraudulent advertisements than the selected
option but increases the Suspicious band from 206 to 506 advertisements.

## Status

The two boundaries were selected from Validation for Risk Band v1. Test was
not used to select or adjust them.

## Test results

The locked boundaries were applied once to the existing ensemble Test
predictions.

| Band | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| Low | 2,087 | 2 | 0.10% |
| Suspicious | 192 | 23 | 11.98% |
| High | 93 | 82 | 88.17% |

Suspicious and High together covered 105 of 107 fraudulent advertisements
(98.13%). The Suspicious band contained 8.09% of all Test advertisements.
High-risk precision was 88.17%, and High alone contained 76.64% of all Test
fraudulent advertisements.

Compared with Validation, fewer fraudulent advertisements fell into Low
(2 instead of 10), while High precision decreased from 93.18% to 88.17%.
The thresholds remain unchanged after this Test evaluation.

## Reproduce the results

Run the Validation step first. It selects the Low/Suspicious boundary using
the saved Validation predictions and writes the frozen risk-band config:

```bash
python src/models/build_risk_bands.py --mode validation
```

Then apply the frozen config to Test:

```bash
python src/models/build_risk_bands.py --mode test
```

The Test step only reads the saved thresholds. It does not search for or
change either boundary.
