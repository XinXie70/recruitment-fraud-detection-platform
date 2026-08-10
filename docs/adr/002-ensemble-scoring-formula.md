# ADR-002: BERT-primary LR false-positive gate

## Status

Accepted.

## Decision

Use the paper-aligned BERT max-length-512 score as the primary risk signal.
Logistic Regression is a conditional false-positive gate, not a weighted
ensemble member. Frozen thresholds and risk boundaries live under
`ensemble_bert_fp_gate_lr_none_bigram_maxlen512/`.

FastAPI consumes the model service's final risk contract and does not recompute
the gate or maintain an alternate scoring formula.
