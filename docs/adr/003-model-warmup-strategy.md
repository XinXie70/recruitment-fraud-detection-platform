# ADR-003: FP-gate service warm-up

## Status

Accepted.

## Decision

At application startup, FastAPI sends one representative request to the
configured FP-gate model API in a background thread. Readiness becomes true
only after the mandatory service returns a valid BERT/LR result. Model loading
and accelerator selection remain the responsibility of the model service.
