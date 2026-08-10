# XAI & Gentle AI — Independent Module

This directory is the independent code scope for the XAI/Gentle AI lead.
Other team members should use the public interface from
`xai_gentle/__init__.py` — do not depend on internal files directly.

```python
from backend.xai_gentle import GentleAIService, RiskContext, XAIService
```

## Files

- `contracts.py`: Stable input/output contracts including `RiskContext`, `XAIResult`,
  and `GentleAIResult`.
- `xai_service.py`: Explains the formal ensemble risk scorer with SHAP Partition.
  Long advertisements use sentence-aware phrase spans of at most eight words across
  the complete input. This avoids a second masking pass that can collapse useful
  evidence to zero when the formal risk probability is saturated. Displayed
  evidence combines isolated high-impact fragments with nearby context and removes
  standalone numeric, generic, calendar, and relatively insignificant noise without
  changing model scores. Long-text analysis never substitutes a full sentence when
  no reliable phrase-level attribution survives filtering; it returns an empty
  evidence list with an explicit message instead.
- `gentle_ai_service.py`: Reads structured evidence, optionally calls local Ollama for rewriting.
- `gentle_fallback.py`: Deterministic templates used when Ollama is unavailable.
- `knowledge/education_en.json`: Local education knowledge base.

## Boundaries with Other Team Members

XAI public API:

```python
XAIService.explain(text, score_batch, expected_output) -> XAIResult
```

It does not import model weights; it calls the `score_batch` callback supplied
by `FPGatePredictor`. Every perturbation therefore uses the same production
BERT-primary + LR-gate `risk_score` as the application.

SHAP is the only attribution method. If it cannot run, XAI returns a structured
`unavailable` result; it does not substitute keyword rules or occlusion scores.

Gentle AI public API:

```python
GentleAIService.generate(risk, xai) -> GentleAIResult
```

`risk` is a `RiskContext` with only four fields. Gentle AI does not receive the model,
the raw scorer, or the full EnsembleResult, so it cannot re-judge real vs. fake.

The only backend integration point is `backend/services/analysis_service.py`.
Independent test command:

```bash
python -m pytest -q backend/tests/test_xai_service.py backend/tests/test_gentle_ai.py
```
