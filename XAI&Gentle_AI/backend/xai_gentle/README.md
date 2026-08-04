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
  Long advertisements use hierarchical SHAP: coarse sentence/paragraph selection,
  followed by word-level attribution inside the most important segments. Displayed
  evidence combines isolated high-impact fragments with nearby context and removes
  standalone numeric, generic, calendar, and relatively insignificant noise without
  changing model scores. When every attribution is generic, the strongest items are
  expanded into readable context phrases instead of returning an empty explanation.
- `gentle_ai_service.py`: Reads structured evidence, optionally calls local Ollama for rewriting.
- `gentle_fallback.py`: Deterministic templates used when Ollama is unavailable.
- `knowledge/education_en.json`: Local education knowledge base.

## Boundaries with Other Team Members

XAI public API:

```python
XAIService.explain(text, score_batch, expected_output) -> XAIResult
```

It does not import any concrete model or EnsemblePredictor — it only calls the
`score_batch` provided by the backend. With the final BERT + LR FP-gate model,
this scorer is backed by the remote `/predict/batch` endpoint and returns the
same formal `risk_score` used by the application.

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
