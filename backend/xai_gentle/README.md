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
- `xai_service.py`: Explains the formal ensemble scorer, generating SHAP or occlusion evidence.
- `gentle_ai_service.py`: Reads structured evidence, optionally calls local Ollama for rewriting.
- `gentle_fallback.py`: Deterministic templates used when Ollama is unavailable.
- `knowledge/education_en.json`: Local education knowledge base.

## Boundaries with Other Team Members

XAI public API:

```python
XAIService.explain(text, score_batch, expected_output) -> XAIResult
```

It does not import any concrete model or EnsemblePredictor — it only calls the
`score_batch` provided by the backend.

Gentle AI public API:

```python
GentleAIService.generate(risk, xai) -> GentleAIResult
```

`risk` is a `RiskContext` with only four fields. Gentle AI does not receive the model,
the raw scorer, or the full EnsembleResult, so it cannot re-judge real vs. fake.

The only backend integration point is `backend/services/analysis_service.py`.
Independent test command:

```bash
python -m pytest -q tests/xai_gentle
```
