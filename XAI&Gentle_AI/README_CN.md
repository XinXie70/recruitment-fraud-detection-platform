# XAI and Gentle AI Submission Package

This directory contains the XAI and Gentle AI modules, their frontend display
components, and related tests. It does not include model weights, databases,
virtual environments, `node_modules`, or unrelated frontend and backend code.

## Directory Structure

```text
backend/
  xai_gentle/
    __init__.py
    contracts.py
    xai_service.py
    gentle_ai_service.py
    gentle_fallback.py
    knowledge/education_en.json
    README.md
  tests/
    test_xai_service.py
    test_gentle_ai.py

frontend/src/features/
  xai_gentle/
    AttributionTable.jsx
    ExplanationText.jsx
    GentleGuidance.jsx
    ModelContributions.jsx
    attributionFormatting.js
  analysis/
    AttributionTable.test.jsx
    ExplanationText.test.jsx
    ModelContributions.test.jsx
```

## Backend Features

- `xai_service.py` uses SHAP Partition to explain the production ensemble risk
  score. It uses hierarchical SHAP for long text.
- `contracts.py` defines stable input and output structures for XAI and Gentle AI.
- `gentle_ai_service.py` generates safety guidance from the ensemble risk and
  structured XAI evidence. It can optionally use a local Ollama model.
- `gentle_fallback.py` returns deterministic template content when Ollama is
  unavailable.
- `knowledge/education_en.json` stores the local educational knowledge base.

The backend modules expose the following public interface:

```python
from xai_gentle import GentleAIService, RiskContext, XAIService

xai_result = XAIService().explain(text, score_batch, risk_score)
gentle_result = GentleAIService().generate(risk_context, xai_result)
```

The production project must provide the following values through
`backend/services/analysis_service.py`:

- The original job advertisement text, `text`.
- The production ensemble batch-scoring function, `score_batch`.
- The final production ensemble risk score, `risk_score`.

The backend dependencies must include:

```text
shap>=0.46.0,<1.0.0
```

## Frontend Features

- `ExplanationText.jsx` uses the backend `start` and `end` offsets to highlight
  evidence in the original text.
- `AttributionTable.jsx` displays phrases, risk direction, and SHAP contribution
  values.
- `ModelContributions.jsx` displays the decision roles of the primary BERT model
  and the LR false-positive gate.
- `GentleGuidance.jsx` displays the Gentle AI summary, safety guidance, and
  disclaimer.
- `attributionFormatting.js` formats SHAP contributions as percentage points.

The production `frontend/src/App.jsx` must import these components:

```javascript
import AttributionTable from './features/xai_gentle/AttributionTable';
import ExplanationText from './features/xai_gentle/ExplanationText';
import GentleGuidance from './features/xai_gentle/GentleGuidance';
import ModelContributions from './features/xai_gentle/ModelContributions';
```

## Testing

Backend tests:

```bash
python -m pytest -q backend/tests/test_xai_service.py backend/tests/test_gentle_ai.py
```

After placing the frontend files in the complete project, run the frontend tests
from the `frontend` directory:

```bash
pnpm test
```
