# ADR-004: XAI Method Selection

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

Users need to understand *why* a job posting was flagged as suspicious. We need an explainability method that:

- Works with an ensemble of 8 heterogeneous models
- Produces human-readable evidence spans
- Is computationally feasible within a synchronous API response (< 5 seconds)

## Decision

Use a **two-tier XAI strategy** (`backend/xai_gentle/`):

1. **Primary: SHAP Partition Explainer**
   - Runs against the ensemble's aggregated score function
   - Produces token-level contribution values
   - Identifies text spans that raise or lower risk

2. **Fallback: Occlusion**
   - Used when SHAP times out or fails (e.g., very long texts)
   - Simpler but still produces directional evidence

3. **Gentle AI layer**: Converts raw XAI output into user-friendly educational language
   - Template-based by default (no external dependency)
   - Optional Ollama integration for LLM rewrites

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| LIME | Slower than SHAP partition for text; less stable across runs |
| Integrated Gradients | Requires gradient access to all models; not feasible for sklearn/xgboost |
| Attention weights only | Does not explain the final BERT + LR gate decision contract |
| Pure LLM explanation | Non-deterministic; may hallucinate; adds latency + cost |

## Consequences

- SHAP partition is the most principled method for ensemble text models
- Two-tier design ensures graceful degradation (occlusion fallback)
- Gentle AI layer separates *explanation* from *communication*, making each independently testable
