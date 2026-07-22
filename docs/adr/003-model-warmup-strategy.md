# ADR-003: Model Warm-Up Strategy

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

Loading 8 ML models (including BERT and RoBERTa with large weight files) takes 30–60 seconds. During this time:
- Auth endpoints should still work (users can log in)
- Analysis endpoints should return a clear 503 until ready

Cloud Run has a cold-start timeout: if the server doesn't respond within the configured timeout, the request fails.

## Decision

**Warm up models in a background thread at startup.**

```python
# main.py — lifespan handler
threading.Thread(target=_warm_up_models_background, daemon=True).start()
```

The app returns HTTP responses immediately (auth works). Analysis endpoints check `analysis_service.ready` and return 503 if not ready.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Lazy loading (load on first request) | First user request would timeout (30–60s load time) |
| Pre-warm all models in `@app.on_event("startup")` | Blocks startup; Cloud Run cold-start would time out before auth works |
| Separate warm-up service | Over-engineering; no async use case |

## Consequences

- Auth is available ~2 seconds after cold start
- Analysis endpoints return 503 for ~30–60 seconds after cold start
- Cloud Run health checks pass immediately (they hit `/api/health` which is always available)
- Failed model members are tolerated — the ensemble degrades gracefully
