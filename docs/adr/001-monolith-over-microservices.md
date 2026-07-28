# ADR-001: Monolith over Microservices

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

The system serves a single core domain — fake job detection — with these characteristics:

- Single bounded context: job advertisement risk analysis
- Synchronous request-response flow: submit text → validate → ensemble inference → XAI → response
- Small team (3–5 developers)
- Deployed on Cloud Run / Render (serverless, auto-scaling)
- 8 ML models loaded in-process for low-latency inference

## Decision

**Use a modular monolith**, not microservices.

The backend is a single FastAPI application with clear internal boundaries:

- `routers/` — HTTP layer
- `services/` — business logic + ML orchestration
- `schemas/` — data contracts (Pydantic)
- `xai_gentle/` — explainability + education subdomain
- `config.py` — centralized configuration
- `middleware.py` — cross-cutting concerns

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Microservices (one per model) | Network overhead kills ensemble latency; 8 model servers is unmanageable for a small team |
| Separate XAI service | XAI needs direct access to ensemble internals (SHAP values); separating would duplicate model loading |
| Message queue for async analysis | No async use case; all requests need synchronous responses |

## Consequences

**Positive:**
- Zero inter-service network latency for ensemble inference
- Single deployment unit → simpler CI/CD, monitoring, debugging
- All design patterns (DI, caching, resilience) work in-process with no distributed coordination

**Negative:**
- Cannot scale models independently (mitigated: Cloud Run scales the whole app horizontally)
- Coupling at deploy time (mitigated: internal interfaces are clean — could extract a service later if needed)
