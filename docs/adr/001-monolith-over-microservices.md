# ADR-001: Monolith over Microservices

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-07-22 |
| Deciders | Capstone Team |

## Context

The system serves a single core domain — fake job detection — with these characteristics:

- Single bounded context: job advertisement risk analysis
- Progressive request-response flow: score first, then load detailed XAI
- Small team (3–5 developers)
- Deployed on Cloud Run / Render (serverless, auto-scaling)
- 8 ML models accessed through a stable adapter interface; they can be loaded
  in-process or hosted by a separate inference runtime

## Decision

**Use a modular monolith for the business application**, while allowing the
compute-heavy model runtime to be deployed separately.

The backend is a single FastAPI application with clear internal boundaries:

- `routers/` — HTTP layer
- `services/` — business logic + ML orchestration
- `schemas/` — data contracts (Pydantic)
- `xai_gentle/` — explainability + education subdomain
- `config.py` — centralized configuration
- `middleware.py` — cross-cutting concerns
- `services/fp_gate_predictor.py` — mandatory model-service boundary

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Microservices (one per model) | Eight independently operated services would add excessive networking and deployment complexity |
| Separate XAI service | XAI needs direct access to ensemble internals (SHAP values); separating would duplicate model loading |
| Message queue for XAI | The browser can request the second phase directly; a queue would add operational complexity before durable background jobs are required |

## Consequences

**Positive:**
- Simple in-process inference is available for environments with local artifacts
- Single deployment unit → simpler CI/CD, monitoring, debugging
- All design patterns (DI, caching, resilience) work in-process with no distributed coordination
- The model runtime may use independent compute without changing route/service contracts

**Negative:**
- Remote inference adds latency and another security boundary (mitigated with
  timeouts, health reporting, HTTPS, authentication, and network restrictions)
- Coupling at deploy time (mitigated: internal interfaces are clean — could extract a service later if needed)
