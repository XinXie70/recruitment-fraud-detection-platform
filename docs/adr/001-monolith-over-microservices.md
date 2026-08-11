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
- The business application is deployed on Cloud Run
- BERT and Logistic Regression are served by a mandatory, separately deployed
  FP-gate inference runtime

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
| Microservices (one per model) | Independently operating BERT and LR services would add networking and deployment complexity without improving the FP-gate contract |
| Separate XAI service | XAI uses the model API's batch endpoint; another service would add an unnecessary network boundary |
| Message queue for XAI | The browser can request the second phase directly; a queue would add operational complexity before durable background jobs are required |

## Consequences

**Positive:**
- The business application remains one deployment unit, simplifying CI/CD and debugging
- Authentication, persistence, validation, and explanation orchestration retain clear module boundaries
- The model runtime can use independent CPU/GPU resources without changing application routes

**Negative:**
- Remote inference adds latency and another security boundary (mitigated with
  timeouts, health reporting, HTTPS, authentication, and network restrictions)
- Coupling at deploy time between FastAPI and the versioned model API contract
