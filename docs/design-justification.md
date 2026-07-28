# Design Justification

This document summarises the major technical decisions behind the implemented
system. Detailed decision records are linked where available.

## 1. Modular monolith with a separately deployable model runtime

**Context.** The product has one bounded domain and a synchronous user journey,
while ML inference has substantially different artifact and compute needs.

**Decision.** Keep authentication, analysis orchestration, history, education,
and administration in one modular FastAPI application. Access inference through
a stable model-adapter interface that supports either local pipelines or a
remote model API.

**Why.** A modular monolith minimises distributed-system overhead for a small
team. Separating only the compute-heavy model runtime permits independent VM/GPU
placement without splitting every business capability into a microservice.

**Trade-offs.** Business modules deploy together, and the remote inference path
introduces network latency and an additional security boundary. Timeouts,
health reporting, graceful degradation, HTTPS, and service authentication are
therefore required.

**Evidence.** [ADR-001](adr/001-monolith-over-microservices.md),
[`backend/services/model_adapter.py`](../backend/services/model_adapter.py), and
the [deployment diagram](architecture/deployment.md).

## 2. Synchronous API instead of a message queue

**Context.** Users need an immediate risk result after submitting a job advert.

**Decision.** Use synchronous HTTP and parallelise model inference inside the
request. Do not add a message broker to the interactive analysis path.

**Why.** A queue would add polling, correlation, retry, and operational
complexity without improving the current user journey. Parallel adapters and a
bounded timeout address latency directly.

**Trade-offs.** Requests remain open during inference and are sensitive to model
latency. A queue becomes appropriate if the product later adds bulk imports,
scheduled processing, or long-running retraining jobs.

## 3. Heterogeneous configurable ensemble

**Context.** Linear, tree, recurrent, dense, and transformer models capture
different fraud signals and have different failure modes.

**Decision.** Calibrate each available model score, apply configuration-driven
weights, and map the weighted probability through low/high thresholds. If a
member fails, renormalise the successful members' weights; return 503 only if
all members fail.

**Why.** Architectural diversity reduces dependence on a single model and the
explicit weighted formula remains inspectable. Configuration files allow
weights and thresholds to be frozen from validation results without changing
application code.

**Trade-offs.** More models increase latency, memory use, and operational
surface area. The production configuration must be fitted and locked before the
Final Demo; the current checked-in `ensemble-v2-eight-model-dev-fallback`
configuration identifies itself as equal-weight and unfitted.

**Evidence.** [ADR-002](adr/002-ensemble-scoring-formula.md),
[`backend/services/ensemble_predictor.py`](../backend/services/ensemble_predictor.py),
and the saved ensemble configuration.

## 4. Model-agnostic explainability with deterministic fallback

**Context.** Users need evidence for a risk result, but the ensemble contains
models for which gradient-only methods do not work.

**Decision.** Explain the aggregated scoring function using SHAP Partition,
fall back to occlusion when necessary, then convert evidence into deterministic
educational language. Ollama rewriting is optional.

**Why.** The approach works across heterogeneous models and separates factual
attribution from presentation. A deterministic fallback prevents an external
LLM outage or hallucination from blocking the core result.

**Trade-offs.** Perturbation-based explanation adds inference calls and is an
approximation of model behaviour rather than a causal explanation.

**Evidence.** [ADR-004](adr/004-xai-method-selection.md) and
`backend/xai_gentle/`.

## 5. PostgreSQL with explicit ownership and minimal history data

**Context.** Accounts, history, and admin statistics require relational
integrity. Submitted job advertisements may contain personal contact details.

**Decision.** Use PostgreSQL through SQLAlchemy and Alembic. Store a redacted
500-character preview and SHA-256 input hash rather than the full submission;
enforce ownership when users read or delete history.

**Why.** Relational constraints protect user/history consistency, and data
minimisation reduces privacy exposure while retaining useful audit metadata.

**Trade-offs.** A preview may still contain personal information not covered by
email/phone patterns. A retention period and deletion policy should be defined
before production use.

## 6. JWT, bcrypt, validation, and layered request protection

**Context.** Analysis history and admin information must not be exposed to
unauthorised users, and inference is computationally expensive.

**Decision.** Hash passwords with bcrypt; issue short-lived signed JWTs with
issuer, audience, expiry, and unique token ID; enforce role/ownership checks;
apply rate limits, body-size controls, explicit production CORS, and security
headers.

**Why.** These controls address credential storage, token substitution,
resource abuse, cross-origin access, and accidental information disclosure at
appropriate layers.

**Trade-offs.** Stateless JWTs are simple to scale but cannot be revoked without
additional state. Production secrets must come from a secret manager rather
than source control.

## 7. Docker Compose and health-aware startup

**Context.** Team members and assessors need a reproducible way to start the
multi-container application.

**Decision.** Compose PostgreSQL, a one-shot migration container, FastAPI, and
the React development server. Gate startup using database and application
health checks.

**Why.** A single command reduces environment drift and makes dependency order
explicit. Separate liveness, readiness, and rich health endpoints support
diagnosis without conflating process health with dependency readiness.

**Trade-offs.** The development frontend container is convenient for local
demonstration but a production deployment should serve a static build through a
web server/CDN. Secrets and public endpoint configuration must be supplied by
the deployment platform.

**Evidence.** [`compose.yaml`](../compose.yaml), [ADR-003](adr/003-model-warmup-strategy.md),
and the [deployment diagram](architecture/deployment.md).

## 8. Leakage-resistant evaluation

**Context.** Duplicate or near-duplicate advertisements across train and test
sets can inflate model metrics and make an ensemble appear stronger than it is.

**Decision.** Deduplicate records and use group-aware splitting experiments,
then fit ensemble decisions using validation data and report final performance
on untouched test data.

**Why.** This makes reported performance better reflect generalisation to new
advertisements and is more defensible than optimising against the test set.

**Trade-offs.** Leakage-resistant splits are harder and may report lower scores,
but those scores are more credible. The exact production ensemble artifact must
still be frozen and labelled with the evaluation version before the demo.

**Evidence.** `src/experiments/split_leakage/`, `data/experiment_splits/`, and
`reports/models/`.

## Remaining decisions before production

1. Replace the unfitted equal-weight ensemble fallback with a validation-fitted,
   versioned, and locked configuration.
2. Put the remote model API behind HTTPS, service authentication, and restrictive
   firewall rules.
3. Define history-data retention and deletion policy.
4. Decide whether the production frontend uses a CDN/static host or Nginx.
5. Record frontend coverage evidence and extend the Playwright suite to the
   admin API flow (the core authentication, analysis, history, and 503 paths
   are automated).
