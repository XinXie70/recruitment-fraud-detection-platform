# Final Demo readiness checklist

Use this document as the single source of truth for the Final Demo. Replace every
`TBD` before the final rehearsal and keep the evidence links current.

## User story traceability

Copy every committed user story and non-functional requirement from the proposal
into this table. A story is only complete when its acceptance criteria can be
demonstrated and supported by automated or repeatable test evidence.

| ID  | User story / requirement | Acceptance criteria | Demo route | Implementation | Test evidence | Owner | Status       |
| --- | ------------------------ | ------------------- | ---------- | -------------- | ------------- | ----- | ------------ |
| TBD | TBD                      | TBD                 | TBD        | TBD            | TBD           | TBD   | Not verified |

Allowed status values: `Not started`, `In progress`, `Implemented`, and
`Demo verified`.

## Quality evidence

| Area               | Evidence to prepare                                                   | Current status                                                                                                       |
| ------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Backend tests      | Test count and coverage report                                        | 74 tests passed; 90.54% measured locally                                                                             |
| Frontend tests     | Unit/component coverage report                                        | 11 tests pass; 41.20% statements / 42.41% lines                                                                      |
| End-to-end tests   | Authentication, analyse, history, admin routing, and failure paths    | 5 Playwright tests pass                                                                                              |
| Responsive UI      | Phone, tablet, and desktop screenshots                                | Implemented; rehearsal verification required                                                                         |
| Loading and errors | Slow request, invalid input, and unavailable model service            | Implemented; automated 503 path passes; rehearsal still required                                                     |
| Live status        | Refresh service status without a full page reload                     | Implemented on Admin Dashboard                                                                                       |
| Security           | Password hashing, JWT validation, rate limits, CORS, model API access | Backend implemented; model API hardening and [router risk tracking](security/react-router-risk-acceptance.md) remain |
| Containers         | One-command frontend/backend/database startup                         | Implemented in `compose.yaml`; smoke test required                                                                   |
| Ensemble release   | Fitted config, provenance, held-out evaluation                        | Follow the [ensemble release checklist](ensemble-release-checklist.md)                                               |

## Architecture and design evidence

- [System context and container/deployment diagrams](architecture/README.md).
- [Prediction-request sequence diagram](architecture/prediction-sequence.md).
- [Consolidated Design Justification](design-justification.md).
- ADR 001: modular monolith instead of microservices.
- ADR 002: ensemble scoring formula.
- ADR 003: model warm-up strategy.
- ADR 004: XAI method selection.
- Data leakage prevention: deduplication and group-aware splitting.

## Rehearsal checklist

- [ ] Every member has a speaking section.
- [ ] The complete presentation takes 12–15 minutes.
- [ ] Each feature is linked to a user story and acceptance criteria.
- [ ] The model service is warmed up before the demo.
- [ ] Demo accounts and deterministic sample inputs are ready.
- [ ] Failure and edge-case behaviour has been rehearsed (automated 503 path passes).
- [ ] Backup screenshots or a short local recording are available.
- [ ] The team can explain architecture choices, ensemble logic, security,
      testing, deployment, and data-leakage controls.

## Suggested timing

| Section                                   |        Time |
| ----------------------------------------- | ----------: |
| Problem, users, and architecture          |    1 minute |
| User stories and acceptance criteria      | 7–8 minutes |
| Algorithm, XAI, and leakage prevention    | 2–3 minutes |
| Testing, security, Docker, and deployment | 1–2 minutes |
| Buffer                                    |    1 minute |
