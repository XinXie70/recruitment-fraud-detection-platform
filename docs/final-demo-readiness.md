# Final Demo readiness checklist

Use this document as the single source of truth for the Final Demo and keep the
evidence links current.

## User story traceability

The original proposal was not available when this matrix was prepared. The
entries marked `INF` are inferred from the implemented product and must not be
presented as verbatim proposal commitments. If the proposal becomes available,
compare its wording and identifiers with this table before the final rehearsal.
A story is only demo-verified when its acceptance criteria have been demonstrated
and supported by automated or repeatable evidence.

| ID     | Inferred user story / requirement                                      | Acceptance criteria                                                                    | Demo route          | Implementation                                      | Test evidence                                | Owner | Status      |
| ------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------- | -------------------------------------------- | ----- | ----------- |
| INF-01 | A visitor can create an account and log in securely.                    | Valid credentials create a session; invalid credentials show a recoverable error.     | `/register`, `/login` | FastAPI auth, password hashing, JWT; React auth flow | `App.test.jsx`, Playwright registration/login | Team  | Implemented |
| INF-02 | An authenticated user can submit a job advertisement for analysis.      | Blank input is rejected; a valid advert is sent with JWT and produces a result.        | `/analyze`          | Analysis API client and protected analyser          | `App.test.jsx`, `api.test.js`, Playwright     | Team  | Implemented |
| INF-03 | A user receives an ensemble risk score and actionable classification.   | Result shows score, risk tier, recommendation, active models, and ensemble version.    | Analysis report     | Backend ensemble scoring and React report           | Backend scoring/API tests; `App.test.jsx`     | Team  | Implemented |
| INF-04 | A user can understand why an advertisement was classified as risky.     | Structured XAI evidence and plain-language safety guidance are displayed.              | Analysis report     | XAI service, explanation and guidance components    | XAI tests; explanation component tests       | Team  | Implemented |
| INF-05 | A user can review previous scans across sessions and devices.            | Authenticated history loads from the backend; local full results remain viewable.      | `/dashboard`        | History API plus server/local dashboard merge       | `DashboardPage.test.jsx`, API tests, E2E      | Team  | Implemented |
| INF-06 | A user can access educational material about recruitment scams.          | Education resources render and can be filtered by topic.                              | `/learn`            | Education API and education library                 | `EducationLibrary.test.jsx`, API tests        | Team  | Implemented |
| INF-07 | An administrator can view research and service-health information.       | Non-admin users see the user dashboard; admins see health and model information.       | `/dashboard`        | Role-aware routing and Admin Dashboard              | `App.test.jsx`, admin API tests, Playwright   | Team  | Implemented |
| INF-08 | The application handles service failure without crashing or losing input. | 401 logs out safely; 503/network failures show retryable messages; local history remains. | Analyse/Dashboard | API error mapping, loading/error UI, offline fallback | API, App, Dashboard and Playwright tests      | Team  | Implemented |
| INF-09 | The system protects user data and privileged operations.                 | Passwords are hashed; JWT, ownership and admin checks are server-enforced.             | API boundary        | Auth dependencies, ownership filters, rate limits   | Backend auth/history/admin tests              | Team  | Implemented |
| INF-10 | The application can be deployed reproducibly as coordinated services.    | Database, migration, backend and frontend start in dependency order with health checks. | Docker Compose      | `compose.yaml` and service Dockerfiles              | Local infrastructure smoke test passed; model integration pending | Team  | In progress |
| INF-11 | The UI remains usable on phone, tablet and desktop layouts.               | Core routes have responsive layouts, visible loading states and no blocking UI defects. | All primary routes  | Responsive CSS and explicit async states            | Playwright plus rehearsal screenshots needed | Team  | In progress |
| INF-12 | Model evaluation avoids train/validation/test leakage.                    | Group-aware splits and provenance checks prevent row mismatch and test-set fitting.    | Offline pipeline    | Safe ensemble fitting and split validation          | Pipeline tests; fitted release evidence needed | Team | In progress |

Allowed status values: `Not started`, `In progress`, `Implemented`, and
`Demo verified`.

## Quality evidence

| Area               | Evidence to prepare                                                   | Current status                                                                                                       |
| ------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Backend tests      | Test count and coverage report                                        | 74 tests passed; 90.54% measured locally                                                                             |
| Frontend tests     | Unit/component coverage report                                        | 16 tests pass; 47.79% statements / 48.93% lines; reachable `App.jsx` is 91.57% lines                                |
| End-to-end tests   | Authentication, analyse, history, admin routing, and failure paths    | 5 Playwright tests pass                                                                                              |
| Responsive UI      | Phone, tablet, and desktop screenshots                                | Implemented; rehearsal verification required                                                                         |
| Loading and errors | Slow request, invalid input, and unavailable model service            | Implemented; automated 503 path passes; rehearsal still required                                                     |
| Live status        | Refresh service status and scan history without a full page reload    | Implemented on Admin Dashboard and user Dashboard; history falls back to browser-cached results when offline         |
| Security           | Password hashing, JWT validation, rate limits, CORS, model API access | Backend implemented; model API hardening and [router risk tracking](security/react-router-risk-acceptance.md) remain |
| Containers         | One-command frontend/backend/database startup                         | Local infrastructure smoke test passed; model readiness requires `MODEL_SERVER_URL`                                 |
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

## Docker smoke-test evidence

Tested locally on 2026-07-28 with Docker Engine 29.6.1:

- `docker compose config --quiet` passed.
- `docker compose up --build -d` built the frontend and remote-backend images.
- PostgreSQL became healthy and both Alembic migrations exited successfully.
- Backend `/api/live` returned HTTP 200 through ports 8000 and the frontend proxy.
- Frontend port 5190 returned HTTP 200 and its container became healthy.
- Registration, JWT issuance, and authenticated history retrieval succeeded through
  the frontend proxy, proving the frontend/backend/database path.
- `/api/health` returned HTTP 503 because `MODEL_SERVER_URL` was unset and the
  lightweight remote-backend image intentionally lacks local model dependencies and
  complete Git LFS model weights. Repeat the analysis smoke test with the deployed
  model endpoint configured before the Final Demo.

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
