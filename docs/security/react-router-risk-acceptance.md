# React Router security review

| Field | Value |
|---|---|
| Status | Resolved; continue routine dependency monitoring |
| Reviewed | 2026-08-09 |
| Dependency | `react-router` 8.3.0 |
| Application mode | Client-only Declarative SPA |

## Decision

The frontend now pins `react-router` 8.3.0. The earlier temporary acceptance
for version 7.18.1 is closed.

The application remains a Vite-built, client-only SPA. It does not use React
Router Framework Mode, React Server Components, loaders, actions, server
actions, `@react-router/serve`, or React Router SSR. State-changing requests go
to the separate FastAPI backend, which enforces JWT validation, authorisation,
input validation, CORS, and rate limiting.

## Verification

The following checks were run from `frontend/` on 2026-08-09:

- `npm ls react-router react-router-dom --depth=0` resolved `react-router` to
  8.3.0; `react-router-dom` is not a direct dependency.
- `npm audit --omit=dev` reported zero production vulnerabilities.

Functional lint, test, end-to-end, and build checks are maintained in the
[CI workflow](../../.github/workflows/frontend-ci.yml), rather than copied into
this security record where counts can become stale.

## Required controls

- Keep router dependencies pinned through `package-lock.json`.
- Review Dependabot alerts and React Router release notes before upgrades.
- Run `npm audit --omit=dev` and the complete frontend quality gate after every
  router upgrade.
- Treat FastAPI authentication and authorisation as the security boundary;
  client-side routing is never an authorisation control.
- Perform a new architecture and security review before adopting SSR, React
  Server Components, loaders, actions, or Framework Mode.

References:

- <https://github.com/advisories/GHSA-qwww-vcr4-c8h2>
- <https://reactrouter.com/changelog>
