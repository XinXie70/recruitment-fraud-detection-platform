# React Router security risk acceptance

| Field            | Value                                               |
| ---------------- | --------------------------------------------------- |
| Status           | Temporarily accepted; monitor for a patched release |
| Reviewed         | 2026-07-28                                          |
| Dependency       | `react-router-dom` / `react-router` 7.18.1          |
| Application mode | Client-only Declarative SPA                         |

## Decision

Upgrade from 6.30.4 to 7.18.1 and pin the exact version.

Version 7.18.1 resolves the client navigation/open-redirect advisories that
affected the previous dependency range. At the review date, `npm audit
--omit=dev` still reports two high-severity findings under
`GHSA-qwww-vcr4-c8h2`. That advisory affects React Server Components and Action
request processing.

The application does not use React Router Framework Mode, RSC, loaders,
actions, server actions, `@react-router/serve`, or React Router SSR. It is built
by Vite as static client assets; all state-changing requests go directly to the
separate FastAPI backend, which performs its own JWT validation, authorization,
input validation, CORS enforcement, and rate limiting.

For this architecture, the vulnerable RSC/Action execution path is not present.
The finding is therefore accepted temporarily rather than downgrading to a
version with vulnerabilities in the client APIs the application actually uses.

## Verification

- `npm ls react-router react-router-dom` resolves both packages to 7.18.1.
- Frontend lint passes.
- 11 Vitest unit/component tests pass with coverage thresholds.
- 5 Playwright end-to-end tests pass, including protected-route redirects,
  authentication, analysis navigation, and admin routing.
- The Vite production build succeeds.

## Required controls

- Keep the application in client-only Declarative Mode.
- Do not add React Router loaders, actions, RSC, SSR, or framework server
  packages while this acceptance is active.
- Continue validating every state-changing API operation in FastAPI; client
  routing is never an authorization boundary.
- Monitor React Router releases and GitHub Dependabot alerts.
- Re-run `npm audit --omit=dev` and the complete frontend gate when a patched
  version is published.

## Expiry

This acceptance expires when any of the following occurs:

1. React Router publishes a version outside the affected advisory range.
2. The application adopts SSR, RSC, loaders, actions, or Framework Mode.
3. New evidence shows the affected code is reachable in Declarative SPA mode.
4. The project reaches production release review, whichever comes first.

References:

- <https://github.com/advisories/GHSA-qwww-vcr4-c8h2>
- <https://reactrouter.com/changelog>
