# Single-host portfolio deployment

This deployment profile runs the complete application on one Docker host. Only
the frontend port is published. Nginx serves the React application and proxies
`/api` to FastAPI on the private Compose network. PostgreSQL and the BERT/LR
model service are not exposed to the internet.

## Requirements

- A Linux host with Docker Engine and Docker Compose v2
- At least 8 GB RAM and 10 GB free disk space
- Git LFS, with the model artifacts pulled before the image is built
- One inbound TCP port for the website; use a TLS reverse proxy or a hosting
  provider that terminates HTTPS before making the service public

## Prepare the environment

Copy `deployment.env.example` to `deployment.env`. Generate three different
random hexadecimal values for `POSTGRES_PASSWORD`, `SECRET_KEY`, and
`MODEL_API_KEY`. Set `PUBLIC_ORIGIN` to the exact public HTTPS origin. The real
`deployment.env` file is ignored by Git and must stay on the host.

The PostgreSQL password must be URL-safe because it is included in the internal
database connection URL. A hexadecimal value is suitable.

## Start and verify

Run the production profile with the deployment environment file:

```bash
git lfs pull
docker compose --env-file deployment.env -f compose.production.yaml up --build -d
docker compose --env-file deployment.env -f compose.production.yaml ps -a
```

The database, model service, backend, and frontend should report healthy. The
one-time `migrate` container should exit with code 0. With the example local
configuration, open `http://localhost:8080/login` and verify backend readiness
at `http://localhost:8080/api/ready`.

## Operations

View recent logs:

```bash
docker compose --env-file deployment.env -f compose.production.yaml logs --tail=200
```

Stop the application without deleting stored data:

```bash
docker compose --env-file deployment.env -f compose.production.yaml down
```

Do not add `--volumes` unless the PostgreSQL data is intentionally being
deleted. Back up the `postgres_data` volume before server migration or major
database changes.
