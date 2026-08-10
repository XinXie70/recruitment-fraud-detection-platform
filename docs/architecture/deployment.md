# Deployment diagram

```mermaid
flowchart TB
    browser["User browser"] --> frontend["React frontend"]
    frontend -->|"/api"| backend["FastAPI application"]
    backend --> database[("PostgreSQL")]
    backend -->|"MODEL_SERVER_URL: /predict/all and /predict/batch"| model["BERT + LR FP-gate service"]
    model --> bert["BERT maxlen=512 checkpoint"]
    model --> lr["LR bigram artifact"]
    model --> config["FP-gate and risk-boundary config"]
```

Startup order is PostgreSQL, migrations, model API, FastAPI, then frontend.
`MODEL_SERVER_URL` is mandatory. Model service failure is surfaced as `503`;
there is no local model fallback.
