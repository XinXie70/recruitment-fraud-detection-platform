# Deployment diagram

The checked-in Compose configuration provides a reproducible local deployment.
The model inference workload can run remotely because its model artifacts and
compute requirements differ from the business API.

```mermaid
flowchart TB
    browser["User browser"]

    subgraph host["Application host / local Docker Compose"]
        frontend["frontend container<br/>React + Vite :5190"]
        backend["backend container<br/>FastAPI :8000"]
        migration["migrate one-shot container<br/>Alembic migrations"]
        database[("database container<br/>PostgreSQL :5432")]

        frontend -->|"Proxy /api"| backend
        backend -->|"SQLAlchemy"| database
        migration -->|"Schema migration"| database
    end

    subgraph gcp["Google Cloud project"]
        vm["Compute Engine VM"]
        model["Model Inference API<br/>/health and /predict/batch"]
        artifacts["Model artifacts<br/>LR, SVM, XGBoost, DNN,<br/>RNN, BiLSTM, BERT, RoBERTa"]
        vm --- model
        model --- artifacts
    end

    browser -->|"HTTP locally / HTTPS in production"| frontend
    backend -->|"MODEL_SERVER_URL<br/>HTTPS + service authentication required"| model
```

## Startup order

`database healthy` → `migrations completed` → `backend healthy` → `frontend`

Docker Compose encodes this order using health checks and `depends_on`. The
remote model service is configured using `MODEL_SERVER_URL`; when it is unset,
the backend can load local model adapters instead.
