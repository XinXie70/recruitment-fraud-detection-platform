# Backend component diagram

```mermaid
flowchart LR
    ui["React frontend"]

    subgraph api["FastAPI application"]
        middleware["Middleware<br/>request ID, body guard,<br/>CORS, security headers"]
        auth["Authentication routes<br/>bcrypt + JWT"]
        analysis["Analysis routes<br/>validation, history ownership"]
        admin["Admin routes<br/>role-protected reporting"]
        health["Health endpoints<br/>live, ready, health"]

        service["AnalysisService<br/>workflow orchestration + TTL cache"]
        ensemble["EnsemblePredictor<br/>calibration, weighting,<br/>thresholds, graceful degradation"]
        registry["ModelRegistry<br/>parallel adapters + timeout"]
        xai["XAIService<br/>SHAP with occlusion fallback"]
        gentle["GentleAIService<br/>deterministic education + optional Ollama"]
        url["URL analyzer"]
        persistence["SQLAlchemy persistence"]
    end

    db[("PostgreSQL")]
    models["Local pipelines or<br/>remote model API"]
    ollama["Optional Ollama"]

    ui --> middleware
    middleware --> auth
    middleware --> analysis
    middleware --> admin
    middleware --> health
    auth --> persistence
    admin --> persistence
    analysis --> service
    analysis --> persistence
    service --> ensemble
    service --> xai
    service --> gentle
    service --> url
    ensemble --> registry
    registry --> models
    xai --> ensemble
    gentle -.-> ollama
    persistence --> db
```

The dependency direction keeps HTTP concerns in routers and business logic in
services. `ModelRegistry` hides whether inference is local or remote, allowing
deployment changes without changing the analysis route contract.
