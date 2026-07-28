# Prediction sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React frontend
    participant API as FastAPI analysis route
    participant Auth as JWT dependency
    participant Service as AnalysisService
    participant Ensemble as EnsemblePredictor
    participant Models as Model inference API
    participant XAI as XAI + Gentle AI
    participant DB as PostgreSQL

    User->>UI: Submit job advertisement text
    UI->>API: POST /api/v1/analyze + Bearer token
    API->>Auth: Validate signature, expiry, issuer, audience, and user
    alt Invalid or expired token
        Auth-->>UI: 401 Unauthorized
    else Authenticated
        API->>Service: analyze(text)
        Service->>Service: Check cache and validate job relevance
        alt Invalid or irrelevant input
            Service-->>UI: 422 with a safe validation reason
        else Valid input
            Service->>Ensemble: predict(text)
            Ensemble->>Models: Run configured model requests in parallel
            Models-->>Ensemble: Raw probabilities or per-model failures
            alt No model succeeds
                Ensemble-->>UI: 503 prediction service unavailable
            else At least one model succeeds
                Ensemble->>Ensemble: Calibrate scores and renormalise available weights
                Ensemble-->>Service: Risk score, level, member evidence, status
                Service->>XAI: Explain ensemble score and generate guidance
                XAI-->>Service: Evidence spans and user-friendly guidance
                Service->>Service: Analyse URLs and mark degraded dependencies
                Service-->>API: Analysis response
                API->>DB: Save redacted preview, SHA-256 input hash, and result
                Note over API,DB: A database write failure is logged and rolled back;<br/>the completed prediction is still returned.
                API-->>UI: 200 success or degraded result
                UI-->>User: Display risk, explanation, and recommended action
            end
        end
    end
```

The degraded path is intentional: if some ensemble members or URL analysis
fail, the available model weights are renormalised and the response explicitly
reports degraded status. The service returns 503 only when no model succeeds.
