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
    UI->>API: POST /api/v1/analyze/score + Bearer token
    API->>Auth: Validate signature, expiry, issuer, audience, and user
    alt Invalid or expired token
        Auth-->>UI: 401 Unauthorized
    else Authenticated
        API->>Service: score(text)
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
                Service->>Service: Analyse URLs and mark degraded dependencies
                Service-->>API: Score-phase response
                API-->>UI: 200 risk score and model outputs
                UI-->>User: Display the primary risk result immediately
                UI->>API: POST /api/v1/analyze for detailed explanation
                API->>Service: analyze(text)
                Service->>Ensemble: predict(text)
                Ensemble->>Models: Run model requests used by XAI
                Models-->>Ensemble: Raw probabilities
                Service->>XAI: Explain ensemble score and generate guidance
                XAI->>Models: Run batched perturbation predictions
                Models-->>XAI: Perturbed probabilities
                XAI-->>Service: Evidence spans and user-friendly guidance
                Service-->>API: Complete analysis response
                API->>DB: Save redacted preview, SHA-256 input hash, and result
                Note over API,DB: A database write failure is logged and rolled back;<br/>the completed prediction is still returned.
                API-->>UI: 200 complete or degraded result
                UI-->>User: Add explanation and recommended action
            end
        end
    end
```

The two-stage flow intentionally prioritises perceived latency: the score route
does not run perturbation-based XAI, so the user can inspect the primary result
while the full request continues. If the explanation request fails, the UI
keeps the score visible and reports that only the detailed evidence is
unavailable.

The degraded path is also intentional: if some ensemble members or URL
analysis fail, the available model weights are renormalised and the response
explicitly reports degraded status. The service returns 503 only when no model
succeeds.
