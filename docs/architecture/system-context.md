# System context

This diagram shows the people and external systems that interact with the Fake
Job Advertisement Detection System.

```mermaid
flowchart LR
    user["Job seeker<br/>Submits advertisements and reviews risk explanations"]
    admin["Administrator<br/>Reviews users, analyses, and service statistics"]

    system["Fake Job Advertisement Detection System<br/>Authentication, analysis, history, education, and administration"]

    model["FP-gate Model Service<br/>BERT primary signal + Logistic Regression false-positive gate"]
    ollama["Optional Ollama Service<br/>Rewrites deterministic explanations in friendlier language"]

    user -->|"Uses over HTTPS"| system
    admin -->|"Uses over HTTPS"| system
    system -->|"Requests final risk and batched perturbation scores"| model
    system -.->|"Optional rewrite request"| ollama
```

## Boundary and trust assumptions

- Browsers communicate with the application API; they do not call the model
  server directly.
- Authentication and authorization are enforced by the FastAPI backend.
- Ollama is optional. The application retains deterministic template-based
  explanations when it is unavailable.
- Production traffic to both application and model endpoints should use HTTPS.
  The model endpoint should additionally require service-to-service
  authentication and network restrictions.
