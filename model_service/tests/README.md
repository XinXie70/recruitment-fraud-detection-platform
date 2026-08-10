# model_service tests

This directory contains automated tests for the FP-gate inference API in
[`../`](../).

## Test layers

| Layer | Files | What is exercised |
| --- | --- | --- |
| Unit | `test_text_utils.py`, `test_coalesce.py`, `test_ensemble_service.py`, `test_settings.py`, `test_lr_service.py` | Input validation, FP-gate business rules, config loading, request coalescing |
| Integration | `test_app_integration.py` | Flask routes, auth, batch limits, error handling with mocked LR/BERT services |
| End-to-end | `test_app_e2e.py` | Real model weights and `/predict/all` contract (slow; opt-in) |

## Running tests

From the repository root:

```bash
pip install -r model_service/requirements-dev.txt
pytest model_service/tests -q
```

With coverage:

```bash
pytest model_service/tests -m "not e2e" \
  --cov=model_service \
  --cov-report=term-missing \
  --cov-fail-under=55
```

Slow end-to-end tests (load BERT + LR artifacts):

```bash
RUN_MODEL_SERVICE_E2E=1 pytest model_service/tests -m e2e -q
```

On Windows PowerShell:

```powershell
$env:RUN_MODEL_SERVICE_E2E = "1"
pytest model_service/tests -m e2e -q
```

## Mocking strategy

- **Unit tests** call pure functions directly (`build_texts`, `apply_fp_gate`, `apply_risk`).
- **Integration tests** patch the service objects imported by `prediction_routes` so HTTP
  routes can be tested without loading PyTorch or joblib artifacts. Authentication tests
  set `app.config["MODEL_API_KEY"]`, matching the runtime request hook.
- **E2E tests** load the frozen files under `model_service/models/` and exercise the real
  `/predict/all` path. They are skipped unless `RUN_MODEL_SERVICE_E2E=1` because they are
  slow and memory-intensive.

## Coverage notes

Happy-path and sad-path cases are included for:

- JSON validation and text-length limits
- API-key authentication (`401` when missing or wrong)
- Batch size / item-shape validation
- FP-gate demotion when LR blocks a high BERT score
- Concurrent inference coalescing (race-condition protection)
- Sanitized `500` responses (no traceback leakage)

If E2E tests cannot run in CI because of CPU/RAM limits, the skip reason is documented
above and the mocked integration suite still verifies the HTTP contract.

CI enforces an initial 55% model-service coverage floor. This deliberately
includes the production package while excluding the opt-in E2E path; raise the
floor as BERT-service unit coverage is added.
