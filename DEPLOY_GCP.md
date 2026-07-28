# Google Cloud Deployment Guide — Frontend + Backend (Supabase Free PostgreSQL)

> Frontend (React + Nginx) and Backend (FastAPI + Gunicorn) deployed separately to Cloud Run.
> Database uses Supabase free hosted PostgreSQL.

## Architecture

```
User → Cloud Run (frontend, Nginx)  ──/api/*──→  Cloud Run (backend, FastAPI)  →  Supabase (free PostgreSQL)
                                                       ↕
                                                Model Inference Server (existing)
```

- **Frontend** `almond-frontend`: Nginx serves static files + reverse proxies `/api/*` to backend
- **Backend** `almond-backend`: FastAPI + Gunicorn, handles business logic

## One-Time Setup (~15 min)

### 1. Create Supabase Free Database (5 min)

1. Open [supabase.com](https://supabase.com), sign up / log in
2. Click **New project**
3. Enter project name (e.g. `almond-db`), set a database password (save it!)
4. Choose region **ap-southeast-1 (Singapore)** or **us-west-1** (near your Cloud Run region)
5. Select **Free plan**, click Create project
6. Wait for creation (~2 min)
7. Go to **Settings → Database**, find **Connection string**
8. Select the **URI** tab, copy the connection string:

```
postgresql://postgres.[PROJECT_REF]:[YOUR_PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

> ⚠️ **Important**: Use **Session Pooler** (port 6543), not direct connection (port 5432).
> Cloud Run is serverless — pooled connections are preferred.

### 2. Install gcloud CLI and Log In

```bash
brew install google-cloud-sdk
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

### 4. Create Artifact Registry (stores Docker images)

```bash
gcloud artifacts repositories create almond-repo \
  --repository-format=docker \
  --location=asia-southeast1
```

### 5. Store Secrets in Secret Manager

```bash
# SECRET_KEY (generate a random string with Python)
python3 -c "import secrets; print(secrets.token_urlsafe(64))" | \
  gcloud secrets create SECRET_KEY_PRODUCTION --data-file=-

# DATABASE_URL — replace with your Supabase connection string
echo -n "postgresql+psycopg2://postgres.xxxxx:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres" | \
  gcloud secrets create DATABASE_URL_PRODUCTION --data-file=-

# ACCESS_TOKEN_EXPIRE_MINUTES
echo -n "10080" | gcloud secrets create ACCESS_TOKEN_EXPIRE_MINUTES --data-file=-
```

### 6. Grant Cloud Run Access to Secret Manager

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
CLOUD_RUN_SA="service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Deployment

### Method 1: Cloud Build Auto Deploy (recommended)

Push to `main` branch. `cloudbuild.yaml` will automatically:
1. Build and push the versioned backend image
2. Deploy and execute the one-off `almond-migrate` Cloud Run Job
3. Deploy backend `almond-backend` only after migrations succeed
4. Get the backend URL
5. Build + deploy frontend `almond-frontend` (auto-injects `BACKEND_URL`)

The API container runs with `RUN_DATABASE_MIGRATIONS=false`. This prevents
multiple Cloud Run instances from racing to apply schema changes during rollout
or autoscaling. A failed migration stops the build before the API deployment.

### Method 2: Manual Separate Deployment

**Deploy backend first:**
```bash
docker build -f backend/Dockerfile.cloudrun -t gcr.io/YOUR_PROJECT/backend .
docker push gcr.io/YOUR_PROJECT/backend

gcloud run deploy almond-backend \
  --image=gcr.io/YOUR_PROJECT/backend \
  --region=asia-southeast1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars=ENVIRONMENT=production \
  --set-env-vars=MODEL_SERVER_URL="https://your-model-server.com" \
  --set-secrets=DATABASE_URL=DATABASE_URL_PRODUCTION:latest \
  --set-secrets=SECRET_KEY=SECRET_KEY_PRODUCTION:latest
```

**Then deploy frontend:**
```bash
# Get backend URL
BACKEND_URL=$(gcloud run services describe almond-backend \
  --region=asia-southeast1 \
  --format='value(status.url)')

# Build frontend image
docker build -f frontend/Dockerfile.prod -t gcr.io/YOUR_PROJECT/frontend .
docker push gcr.io/YOUR_PROJECT/frontend

# Deploy frontend, inject BACKEND_URL
gcloud run deploy almond-frontend \
  --image=gcr.io/YOUR_PROJECT/frontend \
  --region=asia-southeast1 \
  --platform=managed \
  --allow-unauthenticated \
  --cpu=1 \
  --memory=256Mi \
  --set-env-vars=BACKEND_URL="${BACKEND_URL}"
```

---

## Verify Deployment

```bash
# Get service URLs
gcloud run services describe almond-backend \
  --region=asia-southeast1 \
  --format='value(status.url)'

gcloud run services describe almond-frontend \
  --region=asia-southeast1 \
  --format='value(status.url)'

# Backend health check
curl https://BACKEND_URL/api/health
# Expected: {"status":"healthy","service":"fake_job_detection_api","model_ready":true}

# Frontend (via nginx proxy to backend)
curl https://FRONTEND_URL/api/health
# Expected: same as above
```

---

## Cost Estimate

| Service | Config | Monthly (approx) |
|---------|--------|------------------|
| Cloud Run (backend) | 1 vCPU, 512MB, pay-per-use | $0 — within free tier |
| Cloud Run (frontend) | 1 vCPU, 256MB, pay-per-use | $0 — within free tier |
| Supabase | Free plan (500MB DB) | **$0** |
| Secret Manager | 3 secrets | $0 |
| Artifact Registry | Small amount of images | ~$0 |
| **Total** | | **$0/month 🎉** |

---

## Security Notes

1. **Supabase connection string contains a password** — always store via Secret Manager, never hardcode
2. **Store all secrets in Secret Manager** — never hardcode in code or env vars
3. **Cloud Run auto HTTPS** — no need to configure TLS certificates manually
4. **Frontend Nginx reverse proxy** — browser accesses `/api/*` through the same frontend domain, no CORS needed
5. **Supabase free tier limits** — 500MB database, 2 projects, weekly backups, manual recovery after pause
