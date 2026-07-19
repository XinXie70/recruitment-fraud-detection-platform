# Google Cloud Deployment Guide — Backend (Database uses Supabase Free PostgreSQL)

> Deploy only the backend (FastAPI + Gunicorn) to Cloud Run. Database uses Supabase free hosted PostgreSQL. Frontend is not deployed.

## 📋 Architecture

```
User/Frontend → Cloud Run (backend) → Supabase (free PostgreSQL)
                ↕
         Model Inference Server (existing, not deployed here)
```

## 🚀 One-Time Setup (~15 min)

### 1. Create Supabase Free Database (5 min)

1. Open [supabase.com](https://supabase.com) and sign up / log in
2. Click **New project**
3. Fill in the project name (e.g. `almond-db`), set a database password (write it down!)
4. Select Region **ap-southeast-1 (Singapore)** or **us-west-1** (close to your Cloud Run region)
5. Choose **Free plan**, click Create project
6. Wait for creation to complete (~2 min)
7. Go to **Settings → Database**, find **Connection string**
8. Select **URI** tab, copy the connection string. Format:

```
postgresql://postgres.[PROJECT_REF]:[YOUR_PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

> ⚠️ **Important**: Use **Session Pooler** connection (port 6543), not direct connection (port 5432), because Cloud Run is serverless and connection pooling is more appropriate.

### 2. Install gcloud CLI and Login

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

## 🔧 Deployment

### Method 1: Cloud Build Auto-Deploy (Recommended)

Modify `${_MODEL_SERVER_URL}` in `cloudbuild.yaml` with your actual value, then push code to trigger automatic build and deployment.

### Method 2: Manual Build and Deploy

```bash
# 1. Build image
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_MODEL_SERVER_URL="https://your-model-server.com"

# 2. Or deploy manually with Docker only
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

---

## ✅ Verify Deployment

```bash
# Get service URL
gcloud run services describe almond-backend \
  --region=asia-southeast1 \
  --format='value(status.url)'

# Health check
curl https://YOUR_SERVICE_URL/api/health

# Expected response:
# {"status":"healthy","service":"fake_job_detection_api","model_ready":true}
```

---

## 📊 Cost Estimate

| Service | Config | Monthly Cost (~) |
|------|------|-----------|
| Cloud Run | 0.5 vCPU, 256MB, pay-per-use | $0 — within free tier |
| Supabase | Free plan (500MB DB) | **$0** |
| Secret Manager | 3 secrets | $0 |
| Artifact Registry | Small images | ~$0 |
| **Total** | | **$0/month 🎉** |

---

## 🔒 Security Notes

1. **Supabase connection string contains password** — Always store via Secret Manager, never hardcode
2. **Secret Manager stores all secrets** — Never hardcode in code or environment variables
3. **Cloud Run auto-HTTPS** — No need to configure TLS certificates yourself
4. **Set CORS_ORIGINS** — Restrict allowed frontend domains
5. **Supabase free plan limits** — 500MB database, 2 projects, weekly backups, manual recovery after pausing
