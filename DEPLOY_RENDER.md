# Render.com — Frontend Web Service Deployment

> Both frontend and backend on Render.com as Web Services. Nginx proxy, zero CORS config.

## Architecture

```
User → https://almond-frontend.onrender.com  (Render Web Service, Nginx :80)
         │
         ├─ /                     → index.html (SPA)
         ├─ /assets/*.js          → static files
         └─ /api/*                → Nginx proxy → backend (no CORS needed)
                                                        │
                                                        ▼
                              https://capstone-project-26t2-9900-h09c-almond.onrender.com
```

---

## 1. Backend (already deployed)

```
https://capstone-project-26t2-9900-h09c-almond.onrender.com
```

---

## 2. Deploy Frontend on Render

### Step A: Push latest code to GitHub

Make sure `frontend/Dockerfile.prod` and `frontend/docker-entrypoint.sh` are pushed.

### Step B: Create Web Service

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect the same GitHub repo
3. Configure:

| Setting            | Value                                                              |
|--------------------|--------------------------------------------------------------------|
| Name               | `almond-frontend`                                                  |
| Root Directory     | `frontend`                                                         |
| Dockerfile Path    | `frontend/Dockerfile.render`                                      |
| Port               | `80`                                                               |

4. Add **Environment Variable**:

| Key            | Value                                                               |
|----------------|---------------------------------------------------------------------|
| `BACKEND_URL`  | `https://capstone-project-26t2-9900-h09c-almond.onrender.com`      |

> ⚠️ **Important**: If re-deploying the backend, use `backend/Dockerfile.remote` (slim, no ML deps) instead of `backend/Dockerfile`. The full ML Dockerfile (TensorFlow/PyTorch) will OOM on Render's free tier (512 MB RAM). Set `MODEL_SERVER_URL` to your model inference server, or deploy models separately.

5. Click **Create Web Service**

---

## 3. No CORS Needed

Nginx proxies `/api/*` to the backend internally. The browser only talks to the frontend domain — same origin, no CORS.

---

## 4. Verify

```bash
# Frontend
curl https://almond-frontend.onrender.com

# API through frontend proxy
curl https://almond-frontend.onrender.com/api/health
```

Open `https://almond-frontend.onrender.com` in browser and test.

---

## How It Works

`docker-entrypoint.sh` → `envsubst` injects `BACKEND_URL` into `nginx.conf` → Nginx proxies `/api/*`

| Request | Routed to |
|---------|-----------|
| `GET /` | Nginx → `index.html` |
| `GET /assets/app.js` | Nginx → static file |
| `POST /api/predict` | Nginx → `https://capstone-project-.../api/predict` |
| `GET /api/health` | Nginx → `https://capstone-project-.../api/health` |

---

## Cost

| Service              | Cost       |
|----------------------|------------|
| Render Web Service × 2 | Free tier |
| **Total**            | **$0 🎉**  |


