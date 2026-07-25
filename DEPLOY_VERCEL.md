# Deploy the frontend to Vercel

The frontend is a Vite single-page application in `frontend/`. Its
`vercel.json` sends `/api/*` requests to the existing Render backend and sends
client-side routes such as `/analyze` and `/dashboard` to `index.html`.

## First deployment

1. Push this branch (or merge it) to GitHub.
2. In Vercel, choose **Add New > Project** and import this GitHub repository.
3. Configure the project as follows:

   | Setting | Value |
   | --- | --- |
   | Framework Preset | Vite |
   | Root Directory | `frontend` |
   | Build Command | `npm run build` (detected automatically) |
   | Output Directory | `dist` (detected automatically) |
   | Install Command | `npm ci` (detected automatically) |

4. No frontend environment variable is required while the API rewrite in
   `frontend/vercel.json` points to the current Render backend.
5. Select **Deploy**.

Vercel will then create production deployments from the configured production
branch and preview deployments/checks for pull requests.

## Show the deployed site on GitHub

After the first deployment, copy the production URL (for example,
`https://your-project.vercel.app`). On the GitHub repository page, open the
**About** settings (gear icon), paste it into **Website**, and save. GitHub pull
requests and commits will also show Vercel deployment checks once the Vercel
GitHub integration is connected.

## Verification

After deployment, check:

```bash
curl -I https://your-project.vercel.app
curl https://your-project.vercel.app/api/health
```

Also open `https://your-project.vercel.app/analyze` directly. It should load the
app instead of returning a Vercel 404.

## If the backend address changes

Update the first rewrite destination in `frontend/vercel.json`, commit, and
push. Vercel will redeploy automatically. Because API calls are proxied through
the Vercel domain, the browser does not require an additional backend CORS
origin for this setup.
