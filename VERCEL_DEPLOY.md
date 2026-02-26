# Deploying to Vercel

This project is set up to deploy on Vercel: **frontend** (React/Vite) and **API** (FastAPI via serverless) from one repo.

## Why you saw "CODE NOT FOUND" (404)

- Vercel was building from the repo **root**, which has no `index.html` or build output.
- The app actually lives in **`frontend/`** and the API in **`backend/`** and **`api/`**.

## What’s configured

1. **`vercel.json`**  
   - Builds the app from `frontend/` and uses **`frontend/dist`** as the output directory.  
   - This fixes the 404 so the deployed URL serves the React app.

2. **`api/[...path].py`**  
   - Catch-all serverless function that runs the FastAPI backend for all `/api/*` routes (e.g. `/api/health`, `/api/chat`, `/api/draft-email`, `/api/trial/...`).  
   - Uses Mangum to run the FastAPI app in a serverless way.

3. **Environment variables**  
   - `.env` is gitignored, so you must set variables in **Vercel**, not only locally.

## Steps to deploy

### 1. Connect the repo to Vercel

- [Vercel Dashboard](https://vercel.com/dashboard) → **Add New** → **Project** → import `tylercarty-cloud/docupdate-tial-nav` (or your repo).
- Leave **Root Directory** empty (project root).
- Vercel will use the repo’s `vercel.json` for build and output.

### 2. Add environment variables in Vercel

In the project: **Settings → Environment Variables**, add:

| Name               | Value              | Notes                    |
|--------------------|--------------------|--------------------------|
| `OPENAI_API_KEY`   | `sk-...`           | **Required** for chat and email drafting |

Add them for **Production** (and Preview if you want the same behavior there).

### 3. Deploy

- Push to `main` (or your production branch), or trigger a deploy from the Vercel dashboard.
- After the build, your deployment URL should serve the app and `/api/*` should hit the FastAPI backend.

### 4. Optional: client-side routing (SPA fallback)

If you use React Router (or similar) and want routes like `/trial/NCT123` to load the app instead of 404:

- In Vercel: **Project → Settings → General** (or **vercel.json**), add a **Rewrite**:
  - **Source:** `/:path*`  
  - **Destination:** `/index.html`  
  - Use a **Condition** so this does **not** apply when the path starts with `/api/` (so API routes are unchanged).

## Troubleshooting

- **404 on the app**  
  - Confirm **Build Command** and **Output Directory** match `vercel.json`: build from `frontend/`, output `frontend/dist`.

- **503 or “OPENAI_API_KEY not configured”**  
  - Add `OPENAI_API_KEY` in Vercel **Environment Variables** and redeploy.

- **API returns 404 or 500**  
  - Check **Functions** (or **Logs**) in the Vercel dashboard for the serverless function that handles `/api/*` (`api/[...path].py`).  
  - Ensure the **backend** folder (and CSV data files) are in the repo so the serverless function can load trial data.

- **CORS errors**  
  - The backend allows `https://*.vercel.app`. If you use a custom domain, add it to `allow_origins` or `allow_origin_regex` in `backend/api/main.py` and redeploy.
