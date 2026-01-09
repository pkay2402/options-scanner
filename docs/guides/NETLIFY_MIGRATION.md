# Netlify Migration (Phase 1)

This doc describes a staged migration from the Streamlit app to a Netlify-hosted PWA frontend + Python backend.

## Target Architecture
- Frontend (PWA): CRA TypeScript app in `pwa-app/frontend` hosted on Netlify
- Backend API: FastAPI (`pwa-app/backend`) hosted on a Python-friendly service (Railway/Render/Fly/EC2)
- Realtime: WebSockets from FastAPI (Netlify will not host Python WS). The frontend connects via `REACT_APP_WS_URL`.
- Discord Bot: Deploy separately (Railway/Render/Fly/Docker on VPS). Not on Netlify.

## Phase 1: Frontend on Netlify (Smoke Test)
1. Netlify config added at repo root: `netlify.toml`
   - Builds `pwa-app/frontend`
   - Publishes `pwa-app/frontend/build`
   - Adds SPA redirects
2. Example Netlify Function: `netlify/functions/hello.js` (smoke test only)

### Deploy Steps
1. Push this repo to GitHub (private repos supported by Netlify).
2. In Netlify UI: New Site from Git, pick this repo.
3. Build settings are auto-detected from `netlify.toml`.
4. Set environment variables in Netlify → Site Settings → Build & deploy → Environment:
   - `REACT_APP_API_URL=https://YOUR-BACKEND-URL` (temporarily use `http://localhost:8000` for local dev)
   - `REACT_APP_WS_URL=wss://YOUR-BACKEND-URL` (or `ws://localhost:8000` for local dev)
5. Deploy. Visit the site URL. You should see the PWA shell load.
6. Test function: `${SITE_URL}/.netlify/functions/hello` should return JSON.

## Phase 2: Backend (FastAPI) Hosting
- Use `pwa-app/backend` (already implemented) and deploy to one of:
  - Railway: Dockerfile or `uvicorn main:app` with Python buildpack
  - Render: Web Service (Python), start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
  - Fly.io: Docker (good for WebSockets)
- Expose HTTPS and note the public URL for API + WS.
- Update Netlify env vars to point the frontend to this backend.

## Phase 3: Page-by-Page Migration
- Choose a Streamlit page and expose its data via FastAPI endpoints
  - Example: Port logic from `pages/2_Option_Volume_Walls.py` into `/api/volume-walls/{symbol}` (already scaffolded)
- Add a corresponding React view calling the new API (use `src/services/api.ts`).
- Iterate until the core Streamlit flows are available in the PWA.

## Phase 4: Auth & Tokens
- Schwab OAuth/token operations should remain on the backend.
- Do NOT expose secrets in the frontend.
- Keep `schwab_client.json`/secure tokens in server storage or a secure secret store (Railway/Render secrets).
- If you need a browser-based login flow, implement a small backend OAuth callback endpoint that stores/rotates tokens.

## Notes
- Netlify supports private repos and deploy previews.
- Netlify is ideal for static frontends + serverless functions; Python WS needs a separate host (covered above).
- The existing PWA (`pwa-app/frontend`) is already CRA and PWA-ready, so Phase 1 is minimal.
