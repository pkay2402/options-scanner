# Deploy FastAPI Backend on Railway

This guide deploys the backend in `pwa-app/backend` to Railway with no Docker needed, supports WebSockets, and loads Schwab tokens from an environment variable.

## Why no Docker?
The backend imports from the repo-level `src/` folder. Using Railway's native Python build keeps the full repo available during runtime without custom Docker context.

## Prereqs
- GitHub repo connected to Railway (private repos supported)
- A current `schwab_client.json` from your local auth setup

## Steps
1. Create a new Railway service → Deploy from GitHub → select this repo.
2. Service settings:
   - Root Directory: repository root (keep default).
   - Build Command:
     ```sh
     pip install -r pwa-app/backend/requirements.txt
     ```
   - Start Command:
     ```sh
     uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir pwa-app/backend
     ```
   - Health Check Path: `/`
3. Environment Variables:
   - `SCHWAB_CLIENT_ID`: your Schwab app key
   - `SCHWAB_CLIENT_SECRET`: your Schwab app secret
   - `SCHWAB_REDIRECT_URI`: e.g. `https://127.0.0.1:8182`
   - `SCHWAB_CLIENT_JSON`: paste the full JSON content from your `schwab_client.json` file
   - Optional: `LOG_LEVEL=INFO`

   Note: `main.py` automatically writes `SCHWAB_CLIENT_JSON` to the expected file on startup.

4. Deploy
   - On success, open the public URL.
   - Verify:
     - `GET /` → healthy payload
     - `GET /api/auth/status` → should report authenticated if tokens are valid

## Using WebSockets
- Railway supports WS/WSS on the same service.
- Test endpoint: `wss://<your-railway-host>/ws/flow`

## Connect Netlify Frontend
- In Netlify → Site Settings → Environment:
  - `REACT_APP_API_URL=https://<your-railway-host>`
  - `REACT_APP_WS_URL=wss://<your-railway-host>`
- Redeploy the Netlify site.

## Token Refresh
- Refresh locally using your existing script, then update Railway:
  1. Run locally:
     ```sh
     cd /Users/piyushkhaitan/schwab/options
     python scripts/auth_setup.py
     ```
  2. Copy updated `schwab_client.json`
  3. Paste new JSON into Railway env var `SCHWAB_CLIENT_JSON`
  4. Redeploy service

## Troubleshooting
- 401s on API: Token expired → refresh and update `SCHWAB_CLIENT_JSON`
- Import errors for `src.*`: Ensure Start Command uses `--app-dir pwa-app/backend` and Root Directory is repo root.
- WebSocket disconnects: Confirm WSS URL and no corporate proxies blocking WS.
