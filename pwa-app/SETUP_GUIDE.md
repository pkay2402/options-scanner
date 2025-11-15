# PWA Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to the PWA app directory
cd /Users/piyushkhaitan/schwab/options/pwa-app

# Run the automated setup script
chmod +x start.sh
./start.sh
```

The `start.sh` script will:
- Create Python virtual environment
- Install backend dependencies
- Install frontend dependencies (npm)
- Create .env file (you need to add your Schwab API keys)
- Start both backend and frontend servers

### 2. Configure Environment Variables

Edit `pwa-app/backend/.env`:

```bash
# Schwab API Credentials (same as parent app)
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_SECRET=your_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1

# Optional: Discord Webhook
DISCORD_WEBHOOK_URL=your_webhook_url_here

# VAPID Keys for Push Notifications (generate these)
VAPID_PUBLIC_KEY=your_vapid_public_key
VAPID_PRIVATE_KEY=your_vapid_private_key
```

### 3. Generate VAPID Keys for Push Notifications

```bash
cd frontend
npx web-push generate-vapid-keys
```

Copy the output to your `.env` file.

### 4. Manual Start (if not using start.sh)

**Backend:**
```bash
cd pwa-app/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd pwa-app/frontend
npm install
npm start
```

## Access the App

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (Swagger UI)

## Features

### ✅ Completed Components

1. **Dashboard** (`/`)
   - Market overview with SPY, QQQ, IWM, DIA, VIX
   - Real-time price updates every 30 seconds
   - Quick actions and feature cards

2. **Volume Walls** (`/volume-walls`)
   - Interactive volume walls chart with Plotly
   - Call wall, put wall, and flip level detection
   - Trading range analysis

3. **Flow Scanner** (`/flow-scanner`)
   - Real-time options flow via WebSocket
   - Configurable premium filters ($50K - $1M+)
   - Symbol filtering with chips
   - Push notifications for big trades ($500K+)
   - Trade sentiment indicators

4. **Option Finder** (`/option-finder`)
   - Max gamma scanner with heatmap
   - Strike-level gamma exposure analysis
   - Volume, open interest, and IV metrics
   - Interactive Plotly charts

5. **Alerts Config** (`/alerts`)
   - Custom alert rule creation
   - Push notification management
   - Alert types: premium, volume, price, gamma, IV
   - Rules persist in localStorage

### 🔧 Backend Features

- **REST API Endpoints:**
  - `/api/auth/status` - Check Schwab authentication
  - `/api/quote/{symbol}` - Get real-time quote
  - `/api/options-chain/{symbol}` - Get options chain data
  - `/api/volume-walls/{symbol}` - Calculate volume walls
  - `/api/price-history/{symbol}` - Get historical prices
  - `/api/big-trades/{symbol}` - Detect big trades

- **WebSocket Endpoints:**
  - `/ws/flow` - Real-time options flow stream
  - `/ws/alerts` - Real-time alert notifications

- **Authentication:**
  - Reuses existing `SchwabClient` from parent directory
  - Token management via `../tokens/` directory
  - No duplicate authentication needed

### 📱 PWA Features

- **Service Worker:** Offline caching and background sync
- **Push Notifications:** Real-time alerts with actions
- **App Manifest:** Install prompts for iOS/Android/Desktop
- **Responsive Design:** Mobile-first CSS with breakpoints
- **Bottom Navigation:** Mobile-friendly tab navigation

## Installation on Mobile

### iOS (iPhone/iPad)

1. Open Safari and navigate to `http://your-server:3000`
2. Tap the Share button (square with arrow)
3. Scroll down and tap "Add to Home Screen"
4. Name the app and tap "Add"
5. The app icon will appear on your home screen

### Android

1. Open Chrome and navigate to `http://your-server:3000`
2. Tap the menu (three dots) and select "Add to Home Screen"
3. Or look for the install prompt banner at the bottom
4. Tap "Install" and confirm

### Desktop (Chrome/Edge)

1. Navigate to `http://your-server:3000`
2. Look for the install icon in the address bar (monitor with arrow)
3. Click "Install" to add to desktop

## Deployment

### Option 1: Docker (Recommended)

```bash
cd pwa-app
docker-compose up -d
```

Access at:
- Frontend: http://localhost:80
- Backend: http://localhost:8000

### Option 2: Production Servers

**Backend (Railway/Render):**
```bash
# Deploy backend folder to Railway/Render
# Set environment variables in dashboard
# Ensure parent ../src directory is accessible (may need to copy)
```

**Frontend (Vercel/Netlify):**
```bash
cd frontend
npm run build
# Deploy the build/ folder
# Set environment variable: REACT_APP_API_URL=https://your-backend.railway.app
```

## Testing

### Test Backend

```bash
# Check authentication
curl http://localhost:8000/api/auth/status

# Get quote
curl http://localhost:8000/api/quote/SPY

# View API docs
open http://localhost:8000/docs
```

### Test Frontend

```bash
# Run development server
npm start

# Build for production
npm run build

# Test production build
npm install -g serve
serve -s build
```

### Test Push Notifications

1. Open the app in Chrome/Edge (not Safari - limited support)
2. Navigate to Alerts Config (`/alerts`)
3. Click "Enable Notifications" and allow permission
4. Click "Test Notification" to verify

## Troubleshooting

### Backend won't start

- **Error:** `ModuleNotFoundError: No module named 'src'`
  - **Fix:** The backend uses `sys.path.insert` to import from parent `../src`. Ensure the parent directory structure is intact.

- **Error:** `Schwab authentication failed`
  - **Fix:** Copy tokens from parent app: `cp ../tokens/* backend/tokens/` or ensure the volume mount works.

### Frontend won't build

- **Error:** `Cannot find module 'react'`
  - **Fix:** Run `npm install` in the frontend directory.

- **Error:** TypeScript errors
  - **Fix:** These are expected until dependencies are installed. Run `npm install`.

### WebSocket not connecting

- **Error:** `WebSocket connection failed`
  - **Fix:** Ensure backend is running and CORS is configured. Check browser console for errors.

### Push notifications not working

- **Error:** Notifications don't appear
  - **Fix:** 
    1. Check browser permission (Settings → Site Settings → Notifications)
    2. Generate VAPID keys and add to `.env`
    3. Use HTTPS in production (required for service workers)
    4. Test in Chrome/Edge (Safari has limited support)

### Docker issues

- **Error:** `Cannot connect to backend`
  - **Fix:** Check Docker network: `docker-compose logs backend`

- **Error:** Volume mount issues
  - **Fix:** Ensure paths in `docker-compose.yml` are correct for your system.

## Architecture

```
pwa-app/
├── backend/                 # FastAPI server
│   ├── main.py             # API + WebSocket endpoints
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container
├── frontend/               # React PWA
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API + notification services
│   │   ├── App.tsx         # Main app with routing
│   │   └── index.tsx       # Entry point
│   ├── public/
│   │   ├── manifest.json   # PWA manifest
│   │   └── index.html      # HTML template
│   ├── package.json        # npm dependencies
│   ├── nginx.conf          # Nginx config for Docker
│   └── Dockerfile          # Frontend container
├── docker-compose.yml      # Orchestration
├── start.sh                # Automated setup script
└── README.md               # Comprehensive docs

Parent Directory Integration:
├── ../src/api/schwab_client.py    # Reused via sys.path
├── ../src/analysis/big_trades.py  # Reused for detection
├── ../tokens/                      # Shared authentication tokens
```

## Next Steps

1. **Generate VAPID Keys** - Required for push notifications
2. **Configure .env** - Add Schwab API credentials
3. **Test Locally** - Run `./start.sh` and test all features
4. **Deploy Backend** - Use Railway, Render, or Docker
5. **Deploy Frontend** - Use Vercel, Netlify, or Docker
6. **Enable HTTPS** - Required for service workers in production
7. **Test Mobile Install** - Verify PWA install on iOS/Android

## Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **React Docs:** https://react.dev/
- **PWA Guide:** https://web.dev/progressive-web-apps/
- **Plotly React:** https://plotly.com/javascript/react/
- **Web Push:** https://web.dev/push-notifications/
- **Service Workers:** https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API

## Support

For issues or questions:
1. Check the logs: `docker-compose logs` or console output
2. Review API documentation: http://localhost:8000/docs
3. Check browser console for frontend errors
4. Verify environment variables are set correctly

---

**Status:** ✅ Complete - All components implemented and ready for testing!
