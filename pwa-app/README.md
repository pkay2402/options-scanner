# Options Flow Pro - PWA

A Progressive Web App for real-time options trading intelligence, featuring:
- 🧱 **Volume Walls**: Identify key support/resistance from option volumes
- 🌊 **Flow Scanner**: Monitor live options flow and big trades
- 🎯 **Option Finder**: Search and analyze options with gamma heatmaps
- 🔔 **Push Notifications**: Real-time alerts for trading opportunities

## Architecture

```
pwa-app/
├── backend/          # FastAPI server (Python)
│   ├── main.py      # REST API + WebSocket endpoints
│   └── requirements.txt
│
└── frontend/         # React PWA (TypeScript)
    ├── public/       # Static assets + PWA manifest
    ├── src/
    │   ├── components/      # UI components
    │   ├── services/        # API & notification services
    │   └── App.tsx         # Main application
    └── package.json
```

## Features

### ✅ Implemented
- **Schwab API Integration**: Real authentication, no mock data
- **REST API**: Quote, options chain, volume walls, price history, big trades
- **WebSocket Streams**: Real-time flow monitoring and alerts
- **PWA Manifest**: Install on iOS/Android via browser
- **Service Worker**: Offline capability and background sync
- **Push Notifications**: Alert system with priority levels
- **Responsive Design**: Mobile-first UI with bottom navigation

### 🚀 Core Modules
1. **Volume Walls**: NetSPY indicator with call/put walls, flip levels, GEX analysis
2. **Flow Scanner**: Live options flow with big trade detection
3. **Option Finder**: Gamma heatmap and options analysis
4. **Dashboard**: Overview with quick stats and recent alerts
5. **Alerts Config**: Customizable notification rules

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Existing Schwab API credentials (from parent directory)

### Backend Setup

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start the API server**:
   ```bash
   python main.py
   ```
   
   Server runs on: `http://localhost:8000`

3. **Test authentication**:
   ```bash
   curl http://localhost:8000/api/auth/status
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Create environment file** (`.env`):
   ```env
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_WS_URL=ws://localhost:8000
   REACT_APP_VAPID_PUBLIC_KEY=your-vapid-key-here
   ```

3. **Start development server**:
   ```bash
   npm start
   ```
   
   App runs on: `http://localhost:3000`

4. **Build for production**:
   ```bash
   npm run build
   ```

## API Endpoints

### REST API
- `GET /api/auth/status` - Check Schwab authentication
- `GET /api/quote/{symbol}` - Get current quote
- `GET /api/options-chain/{symbol}` - Get options chain
- `GET /api/volume-walls/{symbol}` - Calculate volume walls
- `GET /api/price-history/{symbol}` - Get intraday price history
- `GET /api/big-trades/{symbol}` - Scan for big trades

### WebSocket
- `/ws/flow` - Real-time options flow stream
- `/ws/alerts` - Real-time trading alerts

## PWA Installation

### iOS (Safari)
1. Open app in Safari
2. Tap Share button
3. Select "Add to Home Screen"
4. Tap "Add"

### Android (Chrome)
1. Open app in Chrome
2. Tap menu (⋮)
3. Select "Add to Home Screen"
4. Tap "Add"

### Desktop (Chrome/Edge)
1. Click install icon in address bar
2. Click "Install"

## Push Notifications

The app supports push notifications for:
- High-priority alerts (price near walls, big trades)
- Medium-priority alerts (gamma zones, flip level crosses)
- Low-priority alerts (range-bound trading)

**Enable notifications**:
1. Tap "🔔 Enable Alerts" in header
2. Allow notification permissions
3. Configure alert rules in Alerts tab

## Development

### Adding New Features

1. **Backend endpoint**:
   ```python
   # backend/main.py
   @app.get("/api/new-feature/{symbol}")
   async def new_feature(symbol: str):
       client = get_schwab_client()
       # Use existing Schwab client and analysis modules
       return {"data": result}
   ```

2. **Frontend service**:
   ```typescript
   // frontend/src/services/api.ts
   export const api = {
       async getNewFeature(symbol: string) {
           const response = await fetch(`${API_BASE_URL}/api/new-feature/${symbol}`);
           return response.json();
       }
   };
   ```

3. **React component**:
   ```typescript
   // frontend/src/components/NewFeature.tsx
   import { api } from '../services/api';
   
   export default function NewFeature() {
       // Component implementation
   }
   ```

### Debugging

- **Backend logs**: Check terminal running `python main.py`
- **Frontend logs**: Open browser DevTools (F12) → Console
- **Network issues**: Check CORS settings in `backend/main.py`
- **WebSocket**: Monitor in DevTools → Network → WS tab

## Deployment

### Backend (Railway/Render)
```bash
# Push to git repository
git add .
git commit -m "Deploy PWA backend"
git push origin main

# Configure environment variables:
# - SCHWAB_API_KEY
# - SCHWAB_API_SECRET
# - SCHWAB_REDIRECT_URI
```

### Frontend (Vercel/Netlify)
```bash
# Build production bundle
npm run build

# Deploy build/ directory
# Configure environment variables:
# - REACT_APP_API_URL=https://your-api.com
# - REACT_APP_WS_URL=wss://your-api.com
```

## Performance Optimization

- **Code splitting**: React.lazy() for route-based splitting
- **Caching**: Service worker caches API responses
- **WebSocket reconnection**: Automatic with exponential backoff
- **Virtual scrolling**: For large lists in Flow Scanner
- **Debouncing**: Search inputs and API calls

## Security

- **CORS**: Configured in backend for frontend domain only
- **HTTPS**: Required for service workers and push notifications
- **API Keys**: Stored server-side, never exposed to client
- **Rate Limiting**: Implement in production for API endpoints

## Browser Support

- **iOS**: Safari 11.3+ (PWA support)
- **Android**: Chrome 67+ (full PWA features)
- **Desktop**: Chrome 67+, Edge 79+, Firefox 65+

## Troubleshooting

**Authentication fails**:
- Check Schwab credentials in parent directory
- Verify token file exists and is valid
- Run `python -c "from src.api.schwab_client import SchwabClient; client = SchwabClient(); print(client.authenticate())"`

**WebSocket won't connect**:
- Check backend is running
- Verify WS_BASE_URL in frontend .env
- Check browser console for connection errors

**Notifications not working**:
- Grant notification permission in browser
- HTTPS required (use ngrok for local testing)
- Check service worker is registered

**iOS install button missing**:
- Must use Safari browser
- HTTPS required
- Clear cache and reload

## Next Steps

1. ✅ **Phase 1**: Backend API + Core Services ← YOU ARE HERE
2. **Phase 2**: Build React components (Dashboard, VolumeWalls, FlowScanner, OptionFinder)
3. **Phase 3**: Implement alert rules engine
4. **Phase 4**: Add charts (Plotly.js integration)
5. **Phase 5**: Production deployment + testing

## License

Proprietary - For personal use only

## Support

For issues or questions:
1. Check existing Streamlit pages for logic reference
2. Review API documentation above
3. Check browser DevTools console for errors
4. Test backend endpoints with curl/Postman first
