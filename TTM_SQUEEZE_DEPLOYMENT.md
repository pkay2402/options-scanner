# TTM Squeeze Scanner Deployment Guide

## Local Deployment (Already Done ✅)
- Scanner script created: `scripts/ttm_squeeze_scanner.py`
- Database support added: `src/data/market_cache.py`
- API endpoint added: `scripts/api_server.py`
- UI integration complete: `pages/2_Trading_Dashboard.py`
- Service file created: `scripts/ttm-squeeze-scanner.service`

## Droplet Deployment Steps

### 1. Pull Latest Code
```bash
cd /root/options-scanner
git pull origin main
```

### 2. Install Service
```bash
# Copy service file
sudo cp scripts/ttm-squeeze-scanner.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable ttm-squeeze-scanner
sudo systemctl start ttm-squeeze-scanner

# Check status
sudo systemctl status ttm-squeeze-scanner
```

### 3. Verify Service is Running
```bash
# Check logs
tail -f /root/options-scanner/logs/ttm_squeeze_scanner.log

# Check if data is being written
sqlite3 /root/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM ttm_squeeze_scanner;"
```

### 4. Test API Endpoint
```bash
# Test the endpoint
curl http://localhost:8000/api/ttm_squeeze_scanner?filter=all&limit=10

# Restart API server to load new endpoint
sudo systemctl restart api-server
```

## What the Scanner Does

### Detection Logic
- **Bollinger Bands**: 20-period, 2 std dev
- **Keltner Channels**: 20-period, 1.5 ATR multiplier
- **Squeeze ON**: BB inside KC (consolidation)
- **Squeeze OFF**: BB breaks outside KC (breakout)
- **Momentum**: Linear regression normalized by ATR

### Scanner Output
1. **Active Squeeze** (⚡): Consolidating, waiting for breakout
2. **Bullish Fire** (🟢): Squeeze just fired bullish (last 5 days)
3. **Bearish Fire** (🔴): Squeeze just fired bearish (last 5 days)

### Data Tracked
- Current squeeze status (active/fired/none)
- Squeeze duration (days)
- Momentum direction
- Fire date (if recently fired)
- BB and KC levels
- Price action

## Usage in Platform

### Watchlist Icons
- ⚡ = Active squeeze (stock consolidating)
- 🟢 = Bullish squeeze fire (breakout up)
- 🔴 = Bearish squeeze fire (breakdown)

### Advanced Filter
Click the ⚡ icon in "Show only" filters to see:
- All stocks with active squeezes
- Recent squeeze fires (last 5 days)
- Combines with bull/bear filter

### API Filters
- `?filter=all` - All squeeze signals
- `?filter=active` - Only active squeezes
- `?filter=fired` - Recently fired
- `?filter=bullish` - Bullish fires only
- `?filter=bearish` - Bearish fires only

## Monitoring

### Check Scanner Health
```bash
# Service status
sudo systemctl status ttm-squeeze-scanner

# Recent logs
tail -n 50 /root/options-scanner/logs/ttm_squeeze_scanner.log

# Database stats
sqlite3 /root/options-scanner/data/market_cache.db "SELECT signal, COUNT(*) FROM ttm_squeeze_scanner GROUP BY signal;"
```

### Expected Results
- Scanner runs every 1 hour
- Scans 150 stocks
- Typical: 5-15 active squeezes, 2-8 recent fires
- Each scan takes ~2-3 minutes

## Troubleshooting

### Scanner Not Running
```bash
sudo journalctl -u ttm-squeeze-scanner -n 100
sudo systemctl restart ttm-squeeze-scanner
```

### No Data in API
```bash
# Check if table exists
sqlite3 /root/options-scanner/data/market_cache.db ".tables"

# Manual scan test
cd /root/options-scanner
python3 scripts/ttm_squeeze_scanner.py
```

### Service Keeps Restarting
```bash
# Check for errors
tail -f /root/options-scanner/logs/ttm_squeeze_scanner.log

# Check permissions
ls -la /root/options-scanner/data/
```

## Notes
- Scanner uses 60 days of data for accurate squeeze detection
- Rate limited: 0.5s between stocks, 2s every 20 stocks
- Stores results in SQLite for fast API access
- Automatically handles weekends/holidays
- Detects fires up to 5 days old for fresh setups
