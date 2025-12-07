# MACD Scanner Service Deployment

## Overview
MACD scanner service that scans 150 stocks for bullish and bearish MACD crossovers on daily charts.

## Files Created
- `scripts/macd_scanner.py` - Main scanner service
- `scripts/macd-scanner.service` - Systemd service file
- Updated `src/data/market_cache.py` - Added MACD scanner table and methods
- Updated `scripts/api_server.py` - Added `/api/macd_scanner` endpoint
- Updated Trading Dashboard - Added MACD scanner display

## Deployment Steps

### 1. SSH into Droplet
```bash
ssh root@138.197.210.166
```

### 2. Pull Latest Code
```bash
cd /root/options-scanner
git pull
```

### 3. Install pandas (if not already installed)
```bash
source venv/bin/activate
pip install pandas numpy
```

### 4. Copy Service File
```bash
sudo cp scripts/macd-scanner.service /etc/systemd/system/
```

### 5. Reload Systemd and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable macd-scanner
sudo systemctl start macd-scanner
```

### 6. Verify Service is Running
```bash
sudo systemctl status macd-scanner
```

### 7. Check Logs
```bash
# Real-time logs
sudo journalctl -u macd-scanner -f

# Recent logs
sudo journalctl -u macd-scanner -n 100 --no-pager
```

### 8. Restart API Server
The API server needs to restart to pick up the new endpoint:
```bash
sudo systemctl restart api-server
```

## Service Management

### Check Status
```bash
sudo systemctl status macd-scanner
```

### Stop Service
```bash
sudo systemctl stop macd-scanner
```

### Start Service
```bash
sudo systemctl start macd-scanner
```

### Restart Service
```bash
sudo systemctl restart macd-scanner
```

### View Logs
```bash
# Application logs
tail -f /root/options-scanner/logs/macd_scanner.log

# Error logs
tail -f /root/options-scanner/logs/macd_scanner_error.log

# Systemd logs
sudo journalctl -u macd-scanner -f
```

## How It Works

1. **Scanner Service** (`macd_scanner.py`):
   - Scans 150 stocks from watchlist
   - Calculates MACD indicators on daily timeframe
   - Detects bullish/bearish crossovers
   - Stores results in SQLite database
   - Runs every 30 minutes during market hours, hourly after hours

2. **Database** (`market_cache.db`):
   - Table: `macd_scanner`
   - Stores: symbol, price, MACD values, crossover flags, trend
   - Indexed for fast queries

3. **API Endpoint** (`/api/macd_scanner`):
   - Filter by: all, bullish, bearish
   - Returns top results with MACD data

4. **Dashboard Display**:
   - Shows in Market News & Alerts expander
   - Toggle between bullish/bearish crosses
   - Live table with symbol, price, change %, MACD, trend

## API Usage

### Get All MACD Signals
```bash
curl http://138.197.210.166:5000/api/macd_scanner
```

### Get Bullish Crosses Only
```bash
curl http://138.197.210.166:5000/api/macd_scanner?filter=bullish&limit=10
```

### Get Bearish Crosses Only
```bash
curl http://138.197.210.166:5000/api/macd_scanner?filter=bearish&limit=10
```

## Verification

After deployment, verify:

1. Service is running:
   ```bash
   sudo systemctl status macd-scanner
   ```

2. Scanner is processing stocks:
   ```bash
   tail -f /root/options-scanner/logs/macd_scanner.log
   ```

3. API endpoint works:
   ```bash
   curl http://138.197.210.166:5000/api/macd_scanner?filter=bullish
   ```

4. Dashboard displays data:
   - Open Trading Hub page
   - Expand "Market News & Alerts"
   - See MACD Scanner on right side

## Troubleshooting

### Service Won't Start
```bash
# Check for errors
sudo journalctl -u macd-scanner -n 50 --no-pager

# Check Python path
which python3

# Test manually
cd /root/options-scanner
source venv/bin/activate
python3 scripts/macd_scanner.py
```

### No Data in Dashboard
```bash
# Check if scanner ran
tail -n 100 /root/options-scanner/logs/macd_scanner.log

# Test API directly
curl http://138.197.210.166:5000/api/macd_scanner

# Check database
sqlite3 /root/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM macd_scanner;"
```

### Rate Limit Issues
The scanner includes rate limiting (0.5s between stocks, 2s every 20 stocks) to stay within Schwab's 120 calls/min limit.

## Future Scanners

This architecture supports adding more scanners:
1. Create new scanner script (e.g., `rsi_scanner.py`)
2. Add table to `market_cache.py`
3. Add API endpoint
4. Create systemd service
5. Display in dashboard

The MACD scanner is the template for all future scanners!
