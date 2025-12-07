# Volume-Price Break (VPB) Scanner Deployment Guide

## Overview
The VPB Scanner detects bullish breakouts and bearish breakdowns based on volume surge and price action patterns.

### Criteria
**Bullish Breakout (Buy Signal):**
- Current volume > 30-day volume MA
- Current close > highest high of last 7 days (excluding current bar)

**Bearish Breakdown (Sell Signal):**
- Current volume > 30-day volume MA
- Current close < lowest low of last 7 days (excluding current bar)

## Deployment Steps

### 1. SSH into Digital Ocean Droplet
```bash
ssh root@138.197.210.166
```

### 2. Navigate to Project Directory
```bash
cd /root/options-scanner
```

### 3. Pull Latest Code
```bash
git pull origin main
```

### 4. Install Dependencies (if needed)
```bash
source venv/bin/activate
pip install pandas numpy
```

### 5. Copy Service File
```bash
sudo cp scripts/vpb-scanner.service /etc/systemd/system/
```

### 6. Enable and Start Service
```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable vpb-scanner

# Start the service
sudo systemctl start vpb-scanner
```

### 7. Restart API Server (to load new endpoint)
```bash
sudo systemctl restart api-server
```

### 8. Verify Service Status
```bash
# Check service status
sudo systemctl status vpb-scanner

# View live logs
sudo journalctl -u vpb-scanner -f

# Check last 50 lines
sudo journalctl -u vpb-scanner -n 50
```

## Service Management

### Start Service
```bash
sudo systemctl start vpb-scanner
```

### Stop Service
```bash
sudo systemctl stop vpb-scanner
```

### Restart Service
```bash
sudo systemctl restart vpb-scanner
```

### View Service Status
```bash
sudo systemctl status vpb-scanner
```

### View Logs
```bash
# Real-time logs
sudo journalctl -u vpb-scanner -f

# Last 100 lines
sudo journalctl -u vpb-scanner -n 100

# Logs from today
sudo journalctl -u vpb-scanner --since today

# Check log file
tail -f /root/options-scanner/logs/vpb_scanner.log
```

## API Usage

### Endpoint
```
GET http://138.197.210.166:5000/api/vpb_scanner
```

### Query Parameters
- `filter`: Filter type (default: 'all')
  - `all`: All signals
  - `bullish`: Bullish breakouts only
  - `bearish`: Bearish breakdowns only
- `limit`: Number of results (default: 20)

### Example Requests
```bash
# Get all signals
curl "http://138.197.210.166:5000/api/vpb_scanner"

# Get top 10 bullish breakouts
curl "http://138.197.210.166:5000/api/vpb_scanner?filter=bullish&limit=10"

# Get top 10 bearish breakdowns
curl "http://138.197.210.166:5000/api/vpb_scanner?filter=bearish&limit=10"
```

### Response Format
```json
{
  "success": true,
  "filter": "bullish",
  "count": 10,
  "data": [
    {
      "symbol": "NVDA",
      "price": 495.50,
      "price_change": 12.30,
      "price_change_pct": 2.55,
      "buy_signal": 1,
      "sell_signal": 0,
      "volume_surge": 1,
      "current_volume": 45000000,
      "volume_ma30": 35000000,
      "volume_surge_pct": 28.57,
      "highest_high_7": 492.00,
      "lowest_low_7": 475.20,
      "breakout_distance_pct": 0.71,
      "breakdown_distance_pct": 4.27,
      "pattern": "bullish_breakout",
      "scanned_at": "2025-12-07T15:30:00"
    }
  ]
}
```

## Database Verification

### Check Scanner Results
```bash
sqlite3 /root/options-scanner/data/market_cache.db

# Count total signals
SELECT COUNT(*) FROM vpb_scanner;

# View bullish breakouts
SELECT symbol, price, volume_surge_pct, pattern 
FROM vpb_scanner 
WHERE buy_signal = 1 
ORDER BY volume_surge_pct DESC 
LIMIT 10;

# View bearish breakdowns
SELECT symbol, price, volume_surge_pct, pattern 
FROM vpb_scanner 
WHERE sell_signal = 1 
ORDER BY volume_surge_pct DESC 
LIMIT 10;

# Exit sqlite
.quit
```

## Scanner Details

### Watchlist
Scans 150 stocks including:
- Mega cap tech (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
- Large cap growth (CRM, NFLX, AMD, QCOM)
- AI & Cloud (PLTR, AI, SMCI, ARM)
- Semiconductors (TSM, ASML, MU, LRCX)
- EVs (RIVN, LCID, NIO)
- Fintech (V, MA, HOOD, SOFI)
- Index ETFs (SPY, QQQ, IWM, DIA)

### Scan Frequency
- **Market Hours (9 AM - 4 PM):** Every 30 minutes
- **After Hours:** Every 1 hour

### Rate Limiting
- 0.5 seconds between stocks
- 2 second pause every 20 stocks
- Complies with Schwab API limits (120 calls/minute)

### Data Requirements
- Fetches 2 months of daily data
- Needs 30+ days for volume MA calculation
- Uses last 7 days for price breakout/breakdown detection

## Troubleshooting

### Service Won't Start
```bash
# Check service logs
sudo journalctl -u vpb-scanner -n 50

# Check if port conflicts exist
sudo netstat -tulpn | grep python

# Verify Python environment
/root/options-scanner/venv/bin/python --version

# Test script manually
cd /root/options-scanner
source venv/bin/activate
python scripts/vpb_scanner.py
```

### No Data in Database
```bash
# Check if scanner is running
sudo systemctl status vpb-scanner

# Check logs for errors
tail -f /root/options-scanner/logs/vpb_scanner.log

# Verify Schwab authentication
ls -la /root/options-scanner/schwab_client.json

# Test database connection
sqlite3 /root/options-scanner/data/market_cache.db "SELECT * FROM vpb_scanner LIMIT 5;"
```

### API Endpoint Not Working
```bash
# Check API server status
sudo systemctl status api-server

# Restart API server
sudo systemctl restart api-server

# Test endpoint
curl "http://localhost:5000/api/vpb_scanner?filter=all&limit=5"
```

### High Memory/CPU Usage
```bash
# Check resource usage
top -p $(pgrep -f vpb_scanner.py)

# Check scanner timing
grep "Scan complete" /root/options-scanner/logs/vpb_scanner.log

# Adjust scan frequency in vpb_scanner.py if needed
```

## Dashboard Integration

The VPB Scanner is integrated into the Trading Dashboard at:
- **Page:** Trading Hub (2_Trading_Dashboard.py)
- **Section:** Market News & Alerts expander
- **Location:** Below MACD Scanner

### Display Features
- Filter toggle: Bullish Breakouts / Bearish Breakdowns
- Compact table with: Symbol, Price, Change %, Volume Surge %, Signal
- Auto-refresh every 3 minutes
- Shows last scan timestamp

## Next Steps

After deploying the VPB Scanner:
1. Monitor logs for the first few scan cycles
2. Verify data is being stored in database
3. Check dashboard displays results correctly
4. Use scanner to identify high-volume breakout/breakdown opportunities
5. Consider adding alerts for specific volume surge thresholds

## Scanner Architecture

The VPB Scanner serves as a template for future scanners. To add more scanners:
1. Create new scanner script (e.g., `scripts/rsi_scanner.py`)
2. Add database table in `src/data/market_cache.py`
3. Add API endpoint in `scripts/api_server.py`
4. Create systemd service file
5. Update dashboard to display results

This modular approach allows easy addition of RSI, EMA Cloud, Bollinger Bands, and other scanners.
