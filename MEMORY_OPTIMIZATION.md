# Memory Optimization Summary

**Date:** December 7, 2025  
**Server:** Learn2Trade (138.197.210.166)  
**RAM:** 961MB (1GB Droplet)

## Problem Identified

Memory usage was at **91%** (885MB/961MB) with only **75MB available**, caused by:
- 2 duplicate market data workers (138MB each)
- VPB Scanner (145MB)
- MACD Scanner (145MB)  
- Discord Bot (141MB)
- API Server (35MB)

**Total Python processes:** ~743MB

## Solution Implemented

### 1. **Systemd Memory Limits**
Added resource constraints to all services:

| Service | Memory Max | Memory High | CPU Quota |
|---------|-----------|-------------|-----------|
| API Server | 80MB | 60MB | Default |
| Market Data Worker | 120MB | 100MB | 80% |
| Discord Bot | 130MB | 110MB | Default |
| VPB Scanner | 120MB | 100MB | 50% |
| MACD Scanner | 120MB | 100MB | 50% |
| TTM Squeeze Scanner | 120MB | 100MB | 50% |

### 2. **Python Optimization**
Applied environment variables to all services:
```bash
PYTHONHASHSEED=0           # Consistent hashing
PYTHONOPTIMIZE=2           # Maximum optimization
PYTHONDONTWRITEBYTECODE=1  # No .pyc files
MALLOC_TRIM_THRESHOLD_=100000   # Aggressive memory return
MALLOC_MMAP_THRESHOLD_=100000   # Memory mapping threshold
```

### 3. **Worker Consolidation**
- Removed duplicate market-data-worker instances (2 → 1)
- Set MAX_WORKERS=1 for single-threaded operation
- Reduced SCANNER_BATCH_SIZE to 5

### 4. **Service Dependencies**
Configured proper startup order:
1. API Server (essential)
2. Market Data Worker (after API)
3. Discord Bot (after API)
4. Scanners (after API, can restart independently)

### 5. **Bug Fixes**
- Added missing `set_metadata()` and `get_metadata()` methods to MarketCache class
- Fixed TTM Squeeze Scanner AttributeError

## Results

### Before Optimization
- Memory Usage: **91%** (885MB/961MB)
- Available: **75MB**
- Python Processes: 6 (including duplicates)
- TTM Scanner: Not working (AttributeError)

### After Optimization
- Memory Usage: **73%** (699MB/961MB)
- Available: **262MB**
- Python Processes: 6 (all functional, no duplicates)
- TTM Scanner: ✅ Working
- Free Memory: **+187MB improvement**

### Current Service Memory
```
api-server:             31MB (limit: 80MB)
market-data-worker:     95MB (limit: 120MB)
discord-bot:           109MB (limit: 130MB)
vpb-scanner:            71MB (limit: 120MB)
macd-scanner:           72MB (limit: 120MB)
ttm-squeeze-scanner:    70MB (limit: 120MB)
──────────────────────────────────────────
TOTAL:                 448MB / 961MB (47%)
```

## Dashboard Integration

### Added Scanner Filters
Enhanced the Trading Dashboard with filters for all three scanners:

**Advanced Filters:**
- ✨ None - Show all stocks
- 🐋 Whale - Stocks with 2+ whale flows (last 6h)
- 📞 Flow - Strong options flow (>$50k net premium)
- 🌅 Premarket - >1% premarket move
- 📰 News - Analyst upgrades/downgrades
- ⚡ **Squeeze - TTM Squeeze (active or fired)**
- 🚀 **VPB - Volume-Price Breakouts**
- 📊 **MACD - MACD Crossovers**

### Added Scanner Results Section
New tabbed section below Advanced Analytics showing:

1. **⚡ TTM Squeeze Tab**
   - Symbol, Signal, Momentum Direction, Duration, Fire Direction, Price
   - Shows active squeezes and recent fires
   - Real-time data from scanner API

2. **🚀 VPB Scanner Tab**
   - Symbol, Signal (Buy/Sell), Price, Change %, Volume Surge %, Pattern
   - Shows volume breakouts and breakdowns
   - Color-coded buy/sell signals

3. **📊 MACD Scanner Tab**
   - Symbol, Signal (Bull/Bear Cross), Price, Change %, MACD, Histogram, Trend
   - Shows bullish and bearish MACD crossovers
   - Technical momentum indicators

## Monitoring

Created `/root/check_memory.sh` on the server to monitor memory usage:

```bash
ssh root@138.197.210.166
/root/check_memory.sh
```

This shows:
- Overall memory status
- Top 5 memory consumers
- Per-service memory usage vs limits

## Key Features Maintained

✅ All functionality preserved:
- Discord bot alerts
- VPB scanner
- MACD scanner  
- TTM Squeeze scanner
- Market data collection
- API server for data access
- Trading Dashboard integration

✅ Automatic restart on failure
✅ Systemd memory enforcement
✅ Proper logging maintained
✅ Real-time data in dashboard

## Memory Limits Explained

**MemoryMax:** Hard limit - process is killed if exceeded  
**MemoryHigh:** Soft limit - triggers throttling before reaching max  

Services stay well below limits with room for spike handling.

## Future Recommendations

### If Memory Issues Return:
1. **Check for leaks:** Look for gradual memory growth
2. **Restart services:** `systemctl restart <service>`
3. **Upgrade droplet:** Consider 2GB ($12/mo) if adding more features

### To Add New Services:
1. Always set MemoryMax and MemoryHigh
2. Test memory usage first: `systemd-cgtop`
3. Ensure total doesn't exceed ~700MB to leave headroom

### To Reduce Memory Further:
1. Run scanners on schedule instead of continuously
2. Use cron jobs for periodic scans
3. Implement result caching to reduce API calls

## Commands Reference

```bash
# Check overall memory
free -h

# Check service memory
systemctl status <service> | grep Memory

# Monitor in real-time
systemd-cgtop

# View service limits
systemctl show <service> | grep Memory

# Restart a service
systemctl restart <service>

# View service logs
journalctl -u <service> -f

# Check scanner data
sqlite3 /root/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM ttm_squeeze_scanner;"
sqlite3 /root/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM vpb_scanner;"
sqlite3 /root/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM macd_scanner;"
```

## Configuration Files Updated

All service files in `/etc/systemd/system/`:
- `api-server.service`
- `market-data-worker.service`
- `discord-bot.service`
- `vpb-scanner.service`
- `macd-scanner.service`
- `ttm-squeeze-scanner.service`

Code files updated:
- `src/data/market_cache.py` - Added set_metadata() and get_metadata() methods
- `pages/2_Trading_Dashboard.py` - Added scanner filters and results section

Environment config: `/root/options-scanner/.env.memory`

## Success Metrics

- ✅ Memory usage reduced from 91% to 73%
- ✅ Available memory increased from 75MB to 262MB
- ✅ All 6 services running and functional
- ✅ Memory limits enforced by systemd
- ✅ Duplicate workers eliminated
- ✅ CPU quotas applied to scanners
- ✅ TTM Scanner bug fixed
- ✅ Dashboard enhanced with all scanner filters
- ✅ Scanner results tables added
- ✅ No functionality compromised

## Emergency Procedures

If memory reaches 90%+:

```bash
# Quick fix - restart heavy services
systemctl restart vpb-scanner macd-scanner ttm-squeeze-scanner

# If still high - restart market data worker
systemctl restart market-data-worker

# Nuclear option - restart all
systemctl restart api-server market-data-worker discord-bot vpb-scanner macd-scanner ttm-squeeze-scanner
```

The memory limits will prevent OOM (Out of Memory) kills by systemd managing resources proactively.
