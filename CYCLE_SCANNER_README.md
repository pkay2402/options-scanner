# Cycle Indicator Scanner

Automated scanner that detects cycle peaks and bottoms across 150+ stocks using John Ehlers' methodology.

## Features

- **150 Stock Watchlist**: Scans major tech, growth, semiconductors, financials, and ETFs
- **4 Signal Types**:
  - 🔴 **Peak** (SELL): Strong peak detected, consider selling
  - 🟢 **Bottom** (BUY): Strong bottom detected, consider buying
  - 🟠 **Approaching Peak** (PREPARE SELL): Peak coming, tighten stops
  - 💡 **Approaching Bottom** (PREPARE BUY): Bottom coming, prepare entry

- **Smart Filtering**:
  - Peaks: Cycle >+1.5σ, strength >1.0, phase 315-45°, local maximum
  - Bottoms: Cycle <-1.3σ, strength >0.8, phase 120-240°, negative momentum, local minimum
  - Only reports actionable signals

## Deployment

### 1. Run Locally (Test)
```bash
python scripts/cycle_scanner.py
```

Results saved to: `data/cycle_signals.json`

### 2. Deploy to Droplet (Production)

#### Option A: Automated Deployment
```bash
# Update DROPLET_IP in deploy_cycle_scanner.sh
chmod +x scripts/deploy_cycle_scanner.sh
./scripts/deploy_cycle_scanner.sh
```

#### Option B: Manual Deployment

1. **Copy files to droplet:**
```bash
scp scripts/cycle_scanner.py root@your-droplet:/root/options-scanner/scripts/
scp services/cycle-scanner.service root@your-droplet:/etc/systemd/system/
scp services/cycle-scanner.timer root@your-droplet:/etc/systemd/system/
```

2. **SSH to droplet and setup:**
```bash
ssh root@your-droplet

# Install dependencies
pip3 install yfinance pandas numpy

# Create directories
mkdir -p /root/options-scanner/data
mkdir -p /root/options-scanner/logs

# Enable service
systemctl daemon-reload
systemctl enable cycle-scanner.timer
systemctl start cycle-scanner.timer
```

3. **Verify deployment:**
```bash
# Check timer status
systemctl status cycle-scanner.timer

# List all timers
systemctl list-timers

# Run manually to test
systemctl start cycle-scanner.service

# View logs
journalctl -u cycle-scanner.service -f
tail -f /root/options-scanner/logs/cycle_scanner.log
```

## Schedule

Scanner runs **every 2 hours during market hours**:
- 9:30 AM EST (Market open)
- 11:30 AM EST (Mid-morning)
- 1:30 PM EST (After lunch)
- 3:30 PM EST (Before close)

Only runs Monday-Friday (market days).

## Output Format

```json
{
  "peak": [
    {
      "symbol": "NVDA",
      "price": 450.25,
      "cycle_value": 2.1,
      "phase": 355.0,
      "strength": 1.35,
      "action": "SELL",
      "signal_type": "PEAK"
    }
  ],
  "bottom": [...],
  "approaching_peak": [...],
  "approaching_bottom": [...],
  "metadata": {
    "scan_time": "2025-12-23T14:30:00",
    "total_stocks": 150,
    "stocks_scanned": 150
  }
}
```

## Integration with Dashboard

The Cycle Indicator page automatically loads and displays scanner results:

1. **Live Scanner Panel**: Shows latest signals at top of page
2. **Auto-refresh**: Results cache refreshes every 5 minutes
3. **Sortable Tables**: Buy and sell opportunities sorted by signal strength
4. **Click to Analyze**: Click any symbol to view detailed chart

## Monitoring

### Check Scanner Status
```bash
# Is the timer running?
systemctl status cycle-scanner.timer

# When is next run?
systemctl list-timers --all | grep cycle

# View recent runs
journalctl -u cycle-scanner.service --since "1 day ago"
```

### View Logs
```bash
# Real-time logs
journalctl -u cycle-scanner.service -f

# Recent scans
tail -100 /root/options-scanner/logs/cycle_scanner.log

# Errors
tail -50 /root/options-scanner/logs/cycle_scanner.error.log
```

### Manual Trigger
```bash
# Run scanner immediately
systemctl start cycle-scanner.service

# Check results
cat /root/options-scanner/data/cycle_signals.json | jq .
```

## Troubleshooting

### Scanner not running
```bash
# Check timer is enabled
systemctl is-enabled cycle-scanner.timer

# Check for errors
systemctl status cycle-scanner.service

# View full logs
journalctl -u cycle-scanner.service -n 100
```

### No results / Empty signals
- Scanner only reports actionable signals
- During low volatility, few signals are normal
- Check `metadata.stocks_scanned` to verify it ran

### Rate limiting (429 errors)
- yfinance has rate limits
- Scanner includes 1s pause every 10 stocks
- If issues persist, increase delay in script

### Old data showing
- Check `scan_time` in results
- Verify timer is running: `systemctl status cycle-scanner.timer`
- Manually trigger: `systemctl start cycle-scanner.service`

## Performance

- **Scan time**: ~3-5 minutes for 150 stocks
- **Memory**: ~200MB during scan
- **Network**: ~15MB data per scan
- **CPU**: Low (mostly I/O bound)

## Customization

### Change watchlist
Edit `WATCHLIST` array in `cycle_scanner.py`

### Change scan frequency
Edit `cycle-scanner.timer`:
```ini
# Every hour during market hours
OnCalendar=Mon-Fri 09:30,10:30,11:30,12:30,13:30,14:30,15:30

# Every 30 minutes
OnCalendar=Mon-Fri *:00,30
```

Then reload: `systemctl daemon-reload && systemctl restart cycle-scanner.timer`

### Adjust signal thresholds
Edit thresholds in `calculate_cycle_indicator()`:
- Peak threshold: `> 1.5` → change to 1.7 for fewer signals
- Bottom threshold: `< -1.3` → change to -1.5 for stricter signals

## API Access (Future)

Results file is JSON and can be accessed via:
- Direct file read: `data/cycle_signals.json`
- HTTP endpoint (if serving): `/api/cycle-signals`
- Websocket (real-time): Stream on each scan completion

## Contributing

To add new signal types or improve detection:
1. Update `calculate_cycle_indicator()` logic
2. Add new signal category in `scan_watchlist()`
3. Update page display in `pages/6_Cycle_Indicator.py`
4. Test thoroughly before deploying

## Support

- Scanner logs: `logs/cycle_scanner.log`
- System logs: `journalctl -u cycle-scanner.service`
- Service status: `systemctl status cycle-scanner.timer`
- Results: `data/cycle_signals.json`
