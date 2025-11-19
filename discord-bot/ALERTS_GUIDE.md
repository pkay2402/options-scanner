# 🔔 Automated Discord Bot Alerts

This Discord bot provides automated alerts for options trading activity during market hours.

## Features

### 🐋 Whale Flow Alerts
- Scans for whale scores > 300 every 15 minutes
- Tracks top 22 tech stocks (AAPL, MSFT, GOOGL, etc.)
- Prevents duplicate alerts within the same market session
- Shows top 10 results per scan

### 📊 0DTE Levels Updates
- SPY, QQQ, and $SPX real-time levels
- Current price vs Call/Put walls
- Max pain levels
- Call/Put volume ratios
- Updates every 15 minutes

## Setup

### 1. Configure the Bot

The bot needs a Discord channel to send alerts to. You have two options:

**Option A: Use Commands (Recommended)**
```
/alerts_setup    # Run this in the channel where you want alerts
/alerts_start    # Start the automated scanning
```

**Option B: Environment Variable**
Add to your `.env` file:
```env
DISCORD_ALERT_CHANNEL_ID=1234567890123456789
```

To get a channel ID:
1. Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
2. Right-click the channel → Copy ID

### 2. Start Alerts

Once configured, start the alert service:
```
/alerts_start
```

The bot will now:
- Scan every 15 minutes during market hours (9:30 AM - 4:00 PM ET)
- Skip weekends automatically
- Clear duplicate tracking when market closes

## Commands

### Alert Management
- `/alerts_setup` - Configure alerts in current channel (Admin only)
- `/alerts_start` - Start automated scanning (Admin only)
- `/alerts_stop` - Stop automated scanning (Admin only)
- `/alerts_status` - Check current status
- `/alerts_test` - Send a test alert (Admin only)

### Manual Analysis (Existing)
- `/whalescan [min_score]` - Manual whale flow scan
- `/dte [symbol]` - 0DTE analysis for a symbol
- `/gammamap [symbol]` - Gamma exposure heatmap
- `/emacloud [symbol]` - EMA cloud analysis

## Alert Examples

### Whale Flow Alert
```
🐋 Whale Flow Alert
Found 3 new whale flows (Score > 300)

NVDA $850.00 CALL
Score: 1,245
Vol: 15,000 | OI: 3,200
Distance: +2.3% | IV: 45.2%

AAPL $175.00 PUT
Score: 892
Vol: 8,500 | OI: 2,100
Distance: -1.1% | IV: 32.8%
```

### 0DTE Levels Update
```
📊 0DTE Levels Update
Current price vs Call/Put Walls

SPY
Current: $585.25 🟡 Between Walls
Call Wall: $586.00 (125,000 vol)
Put Wall: $584.00 (98,500 vol)
Max Pain: $585.00
Call/Put Vol: 245,000 / 198,000

QQQ
Current: $512.80 🟢 Above Call Wall
Call Wall: $512.00 (85,000 vol)
Put Wall: $510.00 (67,000 vol)
Max Pain: $511.50
Call/Put Vol: 180,000 / 142,000
```

## Configuration

### Adjust Thresholds

Edit `discord-bot/bot/services/alert_service.py`:

```python
# In AutomatedAlertService.__init__():
self.whale_score_threshold = 300  # Change whale score threshold
self.scan_interval_minutes = 15   # Change scan frequency
```

### Market Hours

The bot uses Eastern Time for market hours (9:30 AM - 4:00 PM ET). To adjust:

```python
# In AutomatedAlertService.__init__():
self.market_open = time(9, 30)   # 9:30 AM ET
self.market_close = time(16, 0)  # 4:00 PM ET
```

## Troubleshooting

### Alerts not sending?

1. Check status: `/alerts_status`
2. Verify market hours (no alerts on weekends or after 4 PM ET)
3. Check bot permissions in the alert channel
4. Review logs: `discord-bot/bot.log`

### Bot not finding whale flows?

- Whale score threshold is 300 (lower than manual `/whalescan`)
- Only NEW flows are alerted (duplicates filtered)
- Cache clears at market close

### 0DTE data missing?

- Ensure Schwab API credentials are valid
- Check that symbols have 0DTE options available
- SPY/QQQ have daily expiries, SPX may vary

## Performance

- Each scan takes ~5-10 seconds for 22 stocks
- Minimal API rate limit impact (one scan per 15 min)
- Memory efficient (duplicate tracking cleared daily)

## Logs

View bot activity:
```bash
tail -f discord-bot/bot.log
```

Look for:
- `"Running scheduled scans..."` - Scan started
- `"Sent whale flow alert: X flows"` - Alerts sent
- `"No new whale flows detected"` - No new activity

## Security

- Alert commands require Admin permissions
- Channel configuration persists across bot restarts
- No data is stored permanently (session-based cache only)
