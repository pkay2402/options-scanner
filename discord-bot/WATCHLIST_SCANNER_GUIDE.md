# Watchlist Flip Level Scanner - Discord Bot Guide

## Overview
The Watchlist Scanner monitors your watchlist stocks for flip level crossings every 30 minutes during market hours. When a stock crosses above or below its flip level, you'll receive an instant Discord alert with comprehensive options data.

## Features
- ✅ Monitors all stocks in `user_preferences.json` watchlist
- ✅ Scans every 30 minutes during market hours (9:30 AM - 4:00 PM ET)
- ✅ Detects bullish crossings (price crosses ABOVE flip level)
- ✅ Detects bearish crossings (price crosses BELOW flip level)
- ✅ Provides comprehensive options data per alert
- ✅ Uses daily timeframe for flip level calculations

## Setup Instructions

### 1. Configure Discord Channel
In your Discord server, run:
```
/setup_watchlist_scanner
```

This designates the current channel for watchlist crossing alerts.

### 2. Start the Scanner
To begin monitoring, run:
```
/start_watchlist_scanner
```

The bot will:
- Load your watchlist from `user_preferences.json`
- Begin scanning every 30 minutes
- Only send alerts during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)

### 3. Monitor Status
Check scanner status at any time:
```
/watchlist_status
```

Shows:
- Current running status (🟢 Running / 🔴 Stopped)
- Configured channel
- Scan interval (30 minutes)
- Current watchlist stocks
- Current positions relative to flip levels

### 4. Stop the Scanner
To stop monitoring:
```
/stop_watchlist_scanner
```

## Alert Information

Each crossing alert includes:

### Price Data
- **Current Price**: Real-time stock price
- **Flip Level**: The calculated flip level (gamma neutral point)
- **Distance**: Absolute and percentage distance from flip level

### Key Levels
- **Call Wall**: Highest call gamma strike (resistance)
- **Put Wall**: Highest put gamma strike (support)
- **Max GEX**: Strike with maximum gamma exposure

### Crossing Direction
- 🚀 **Bullish Crossing**: Price moved from below to above flip level
- 📉 **Bearish Crossing**: Price moved from below to above flip level

## Example Alert

```
🚀 AAPL Crossed ABOVE Flip Level
Price moved from below to above flip level

💰 Current Price: $277.10
🎯 Flip Level: $275.00 (+2.10)
📊 Distance: +0.76%

📞 Call Wall: $280.00 (+2.90)
📱 Put Wall: $275.00 (-2.10)
⚡ Max GEX: $275.00 (45.2M)

Watchlist Scanner • Daily Timeframe
```

## Configuration

### Modifying Watchlist
Edit `user_preferences.json` in the project root:

```json
{
  "watchlist": [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA"
  ]
}
```

After changing the watchlist:
1. Stop the scanner: `/stop_watchlist_scanner`
2. Restart the bot or reload the command
3. Start the scanner: `/start_watchlist_scanner`

### Scan Interval
Default: 30 minutes (hardcoded)

To change, edit `discord-bot/bot/commands/watchlist_scanner.py`:
```python
self.scan_interval_minutes = 30  # Change to desired interval
```

### Market Hours
Default: 9:30 AM - 4:00 PM ET, Monday-Friday

Scanner automatically skips weekends and after-hours.

## Commands Summary

| Command | Description |
|---------|-------------|
| `/setup_watchlist_scanner` | Configure the alert channel |
| `/start_watchlist_scanner` | Start automated scanning |
| `/stop_watchlist_scanner` | Stop automated scanning |
| `/watchlist_status` | Check scanner status and positions |

## Technical Details

### Flip Level Detection
- Uses daily timeframe options data
- Calculates comprehensive GEX analysis per stock
- Tracks previous position (above/below flip)
- Detects crossings by comparing current vs previous position

### Rate Limiting
- 0.5 second delay between stock checks
- 1 second delay between alert messages
- Prevents API throttling and Discord rate limits

### Error Handling
- Individual stock errors don't stop the scan
- Scanner continues on network failures (retries after 1 minute)
- Logs all errors for debugging

## Troubleshooting

### No Alerts Received
1. Check scanner is running: `/watchlist_status`
2. Verify market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
3. Confirm watchlist has stocks in `user_preferences.json`
4. Check bot logs: `logs/discord_bot.log`

### "No stocks in watchlist" Error
1. Verify `user_preferences.json` exists in project root
2. Check JSON format is valid
3. Ensure `watchlist` array contains stock symbols

### Scanner Stops Unexpectedly
1. Check bot logs for errors
2. Verify Schwab API credentials are valid
3. Restart scanner: `/start_watchlist_scanner`

## Deployment

### On DigitalOcean Droplet
The scanner runs as part of the Discord bot service.

1. Copy updated files to server:
```bash
cd discord-bot
./deploy_to_droplet.sh
```

2. Restart the bot service:
```bash
ssh root@138.197.210.166
systemctl restart discord-bot
systemctl status discord-bot
```

3. Check logs:
```bash
tail -f /root/options-scanner/logs/discord_bot.log
```

### Memory Considerations
- Scanner adds minimal overhead (~5-10MB)
- Caches previous positions in memory
- Watchlist of 10 stocks uses ~0.5MB additional memory

## Integration with Existing Alerts

The Watchlist Scanner works alongside existing alert channels:
- **Whale Flow Alerts**: Individual stock whale flows (score > threshold)
- **0DTE Alerts**: SPY/QQQ/SPX wall levels and positioning
- **Market Intelligence**: Aggregate SPY/QQQ market signals
- **Watchlist Scanner**: Flip level crossings on your watchlist

All systems run independently and can be configured separately.

## Future Enhancements

Potential improvements:
- [ ] Configurable scan intervals per channel
- [ ] Custom flip level thresholds
- [ ] Volume and OI confirmation filters
- [ ] Multi-timeframe analysis (daily, weekly)
- [ ] Historical crossing tracking and statistics
- [ ] Backtesting flip level crossing strategies
