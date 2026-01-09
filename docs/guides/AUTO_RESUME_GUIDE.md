# Discord Bot Auto-Resume Feature

All Discord bot scanners now automatically resume after bot restarts! No more manual setup needed.

## How It Works

When the bot restarts, each scanner checks its saved configuration:
- If it was running before restart → **auto-resumes automatically**
- If it was stopped before restart → stays stopped
- Sends a notification to the configured channel when auto-resuming

## Available Scanners

### 1. Z-Score Scanner ✅
**Commands:**
- `/setup_zscore` - Configure channel (first time only)
- `/start_zscore` - Start monitoring
- `/stop_zscore` - Stop monitoring
- `/zscore_status` - Check current status

**Features:**
- Scans watchlist stocks every 15 minutes
- Detects z-score crossings during market hours
- Auto-resumes after bot restart

### 2. TOS Alerts Monitor ✅  
**Commands:**
- `/setup_tos_alerts` - Configure channel (first time only)
- `/start_tos_alerts` - Start monitoring
- `/stop_tos_alerts` - Stop monitoring
- `/tos_alerts_status` - Check current status

**Features:**
- Monitors Gmail for ThinkorSwim scan alerts every 3 minutes
- Sends HG_30mins_L (LONG) and HG_30mins_S (SHORT) signals
- Auto-resumes after bot restart

### 3. Whale Flow Scanner ✅
**Commands:**
- `/setup_whale_scanner [min_score]` - Configure channel (first time only, default min_score=50)
- `/start_whale_scanner` - Start monitoring
- `/stop_whale_scanner` - Stop monitoring
- `/whale_scanner_status` - Check current status

**On-demand commands still available:**
- `/whalescan [min_score]` - Manual scan of all stocks
- `/whalestock <symbol> [min_score]` - Scan specific stock

**Features:**
- Scans 30 top stocks every 30 minutes
- Detects new whale flows during market hours
- Auto-resumes after bot restart

## First-Time Setup

For each scanner you want to use:

1. **Setup channel** (one time):
   ```
   /setup_zscore
   /setup_tos_alerts
   /setup_whale_scanner 50
   ```

2. **Start monitoring**:
   ```
   /start_zscore
   /start_tos_alerts
   /start_whale_scanner
   ```

3. **Done!** Scanners will now:
   - Run during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
   - Auto-resume after bot restarts
   - Stay running until you manually stop them

## Checking Status

Use status commands anytime:
```
/zscore_status
/tos_alerts_status
/whale_scanner_status
```

Shows:
- Running/Stopped status
- Configured channel
- Market hours status
- Scan intervals
- Alert counts

## Stopping Scanners

To stop any scanner:
```
/stop_zscore
/stop_tos_alerts
/stop_whale_scanner
```

Scanner will:
- Stop immediately
- Save "stopped" state
- NOT auto-resume on next restart

## Config Files

Each scanner saves its state in:
- `discord-bot/zscore_config.json`
- `discord-bot/tos_alerts_config.json`
- `discord-bot/whale_score_config.json`

These files persist:
- Channel ID
- Running state (is_running: true/false)
- Scanner settings (intervals, thresholds, etc.)

## Auto-Resume Behavior

**When bot restarts:**

1. Bot loads all config files
2. For each scanner with `is_running: true`:
   - Automatically starts background task
   - Sends "Auto-Resumed" notification to channel
   - Continues monitoring where it left off

**Example notification:**
```
🐋 Whale Scanner Auto-Resumed
Monitoring resumed after bot restart
```

## Benefits

✅ **No manual intervention** - Set it once, forget it  
✅ **Survives restarts** - Token updates, server reboots, etc.  
✅ **Transparent** - Get notified when auto-resume happens  
✅ **Flexible** - Stop/start anytime, state persists  

## Token Updates

When Schwab tokens are refreshed (every 7 days):
1. Tokens are updated on droplet
2. Bot is restarted
3. **All active scanners auto-resume automatically**
4. You'll see auto-resume notifications in Discord

No need to manually restart each scanner! 🎉

## Troubleshooting

**Scanner not auto-resuming?**
- Check status command to see if it was running before restart
- Look for config file in `discord-bot/` directory
- Check bot logs: `journalctl -u discord-bot -n 100`

**Want to reset a scanner?**
- Stop it: `/stop_<scanner>`
- Delete config file on server
- Setup and start fresh

**Multiple restarts causing spam?**
- Each restart sends one notification per active scanner
- Normal behavior, indicates scanners are working properly
