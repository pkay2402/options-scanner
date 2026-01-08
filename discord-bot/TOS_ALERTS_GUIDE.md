# TOS Alerts Discord Bot Command

Automatically monitors ThinkorSwim email alerts and sends them to Discord during market hours.

## Features

- ✅ Monitors Gmail inbox for TOS scan alerts
- ✅ Checks every 3 minutes during market hours (9:30 AM - 4:00 PM ET)
- ✅ Automatic deduplication (won't send same ticker+signal twice in a day)
- ✅ Supports HG_30mins_L (LONG) and HG_30mins_S (SHORT) scans
- ✅ Beautiful Discord embeds with color coding
- ✅ Runs as background task on droplet

## Setup

### 1. Configure Email Credentials on Droplet

SSH into your droplet and edit the environment file:

```bash
ssh root@138.197.210.166
cd /root/discord-bot
nano .env
```

Add these lines:

```bash
# TOS Email Configuration
TOS_EMAIL_ADDRESS="your.email@gmail.com"
TOS_EMAIL_PASSWORD="your_app_password"
```

**Note**: Use a Gmail App Password, not your regular password:
1. Go to Google Account settings
2. Security → 2-Step Verification → App passwords
3. Generate new app password for "Mail"
4. Use that 16-character password

### 2. Restart Discord Bot

```bash
systemctl restart discord-bot
```

### 3. Discord Commands

In your Discord channel, run:

```
/setup_tos_alerts        # Configure this channel for alerts
/start_tos_alerts        # Start monitoring (runs during market hours)
/tos_alerts_status       # Check status and stats
/stop_tos_alerts         # Stop monitoring
```

## How It Works

1. **Every 3 minutes** during market hours, checks Gmail for new TOS alerts
2. **Parses email body** to extract tickers and signal types
3. **Checks deduplication** - skips if same ticker+signal already sent today
4. **Sends Discord embed** with:
   - 🟢 Green for LONG signals
   - 🔴 Red for SHORT signals
   - Ticker, signal type, scan name
   - Timestamp

5. **Auto-resets** at midnight ET for next trading day

## Alert Format

```
🟢 TOS LONG Alert: AAPL
ThinkorSwim High Grade 30-Min Signal
HG_30mins_L

Ticker: AAPL
Signal Type: LONG
Scan: HG_30mins_L
```

## Monitored Scans

- `HG_30mins_L` - High Grade 30-minute LONG signals
- `HG_30mins_S` - High Grade 30-minute SHORT signals

## Technical Details

- **Check Interval**: 3 minutes
- **Market Hours**: 9:30 AM - 4:00 PM ET (Mon-Fri)
- **Email Source**: alerts@thinkorswim.com
- **Deduplication**: By ticker + signal + date
- **Rate Limiting**: 0.5s delay between Discord messages

## Troubleshooting

### "Email credentials not configured"
- Check `.env` file has TOS_EMAIL_ADDRESS and TOS_EMAIL_PASSWORD
- Restart bot: `systemctl restart discord-bot`

### "No alerts received"
- Verify TOS alerts are being sent to your Gmail
- Check spam folder
- Verify email subject contains "HG_30mins_L" or "HG_30mins_S"
- Check bot logs: `journalctl -u discord-bot -f`

### "Market is CLOSED" status
- Bot only runs during market hours (9:30 AM - 4:00 PM ET)
- Outside hours, it waits until market opens
- Automatically clears daily cache at midnight

## Logs

```bash
# View bot logs
journalctl -u discord-bot -f

# Filter for TOS alerts only
journalctl -u discord-bot | grep "TOS"
```

## Files

- `/root/discord-bot/bot/commands/tos_alerts.py` - Main command file
- `/root/discord-bot/.env` - Email credentials
- Bot automatically loads this command on startup
