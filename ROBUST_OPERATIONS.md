# Robust Operations Guide

## Overview
This system now includes automatic error recovery, health monitoring, and proactive alerting to minimize manual intervention.

## What's Been Improved

### 1. **Automatic Error Recovery**
- Worker automatically recovers from failures with exponential backoff
- Client reconnection on persistent failures
- Maximum 5 consecutive failures before 30-minute cooldown
- Graceful degradation to yfinance when Schwab API unavailable

### 2. **Health Monitoring**
- Automated checks every 15 minutes
- Discord alerts for critical issues
- Monitors:
  - Schwab token expiration (warns 2 days before 7-day expiry)
  - Database update freshness (alerts if stale > 15 min)
  - Memory usage (warns at 95% capacity)
  - System component health

### 3. **Better Error Handling**
- Discord bot validates database schema
- Worker catches and logs all errors
- No more silent failures
- Detailed error context in logs

### 4. **Proactive Alerts**
- Get notified **before** things break
- Token expiration warnings
- Database staleness alerts
- Memory pressure warnings

## Setup

### 1. Deploy Updated Worker
```bash
# Copy updated worker with error recovery
scp scripts/market_data_worker.py root@138.197.210.166:/root/options-scanner/scripts/
ssh root@138.197.210.166 'systemctl restart market-data-worker'
```

### 2. Install Health Monitor
```bash
# Copy health monitor script
scp scripts/health_monitor.py root@138.197.210.166:/root/options-scanner/scripts/

# Copy and run setup script
scp scripts/setup_health_monitor.sh root@138.197.210.166:/root/options-scanner/scripts/
ssh root@138.197.210.166 'chmod +x /root/options-scanner/scripts/setup_health_monitor.sh'
ssh root@138.197.210.166'/root/options-scanner/scripts/setup_health_monitor.sh'
```

### 3. Configure Discord Webhook
```bash
# Edit config file
ssh root@138.197.210.166 'nano /root/options-scanner/config/health_monitor.json'

# Add your Discord webhook URL:
# 1. Open Discord server
# 2. Go to: Server Settings > Integrations > Webhooks
# 3. Create "New Webhook" named "Health Monitor"
# 4. Copy webhook URL
# 5. Paste into health_monitor.json
```

### 4. Test Health Monitor
```bash
# Run manual health check
ssh root@138.197.210.166 'systemctl start health-monitor.service'

# View results
ssh root@138.197.210.166 'tail -20 /root/options-scanner/logs/health_monitor.log'

# Check timer status
ssh root@138.197.210.166 'systemctl status health-monitor.timer'
```

## What You'll Get Notified About

### 🔴 Critical Alerts
- **Token expired**: Access token expired (should auto-refresh, check worker)
- **Refresh token expired**: MANUAL REAUTHORIZATION REQUIRED
- **Database not updating**: Watchlist data stale > 15 minutes
- **Database file missing**: Database file not found
- **Watchlist empty**: No data in watchlist table

### 🟡 Warning Alerts  
- **Token expiring soon**: Refresh token expires in 2 days
- **Whale flows stale**: Last whale scan > 15 minutes ago
- **Memory pressure**: Service using 95%+ of memory limit

### Example Alert
```
🚨 System Health Alert
Found 2 issue(s) requiring attention

🟡 Schwab Refresh Token
Issue: Refresh token expires in 2 days
Action: Prepare to reauthorize by 2025-12-26

🔴 Watchlist Updates
Issue: Last update 18 minutes ago  
Action: Check market-data-worker service
```

## Maintenance Schedule

### Daily (Automated)
- ✅ Token refresh every 30 minutes
- ✅ Market data updates every 5 minutes
- ✅ Health checks every 15 minutes
- ✅ Old data cleanup

### Weekly (Manual - You'll Get Reminded)
- 🔔 Token reauthorization every 7 days
  - You'll receive 2-day warning via Discord
  - Action: Run token refresh script or reauthorize

### As Needed (You'll Get Alerted)
- 🔔 Service restarts (only if health check fails)
- 🔔 Memory increase (only if usage consistently high)

## Monitoring Commands

### Check Worker Status
```bash
# Service status
ssh root@138.197.210.166 'systemctl status market-data-worker'

# Recent logs (actual work, not auth prompts)
ssh root@138.197.210.166 'tail -100 /root/options-scanner/logs/market_data_worker_error.log | tail -30'

# Database freshness
ssh root@138.197.210.166 "sqlite3 /root/options-scanner/data/market_cache.db 'SELECT key, updated_at FROM cache_metadata ORDER BY updated_at DESC LIMIT 5'"
```

### Check Discord Bot Status
```bash
# Service status
ssh root@138.197.210.166 'systemctl status discord-bot'

# Recent logs
ssh root@138.197.210.166 'tail -50 /root/options-scanner/logs/discord_bot.log | grep -E "(whale|alert|flow)"'
```

### Check Health Monitor
```bash
# Timer status (should be active/running)
ssh root@138.197.210.166 'systemctl status health-monitor.timer'

# Last health check results
ssh root@138.197.210.166 'tail -50 /root/options-scanner/logs/health_monitor.log'

# Manually trigger health check
ssh root@138.197.210.166 'systemctl start health-monitor.service'
```

### Check Token Status
```bash
ssh root@138.197.210.166 "python3 -c \"
import json
from datetime import datetime, timedelta

with open('/root/options-scanner/schwab_client.json') as f:
    data = json.load(f)
token = data.get('token', {})

if 'expires_at' in token:
    expires = datetime.fromtimestamp(token['expires_at'])
    print(f'Access token expires: {expires}')
    
if 'refresh_token_created_at' in token:
    created = datetime.fromtimestamp(token['refresh_token_created_at'])
    expires_at = created + timedelta(days=7)
    days_left = (expires_at - datetime.now()).days
    print(f'Refresh token created: {created}')
    print(f'Refresh token expires: {expires_at} ({days_left} days left)')
\""
```

## Troubleshooting

### "Everything is stuck" - Quick Diagnosis
```bash
# 1. Check if services are running
ssh root@138.197.210.166 'systemctl is-active market-data-worker discord-bot'

# 2. Check database updates
ssh root@138.197.210.166 "sqlite3 /root/options-scanner/data/market_cache.db 'SELECT key, updated_at FROM cache_metadata WHERE key IN (\"watchlist\", \"whale_flows\") ORDER BY updated_at DESC'"

# 3. Run health check
ssh root@138.197.210.166 'systemctl start health-monitor.service && sleep 2 && tail -30 /root/options-scanner/logs/health_monitor.log'

# 4. Check Discord for health alerts
# (You should have received notification if something is wrong)
```

### Worker Not Updating
```bash
# Check for errors
ssh root@138.197.210.166 'journalctl -u market-data-worker -n 50 --no-pager | grep -i error'

# Check recent activity
ssh root@138.197.210.166 'tail -50 /root/options-scanner/logs/market_data_worker_error.log'

# Restart worker
ssh root@138.197.210.166 'systemctl restart market-data-worker'

# Health monitor will alert you if it doesn't recover
```

### Discord Bot Not Sending Alerts
```bash
# Check bot status
ssh root@138.197.210.166 'systemctl status discord-bot'

# Check recent whale flow scans
ssh root@138.197.210.166 'tail -50 /root/options-scanner/logs/discord_bot.log | grep whale'

# Check if whale flows exist in database
ssh root@138.197.210.166 "sqlite3 /root/options-scanner/data/market_cache.db 'SELECT COUNT(*) FROM whale_flows WHERE whale_score >= 50'"

# Restart bot
ssh root@138.197.210.166 'systemctl restart discord-bot'
```

### Token Expired
```bash
# You'll receive Discord alert 2 days before expiration
# To reauthorize:
ssh root@138.197.210.166
cd /root/options-scanner
source venv/bin/activate
python -c "from src.api.schwab_client import SchwabClient; SchwabClient(interactive=True).setup()"
# Follow the OAuth flow in your browser
```

## Files Changed

### New Files
- `scripts/health_monitor.py` - Health monitoring script
- `scripts/setup_health_monitor.sh` - Health monitor installer
- `config/health_monitor.json.template` - Config template

### Updated Files
- `scripts/market_data_worker.py` - Added auto-recovery, exponential backoff
- `discord-bot/bot/services/multi_channel_alert_service.py` - Fixed column names, added schema validation

## Success Metrics

Your system is healthy when:
- ✅ Worker updates every 5 minutes (check cache_metadata)
- ✅ Whale flows detected during market hours
- ✅ Discord bot sends alerts for new flows
- ✅ No critical health alerts in Discord
- ✅ Token shows 3-7 days until expiration
- ✅ Memory usage < 95% of limits

## Need Help?

1. **Check Discord health channel** - Automated alerts tell you what's wrong
2. **Run health check** - `systemctl start health-monitor.service`
3. **Check logs** - Worker: `market_data_worker_error.log`, Bot: `discord_bot.log`
4. **Restart services** - `systemctl restart market-data-worker discord-bot`

The system will now alert you **before** things break, not after. 🎯
