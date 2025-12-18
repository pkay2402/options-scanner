# Schwab API Token Management

## Problem Overview

Schwab's OAuth2 authentication uses two types of tokens:

- **Access Token**: Expires in 30 minutes, automatically refreshed
- **Refresh Token**: Expires in 7 days (604,800 seconds), requires manual re-authentication

The background worker on the droplet cannot automatically handle refresh token expiration because it requires browser-based OAuth flow with manual URL pasting.

## Symptoms of Token Expiration

When refresh tokens expire, the worker gets stuck in an authentication loop:
- Logs fill with "Paste URL:" prompts
- No data updates occur
- Service appears "running" but isn't fetching data
- Error log grows rapidly (can reach 100+ MB)

## Solution: Three-Layer Approach

### 1. Automated Monitoring (Preventive)

**Token Monitor Script**: `scripts/token_monitor.py`
- Checks token age twice daily
- Alerts when < 2 days remaining (WARNING)
- Alerts when < 1 day remaining (CRITICAL)
- Logs to systemd journal

**Setup**:
```bash
./scripts/setup_token_monitor.sh
```

This creates a systemd timer that runs at 9 AM and 9 PM UTC.

**Manual Check**:
```bash
ssh root@138.197.210.166 'cd /root/options-scanner && venv/bin/python scripts/token_monitor.py'
```

### 2. Quick Manual Refresh (Reactive)

**Refresh Script**: `scripts/refresh_worker_auth.sh`

When you receive a token expiration warning:
```bash
./scripts/refresh_worker_auth.sh
```

This script:
1. Stops the worker service
2. Runs local authentication (you paste the callback URL)
3. Copies fresh tokens to droplet
4. Restarts the service
5. Verifies it's working

**Time Required**: ~2 minutes

### 3. Token Creation Timestamp Tracking

The `schwab_client.py` now stores `refresh_token_created_at` timestamp when tokens are created. This enables:
- Accurate calculation of token age
- Precise expiration date prediction
- Automated monitoring without guessing

## Maintenance Schedule

### Every 6 Days (Proactive)
Run the refresh script before tokens expire:
```bash
./scripts/refresh_worker_auth.sh
```

**Calendar reminder**: Set a recurring reminder for every Monday morning

### Daily (Automated)
The monitoring timer checks token status automatically:
- 9:00 AM UTC
- 9:00 PM UTC

### When Alerts Fire

**WARNING** (< 2 days remaining):
- Schedule time to run refresh script within 24 hours
- Non-urgent but don't ignore

**CRITICAL** (< 1 day remaining):
- Run refresh script immediately
- Service will fail within hours

## Viewing Logs

### Worker Status
```bash
ssh root@138.197.210.166 'systemctl status market-data-worker.service'
```

### Recent Worker Activity
```bash
ssh root@138.197.210.166 'tail -n 50 /root/options-scanner/logs/market_data_worker.log'
```

### Token Monitor Logs
```bash
ssh root@138.197.210.166 'journalctl -u schwab-token-monitor.service -n 20'
```

### Token Monitor Timer Status
```bash
ssh root@138.197.210.166 'systemctl list-timers schwab-token-monitor.timer'
```

## Token Information

### Check Token Details Locally
```bash
cat schwab_client.json | jq '{
  refresh_token_created_at: .token.refresh_token_created_at,
  created_date: (.token.refresh_token_created_at | strftime("%Y-%m-%d %H:%M:%S")),
  expires_at: ((.token.refresh_token_created_at + 604800) | strftime("%Y-%m-%d %H:%M:%S")),
  days_remaining: (((.token.refresh_token_created_at + 604800) - now) / 86400 | floor)
}'
```

### Check Token Details on Droplet
```bash
ssh root@138.197.210.166 'cat /root/options-scanner/schwab_client.json | python3 -c "
import json, sys, time
data = json.load(sys.stdin)
created = data[\"token\"][\"refresh_token_created_at\"]
expires = created + 604800
remaining = expires - time.time()
print(f\"Created: {time.strftime(\"%Y-%m-%d %H:%M:%S\", time.localtime(created))}\")
print(f\"Expires: {time.strftime(\"%Y-%m-%d %H:%M:%S\", time.localtime(expires))}\")
print(f\"Days remaining: {remaining / 86400:.1f}\")
"'
```

## Troubleshooting

### Worker Stuck in Auth Loop

**Symptoms**:
- Logs show repeated "Paste URL:" messages
- No data updates
- Error log growing rapidly

**Fix**:
```bash
./scripts/refresh_worker_auth.sh
```

### Token Monitor Not Running

**Check timer status**:
```bash
ssh root@138.197.210.166 'systemctl status schwab-token-monitor.timer'
```

**Restart timer**:
```bash
ssh root@138.197.210.166 'systemctl restart schwab-token-monitor.timer'
```

**Test monitor manually**:
```bash
ssh root@138.197.210.166 'cd /root/options-scanner && venv/bin/python scripts/token_monitor.py'
```

### Authentication Failed During Refresh

**Common causes**:
- Incorrect callback URL pasted
- OAuth redirect not properly captured
- Schwab API credentials changed

**Steps**:
1. Ensure you copy the ENTIRE callback URL from browser
2. URL should start with `https://127.0.0.1/`
3. Paste immediately after authorization (within 60 seconds)
4. If it fails, try again from the start

## Future Improvements

### Option 1: Automated Browser-Based Re-auth
Use Selenium or Playwright to automate the OAuth flow entirely:
- Runs on schedule (every 6 days)
- No manual intervention needed
- Requires secure credential storage

### Option 2: Alternative Data Sources
Implement fallback to yfinance when Schwab tokens expire:
- Worker continues operating
- Uses free data sources temporarily
- Alerts you to refresh tokens at convenience

### Option 3: Token Sharing from Main App
Sync tokens from the main Streamlit app (which you interact with):
- Main app refreshes tokens during normal use
- Automatically syncs to worker
- Reduces manual refresh frequency

## Best Practices

1. **Set Calendar Reminder**: Monday mornings to refresh tokens
2. **Monitor Alerts**: Check monitoring logs weekly
3. **Keep Error Log Small**: The error log can grow to 100+ MB
   ```bash
   ssh root@138.197.210.166 'echo "" > /root/options-scanner/logs/market_data_worker_error.log'
   ```
4. **Test Refresh Script**: Run it once now to ensure it works
5. **Document Changes**: Update this guide if you modify the auth flow

## Quick Reference Commands

```bash
# Check token age
./scripts/token_monitor.py

# Refresh tokens
./scripts/refresh_worker_auth.sh

# Check worker status
ssh root@138.197.210.166 'systemctl status market-data-worker.service'

# View recent worker logs
ssh root@138.197.210.166 'tail -n 100 /root/options-scanner/logs/market_data_worker.log'

# Clear error log
ssh root@138.197.210.166 'echo "" > /root/options-scanner/logs/market_data_worker_error.log'

# Manual token check
ssh root@138.197.210.166 'cd /root/options-scanner && venv/bin/python scripts/token_monitor.py'

# View monitor schedule
ssh root@138.197.210.166 'systemctl list-timers schwab-token-monitor.timer'
```

## Support

If you continue experiencing token issues:
1. Check this document first
2. Review monitor logs for patterns
3. Verify token file permissions on droplet
4. Consider implementing automated re-auth (Option 1 above)
