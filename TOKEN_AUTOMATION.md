# Automated Token Refresh Guide

## The Problem
Schwab API tokens expire every 7 days, requiring a tedious manual process:
1. Delete old token
2. Run auth script
3. Generate Streamlit secrets
4. Copy-paste to Streamlit Cloud
5. Update droplet
6. Restart services
7. Restart Discord bot

## The Solution
**One script does it all!**

## Quick Start

### Run Token Refresh
```bash
./scripts/refresh_tokens_everywhere.sh
```

This script will:
1. ✅ Backup your old token
2. ✅ Delete old token
3. ✅ Run authentication (browser opens)
4. ✅ Copy new token to droplet
5. ✅ Restart all services on droplet
6. ✅ Verify services are running
7. ✅ Generate Streamlit secrets (displays on screen)
8. ⏳ Wait for you to paste secrets to Streamlit Cloud

**Only manual step:** Copy-paste the Streamlit secrets shown at the end into your Streamlit Cloud settings.

### Check Token Status
```bash
./scripts/check_token_expiry.sh
```

Shows:
- 🟢 Token OK - X days remaining
- 🟡 WARNING! Token expires in X days (< 2 days)
- 🔴 EXPIRED! Token expired X days ago

### Set Up Reminders (Optional)

Add to your crontab to get daily reminders:
```bash
crontab -e
```

Add this line:
```bash
# Check Schwab token expiry daily at 9 AM
0 9 * * * cd /Users/piyushkhaitan/schwab/options && ./scripts/check_token_expiry.sh
```

## What Gets Updated

### ✅ Automatically Updated:
- Local `schwab_client.json`
- Droplet `/root/options-scanner/schwab_client.json`
- Discord bot (restarted)
- API server (restarted)
- Scanner services (use token automatically)

### ⏳ Manual Step Required:
- Streamlit Cloud secrets (script shows what to paste)

## Troubleshooting

### Authentication fails
- Make sure your Schwab API credentials are correct
- Browser should open automatically for OAuth
- Complete the login and authorization

### Services won't start
```bash
ssh root@138.197.210.166 "systemctl status discord-bot api-server"
```

### Token file not found after auth
- Check if auth_setup.py ran successfully
- Look for error messages in terminal output

## Files

- `refresh_tokens_everywhere.sh` - Main automation script
- `check_token_expiry.sh` - Check token status
- `auth_setup.py` - Schwab OAuth authentication
- `generate_streamlit_secrets.py` - Format secrets for Streamlit

## Best Practices

1. **Run check weekly:** `./scripts/check_token_expiry.sh`
2. **Refresh when warned:** When you see 🟡 warning
3. **Keep backup:** Script auto-creates `schwab_client.json.backup`
4. **Test immediately:** After refresh, verify services work

## Timeline

- **Day 0:** Fresh token created
- **Day 5:** 🟡 Warning appears
- **Day 6:** Last safe day to refresh
- **Day 7:** 🔴 Token expires, services stop working

**Pro tip:** Refresh on Day 5 when you see the warning!
