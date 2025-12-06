# Discord Bot Deployment Guide - Digital Ocean

## Quick Deploy

1. **SSH to your droplet:**
```bash
ssh root@138.197.210.166
```

2. **Update code:**
```bash
cd /root/options-scanner
git pull
```

3. **Copy your .env file from PythonAnywhere or create new:**
```bash
cd /root/options-scanner/discord-bot
nano .env
```

Add your Discord credentials:
```env
DISCORD_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=your_server_id_here
```

4. **Run deployment script:**
```bash
cd /root/options-scanner/discord-bot
sudo bash deploy_to_droplet.sh
```

## Service Management

**Check status:**
```bash
sudo systemctl status discord-bot
```

**View logs:**
```bash
tail -f /root/options-scanner/logs/discord_bot.log
# or
sudo journalctl -u discord-bot -f
```

**Restart bot:**
```bash
sudo systemctl restart discord-bot
```

**Stop bot:**
```bash
sudo systemctl stop discord-bot
```

**Start bot:**
```bash
sudo systemctl start discord-bot
```

## Benefits of Digital Ocean vs PythonAnywhere

✅ **Same infrastructure** - Runs on same droplet as API server and market worker
✅ **No extra cost** - Already paying for the droplet
✅ **Better performance** - Direct access to cached data
✅ **Unified monitoring** - All services in one place
✅ **Shared authentication** - Uses same schwab_client.json token
✅ **Lower latency** - Bot can access local cache directly

## Architecture

```
Digital Ocean Droplet
├── API Server (port 5000)
│   └── Serves cached data via REST API
├── Market Data Worker
│   └── Updates cache every 5 min
└── Discord Bot
    └── Uses Schwab client directly
    └── Can also use REST API for cached data
```

## Migrating from PythonAnywhere

1. ✅ Deploy bot to Digital Ocean (above)
2. ✅ Test bot on Discord to ensure it works
3. ❌ Stop PythonAnywhere bot
4. ❌ Cancel PythonAnywhere subscription (optional)

## Troubleshooting

**Bot not starting:**
- Check .env file exists and has correct credentials
- View logs: `tail -f /root/options-scanner/logs/discord_bot.log`
- Check service status: `sudo systemctl status discord-bot`

**Authentication errors:**
- Copy fresh token: `scp schwab_client.json root@138.197.210.166:/root/options-scanner/`
- Restart bot: `sudo systemctl restart discord-bot`

**Bot commands not working:**
- Check bot has proper Discord permissions
- Verify DISCORD_GUILD_ID is correct
- Check logs for errors
