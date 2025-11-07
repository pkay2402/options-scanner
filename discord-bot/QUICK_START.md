# 🚀 Discord Bot - Quick Reference

## Essential Commands

### Setup
```bash
cd discord-bot
cp .env.template .env
# Edit .env with your tokens
pip install -r requirements.txt
./start.sh
```

### Run Bot
```bash
# From project root
python -m discord-bot.bot.main

# Or use the script
cd discord-bot && ./start.sh
```

### Docker
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f discord-bot

# Stop
docker-compose down
```

## Discord Commands Cheatsheet

```
📊 GAMMA ANALYSIS
/gamma-heatmap symbol:SPY expiries:4    # GEX heatmap
/gamma-top symbol:AAPL count:5          # Top gamma strikes

🧱 VOLUME WALLS
/walls symbol:QQQ                        # Full walls analysis
/call-wall symbol:TSLA                   # Call resistance
/put-wall symbol:NVDA                    # Put support

💰 MARKET ANALYSIS
/dark-pool symbol:SPY                    # 7-day sentiment
/ema-trend symbol:AAPL                   # EMA positioning
/quote symbol:SPY                        # Quick quote
```

## Required Environment Variables

```bash
DISCORD_BOT_TOKEN=your_bot_token
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1:8182
```

## Discord Bot Setup Steps

1. **Create Bot:**
   - https://discord.com/developers/applications
   - New Application → Bot tab → Add Bot
   - Copy token to `.env`
   - Enable "Message Content Intent"

2. **Invite to Server:**
   - OAuth2 → URL Generator
   - Scopes: `bot`, `applications.commands`
   - Permissions: Send Messages, Embed Links, Attach Files
   - Open generated URL

3. **Test:**
   - Type `/gamma-heatmap` in Discord
   - Should see autocomplete

## Troubleshooting

**Bot offline?**
```bash
# Check logs
tail -f ../logs/discord_bot.log

# Verify token
grep DISCORD_BOT_TOKEN .env
```

**Commands not showing?**
```bash
# Ensure bot has applications.commands scope
# Re-invite bot with correct permissions
```

**Schwab auth failing?**
```bash
# Delete and re-auth
rm ../schwab_client.json
# Restart bot and complete OAuth flow
```

**Charts not generating?**
```bash
pip install kaleido --upgrade
```

## File Locations

- **Bot code:** `discord-bot/bot/`
- **Logs:** `logs/discord_bot.log`
- **Config:** `discord-bot/.env`
- **Tokens:** `schwab_client.json` (parent dir)

## Common Issues

| Issue | Solution |
|-------|----------|
| "No options data" | Market closed or symbol invalid |
| "Authentication failed" | Check Schwab credentials in .env |
| Bot not responding | Verify bot is online and has permissions |
| Rate limit errors | Wait 60 seconds, bot auto-retries |

## Development

**Test locally:**
```bash
python -m discord-bot.bot.main
```

**Add new command:**
1. Edit `bot/commands/gamma.py` (or walls.py, analysis.py)
2. Add `@app_commands.command()` decorator
3. Restart bot
4. Commands auto-sync

**View command tree:**
```python
# In Discord, all commands show with /
# Use autocomplete to see parameters
```

## Monitoring

```bash
# Live logs
tail -f ../logs/discord_bot.log

# Error search
grep ERROR ../logs/discord_bot.log

# Bot status
ps aux | grep "discord-bot.bot.main"
```

## Production Deployment

**Railway (easiest):**
1. Push to GitHub
2. Connect repo to Railway
3. Add env vars
4. Auto-deploys

**AWS EC2:**
```bash
# On EC2 instance
git clone your_repo
cd options/discord-bot
docker-compose up -d
```

## Architecture

```
Discord User → Slash Command → Bot (Discord.py)
                                  ↓
                            Schwab Service (Auto-refresh)
                                  ↓
                            Schwab API (Shared account)
                                  ↓
                            Analysis Functions (Reused from Streamlit)
                                  ↓
                            Plotly Charts → PNG → Discord Embed
```

## Rate Limits

- Schwab API: 120/min (shared)
- Discord: 50 commands/sec per guild
- Bot queues requests automatically

## Security Checklist

- [ ] `.env` not in git
- [ ] `schwab_client.json` not in git
- [ ] Bot token kept secret
- [ ] Minimal Discord permissions
- [ ] Logs reviewed regularly

## Next Steps After Setup

1. ✅ Test `/quote symbol:SPY` (simplest command)
2. ✅ Test `/gamma-heatmap symbol:SPY` (chart generation)
3. ✅ Test `/walls symbol:QQQ` (full analysis)
4. ✅ Monitor logs for errors
5. ✅ Deploy to cloud if desired

## Support

- **Logs:** `tail -f ../logs/discord_bot.log`
- **Test Schwab API:** Run Streamlit app first
- **Discord Bot Portal:** https://discord.com/developers/applications
- **Test environment:** Use private Discord server for testing

---

**Quick Start:** `cd discord-bot && ./start.sh`

**Stop Bot:** `Ctrl+C` or `docker-compose down`

**Update Bot:** `git pull && pip install -r requirements.txt --upgrade && restart`
