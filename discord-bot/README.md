# 🤖 Options Trading Discord Bot

Discord bot providing real-time options analysis via slash commands. Uses **service account authentication** (Option A) with your Schwab API credentials.

## ✨ Features

### Gamma Analysis
- `/gamma-heatmap` - Net GEX heatmap across strikes and expiries
- `/gamma-top` - Top gamma strikes with expiry dates

### Volume Walls
- `/walls` - Call/put walls with flip level chart
- `/call-wall` - Call wall (resistance) details
- `/put-wall` - Put wall (support) details

### Market Analysis
- `/dark-pool` - 7-day dark pool sentiment from FINRA
- `/ema-trend` - EMA positioning (8/21/50/200)
- `/quote` - Quick price quote

## 🚀 Quick Start

### Prerequisites

1. **Discord Bot Token**
   - Go to https://discord.com/developers/applications
   - Create "New Application"
   - Go to "Bot" tab → "Add Bot"
   - Copy the token
   - Enable "Message Content Intent" under Privileged Gateway Intents

2. **Schwab API Credentials**
   - Same credentials as your main Streamlit app
   - Bot will use existing `schwab_client.json` for authentication

### Installation

1. **Navigate to bot directory:**
   ```bash
   cd discord-bot
   ```

2. **Create environment file:**
   ```bash
   cp .env.template .env
   ```

3. **Edit `.env` with your credentials:**
   ```bash
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   SCHWAB_CLIENT_ID=your_schwab_client_id
   SCHWAB_CLIENT_SECRET=your_schwab_client_secret
   SCHWAB_REDIRECT_URI=https://127.0.0.1:8182
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the bot:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

   Or manually:
   ```bash
   cd ..
   python -m discord-bot.bot.main
   ```

### Invite Bot to Your Server

1. Go to Discord Developer Portal → Your Application → OAuth2 → URL Generator
2. Select scopes: `bot`, `applications.commands`
3. Select permissions:
   - Send Messages
   - Embed Links
   - Attach Files
   - Use Slash Commands
4. Copy the generated URL and open in browser
5. Select your server and authorize

## 📖 Command Usage

### Gamma Commands

**Gamma Heatmap:**
```
/gamma-heatmap symbol:SPY expiries:4
```
Returns a color-coded heatmap showing net gamma exposure across strikes and expiration dates.

**Top Gamma Strikes:**
```
/gamma-top symbol:AAPL count:5
```
Shows the top N strikes with highest gamma exposure.

### Volume Walls Commands

**Full Walls Analysis:**
```
/walls symbol:QQQ
```
Shows call wall (resistance), put wall (support), and gamma flip level with chart.

**Call Wall Only:**
```
/call-wall symbol:TSLA
```
Detailed call wall analysis with interpretation.

**Put Wall Only:**
```
/put-wall symbol:NVDA
```
Detailed put wall analysis with interpretation.

### Analysis Commands

**Dark Pool Sentiment:**
```
/dark-pool symbol:SPY
```
7-day cumulative buy/sell ratio from FINRA dark pool data.

**EMA Trend:**
```
/ema-trend symbol:AAPL
```
EMA positioning (8/21/50/200) with trend determination.

**Quick Quote:**
```
/quote symbol:SPY
```
Current price, change, volume, bid/ask.

## 🐳 Docker Deployment

### Build and Run with Docker Compose

1. **Make sure `.env` is configured**

2. **Start the bot:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f discord-bot
   ```

4. **Stop the bot:**
   ```bash
   docker-compose down
   ```

### Standalone Docker Build

```bash
cd ..
docker build -f discord-bot/Dockerfile -t options-discord-bot .
docker run -d \
  --name options-bot \
  --env-file discord-bot/.env \
  -v $(pwd)/schwab_client.json:/app/schwab_client.json \
  -v $(pwd)/logs:/app/logs \
  options-discord-bot
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DISCORD_BOT_TOKEN` | Discord bot token | ✅ Yes |
| `SCHWAB_CLIENT_ID` | Schwab API client ID | ✅ Yes |
| `SCHWAB_CLIENT_SECRET` | Schwab API secret | ✅ Yes |
| `SCHWAB_REDIRECT_URI` | OAuth redirect URI | ✅ Yes |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | No (default: INFO) |

### Authentication

The bot uses **Option A: Service Account** authentication:
- Single Schwab account shared across all Discord users
- Uses existing `schwab_client.json` from parent directory
- Automatic token refresh every 25 minutes
- No per-user authentication required

If `schwab_client.json` doesn't exist, you'll need to complete OAuth flow on first run.

## 📂 Project Structure

```
discord-bot/
├── bot/
│   ├── main.py                 # Bot entry point
│   ├── commands/
│   │   ├── gamma.py           # Gamma analysis commands
│   │   ├── walls.py           # Volume walls commands
│   │   └── analysis.py        # Market analysis commands
│   ├── services/
│   │   └── schwab_service.py  # Schwab API service with auto-refresh
│   └── utils/
│       └── chart_utils.py     # Plotly to Discord image conversion
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.template
├── start.sh
└── README.md
```

## 🔐 Security Notes

1. **Never commit `.env` file** - It contains sensitive tokens
2. **Keep `schwab_client.json` secure** - Contains OAuth tokens
3. **Use environment variables** in production deployment
4. **Restrict bot permissions** to only what's needed

## 🚀 Deployment Options

### Local Development
```bash
./start.sh
```

### Railway (Recommended - $5/mo)
1. Create new project on Railway.app
2. Connect GitHub repo
3. Add environment variables
4. Deploy automatically

### AWS EC2
1. Launch t3.small instance ($10-20/mo)
2. Clone repo
3. Install Docker
4. Run with docker-compose

### DigitalOcean Droplet
1. Create $6/mo droplet
2. Clone repo
3. Setup with docker-compose

### Heroku
1. Create new app
2. Add worker dyno ($7/mo)
3. Set environment variables
4. Deploy from GitHub

## 📊 Rate Limits

- **Schwab API:** 120 calls/minute (shared across all Discord users)
- **Discord API:** 50 slash commands/second per guild
- Bot implements automatic rate limiting and queuing

## 🐛 Troubleshooting

### Bot not responding to commands
1. Check bot is online in Discord server
2. Verify slash commands are registered: `/gamma-heatmap` should auto-complete
3. Check logs: `tail -f ../logs/discord_bot.log`

### Authentication errors
1. Verify `.env` credentials are correct
2. Check `schwab_client.json` exists and is valid
3. Try deleting `schwab_client.json` and re-authenticating

### Chart generation fails
1. Ensure `kaleido` is installed: `pip install kaleido`
2. Check logs for Plotly errors
3. Verify image conversion in `chart_utils.py`

### "No options data available" errors
1. Ensure market is open (9:30 AM - 4:00 PM ET, Mon-Fri)
2. Verify symbol is valid
3. Check Schwab API rate limits

## 📝 Logs

Logs are written to `../logs/discord_bot.log`:

```bash
# View real-time logs
tail -f ../logs/discord_bot.log

# View last 100 lines
tail -n 100 ../logs/discord_bot.log

# Search for errors
grep ERROR ../logs/discord_bot.log
```

## 🔄 Updates

To update the bot:

```bash
git pull origin main
cd discord-bot
pip install -r requirements.txt --upgrade
# Restart bot
```

## 🤝 Integration with Streamlit App

The bot **shares code** with your Streamlit app:
- Uses `src/api/schwab_client.py` for Schwab API
- Uses `src/utils/dark_pool.py` for FINRA data
- Reuses analysis logic from pages
- **Does not affect** existing Streamlit functionality

Both can run simultaneously without conflicts.

## 💡 Tips

1. **Test commands in private channel** before public use
2. **Monitor rate limits** - bot serves all users with one Schwab account
3. **Use Docker** for production deployment
4. **Set up monitoring** (Uptime Robot, etc.) for 24/7 availability
5. **Regular restarts** (weekly) help maintain stability

## 📞 Support

- Check logs first: `tail -f ../logs/discord_bot.log`
- Verify environment variables in `.env`
- Ensure Schwab credentials are valid
- Test Schwab API with main Streamlit app first

## 🎯 Future Enhancements

Potential additions (not included yet):
- Alert system (`/alert add SPY gamma > 1B`)
- Options flow monitoring
- Multi-symbol comparison charts
- Custom watchlists per Discord server
- Per-user authentication (Option B)

---

**Built with:**
- discord.py 2.3+
- Python 3.11+
- Schwab API
- Plotly for charts
- Your existing options analysis code

**License:** Same as parent project
