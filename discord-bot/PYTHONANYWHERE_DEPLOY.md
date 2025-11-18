# Discord Bot Deployment to PythonAnywhere

## Required Files Structure

```
discord-bot/
├── bot/
│   ├── __init__.py
│   ├── main.py
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── dte_commands.py
│   │   ├── gamma_map.py
│   │   ├── whale_score.py
│   │   └── ema_cloud.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── schwab_service.py
│   └── utils/
│       ├── __init__.py
│       └── chart_utils.py
├── .env (create from template)
├── requirements.txt
└── README.md

src/  (from parent project)
├── __init__.py
├── api/
│   ├── __init__.py
│   └── schwab_client.py
└── utils/
    ├── __init__.py
    └── config.py

schwab_client.json (your token file)
```

## Step-by-Step Deployment

### 1. Prepare Files Locally

Create a deployment package with these files:
- All files in `discord-bot/` folder
- `src/api/schwab_client.py` 
- `src/utils/config.py`
- `schwab_client.json` (your Schwab tokens)
- `.env` file with Discord token and Schwab credentials

### 2. PythonAnywhere Setup

**Console Commands:**
```bash
# Navigate to home directory
cd ~

# Create project directory
mkdir discord-bot
cd discord-bot

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Upload Files

**Option A: Using PythonAnywhere File Browser**
- Upload zip of discord-bot folder
- Extract in ~/discord-bot/

**Option B: Using Git (Recommended)**
```bash
git clone https://github.com/pkay2402/options-scanner.git
cd options-scanner
```

### 4. Install Dependencies

```bash
source venv/bin/activate
pip install discord.py python-dotenv pandas numpy plotly kaleido yfinance requests aiohttp authlib httpx
```

### 5. Configure Environment

Create `.env` file in `~/discord-bot/`:
```env
DISCORD_BOT_TOKEN=your_token_here
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1:3000/callback
```

Copy `schwab_client.json` to project root (contains refresh tokens)

### 6. Create Startup Script

Create `~/discord-bot/start_bot.sh`:
```bash
#!/bin/bash
cd ~/discord-bot
source venv/bin/activate
python bot/main.py >> bot.log 2>&1
```

Make it executable:
```bash
chmod +x start_bot.sh
```

### 7. Keep Bot Running (Always-On Task)

**PythonAnywhere Paid Account:**
Add to "Always-on tasks":
```bash
/home/yourusername/discord-bot/start_bot.sh
```

**Free Account Alternative:**
Create scheduled task to restart every hour:
```bash
/home/yourusername/discord-bot/start_bot.sh
```

### 8. Monitor & Debug

**View logs:**
```bash
tail -f ~/discord-bot/bot.log
```

**Check if running:**
```bash
ps aux | grep python
```

**Stop bot:**
```bash
pkill -f "bot/main.py"
```

## Important Notes

### Token Management
- **schwab_client.json** contains refresh tokens (valid 7 days)
- Bot auto-refreshes access tokens every 25 minutes
- If refresh token expires, you'll need to re-authenticate locally and upload new `schwab_client.json`

### PythonAnywhere Limitations
- **Free accounts**: No always-on tasks (use scheduled tasks instead)
- **Paid accounts**: Can run bot 24/7 with always-on tasks
- **Outbound HTTPS**: Whitelist `api.schwabapi.com` (usually allowed)

### Files NOT to Include
- `__pycache__/` folders
- `.pyc` files
- `logs/` folder (will be created)
- Virtual environment (`venv/`)
- Local `.env` (create fresh on server)

## Minimal File Checklist

Essential files only:
```
✅ discord-bot/bot/main.py
✅ discord-bot/bot/commands/*.py (all 4 command files)
✅ discord-bot/bot/services/schwab_service.py
✅ discord-bot/bot/utils/chart_utils.py
✅ src/api/schwab_client.py
✅ src/utils/config.py
✅ schwab_client.json (token file)
✅ .env (Discord token + Schwab credentials)
✅ requirements.txt
```

## Quick Deploy Script

Create `deploy.sh` locally:
```bash
#!/bin/bash
# Package files for deployment
mkdir -p deploy_package
cp -r discord-bot deploy_package/
cp -r src deploy_package/
cp schwab_client.json deploy_package/
cp .env deploy_package/discord-bot/
cd deploy_package
zip -r ../discord_bot_deploy.zip .
cd ..
rm -rf deploy_package
echo "✅ Package created: discord_bot_deploy.zip"
```

Run: `bash deploy.sh`

Upload `discord_bot_deploy.zip` to PythonAnywhere and extract.

## Troubleshooting

**Bot not starting:**
- Check logs: `tail -100 ~/discord-bot/bot.log`
- Verify Python version: `python --version` (need 3.9+)
- Check installed packages: `pip list`

**Schwab API errors:**
- Verify token file exists: `ls -la schwab_client.json`
- Check token expiry: Run locally first to refresh
- Ensure credentials in `.env` are correct

**Discord commands not syncing:**
- Bot needs to restart to sync commands
- Global sync can take up to 1 hour
- Use guild-specific sync for faster testing

## Alternative: Docker Deployment

If PythonAnywhere doesn't work, consider:
- AWS EC2 free tier
- Google Cloud Run
- Heroku
- Railway.app
- Fly.io

These support Docker which makes deployment cleaner.
