#!/bin/bash
# Deploy Package Creator for PythonAnywhere
# Creates a zip file with all necessary files for Discord bot deployment

echo "🚀 Creating Discord Bot Deployment Package..."

# Create temporary directory
DEPLOY_DIR="discord_bot_deploy"
mkdir -p "$DEPLOY_DIR"

# Copy Discord bot files
echo "📦 Copying bot files..."
mkdir -p "$DEPLOY_DIR/bot"
cp -r discord-bot/bot/*.py "$DEPLOY_DIR/bot/" 2>/dev/null
cp -r discord-bot/bot/commands "$DEPLOY_DIR/bot/"
cp -r discord-bot/bot/services "$DEPLOY_DIR/bot/"
cp -r discord-bot/bot/utils "$DEPLOY_DIR/bot/" 2>/dev/null

# Copy Python runner script
cp discord-bot/run_bot.py "$DEPLOY_DIR/"

# Copy source dependencies
echo "📦 Copying source dependencies..."
mkdir -p "$DEPLOY_DIR/src/api"
mkdir -p "$DEPLOY_DIR/src/utils"
cp src/__init__.py "$DEPLOY_DIR/src/" 2>/dev/null || echo "" > "$DEPLOY_DIR/src/__init__.py"
cp src/api/__init__.py "$DEPLOY_DIR/src/api/" 2>/dev/null || echo "" > "$DEPLOY_DIR/src/api/__init__.py"
cp src/api/schwab_client.py "$DEPLOY_DIR/src/api/"
cp src/utils/__init__.py "$DEPLOY_DIR/src/utils/" 2>/dev/null || echo "" > "$DEPLOY_DIR/src/utils/__init__.py"
cp src/utils/config.py "$DEPLOY_DIR/src/utils/"

# Copy token file (IMPORTANT!)
echo "🔑 Copying Schwab token file..."
if [ -f "schwab_client.json" ]; then
    cp schwab_client.json "$DEPLOY_DIR/"
    echo "✅ Token file copied"
else
    echo "⚠️  WARNING: schwab_client.json not found!"
fi

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > "$DEPLOY_DIR/requirements.txt" << 'EOF'
# Discord Bot Requirements
discord.py>=2.3.2
python-dotenv>=1.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Plotting
plotly>=5.14.0
kaleido>=0.2.1

# Market data
yfinance>=0.2.0

# API clients
requests>=2.31.0
aiohttp>=3.9.0
httpx>=0.25.0

# Auth
authlib>=1.2.0
EOF

# Create .env template
echo "📝 Creating .env template..."
cat > "$DEPLOY_DIR/.env.template" << 'EOF'
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# Schwab API Credentials
SCHWAB_CLIENT_ID=your_schwab_client_id
SCHWAB_CLIENT_SECRET=your_schwab_client_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1:3000/callback

# Optional Settings
LOG_LEVEL=INFO
EOF

# Create startup script
echo "📝 Creating startup script..."
cat > "$DEPLOY_DIR/start_bot.sh" << 'EOF'
#!/bin/bash
# Discord Bot Startup Script for PythonAnywhere

# Navigate to bot directory
cd "$(dirname "$0")"

# Activate virtual environment (adjust path as needed)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run bot
python3 bot/main.py >> bot.log 2>&1
EOF

chmod +x "$DEPLOY_DIR/start_bot.sh"

# Create README
echo "📝 Creating deployment README..."
cat > "$DEPLOY_DIR/README_DEPLOY.md" << 'EOF'
# Discord Bot Deployment Package

## Quick Start

1. **Upload to PythonAnywhere**
   - Upload this entire folder to ~/discord-bot/

2. **Create Virtual Environment**
   ```bash
   cd ~/discord-bot
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   - Copy .env.template to .env
   - Fill in your Discord bot token and Schwab credentials
   - Verify schwab_client.json exists and has valid tokens

5. **Test Run**
   ```bash
   python3 bot/main.py
   ```
   Press Ctrl+C to stop

6. **Keep Running (Always-On Task)**
   - PythonAnywhere Dashboard → Tasks → Add new always-on task
   - Command: `/home/yourusername/discord-bot/start_bot.sh`

## Files Included

- bot/ - Discord bot code
- src/ - Schwab API client
- schwab_client.json - Token file (KEEP SECURE!)
- requirements.txt - Python dependencies
- .env.template - Environment variables template
- start_bot.sh - Startup script

## Troubleshooting

**View Logs:**
```bash
tail -f ~/discord-bot/bot.log
```

**Check Process:**
```bash
ps aux | grep python
```

**Stop Bot:**
```bash
pkill -f "bot/main.py"
```

## Token Refresh

Schwab refresh tokens expire after 7 days. If bot stops authenticating:
1. Run bot locally to refresh tokens
2. Copy updated schwab_client.json to server
3. Restart bot
EOF

# Create .gitignore for deployment package
cat > "$DEPLOY_DIR/.gitignore" << 'EOF'
# Environment
.env
venv/

# Logs
*.log
logs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Tokens (NEVER commit these!)
schwab_client.json
EOF

# Create zip file
echo "🗜️  Creating zip archive..."
zip -r discord_bot_deploy.zip "$DEPLOY_DIR" -q

# Cleanup
rm -rf "$DEPLOY_DIR"

echo ""
echo "✅ Deployment package created: discord_bot_deploy.zip"
echo ""
echo "📋 Next steps:"
echo "1. Upload discord_bot_deploy.zip to PythonAnywhere"
echo "2. Extract: unzip discord_bot_deploy.zip"
echo "3. Follow instructions in README_DEPLOY.md"
echo ""
echo "⚠️  IMPORTANT: Don't forget to:"
echo "   - Copy .env.template to .env and fill in credentials"
echo "   - Verify schwab_client.json has valid tokens"
echo ""
