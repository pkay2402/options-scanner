#!/bin/bash
# Quick start script for Discord bot

set -e

echo "🤖 Options Trading Discord Bot - Quick Start"
echo "==========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📝 Creating .env from template..."
    cp .env.template .env
    echo "✅ .env created. Please edit it with your credentials:"
    echo "   - DISCORD_BOT_TOKEN"
    echo "   - SCHWAB_CLIENT_ID"
    echo "   - SCHWAB_CLIENT_SECRET"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if schwab_client.json exists
if [ ! -f ../schwab_client.json ]; then
    echo "⚠️  schwab_client.json not found in parent directory"
    echo "🔑 You'll need to authenticate when the bot starts"
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🚀 Starting Discord bot..."
echo "   (Press Ctrl+C to stop)"
echo ""

cd ..
python -m discord-bot.bot.main
