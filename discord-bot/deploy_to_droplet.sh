#!/bin/bash
# Deploy Discord Bot to Digital Ocean Droplet

echo "=== Discord Bot Deployment to Digital Ocean ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

PROJECT_DIR="/root/options-scanner"
DISCORD_DIR="$PROJECT_DIR/discord-bot"

# Verify directories exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$DISCORD_DIR" ]; then
    echo "Error: Discord bot directory not found at $DISCORD_DIR"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Discord bot directory: $DISCORD_DIR"
echo ""

# Check if .env file exists
if [ ! -f "$DISCORD_DIR/.env" ]; then
    echo "⚠ Warning: .env file not found at $DISCORD_DIR/.env"
    echo "You'll need to create it with your Discord credentials:"
    echo ""
    echo "DISCORD_TOKEN=your_bot_token_here"
    echo "DISCORD_GUILD_ID=your_server_id_here"
    echo ""
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Discord bot dependencies
echo "Installing Discord bot dependencies..."
cd $DISCORD_DIR
$PROJECT_DIR/venv/bin/pip install -r requirements.txt
echo "✓ Dependencies installed"

# Copy service file to systemd
echo ""
echo "Installing Discord bot systemd service..."
cp $DISCORD_DIR/discord-bot.service /etc/systemd/system/
echo "✓ Service file copied to /etc/systemd/system/"

# Reload systemd
systemctl daemon-reload
echo "✓ Systemd daemon reloaded"

# Enable service
systemctl enable discord-bot.service
echo "✓ Service enabled"

# Start service
echo ""
echo "Starting Discord bot service..."
systemctl start discord-bot.service
echo "✓ Service started"

# Show status
echo ""
echo "=== Service Status ==="
systemctl status discord-bot.service --no-pager -l

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status discord-bot"
echo "  sudo systemctl restart discord-bot"
echo "  sudo systemctl stop discord-bot"
echo "  sudo journalctl -u discord-bot -f"
echo "  tail -f $PROJECT_DIR/logs/discord_bot.log"
echo ""
echo "Don't forget to:"
echo "1. Set up your .env file with Discord credentials"
echo "2. Restart the service after updating .env: sudo systemctl restart discord-bot"
