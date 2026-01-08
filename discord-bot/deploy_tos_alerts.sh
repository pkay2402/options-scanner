#!/bin/bash
# Deploy TOS Alerts Command to Droplet

echo "🚀 Deploying TOS Alerts Discord Bot Command..."

# Check if we have email credentials
echo ""
echo "📧 Email Configuration Check:"
read -p "Enter TOS Email Address (Gmail): " EMAIL_ADDRESS
read -sp "Enter Gmail App Password (16 chars): " EMAIL_PASSWORD
echo ""

if [ -z "$EMAIL_ADDRESS" ] || [ -z "$EMAIL_PASSWORD" ]; then
    echo "❌ Error: Email credentials required"
    exit 1
fi

echo ""
echo "📡 Connecting to droplet..."

ssh root@138.197.210.166 << EOF
    cd /root/discord-bot
    
    echo "📥 Pulling latest code..."
    git pull
    
    echo "⚙️  Updating .env file..."
    
    # Check if .env exists
    if [ ! -f .env ]; then
        echo "Creating .env from template..."
        cp .env.template .env
    fi
    
    # Add or update TOS credentials
    if grep -q "TOS_EMAIL_ADDRESS" .env; then
        sed -i "s/TOS_EMAIL_ADDRESS=.*/TOS_EMAIL_ADDRESS=$EMAIL_ADDRESS/" .env
    else
        echo "TOS_EMAIL_ADDRESS=$EMAIL_ADDRESS" >> .env
    fi
    
    if grep -q "TOS_EMAIL_PASSWORD" .env; then
        sed -i "s/TOS_EMAIL_PASSWORD=.*/TOS_EMAIL_PASSWORD=$EMAIL_PASSWORD/" .env
    else
        echo "TOS_EMAIL_PASSWORD=$EMAIL_PASSWORD" >> .env
    fi
    
    echo "🔄 Restarting Discord bot..."
    systemctl restart discord-bot
    
    echo "⏳ Waiting for bot to start..."
    sleep 3
    
    echo "📊 Checking bot status..."
    systemctl status discord-bot --no-pager -l
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "📝 Next steps in Discord:"
    echo "   1. /setup_tos_alerts       - Configure channel"
    echo "   2. /start_tos_alerts       - Start monitoring"
    echo "   3. /tos_alerts_status      - Check status"
    echo ""
EOF

echo ""
echo "🎉 Done! TOS Alerts command is ready on the droplet."
