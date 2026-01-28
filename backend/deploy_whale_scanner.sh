#!/bin/bash
# Deploy Whale Flow Scanner to Droplet
# Run: ./deploy_whale_scanner.sh

set -e

echo "🐋 Deploying Whale Flow Scanner..."

# Configuration
DROPLET_IP="138.197.210.166"
REMOTE_DIR="/root/options-scanner"

# 1. Copy files to droplet
echo "📦 Copying files to droplet..."
scp backend/whale_flow_scanner.py root@${DROPLET_IP}:${REMOTE_DIR}/backend/
scp backend/whale-flow-scanner.service root@${DROPLET_IP}:/etc/systemd/system/
scp backend/whale-flow-scanner.timer root@${DROPLET_IP}:/etc/systemd/system/

# 2. Setup on droplet
echo "⚙️ Setting up systemd services..."
ssh root@${DROPLET_IP} << 'EOF'
    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start timer
    systemctl enable whale-flow-scanner.timer
    systemctl start whale-flow-scanner.timer
    
    # Check status
    echo ""
    echo "📊 Timer Status:"
    systemctl status whale-flow-scanner.timer --no-pager
    
    echo ""
    echo "⏰ Next scheduled runs:"
    systemctl list-timers whale-flow-scanner.timer --no-pager
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "1. SSH to droplet: ssh root@${DROPLET_IP}"
echo "2. Set Telegram credentials:"
echo "   Edit: /etc/systemd/system/whale-flow-scanner.service"
echo "   Set: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
echo "   Run: systemctl daemon-reload"
echo ""
echo "3. Test manually:"
echo "   cd ${REMOTE_DIR} && python backend/whale_flow_scanner.py"
echo ""
echo "4. Check logs:"
echo "   tail -f /var/log/whale_flow_scanner.log"
