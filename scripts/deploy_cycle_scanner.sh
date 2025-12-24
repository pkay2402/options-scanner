#!/bin/bash

# Deploy Cycle Scanner to Droplet
# Run this script to deploy the scanner service

set -e

echo "🚀 Deploying Cycle Scanner to Droplet..."

# Configuration
DROPLET_IP="138.197.210.166"
DROPLET_USER="root"
REMOTE_DIR="/root/options-scanner"

echo "📦 Copying scanner script..."
scp scripts/cycle_scanner.py $DROPLET_USER@$DROPLET_IP:$REMOTE_DIR/scripts/

echo "📋 Copying service files..."
scp services/cycle-scanner.service $DROPLET_USER@$DROPLET_IP:/etc/systemd/system/
scp services/cycle-scanner.timer $DROPLET_USER@$DROPLET_IP:/etc/systemd/system/

echo "🔧 Setting up service on droplet..."
ssh $DROPLET_USER@$DROPLET_IP << 'EOF'
# Create data directory if it doesn't exist
mkdir -p /root/options-scanner/data
mkdir -p /root/options-scanner/logs

# Install dependencies
pip3 install --break-system-packages yfinance pandas numpy

# Reload systemd
systemctl daemon-reload

# Enable and start timer
systemctl enable cycle-scanner.timer
systemctl start cycle-scanner.timer

# Check status
systemctl status cycle-scanner.timer

echo ""
echo "✅ Cycle Scanner deployed successfully!"
echo ""
echo "Commands:"
echo "  Start scanner now:  sudo systemctl start cycle-scanner.service"
echo "  Check status:       sudo systemctl status cycle-scanner.timer"
echo "  View logs:          sudo journalctl -u cycle-scanner.service -f"
echo "  List timers:        sudo systemctl list-timers"
EOF

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next scans scheduled for market hours:"
echo "  9:30 AM, 11:30 AM, 1:30 PM, 3:30 PM EST"
