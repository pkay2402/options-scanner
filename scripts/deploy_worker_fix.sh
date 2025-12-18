#!/bin/bash
# Deploy fixed market data worker to droplet

DROPLET_IP="138.197.210.166"
REMOTE_PATH="/root/options-scanner"

echo "=== Deploying Worker Fix to Droplet ==="
echo ""

# Check if SSH key exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Error: SSH key not found at ~/.ssh/id_ed25519"
    echo "Please set up SSH key-based authentication first"
    exit 1
fi

echo "1. Copying fixed worker script to droplet..."
scp scripts/market_data_worker.py root@$DROPLET_IP:$REMOTE_PATH/scripts/

if [ $? -ne 0 ]; then
    echo "✗ Failed to copy worker script"
    exit 1
fi

echo "✓ Worker script copied"
echo ""

echo "2. Restarting market-data-worker service..."
ssh root@$DROPLET_IP << 'EOF'
cd /root/options-scanner
sudo systemctl restart market-data-worker
sleep 3
sudo systemctl status market-data-worker --no-pager -l
echo ""
echo "Recent logs:"
sudo journalctl -u market-data-worker -n 50 --no-pager
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Service restarted successfully"
    echo ""
    echo "Monitor logs with:"
    echo "  ssh root@$DROPLET_IP 'sudo journalctl -u market-data-worker -f'"
else
    echo "✗ Failed to restart service"
    exit 1
fi
