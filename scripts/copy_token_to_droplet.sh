#!/bin/bash
# Copy schwab_client.json token file to droplet

DROPLET_IP="your-droplet-ip"  # Update this with your droplet IP
LOCAL_TOKEN="schwab_client.json"
REMOTE_PATH="/root/options-scanner/"

echo "=== Copying Schwab Token to Droplet ==="
echo ""

# Check if local token file exists
if [ ! -f "$LOCAL_TOKEN" ]; then
    echo "Error: $LOCAL_TOKEN not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "Found token file: $LOCAL_TOKEN"
echo "Droplet IP: $DROPLET_IP"
echo ""

# Copy token file
echo "Copying token file to droplet..."
scp $LOCAL_TOKEN root@$DROPLET_IP:$REMOTE_PATH

if [ $? -eq 0 ]; then
    echo "✓ Token file copied successfully"
    echo ""
    echo "Now restart the services on the droplet:"
    echo "  ssh root@$DROPLET_IP"
    echo "  cd /root/options-scanner"
    echo "  sudo systemctl restart market-data-worker"
    echo "  sudo systemctl restart api-server"
else
    echo "✗ Failed to copy token file"
    exit 1
fi
