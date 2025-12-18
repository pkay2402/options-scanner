#!/bin/bash

# Script to refresh Schwab authentication and deploy to droplet
# Run this when the worker stops updating due to expired tokens

echo "🔄 Refreshing Schwab Authentication and Deploying to Droplet"
echo "=============================================================="

# Step 1: Stop the worker on droplet
echo "1️⃣ Stopping worker service on droplet..."
ssh root@138.197.210.166 "systemctl stop market-data-worker.service"
sleep 2

# Step 2: Run local authentication
echo ""
echo "2️⃣ Authenticating locally (you'll need to paste the callback URL)..."
cd "$(dirname "$0")/.."
python3 -c "
from src.api.schwab_client import SchwabClient
client = SchwabClient()
if client.authenticate():
    print('✅ Authentication successful!')
    print('Token file location:', client.filepath)
else:
    print('❌ Authentication failed!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Authentication failed. Please try again."
    exit 1
fi

# Step 3: Copy token file to droplet
echo ""
echo "3️⃣ Copying fresh token file to droplet..."
TOKEN_FILE="schwab_client.json"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Token file not found at $TOKEN_FILE"
    exit 1
fi

scp "$TOKEN_FILE" root@138.197.210.166:/root/options-scanner/schwab_client.json

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy token file to droplet"
    exit 1
fi

echo "✅ Token file copied successfully"

# Step 4: Restart the worker
echo ""
echo "4️⃣ Restarting worker service..."
ssh root@138.197.210.166 "systemctl start market-data-worker.service"
sleep 2

# Step 5: Check status
echo ""
echo "5️⃣ Checking service status..."
ssh root@138.197.210.166 "systemctl status market-data-worker.service --no-pager -l"

echo ""
echo "6️⃣ Checking recent logs..."
ssh root@138.197.210.166 "tail -20 /root/options-scanner/logs/market_data_worker.log"

echo ""
echo "✅ Done! Worker should be updating now."
echo "💡 Monitor logs: ssh root@138.197.210.166 'tail -f /root/options-scanner/logs/market_data_worker.log'"
