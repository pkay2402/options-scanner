#!/bin/bash
# Automated Schwab Token Refresh Script
# Refreshes tokens locally and updates all services

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DROPLET_IP="138.197.210.166"
DROPLET_USER="root"
DROPLET_PATH="/root/options-scanner"
DROPLET_OPT_PATH="/opt/options-scanner"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Schwab Token Refresh & Deployment Automation        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Backup old token
echo -e "${YELLOW}[1/8]${NC} Backing up old token..."
if [ -f "$PROJECT_DIR/schwab_client.json" ]; then
    cp "$PROJECT_DIR/schwab_client.json" "$PROJECT_DIR/schwab_client.json.backup"
    echo -e "${GREEN}✓${NC} Backup created"
else
    echo -e "${YELLOW}⚠${NC}  No existing token to backup"
fi

# Step 2: Delete old token
echo -e "\n${YELLOW}[2/8]${NC} Removing old token..."
rm -f "$PROJECT_DIR/schwab_client.json"
echo -e "${GREEN}✓${NC} Old token removed"

# Step 3: Run auth setup
echo -e "\n${YELLOW}[3/8]${NC} Running Schwab authentication..."
echo -e "${BLUE}ℹ${NC}  Browser will open for OAuth. Please complete authentication."
cd "$PROJECT_DIR"
python3 scripts/auth_setup.py

if [ ! -f "$PROJECT_DIR/schwab_client.json" ]; then
    echo -e "${RED}✗${NC} Authentication failed! Token file not created."
    exit 1
fi
echo -e "${GREEN}✓${NC} New token generated"

# Step 4: Copy token to droplet (both locations)
echo -e "\n${YELLOW}[4/8]${NC} Copying token to droplet..."
scp "$PROJECT_DIR/schwab_client.json" "$DROPLET_USER@$DROPLET_IP:$DROPLET_PATH/schwab_client.json"
ssh "$DROPLET_USER@$DROPLET_IP" "cp $DROPLET_PATH/schwab_client.json $DROPLET_OPT_PATH/schwab_client.json && chown options:options $DROPLET_OPT_PATH/schwab_client.json && chmod 600 $DROPLET_OPT_PATH/schwab_client.json"
echo -e "${GREEN}✓${NC} Token copied to droplet (both /root and /opt)"

# Step 5: Restart services on droplet
echo -e "\n${YELLOW}[5/8]${NC} Restarting services on droplet..."
ssh "$DROPLET_USER@$DROPLET_IP" << 'EOF'
    systemctl restart discord-bot api-server
    sleep 3
    echo "Services restarted"
EOF
echo -e "${GREEN}✓${NC} Services restarted"

# Step 6: Verify services
echo -e "\n${YELLOW}[6/8]${NC} Verifying services..."
ssh "$DROPLET_USER@$DROPLET_IP" << 'EOF'
    if systemctl is-active --quiet discord-bot && systemctl is-active --quiet api-server; then
        echo "✓ All services running"
    else
        echo "✗ Some services failed to start"
        exit 1
    fi
EOF

# Step 7: Generate Streamlit secrets
echo -e "\n${YELLOW}[7/8]${NC} Generating Streamlit secrets format..."
python3 "$PROJECT_DIR/scripts/generate_streamlit_secrets.py" > /tmp/streamlit_secrets_output.txt
echo -e "${GREEN}✓${NC} Streamlit secrets generated"

# Step 8: Show Streamlit secrets
echo -e "\n${YELLOW}[8/8]${NC} Streamlit Cloud Update Required:"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
cat /tmp/streamlit_secrets_output.txt
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📋 Action Required:${NC}"
echo -e "   1. Copy the secrets above"
echo -e "   2. Go to: https://share.streamlit.io/[your-app]/settings/secrets"
echo -e "   3. Paste and save"
echo -e "   4. App will auto-restart"
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Token Refresh Complete! ✓                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓${NC} Local token: Updated"
echo -e "${GREEN}✓${NC} Droplet services: Restarted"
echo -e "${YELLOW}⏳${NC} Streamlit Cloud: Awaiting manual update (see above)"
echo -e "${GREEN}✓${NC} Discord bot: Ready (commands already loaded)"
echo ""
echo -e "${BLUE}ℹ${NC}  Token expires in 7 days. Run this script again before expiration."
echo -e "${BLUE}ℹ${NC}  Backup saved to: schwab_client.json.backup"
echo ""
