#!/bin/bash
# Generate Streamlit secrets.toml for deployment
# Run this after setting up your droplet

echo "=========================================="
echo "Streamlit Secrets Generator"
echo "=========================================="
echo ""

# Get droplet IP
read -p "Enter your droplet IP address (e.g., 138.197.210.166): " DROPLET_IP

# Validate IP format
if [[ ! $DROPLET_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid IP address format"
    exit 1
fi

# Test API connectivity
echo ""
echo "Testing API connectivity..."
if curl -s --connect-timeout 5 "http://$DROPLET_IP:8000/health" > /dev/null; then
    echo "✓ API is reachable at http://$DROPLET_IP:8000"
else
    echo "✗ Warning: Could not reach API at http://$DROPLET_IP:8000"
    echo "  Make sure:"
    echo "  1. API server is running on droplet (sudo systemctl status api-server)"
    echo "  2. Port 8000 is open in firewall (sudo ufw allow 8000)"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ $CONTINUE != "y" ]]; then
        exit 1
    fi
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Generate secrets.toml
cat > .streamlit/secrets.toml << EOF
# Streamlit Secrets Configuration
# Generated: $(date)

# Droplet API Configuration
DROPLET_API_URL = "http://$DROPLET_IP:8000"

# Optional: Schwab API (if needed for some pages)
# SCHWAB_APP_KEY = "your_app_key_here"
# SCHWAB_APP_SECRET = "your_app_secret_here"
# SCHWAB_CALLBACK_URL = "https://127.0.0.1"
EOF

echo ""
echo "=========================================="
echo "✓ Created .streamlit/secrets.toml"
echo "=========================================="
echo ""
echo "Contents:"
cat .streamlit/secrets.toml
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. LOCAL TESTING:"
echo "   streamlit run Main_Dashboard.py"
echo ""
echo "2. STREAMLIT CLOUD DEPLOYMENT:"
echo "   - Go to: https://share.streamlit.io/"
echo "   - Click on your app settings"
echo "   - Go to 'Secrets' section"
echo "   - Paste this content:"
echo ""
echo "   DROPLET_API_URL = \"http://$DROPLET_IP:8000\""
echo ""
echo "3. VERIFY API ACCESS:"
echo "   curl http://$DROPLET_IP:8000/api/stats"
echo ""
echo "Note: .streamlit/secrets.toml is in .gitignore"
echo "      (secrets are not committed to git)"
echo ""
