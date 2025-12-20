#!/bin/bash
# One-command setup for robust monitoring

echo "📊 Setting up robust monitoring system..."
echo ""

# Check if we can SSH to droplet
if ! ssh -o ConnectTimeout=5 root@138.197.210.166 'exit' 2>/dev/null; then
    echo "❌ Cannot connect to droplet. Check SSH access."
    exit 1
fi

echo "✓ Connected to droplet"

# Copy health monitor files
echo "Copying health monitor files..."
scp scripts/health_monitor.py scripts/setup_health_monitor.sh root@138.197.210.166:/root/options-scanner/scripts/

# Make setup script executable
ssh root@138.197.210.166 'chmod +x /root/options-scanner/scripts/setup_health_monitor.sh'

# Run setup
echo "Installing health monitoring..."
ssh root@138.197.210.166 '/root/options-scanner/scripts/setup_health_monitor.sh'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Robust monitoring system installed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1️⃣  Set up Discord webhook for health alerts:"
echo "   • Open Discord server"
echo "   • Server Settings → Integrations → Webhooks"
echo "   • Create New Webhook named 'Health Monitor'"
echo "   • Copy webhook URL"
echo ""
echo "2️⃣  Add webhook to config:"
echo "   ssh root@138.197.210.166"
echo "   nano /root/options-scanner/config/health_monitor.json"
echo "   # Replace YOUR_DISCORD_WEBHOOK_URL_HERE with your URL"
echo ""
echo "3️⃣  Test health monitor:"
echo "   ssh root@138.197.210.166 'systemctl start health-monitor.service'"
echo "   # Check your Discord channel for test alert"
echo ""
echo "4️⃣  Verify it's running:"
echo "   ssh root@138.197.210.166 'systemctl status health-monitor.timer'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 What you'll get:"
echo "  ✅ Health checks every 15 minutes"
echo "  ✅ Discord alerts before things break"
echo "  ✅ Token expiration warnings (2 days before)"
echo "  ✅ Database staleness alerts"
echo "  ✅ Memory pressure warnings"
echo ""
echo "📖 Full guide: ROBUST_OPERATIONS.md"
echo ""
