#!/bin/bash
# Health Monitor Setup Script for DigitalOcean Droplet

echo "Setting up health monitoring..."

# Create config directory
mkdir -p /root/options-scanner/config

# Copy health monitor script if not exists
if [ ! -f /root/options-scanner/scripts/health_monitor.py ]; then
    echo "Error: health_monitor.py not found. Please copy it first."
    exit 1
fi

# Create config file if not exists
if [ ! -f /root/options-scanner/config/health_monitor.json ]; then
    echo "Creating health monitor config..."
    cat > /root/options-scanner/config/health_monitor.json << 'EOF'
{
  "discord_webhook": "YOUR_DISCORD_WEBHOOK_URL_HERE",
  "check_interval_minutes": 15,
  "alerts": {
    "token_expiration_warning_days": 2,
    "database_stale_minutes": 15,
    "memory_usage_threshold_percent": 95
  }
}
EOF
    echo "⚠️  Please edit /root/options-scanner/config/health_monitor.json and add your Discord webhook URL"
fi

# Create systemd timer for health checks
echo "Creating systemd timer..."

cat > /etc/systemd/system/health-monitor.service << 'EOF'
[Unit]
Description=Options Scanner Health Monitor
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/options-scanner
Environment="PATH=/root/options-scanner/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/root/options-scanner/venv/bin/python /root/options-scanner/scripts/health_monitor.py
StandardOutput=append:/root/options-scanner/logs/health_monitor.log
StandardError=append:/root/options-scanner/logs/health_monitor.log

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/health-monitor.timer << 'EOF'
[Unit]
Description=Run Health Monitor every 15 minutes
Requires=health-monitor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
AccuracySec=1min

[Install]
WantedBy=timers.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable and start timer
systemctl enable health-monitor.timer
systemctl start health-monitor.timer

# Show status
systemctl status health-monitor.timer --no-pager

echo ""
echo "✓ Health monitor installed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit /root/options-scanner/config/health_monitor.json with your Discord webhook URL"
echo "2. Get webhook URL from Discord: Server Settings > Integrations > Webhooks > New Webhook"
echo "3. Test health monitor: systemctl start health-monitor.service"
echo "4. View logs: tail -f /root/options-scanner/logs/health_monitor.log"
echo ""
echo "Timer will run every 15 minutes automatically."
