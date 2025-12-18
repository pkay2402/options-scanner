#!/bin/bash

# Setup automated token monitoring on the droplet
# This creates a systemd timer that checks token expiration daily

echo "🔧 Setting up Schwab Token Monitoring"
echo "===================================="

# Create the service file
cat > /tmp/schwab-token-monitor.service << 'EOF'
[Unit]
Description=Schwab API Token Expiration Monitor
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/options-scanner
ExecStart=/root/options-scanner/venv/bin/python /root/options-scanner/scripts/token_monitor.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create the timer file (runs twice daily: 9 AM and 9 PM UTC)
cat > /tmp/schwab-token-monitor.timer << 'EOF'
[Unit]
Description=Run Schwab Token Monitor twice daily
Requires=schwab-token-monitor.service

[Timer]
OnCalendar=09:00
OnCalendar=21:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo "📤 Copying files to droplet..."
scp /tmp/schwab-token-monitor.service root@138.197.210.166:/etc/systemd/system/
scp /tmp/schwab-token-monitor.timer root@138.197.210.166:/etc/systemd/system/

echo "📤 Copying monitor script to droplet..."
scp scripts/token_monitor.py root@138.197.210.166:/root/options-scanner/scripts/

echo "⚙️  Enabling and starting timer on droplet..."
ssh root@138.197.210.166 << 'ENDSSH'
# Reload systemd
systemctl daemon-reload

# Enable and start the timer
systemctl enable schwab-token-monitor.timer
systemctl start schwab-token-monitor.timer

# Show timer status
echo ""
echo "Timer Status:"
systemctl status schwab-token-monitor.timer --no-pager

echo ""
echo "Next scheduled runs:"
systemctl list-timers schwab-token-monitor.timer --no-pager
ENDSSH

# Clean up temp files
rm /tmp/schwab-token-monitor.service
rm /tmp/schwab-token-monitor.timer

echo ""
echo "✅ Token monitoring setup complete!"
echo ""
echo "The monitor will run twice daily (9 AM and 9 PM UTC)"
echo "It will log warnings when tokens are about to expire"
echo ""
echo "To manually check token status:"
echo "  ssh root@138.197.210.166 'cd /root/options-scanner && venv/bin/python scripts/token_monitor.py'"
echo ""
echo "To view monitor logs:"
echo "  ssh root@138.197.210.166 'journalctl -u schwab-token-monitor.service'"
