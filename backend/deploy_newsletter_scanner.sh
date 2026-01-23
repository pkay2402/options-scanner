#!/bin/bash
# Deploy Newsletter Scanner Service to Droplet
# Run this on the droplet as root

set -e

echo "=== Newsletter Scanner Service Deployment ==="
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p /var/log/newsletter-scanner
mkdir -p /root/options/data

# Copy service files
echo "📋 Installing systemd service files..."
cp /root/options/backend/newsletter-scanner.service /etc/systemd/system/
cp /root/options/backend/newsletter-scanner.timer /etc/systemd/system/

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable and start timer
echo "⏰ Enabling scanner timer..."
systemctl enable newsletter-scanner.timer
systemctl start newsletter-scanner.timer

# Run initial scan
echo "🚀 Running initial scan..."
systemctl start newsletter-scanner.service

# Check status
echo ""
echo "=== Service Status ==="
systemctl status newsletter-scanner.timer --no-pager
echo ""
echo "=== Next Scheduled Runs ==="
systemctl list-timers newsletter-scanner.timer --no-pager

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Useful commands:"
echo "  View logs:           journalctl -u newsletter-scanner -f"
echo "  Run manual scan:     systemctl start newsletter-scanner.service"
echo "  Check timer status:  systemctl status newsletter-scanner.timer"
echo "  View scan log:       tail -f /var/log/newsletter-scanner/scanner.log"
