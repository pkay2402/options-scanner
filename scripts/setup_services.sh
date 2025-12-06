#!/bin/bash
# Setup and install systemd services for Options Scanner

echo "=== Options Scanner Service Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Get the actual user's home directory (not root)
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
else
    USER_HOME="/root"
fi

PROJECT_DIR="$USER_HOME/options-scanner"

# Verify project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo ""

# Create logs directory
mkdir -p $PROJECT_DIR/logs
echo "✓ Created logs directory"

# Copy service files to systemd
echo ""
echo "Installing systemd services..."
cp $PROJECT_DIR/scripts/api-server.service /etc/systemd/system/
cp $PROJECT_DIR/scripts/market-data-worker.service /etc/systemd/system/
echo "✓ Service files copied to /etc/systemd/system/"

# Reload systemd
systemctl daemon-reload
echo "✓ Systemd daemon reloaded"

# Enable services
systemctl enable api-server.service
systemctl enable market-data-worker.service
echo "✓ Services enabled"

# Start services
echo ""
echo "Starting services..."
systemctl start api-server.service
systemctl start market-data-worker.service
echo "✓ Services started"

# Show status
echo ""
echo "=== Service Status ==="
systemctl status api-server.service --no-pager -l
echo ""
systemctl status market-data-worker.service --no-pager -l

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status api-server"
echo "  sudo systemctl status market-data-worker"
echo "  sudo systemctl restart api-server"
echo "  sudo systemctl restart market-data-worker"
echo "  sudo journalctl -u api-server -f"
echo "  sudo journalctl -u market-data-worker -f"
echo "  tail -f $PROJECT_DIR/logs/api_server.log"
echo "  tail -f $PROJECT_DIR/logs/market_data_worker.log"
