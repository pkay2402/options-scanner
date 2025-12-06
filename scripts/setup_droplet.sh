#!/bin/bash
# Setup script for DigitalOcean Droplet
# Run this on your droplet after SSH'ing in

set -e  # Exit on error

echo "=========================================="
echo "Setting up Options Scanner Worker"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and dependencies
echo "Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3-pip git sqlite3

# Clone repository
echo "Cloning repository..."
cd ~
if [ -d "options-scanner" ]; then
    echo "Repository already exists, pulling latest..."
    cd options-scanner
    git pull
else
    git clone https://github.com/pkay2402/options-scanner.git
    cd options-scanner
fi

# Create virtual environment
echo "Setting up virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
echo ""
echo "=========================================="
echo "IMPORTANT: Setup your .env file"
echo "=========================================="
echo "Creating .env template..."
cat > .env << 'EOL'
# Schwab API Credentials
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1

# Optional: Database path (defaults to data/market_cache.db)
# DATABASE_PATH=data/market_cache.db
EOL

echo ""
echo "Please edit .env with your Schwab API credentials:"
echo "  nano .env"
echo ""
read -p "Press Enter after you've edited .env..."

# Create logs directory
mkdir -p logs
mkdir -p data

# Test worker (one cycle)
echo ""
echo "Testing worker (running one cycle)..."
python scripts/market_data_worker.py &
WORKER_PID=$!
sleep 120  # Let it run for 2 minutes
kill $WORKER_PID 2>/dev/null || true

# Create systemd service
echo ""
echo "Creating systemd service..."
sudo tee /etc/systemd/system/market-worker.service > /dev/null << EOL
[Unit]
Description=Options Scanner Market Data Worker
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/options-scanner
Environment="PATH=$HOME/options-scanner/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$HOME/options-scanner/venv/bin/python $HOME/options-scanner/scripts/market_data_worker.py
Restart=always
RestartSec=10
StandardOutput=append:$HOME/options-scanner/logs/market_worker.log
StandardError=append:$HOME/options-scanner/logs/market_worker_error.log

[Install]
WantedBy=multi-user.target
EOL

# Enable and start service
echo "Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable market-worker.service
sudo systemctl start market-worker.service

# Show status
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Service Status:"
sudo systemctl status market-worker.service --no-pager
echo ""
echo "Useful Commands:"
echo "  Check status:    sudo systemctl status market-worker"
echo "  View logs:       tail -f ~/options-scanner/logs/market_worker.log"
echo "  Restart worker:  sudo systemctl restart market-worker"
echo "  Stop worker:     sudo systemctl stop market-worker"
echo "  Check database:  sqlite3 ~/options-scanner/data/market_cache.db 'SELECT * FROM cache_metadata;'"
echo ""
echo "Database location: ~/options-scanner/data/market_cache.db"
echo "To access from Streamlit, you'll need to expose this via:"
echo "  1. Set up SSH tunnel, OR"
echo "  2. Use cloud storage (S3/Dropbox), OR"
echo "  3. Create a simple API endpoint"
echo ""
