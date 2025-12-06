# Options Scanner Droplet Setup Guide

## Quick Start

### 1. SSH into your droplet
```bash
ssh root@138.197.210.166
# (use the IP from your DigitalOcean dashboard)
```

### 2. Run the setup script
```bash
# Download and run setup
curl -sSL https://raw.githubusercontent.com/pkay2402/options-scanner/main/scripts/setup_droplet.sh | bash
```

**OR manually:**

```bash
# Clone the repo
git clone https://github.com/pkay2402/options-scanner.git
cd options-scanner

# Make setup script executable
chmod +x scripts/setup_droplet.sh

# Run setup
./scripts/setup_droplet.sh
```

### 3. Edit .env with your Schwab credentials
```bash
nano .env
```

Add:
```env
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### 4. Verify worker is running
```bash
# Check service status
sudo systemctl status market-worker

# View live logs
tail -f logs/market_worker.log

# Check database
sqlite3 data/market_cache.db "SELECT COUNT(*) FROM watchlist;"
```

## Connecting Streamlit to Droplet Database

### Option A: SSH Tunnel (Recommended for testing)
On your local machine:
```bash
# Create SSH tunnel to expose database
ssh -L 5432:localhost:5432 root@138.197.210.166

# In another terminal, test connection
# Your Streamlit app can now access the DB at localhost:5432
```

### Option B: Cloud Storage Sync (Recommended for production)
Set up automatic sync to S3 or Dropbox:

```bash
# Install rclone on droplet
curl https://rclone.org/install.sh | sudo bash

# Configure Dropbox/S3
rclone config

# Add to crontab (sync every 5 minutes)
crontab -e
# Add: */5 * * * * rclone copy ~/options-scanner/data/market_cache.db remote:options-scanner/
```

### Option C: Simple HTTP API (Best for Streamlit Cloud)
Create a simple Flask API on the droplet to serve data:

```python
# scripts/api_server.py
from flask import Flask, jsonify
from src.data.market_cache import MarketCache

app = Flask(__name__)
cache = MarketCache()

@app.route('/watchlist')
def get_watchlist():
    return jsonify(cache.get_watchlist())

@app.route('/whale_flows')
def get_whale_flows():
    sort_by = request.args.get('sort', 'score')
    return jsonify(cache.get_whale_flows(sort_by=sort_by))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

Then in Streamlit:
```python
import requests
watchlist = requests.get('http://138.197.210.166:8000/watchlist').json()
```

## Management Commands

### Service Control
```bash
# Start worker
sudo systemctl start market-worker

# Stop worker
sudo systemctl stop market-worker

# Restart worker
sudo systemctl restart market-worker

# View status
sudo systemctl status market-worker

# Enable auto-start on boot
sudo systemctl enable market-worker

# Disable auto-start
sudo systemctl disable market-worker
```

### Logs
```bash
# View worker logs
tail -f ~/options-scanner/logs/market_worker.log

# View error logs
tail -f ~/options-scanner/logs/market_worker_error.log

# View last 100 lines
tail -n 100 ~/options-scanner/logs/market_worker.log

# Follow systemd journal
sudo journalctl -u market-worker -f
```

### Database
```bash
# Check database size
du -h ~/options-scanner/data/market_cache.db

# Query watchlist count
sqlite3 ~/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM watchlist;"

# Query whale flows count
sqlite3 ~/options-scanner/data/market_cache.db "SELECT COUNT(*) FROM whale_flows;"

# View cache metadata
sqlite3 ~/options-scanner/data/market_cache.db "SELECT * FROM cache_metadata;"

# View latest whale flows
sqlite3 ~/options-scanner/data/market_cache.db "SELECT symbol, type, strike, whale_score FROM whale_flows ORDER BY detected_at DESC LIMIT 10;"

# Check last update times
sqlite3 ~/options-scanner/data/market_cache.db "SELECT key, value, datetime(updated_at, 'localtime') as updated FROM cache_metadata;"
```

### Updates
```bash
# Pull latest code
cd ~/options-scanner
git pull

# Restart worker to apply changes
sudo systemctl restart market-worker
```

## Troubleshooting

### Worker not starting
```bash
# Check logs
sudo journalctl -u market-worker -n 50

# Check if port is already in use
sudo lsof -i :8000

# Manually test worker
cd ~/options-scanner
source venv/bin/activate
python scripts/market_data_worker.py
```

### Authentication issues
```bash
# Verify .env file
cat ~/options-scanner/.env

# Check Schwab API credentials
# Make sure APP_KEY and APP_SECRET are correct
```

### Database issues
```bash
# Backup database
cp ~/options-scanner/data/market_cache.db ~/market_cache_backup.db

# Reset database (removes all data)
rm ~/options-scanner/data/market_cache.db
sudo systemctl restart market-worker
```

### Memory issues
```bash
# Check memory usage
free -h

# Check worker memory
ps aux | grep market_data_worker

# If OOM, add swap space
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Security Recommendations

### 1. Create non-root user
```bash
adduser trader
usermod -aG sudo trader
su - trader
```

### 2. Setup firewall
```bash
sudo ufw allow OpenSSH
sudo ufw enable
```

### 3. Secure .env file
```bash
chmod 600 ~/options-scanner/.env
```

### 4. Regular updates
```bash
# Setup unattended upgrades
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## Monitoring

### Setup email alerts for failures
```bash
# Install mailutils
sudo apt install mailutils

# Edit service to send email on failure
sudo nano /etc/systemd/system/market-worker.service
# Add: OnFailure=status-email@%n.service
```

### Basic monitoring script
```bash
# Create monitor script
cat > ~/monitor.sh << 'EOF'
#!/bin/bash
if ! systemctl is-active --quiet market-worker; then
    echo "Worker is down!" | mail -s "Market Worker Alert" your@email.com
fi
EOF

chmod +x ~/monitor.sh

# Add to crontab (check every hour)
crontab -e
# Add: 0 * * * * ~/monitor.sh
```

## Cost Optimization

Your $6/month droplet runs 24/7, but the market is only open ~6.5 hours/day:
- Market hours: 9:30 AM - 4:00 PM ET (M-F)
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

Consider scheduling the worker to only run during market hours:

```bash
# Stop worker from running as a service
sudo systemctl disable market-worker
sudo systemctl stop market-worker

# Use cron instead (only during market hours)
crontab -e

# Add (runs every 5 minutes from 4 AM to 8 PM ET on weekdays):
*/5 4-20 * * 1-5 cd ~/options-scanner && venv/bin/python scripts/market_data_worker.py >> logs/cron.log 2>&1
```

This won't save money (droplet still charges $6/month) but reduces API usage.
