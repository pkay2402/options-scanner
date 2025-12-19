#!/bin/bash
# Options Scanner Backend Deployment Script
# Run this on your DigitalOcean droplet

set -e

echo "=================================================="
echo "Options Scanner Backend Deployment"
echo "=================================================="

# Configuration
APP_USER="options"
APP_DIR="/opt/options-scanner"
LOG_DIR="/var/log/options-scanner"
DB_NAME="options_scanner"
DB_USER="options_user"
DB_PASSWORD="$(openssl rand -base64 32)"  # Generate secure password

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    log_error "Please run as root (use sudo)"
    exit 1
fi

# Step 1: Update system
log_info "Updating system packages..."
apt-get update
apt-get upgrade -y

# Step 2: Install dependencies
log_info "Installing dependencies..."
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    postgresql \
    postgresql-contrib \
    nginx \
    git \
    supervisor

# Step 3: Create application user
log_info "Creating application user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d /home/$APP_USER -m $APP_USER
    log_info "User $APP_USER created"
else
    log_warn "User $APP_USER already exists"
fi

# Step 4: Create directories
log_info "Creating application directories..."
mkdir -p $APP_DIR
mkdir -p $LOG_DIR
chown -R $APP_USER:$APP_USER $APP_DIR
chown -R $APP_USER:$APP_USER $LOG_DIR

# Step 5: Setup PostgreSQL
log_info "Setting up PostgreSQL..."
sudo -u postgres psql -c "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME"

sudo -u postgres psql -c "SELECT 1 FROM pg_user WHERE usename = '$DB_USER'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD'"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER"
sudo -u postgres psql -d $DB_NAME -c "GRANT ALL ON SCHEMA public TO $DB_USER"

log_info "Database setup complete"
log_info "Database password: $DB_PASSWORD"
echo "$DB_PASSWORD" > /root/.options_db_password
chmod 600 /root/.options_db_password
log_warn "Database password saved to /root/.options_db_password - KEEP THIS SECURE!"

# Step 6: Clone repository (if not exists)
log_info "Setting up application code..."
if [ ! -d "$APP_DIR/.git" ]; then
    log_info "Cloning repository..."
    # Replace with your actual repo URL
    sudo -u $APP_USER git clone https://github.com/pkay2402/options-scanner.git $APP_DIR
else
    log_info "Repository already exists, pulling latest changes..."
    cd $APP_DIR
    sudo -u $APP_USER git pull
fi

# Step 7: Setup Python virtual environment
log_info "Setting up Python virtual environment..."
cd $APP_DIR
if [ ! -d "venv" ]; then
    sudo -u $APP_USER python3.11 -m venv venv
fi

sudo -u $APP_USER venv/bin/pip install --upgrade pip
sudo -u $APP_USER venv/bin/pip install -r requirements.txt

# Install additional backend dependencies
log_info "Installing backend dependencies..."
sudo -u $APP_USER venv/bin/pip install \
    fastapi \
    uvicorn[standard] \
    psycopg2-binary \
    python-dotenv

# Step 8: Initialize database schema
log_info "Initializing database schema..."
sudo -u postgres psql -d $DB_NAME -f $APP_DIR/backend/database.sql

# Step 9: Configure database connection
log_info "Configuring database connection..."
cat > $APP_DIR/backend/.env << EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
EOF
chown $APP_USER:$APP_USER $APP_DIR/backend/.env
chmod 600 $APP_DIR/backend/.env

# Update scanner_worker.py with actual password
sed -i "s/'your_secure_password'/'$DB_PASSWORD'/" $APP_DIR/backend/scanner_worker.py
sed -i "s/'your_secure_password'/'$DB_PASSWORD'/" $APP_DIR/backend/api_service.py

# Step 10: Copy Schwab credentials
log_info "Setting up Schwab API credentials..."
if [ ! -f "$APP_DIR/schwab_client.json" ]; then
    log_warn "Please copy your schwab_client.json to $APP_DIR/"
    log_warn "The scanner will not work without Schwab API credentials"
else
    chown $APP_USER:$APP_USER $APP_DIR/schwab_client.json
    chmod 600 $APP_DIR/schwab_client.json
fi

# Step 11: Install systemd services
log_info "Installing systemd services..."
cp $APP_DIR/backend/options-scanner.service /etc/systemd/system/
cp $APP_DIR/backend/options-scanner.timer /etc/systemd/system/
cp $APP_DIR/backend/options-api.service /etc/systemd/system/

systemctl daemon-reload

# Step 12: Start services
log_info "Starting services..."

# Start API service
systemctl enable options-api.service
systemctl start options-api.service

# Enable scanner timer (will run on schedule)
systemctl enable options-scanner.timer
systemctl start options-scanner.timer

# Step 13: Configure Nginx reverse proxy
log_info "Configuring Nginx..."
cat > /etc/nginx/sites-available/options-scanner << 'EOF'
server {
    listen 80;
    server_name _;

    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Authorization, Content-Type";
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

ln -sf /etc/nginx/sites-available/options-scanner /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl restart nginx

# Step 14: Setup log rotation
log_info "Setting up log rotation..."
cat > /etc/logrotate.d/options-scanner << EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 $APP_USER $APP_USER
    sharedscripts
    postrotate
        systemctl reload options-api.service > /dev/null 2>&1 || true
    endscript
}
EOF

# Step 15: Run initial scan (optional)
log_info "Running initial scan..."
log_warn "This will take about 2 minutes and use ~176 API calls"
read -p "Run initial scan now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo -u $APP_USER $APP_DIR/venv/bin/python $APP_DIR/backend/scanner_worker.py
    log_info "Initial scan complete"
fi

# Step 16: Display status
echo ""
echo "=================================================="
log_info "Deployment Complete!"
echo "=================================================="
echo ""
echo "Services Status:"
systemctl status options-api.service --no-pager | head -n 5
systemctl status options-scanner.timer --no-pager | head -n 5
echo ""
echo "API Endpoints:"
echo "  Health:              http://$(curl -s ifconfig.me)/health"
echo "  Top Opportunities:   http://$(curl -s ifconfig.me)/api/top-opportunities"
echo "  Market Sentiment:    http://$(curl -s ifconfig.me)/api/market-sentiment"
echo ""
echo "Logs:"
echo "  API:      tail -f $LOG_DIR/api.log"
echo "  Scanner:  tail -f $LOG_DIR/scanner.log"
echo ""
echo "Database:"
echo "  Name:     $DB_NAME"
echo "  User:     $DB_USER"
echo "  Password: $DB_PASSWORD (saved in /root/.options_db_password)"
echo ""
echo "Scanner Schedule:"
echo "  Runs every 15 minutes from 9:45 AM - 4:00 PM ET"
echo "  View schedule: systemctl list-timers options-scanner.timer"
echo ""
echo "Management Commands:"
echo "  Restart API:    systemctl restart options-api.service"
echo "  Run scan now:   sudo -u $APP_USER $APP_DIR/venv/bin/python $APP_DIR/backend/scanner_worker.py"
echo "  View logs:      journalctl -u options-api.service -f"
echo ""
log_warn "IMPORTANT: Copy your schwab_client.json to $APP_DIR/ if not done already"
echo "=================================================="
