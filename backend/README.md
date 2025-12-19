# Options Scanner Backend Service

Real-time options flow analysis with whale flows, fresh positioning, and skew metrics. Designed to run on a DigitalOcean droplet with automated 15-minute scans during market hours.

## Architecture

```
┌─────────────────────────────────────────┐
│         Background Scanner               │
│  - Runs every 15 mins (market hours)    │
│  - 44 stocks × 4 expiries                │
│  - Rate-limited: 120 calls/min          │
│  - 2-batch design (2 min total)         │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         PostgreSQL Database              │
│  - whale_flows                           │
│  - oi_flows                              │
│  - skew_metrics                          │
│  - Composite scoring views               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         FastAPI REST Service             │
│  - /api/top-opportunities                │
│  - /api/market-sentiment                 │
│  - /api/stock/{symbol}                   │
│  - /api/historical/{symbol}              │
└─────────────────────────────────────────┘
```

## Features

### 1. **Rate-Limited Scanner**
- Scans 44 stocks (tech + value) across 4 weekly expiries
- Respects Schwab API limit of 120 calls/minute
- Two-batch design: 22 stocks per batch, 1 minute apart
- Total scan time: ~2 minutes

### 2. **Composite Scoring (0-100)**
Combines three metrics for trading edge:
- **Whale Score (0-35 points)**: Institutional positioning using VALR formula
- **Fresh OI (0-35 points)**: New capital deployment (Vol/OI ≥ 3.0x)
- **Skew Alignment (0-30 points)**: Contrarian signals from put-call skew

### 3. **Signal Classification**
- `CONTRARIAN_BULL`: Extreme fear + call buying (bottom signal)
- `CONTRARIAN_BEAR`: Greed + put buying (top signal)
- `MOMENTUM_BULL`: Sustained call volume (trend following)
- `MOMENTUM_BEAR`: Sustained put volume (trend following)
- `NEUTRAL`: Mixed signals

### 4. **Historical Tracking**
- 30 days of data retention
- Track skew changes over time
- Identify accelerating fear/greed
- Pattern recognition opportunities

## Deployment

### Quick Deploy (DigitalOcean Droplet)

```bash
# 1. SSH into your droplet
ssh root@your-droplet-ip

# 2. Clone repository
git clone https://github.com/pkay2402/options-scanner.git
cd options-scanner

# 3. Run deployment script
chmod +x backend/deploy.sh
sudo ./backend/deploy.sh

# 4. Copy Schwab credentials
scp schwab_client.json root@your-droplet-ip:/opt/options-scanner/

# 5. Verify services
systemctl status options-api.service
systemctl status options-scanner.timer
```

The deployment script automatically:
- Installs PostgreSQL, Python, Nginx
- Creates database and user
- Sets up systemd services
- Configures automated scanning
- Starts API service on port 8000

### Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv postgresql nginx

# Create database
sudo -u postgres createdb options_scanner
sudo -u postgres createuser options_user
sudo -u postgres psql -c "ALTER USER options_user WITH PASSWORD 'your_password'"

# Initialize schema
sudo -u postgres psql -d options_scanner -f backend/database.sql

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Update database credentials
# Edit backend/scanner_worker.py and backend/api_service.py
# Change DB_CONFIG password

# Start services
uvicorn backend.api_service:app --host 0.0.0.0 --port 8000 &
python backend/scanner_worker.py &
```

## API Endpoints

### Get Top Opportunities
```bash
curl http://your-droplet-ip/api/top-opportunities?limit=20&min_composite_score=60
```

Returns ranked trading opportunities with composite scores.

### Market Sentiment
```bash
curl http://your-droplet-ip/api/market-sentiment
```

Overall market fear/greed based on options skew.

### Stock Analysis
```bash
curl http://your-droplet-ip/api/stock/NVDA
```

Complete analysis for specific stock (whale flows, OI, skew).

### Historical Skew
```bash
curl http://your-droplet-ip/api/historical/AAPL?hours=24
```

Track skew changes over time.

### Scan Status
```bash
curl http://your-droplet-ip/api/scan-status
```

View recent scan runs and performance.

## Configuration

### Scanner Schedule

Runs every 15 minutes during market hours (9:45 AM - 4:00 PM ET):
- 9:45 AM - Initial scan
- 10:00 AM, 10:15 AM, 10:30 AM, etc.
- Last scan at 4:00 PM

To modify schedule:
```bash
sudo nano /etc/systemd/system/options-scanner.timer
sudo systemctl daemon-reload
sudo systemctl restart options-scanner.timer
```

### Stock Lists

Edit `scanner_worker.py`:
```python
TECH_STOCKS = ['AAPL', 'MSFT', ...]  # 22 stocks
VALUE_STOCKS = ['AXP', 'JPM', ...]   # 22 stocks
```

### Rate Limiting

Schwab API: 120 calls/minute

Current design:
- Batch 1: 22 stocks × 2 calls = 44 calls
- Batch 2: 22 stocks × 2 calls = 44 calls  
- Each stock uses 2 API calls (1 quote + 1 options chain per expiry)
- Total per expiry: 88 calls (73% of limit)
- With 4 expiries: 176 calls total over 2 minutes

To adjust:
```python
rate_limiter = RateLimiter(max_calls_per_minute=115)  # Leave 5-call buffer
```

## Database Schema

### Core Tables

**whale_flows**: High VALR scoring options
- Whale score, volume, OI, gamma, GEX
- Call/put walls, volume ratios

**oi_flows**: Fresh positioning (Vol/OI ≥ 3.0)
- OI score, Vol/OI ratio, notional value

**skew_metrics**: Put-call skew analysis
- 25-delta skew, ATM skew, IV levels
- Implied move, breakout levels
- Put/call ratios

**scan_runs**: Scan metadata
- Timing, status, API call counts

### Views

**top_opportunities**: Composite scoring across all metrics
**market_sentiment**: Aggregate market fear/greed
**stock_skew_history**: Historical skew tracking

## Monitoring

### View Logs
```bash
# API logs
tail -f /var/log/options-scanner/api.log

# Scanner logs
tail -f /var/log/options-scanner/scanner.log

# Systemd logs
journalctl -u options-api.service -f
journalctl -u options-scanner.timer -f
```

### Service Management
```bash
# Restart API
sudo systemctl restart options-api.service

# Run scan manually
sudo -u options /opt/options-scanner/venv/bin/python /opt/options-scanner/backend/scanner_worker.py

# Check timer schedule
systemctl list-timers options-scanner.timer

# View service status
systemctl status options-api.service
systemctl status options-scanner.timer
```

### Database Queries
```bash
# Connect to database
sudo -u postgres psql -d options_scanner

# View latest scan
SELECT * FROM scan_runs ORDER BY start_time DESC LIMIT 1;

# Check data freshness
SELECT symbol, MAX(timestamp) as last_update 
FROM whale_flows 
GROUP BY symbol 
ORDER BY last_update DESC;

# Top opportunities
SELECT * FROM top_opportunities LIMIT 10;
```

## Performance

### Expected Metrics
- **Scan Duration**: ~2 minutes (2 batches)
- **API Calls**: 176 calls per complete scan
- **Database Size**: ~50MB per day (30-day retention)
- **API Response**: <100ms (cached queries)

### Optimization

For faster scans (if you have higher API limits):
```python
# Increase rate limit
rate_limiter = RateLimiter(max_calls_per_minute=200)

# Process all stocks in one batch
scan_batch(client, ALL_STOCKS, expiries, rate_limiter, scan_id, timestamp)
```

## Troubleshooting

### Scanner Not Running
```bash
# Check timer is active
systemctl status options-scanner.timer

# Check last trigger
journalctl -u options-scanner.service | tail -50

# Run manually to test
sudo -u options /opt/options-scanner/venv/bin/python /opt/options-scanner/backend/scanner_worker.py
```

### API Not Responding
```bash
# Check service status
systemctl status options-api.service

# Check nginx
sudo nginx -t
sudo systemctl restart nginx

# Test locally
curl http://localhost:8000/
```

### Database Issues
```bash
# Check PostgreSQL
sudo systemctl status postgresql

# Check connections
sudo -u postgres psql -d options_scanner -c "SELECT * FROM pg_stat_activity;"

# Verify schema
sudo -u postgres psql -d options_scanner -c "\dt"
```

### Rate Limit Errors
If you hit rate limits:
- Reduce batch size (scan fewer stocks per batch)
- Increase delay between batches
- Check rate_limiter configuration

## Security

### Recommendations
1. **Change default database password**
2. **Enable UFW firewall**:
   ```bash
   sudo ufw allow 22
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw enable
   ```
3. **Use SSL certificate** (Let's Encrypt):
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx
   ```
4. **Restrict API access** (edit nginx config)
5. **Secure Schwab credentials**:
   ```bash
   chmod 600 /opt/options-scanner/schwab_client.json
   ```

## Backup

### Database Backup
```bash
# Manual backup
sudo -u postgres pg_dump options_scanner > backup.sql

# Automated daily backup (cron)
0 2 * * * sudo -u postgres pg_dump options_scanner | gzip > /backup/options_scanner_$(date +\%Y\%m\%d).sql.gz
```

### Restore
```bash
sudo -u postgres psql options_scanner < backup.sql
```

## Cost Estimate

### DigitalOcean Droplet
- **Recommended**: 4GB RAM, 2 vCPUs ($24/mo)
- **Minimum**: 2GB RAM, 1 vCPU ($12/mo)

### Schwab API
- Free (120 calls/minute limit)
- Our usage: ~4,500 calls/day (well under limit)

### Total: $12-24/month

## Future Enhancements

1. **Machine Learning**: Train model on successful setups
2. **Alerts**: Discord/Telegram notifications for high scores
3. **Backtesting**: Track composite score accuracy
4. **More Markets**: Add futures, forex options
5. **Real-time Updates**: WebSocket streaming
6. **Custom Watchlists**: User-defined stock lists

## Support

For issues or questions:
- GitHub Issues: https://github.com/pkay2402/options-scanner/issues
- Email: your-email@example.com

## License

MIT License - See LICENSE file for details
