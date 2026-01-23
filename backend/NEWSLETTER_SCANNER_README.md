# AI Newsletter Scanner Service

## Overview

The Newsletter Scanner is a background service that runs every 30 minutes during market hours to:

1. **Scan all theme stocks** for technical setups
2. **Collect options flow data** (if Schwab API available)
3. **Calculate opportunity scores** using multiple factors
4. **Store historical data** in SQLite database
5. **Feed data to AI Copilot** for trade recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DROPLET SERVICE                            │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Newsletter      │    │ SQLite DB       │                    │
│  │ Scanner         │───▶│ newsletter_     │                    │
│  │ (Every 30 min)  │    │ scanner.db      │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                                 │ (Same DB file)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT CLOUD                            │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ AI Copilot      │◀───│ Read Scanner    │                    │
│  │ (LLM Analysis)  │    │ Data            │                    │
│  └─────────────────┘    └─────────────────┘                    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │ Trade           │                                           │
│  │ Recommendations │                                           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scoring Improvements

The enhanced scanner includes these scoring factors:

### Technical Score (100 points)
- **Trend Strength (25 pts)**: Price vs SMA20/50/200 alignment
- **RSI Analysis (20 pts)**: Momentum with oversold/overbought detection
- **Multi-timeframe Momentum (20 pts)**: Week/Month/Quarter returns
- **Volume Analysis (15 pts)**: Volume surge detection
- **Volatility Setup (10 pts)**: Compression = breakout potential
- **Breakout Proximity (10 pts)**: Distance from 52-week high

### Options Enhancement (Bonus points)
- **Options Flow Confluence**: +10 pts when options sentiment matches setup
- **Earnings Catalyst**: +5 pts if earnings within 14 days

### Setup Classification
- **BULLISH**: Strong uptrend + positive momentum + bullish options
- **BEARISH**: Downtrend + negative momentum + bearish options  
- **NEUTRAL**: Mixed signals

### Timeframe Assignment
- **WEEKLY**: High volatility compression, strong weekly momentum
- **MONTHLY**: Strong score with moderate setup
- **QUARTERLY**: Long-term positions

## Database Schema

```sql
-- Full scan history
CREATE TABLE scans (
    id INTEGER PRIMARY KEY,
    scan_time TIMESTAMP,
    ticker TEXT,
    theme TEXT,
    opportunity_score REAL,
    technical_score REAL,
    current_price REAL,
    week_return REAL,
    month_return REAL,
    rsi REAL,
    volume_ratio REAL,
    put_call_ratio REAL,
    options_sentiment TEXT,
    iv_rank REAL,
    gamma_support REAL,
    gamma_resistance REAL,
    setup_type TEXT,    -- BULLISH/BEARISH/NEUTRAL
    timeframe TEXT,     -- WEEKLY/MONTHLY/QUARTERLY
    has_earnings_soon INTEGER,
    ...
);

-- Latest scores (quick lookup)
CREATE TABLE latest_scores (
    ticker TEXT PRIMARY KEY,
    opportunity_score REAL,
    setup_type TEXT,
    timeframe TEXT,
    last_updated TIMESTAMP
);
```

## AI Copilot Integration

The AI Copilot pulls data via these methods:

```python
# Get all bullish setups with score >= 70
copilot.get_scanner_data("bullish", 70)

# Get bearish setups for put plays
copilot.get_scanner_data("bearish", 60)

# Get weekly options plays
copilot.get_scanner_data("weekly")

# Get monthly swing trades
copilot.get_scanner_data("monthly")

# Get stocks with improving scores
copilot.get_improving_from_scanner()

# Generate full trade recommendations (uses LLM)
copilot.get_ai_trade_recommendations()
```

## Deployment

### On Droplet

```bash
# SSH to droplet
ssh root@your-droplet-ip

# Navigate to options folder
cd /root/options

# Pull latest code
git pull

# Run deployment script
chmod +x backend/deploy_newsletter_scanner.sh
./backend/deploy_newsletter_scanner.sh
```

### Service Commands

```bash
# Check timer status
systemctl status newsletter-scanner.timer

# View upcoming runs
systemctl list-timers newsletter-scanner.timer

# Manual scan
systemctl start newsletter-scanner.service

# View logs
journalctl -u newsletter-scanner -f

# View scanner log file
tail -f /var/log/newsletter-scanner/scanner.log
```

## Trade Recommendation Flow

1. **Scanner runs every 30 minutes** during market hours (9:30 AM - 4:00 PM ET)
2. **Stores results** in SQLite database
3. **User clicks "Get Trade Recommendations"** in AI Copilot
4. **Copilot pulls scanner data** (bullish, bearish, weekly, monthly, improving)
5. **LLM synthesizes** data into specific trade recommendations
6. **User gets**: 
   - Top 3 weekly plays with strikes/expiries
   - Top 3 monthly swings with entry/target/stop
   - Bearish setups for puts
   - Avoid list
   - Overall market bias

## Data Freshness

- Scanner runs: Every 30 minutes during market hours
- Data staleness: Max 30 minutes during trading
- Overnight: Last scan from previous day's close
- Weekends: Friday's close data

## Example Output

```
🎯 AI Trade Recommendations

🔥 TOP 3 WEEKLY PLAYS (0-5 day holds)

1. NVDA - BULLISH CALL
   - Score: 85, RSI: 62, Options: BULLISH
   - Entry: $140-142 calls, Feb 2 expiry
   - Target: $148 resistance
   - Stop: Close below $138
   - Why: Strong uptrend, volume surge, bullish options flow

2. AMD - BULLISH CALL
   - Score: 78, RSI: 55, Options: BULLISH
   - Entry: $175 calls, Feb 2 expiry
   ...

📅 TOP 3 MONTHLY SWING TRADES

1. META - Long shares or LEAPS
   - Score: 82, Earnings in 6 days
   ...

🐻 BEARISH PLAYS

1. XYZ - PUT spread candidate
   - Score: 65, RSI: 72 (overbought)
   ...
```

## Maintenance

### Logs
- Service logs: `/var/log/newsletter-scanner/scanner.log`
- Systemd logs: `journalctl -u newsletter-scanner`

### Database
- Location: `/root/options/data/newsletter_scanner.db`
- Backup: Copy file periodically
- Size management: Old scans auto-cleaned (30 days)

### Troubleshooting

**Timer not running:**
```bash
systemctl status newsletter-scanner.timer
systemctl restart newsletter-scanner.timer
```

**Scan failing:**
```bash
# Check logs
journalctl -u newsletter-scanner -n 50

# Run manually to see errors
/root/options/.venv/bin/python /root/options/backend/newsletter_scanner.py
```

**Database locked:**
```bash
# Check for running processes
fuser /root/options/data/newsletter_scanner.db

# Restart service
systemctl restart newsletter-scanner.service
```
