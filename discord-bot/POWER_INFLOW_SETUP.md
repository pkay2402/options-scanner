# Power Inflow Discord Bot Setup Guide

## Overview
The Power Inflow Scanner monitors 20-25 stocks every 3 minutes during market hours and sends alerts to Discord when significant options flow is detected.

## Architecture

### 1. **power_inflow_scanner.py**
Core scanning logic:
- Fetches CBOE options data
- Enriches with Schwab API (bid/ask/OI)
- Detects significant flows
- Tracks reported flows to avoid duplicates
- Formats Discord-friendly messages

### 2. **power_inflow_cmd.py**
Discord bot commands:
- `!flows` - Manual scan
- `!flowstats` - Show statistics
- `!setchannel` - Set alert channel (admin)
- `!startscan` / `!stopscan` - Control auto-scanning (admin)
- Auto-scan every 3 minutes during market hours

### 3. **reported_flows.json** (auto-created)
Tracks reported flows to prevent duplicates:
- Stores flow signatures (hash of symbol/strike/exp/volume)
- Auto-cleans entries older than 24 hours

## Installation

### Step 1: Add to your Discord bot

In your main bot file (e.g., `bot.py` or `main.py`):

```python
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Load the Power Inflow cog
async def load_extensions():
    await bot.load_extension('commands.power_inflow_cmd')

@bot.event
async def on_ready():
    print(f'{bot.user} is ready!')
    await load_extensions()

bot.run('YOUR_DISCORD_TOKEN')
```

### Step 2: Configure watchlist

Edit `power_inflow_scanner.py` line 32-40 to customize symbols:

```python
WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mag 7
    'SPY', 'QQQ', 'IWM',  # ETFs
    # Add your symbols here
]
```

### Step 3: Set alert channel

In Discord, run:
```
!setchannel
```

This sets the current channel for auto-alerts.

## Usage

### Manual Commands

**Scan now:**
```
!flows
```

**Check stats:**
```
!flowstats
```

**Start auto-scanning:**
```
!startscan
```

**Stop auto-scanning:**
```
!stopscan
```

## Alert Format

Example Discord message:
```
🔔 POWER INFLOW ALERTS

🟢 BULLISH **$AAPL** CALL $255 2026-01-16 (12d)
├ Volume: **2,799** | OI: 15,234 | V/OI: 0.18
├ Premium: **$4,567,968** | Side: BUY
└ 📊 ACTIVE

🔴 BEARISH **$NVDA** PUT $185 2026-02-20 (47d)
├ Volume: **11,856** | OI: 4,521 | V/OI: 2.62
├ Premium: **$9,662,640** | Side: BUY
└ 🔥 OPENING
```

## Alert Criteria

Flows are flagged as significant if:
- Volume ≥ 500 contracts AND
- Premium ≥ $100K OR
- Volume ≥ 1,000 contracts OR
- Vol/OI > 2.0 (opening) with clear BUY/SELL direction

## Direction Indicators

- 🟢 **BULLISH** - Call buying or Put selling
- 🔴 **BEARISH** - Put buying
- 🟡 **NEUTRAL** - Call selling (covered calls)
- ⚪ **UNCLEAR** - Mid-price trades

## Position Activity

- 🔥 **OPENING** - Vol/OI > 2.0 (new positions)
- 📊 **ACTIVE** - Normal trading activity
- ❄️ **CLOSING** - Vol/OI < 0.5 (unwinding)

## Market Hours

Auto-scan runs:
- **Time:** 9:30 AM - 4:00 PM ET
- **Days:** Monday - Friday
- **Frequency:** Every 3 minutes

## State Management

The scanner tracks reported flows using MD5 hashes of:
- Symbol + Type + Strike + Expiration + Volume + Price

This prevents duplicate alerts for the same flow. State resets every 24 hours.

## Performance

- **Scan time:** ~30-60 seconds (20-25 stocks)
- **API calls:** 1 CBOE fetch + N Schwab calls (N = # symbols)
- **Rate limits:** Respects Schwab API limits

## Customization

### Adjust significance thresholds

In `power_inflow_scanner.py`, line 173:

```python
significant = stock_df[
    (stock_df['Volume'] >= 500) &  # Minimum volume
    (
        (stock_df['Premium'] >= 100000) |  # $100K+ premium
        (stock_df['Volume'] >= 1000) |  # 1000+ contracts
        ((stock_df['Vol/OI'] > 2.0) & (stock_df['Trade Side'].isin(['BUY', 'SELL'])))
    )
]
```

### Add more symbols

Just add to `WATCHLIST` (line 32-40)

### Change scan frequency

In `power_inflow_cmd.py`, line 26:

```python
@tasks.loop(minutes=3)  # Change to minutes=5 for 5-minute scans
```

## Testing

Run standalone test:
```bash
python3 discord-bot/power_inflow_scanner.py
```

This will:
1. Scan all watchlist symbols
2. Print Discord-formatted messages
3. Show scanner stats

## Troubleshooting

**No alerts:**
- Check market hours
- Verify Schwab API credentials
- Lower significance thresholds
- Check `reported_flows.json` isn't blocking everything

**Duplicate alerts:**
- State file may be corrupted - delete `reported_flows.json`

**API errors:**
- Schwab token may be expired - refresh via SchwabClient
- CBOE may be down - check manually

**Bot not scanning:**
- Check `!flowstats` to see last run
- Restart with `!startscan`
- Verify market hours

## Future Enhancements

Potential additions:
- [ ] Volume vs 20-day average detection
- [ ] Sweep detection (matched vs routed)
- [ ] Greeks analysis (delta, gamma)
- [ ] Strike clustering
- [ ] Sector-wide alerts
- [ ] Web dashboard for historical flows
- [ ] Backtesting framework

## Files Created

```
discord-bot/
├── power_inflow_scanner.py      # Core scanning logic
├── commands/
│   └── power_inflow_cmd.py      # Discord bot commands
└── reported_flows.json          # State tracking (auto-created)
```

## License & Credits

Built on top of Schwab API and CBOE market data.
