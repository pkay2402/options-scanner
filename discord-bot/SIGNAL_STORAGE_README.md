# Signal Storage & Summarization System

## Overview
The Discord bot now automatically stores all trading signals in a SQLite database with a 5-day rolling window. Users can query historical signals and get AI-powered summaries.

## Features

### 1. **Automatic Signal Storage**
All scanner commands now automatically store signals:
- **Whale Flow Scanner** - Large options trades (calls/puts)
- **Z-Score Scanner** - Overbought/oversold reversals and recovery momentum
- **TOS Alerts** - ThinkorSwim scan alerts (long/short)
- **ETF Momentum** - High-momentum ETF moves

### 2. **5-Day Rolling Window**
- Signals are stored for 5 days
- Automatic cleanup of old signals
- Efficient querying with indexed timestamps

### 3. **Two New Commands**

#### `/summarize <symbol> [days]`
Get an AI-powered trading summary for any stock.

**Parameters:**
- `symbol` - Stock ticker (e.g., AAPL, TSLA, SPY)
- `days` - Number of days to look back (1-5, default: 1)

**Example Output:**
```
📊 TSLA Trading Summary - Today

Overall Sentiment: 🐂 BULLISH

💰 Price Action
Current: $245.32
Range: $241.15 - $247.89 (2.8%)

📡 Signal Breakdown
🐋 Whale Flow: 3 signals
   └ CALL(2), PUT(1)

📊 Z-Score: 2 signals
   └ BUY_SIGNAL(1), RECOVERY(1)

🎯 TOS Alerts: 1 signal
   └ LONG(1)

🕐 Recent Activity
🐋 `10:15 AM` WHALE - CALL 🟢
📊 `11:30 AM` ZSCORE - BUY_SIGNAL 🟢
🎯 `02:45 PM` TOS - LONG 🟢

💡 Key Insights
• 🐋 Whales are accumulating - strong buying pressure
• 📊 Hit oversold levels - bounce opportunity
• 💫 Recovery momentum detected - bouncing from lows
• 🎯 Consistent bullish signals - strong uptrend

Total Signals: 6 | 5🟢 1🔴 0⚪
```

#### `/timeline <symbol> [days]`
Show a chronological timeline of all activity.

**Parameters:**
- `symbol` - Stock ticker
- `days` - Number of days to look back (1-5, default: 5)

**Example Output:**
```
📅 AAPL Activity Timeline
Last 5 days of trading signals

📆 2025-01-14 (8 signals)
🐋 `09:45` WHALE - CALL
📊 `10:30` ZSCORE - BUY_SIGNAL
🐋 `11:15` WHALE - PUT
📊 `14:20` ZSCORE - RECOVERY
🎯 `15:30` TOS - LONG

📆 2025-01-13 (5 signals)
🐋 `10:00` WHALE - CALL
📊 `12:45` ZSCORE - SELL_SIGNAL
🐋 `14:15` WHALE - CALL
...
```

## Database Schema

### Table: `signals`
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,       -- WHALE, ZSCORE, TOS, ETF_MOMENTUM
    signal_subtype TEXT,              -- CALL, PUT, BUY_SIGNAL, LONG, etc.
    direction TEXT,                   -- BULLISH, BEARISH, NEUTRAL
    price REAL,                       -- Stock price at time of signal
    data TEXT,                        -- JSON with signal-specific data
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### Indexes
- `idx_symbol_time` - Fast lookups by symbol + timestamp
- `idx_timestamp` - Fast cleanup of old signals

## Signal Types

### 1. WHALE Signals
**Direction:** BULLISH (calls) / BEARISH (puts)  
**Subtypes:** CALL, PUT  
**Data:**
```json
{
  "strike": 250.0,
  "expiry": "2025-01-17",
  "volume": 5000,
  "oi": 12500,
  "mark": 3.45,
  "iv": 0.35,
  "whale_score": 12567,
  "notional": 1725000
}
```

### 2. ZSCORE Signals
**Direction:** BULLISH (oversold) / BEARISH (overbought)  
**Subtypes:** BUY_SIGNAL, SELL_SIGNAL, RECOVERY, WARNING  
**Data:**
```json
{
  "zscore": -2.15,
  "signal": "-2σ",
  "quality": "⭐⭐⭐",
  "rsi": 28.5,
  "trend": -8.3,
  "roc5": -12.1,
  "vol_ratio": 1.87
}
```

### 3. TOS Signals
**Direction:** BULLISH (long) / BEARISH (short)  
**Subtypes:** LONG, SHORT  
**Data:**
```json
{
  "scan_name": "HG_30mins_L",
  "timeframe": "30-Min",
  "alert_time": "2025-01-14T10:45:00"
}
```

### 4. ETF_MOMENTUM Signals
**Direction:** Based on returns  
**Subtypes:** STRONG_MOMENTUM, MODERATE_MOMENTUM  
**Data:**
```json
{
  "day_return": 6.8,
  "week_return": 12.3,
  "month_return": 18.9,
  "volume": 45000000,
  "volatility": 32.5,
  "rank": 1
}
```

## AI Insights Generation

The summary command generates intelligent insights by analyzing patterns:

### Whale Activity
- "🐋 Heavy whale activity" - 3+ large trades detected
- "💪 Whales are accumulating" - More buys than sells
- "⚠️ Whales are distributing" - More sells than buys

### Z-Score Patterns
- "📊 Entered overbought zone" - Potential pullback
- "📊 Hit oversold levels" - Bounce opportunity
- "💫 Recovery momentum detected" - Bouncing from lows
- "🔴 Reversal signal: Crossed down from overbought"

### TOS Alerts
- "🎯 Multiple bullish TOS setups forming" - 2+ long signals
- "🎯 Multiple bearish TOS setups forming" - 2+ short signals

### Overall Patterns
- "🎯 Consistent bullish signals" - 70%+ bullish
- "⚡ High activity stock" - 3+ signals/day
- "📉 Low activity" - Few signals generated

## Usage Examples

### Check today's activity for a stock
```
/summarize TSLA
```

### Get 5-day summary with trends
```
/summarize AAPL 5
```

### View complete timeline
```
/timeline SPY 5
```

### Quick check after seeing an alert
```
/summarize NVDA
```
*Shows if this is part of a pattern or an isolated signal*

## Technical Details

### Storage Service
**Location:** `discord-bot/bot/services/signal_storage.py`

**Key Methods:**
- `store_signal()` - Store a new signal
- `get_signals()` - Query signals by symbol/type/days
- `get_summary()` - Get aggregated statistics
- `get_stock_activity_timeline()` - Day-by-day breakdown
- `cleanup_old_signals()` - Remove signals older than 5 days

### Database Location
`discord-bot/bot/services/trading_signals.db`

### Automatic Integration
All scanner commands automatically store signals when they trigger:
- Whale score scanner → stores on each large trade alert
- Z-score scanner → stores on crossing signals
- TOS alerts → stores on email notification
- ETF momentum → stores on scheduled scans

## Benefits

### For Traders
1. **Pattern Recognition** - See if multiple signals align
2. **Context** - Understand if alerts are isolated or part of a trend
3. **Historical View** - Review what happened over past 5 days
4. **Signal Clustering** - Identify when multiple systems agree
5. **Activity Tracking** - Know which stocks are hot

### For Bot
1. **Persistence** - Signals survive bot restarts
2. **Analysis** - Enable more sophisticated pattern detection
3. **Performance** - Fast indexed queries
4. **Scalability** - Handles thousands of signals efficiently

## Future Enhancements

### Potential Features
- [ ] Alert on signal clustering (3+ signals in 1 hour)
- [ ] Daily digest emails with top signals
- [ ] Correlation analysis between signal types
- [ ] Performance tracking (signal → price outcome)
- [ ] Export to CSV/Excel
- [ ] Web dashboard for signal visualization
- [ ] Custom filters (e.g., only whale calls + z-score buy)
- [ ] Leaderboard of most active stocks

## Troubleshooting

### No signals found
- Make sure the scanners are running (`/start_whale_score`, `/start_zscore`, etc.)
- Signals are only stored when alerts trigger
- Check that the symbol is in the scanner watchlists

### Database errors
- Check file permissions on `trading_signals.db`
- Ensure SQLite3 is installed
- Check available disk space

### Performance issues
- Database has indexes for fast queries
- 5-day limit keeps database small
- Automatic cleanup runs on each new signal

## Credits
Built for high-frequency options traders who need quick context and pattern recognition across multiple signal sources.
