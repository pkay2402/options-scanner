# Signal Storage Quick Reference

## Discord Commands

### Check Today's Signals
```
/summarize TSLA
/summarize SPY
/summarize AAPL
```

### Multi-Day Analysis
```
/summarize NVDA 3    # Last 3 days
/summarize QQQ 5     # Last 5 days (max)
```

### View Timeline
```
/timeline MSFT       # Last 5 days by default
/timeline GOOGL 3    # Last 3 days
```

---

## What Each Scanner Stores

| Scanner | Signal Type | When It Triggers | Data Stored |
|---------|------------|------------------|-------------|
| **Whale Flow** | WHALE | Large options trade detected | Strike, expiry, volume, whale_score, IV |
| **Z-Score** | ZSCORE | Crossing ±2σ / Recovery | Z-score, RSI, trend, quality |
| **TOS Alerts** | TOS | Email from ThinkorSwim | Scan name, timeframe |
| **ETF Momentum** | ETF_MOMENTUM | Top 10 leveraged ETFs | Day/week returns, volume |

---

## Signal Directions

- **BULLISH** 🟢 - Calls, oversold reversals, long signals, positive momentum
- **BEARISH** 🔴 - Puts, overbought reversals, short signals, negative momentum
- **NEUTRAL** ⚪ - Mixed or unclear direction

---

## Understanding Summaries

### Overall Sentiment
- **🐂 BULLISH** - More bullish signals than bearish
- **🐻 BEARISH** - More bearish signals than bullish
- **⚖️ NEUTRAL** - Equal or mixed signals

### Signal Quality
- **⭐⭐⭐** - High-quality signal (strong confirmation)
- **⚠️** - Weak signal (wait for confirmation)
- **🚀** - Recovery momentum (bounce from oversold)

### AI Insights
- **Pattern Detection** - Identifies when multiple signals align
- **Whale Activity** - Accumulation vs distribution
- **Trend Analysis** - Consistent vs mixed signals
- **Activity Level** - High/low signal generation

---

## Common Use Cases

### 1. Verify an Alert
**Scenario:** Bot sends whale flow alert for TSLA  
**Action:** `/summarize TSLA`  
**Check:** Are there other signals supporting this? (z-score, TOS, etc.)

### 2. Morning Routine
```
/summarize SPY    # Market pulse
/summarize QQQ    # Tech sentiment
```

### 3. Stock Research
```
/summarize NVDA 5    # Week-long view
/timeline NVDA 5     # Detailed chronology
```

### 4. Confirm Pattern
**Scenario:** Saw multiple AAPL alerts yesterday  
**Action:** `/timeline AAPL 2`  
**Check:** Were signals clustered? Did direction change?

### 5. Post-Trade Review
**Scenario:** Traded TSLA yesterday  
**Action:** `/summarize TSLA 2`  
**Check:** Did signals support the trade? What happened after?

---

## Reading Insights

### Whale Insights
- "Heavy whale activity (X trades)" - X ≥ 3 large trades
- "Whales accumulating" - More calls than puts
- "Whales distributing" - More puts than calls

### Z-Score Insights
- "Entered overbought zone" - Crossed above +2σ
- "Hit oversold levels" - Crossed below -2σ
- "Recovery momentum" - Bouncing from -1.5σ to -2σ with low RSI
- "Reversal signal" - Crossing back from extremes

### Pattern Insights
- "Consistent X signals" - 70%+ of signals in one direction
- "High activity stock" - 3+ signals/day
- "Low activity" - Few signals generated

---

## Tips

### ✅ Do's
- Check summaries before taking trades (confirmation)
- Use timeline to see signal sequencing
- Look for alignment across multiple signal types
- Check 3-5 day view for trends

### ❌ Don'ts
- Don't rely on single signal type
- Don't ignore price action in summary
- Don't trade on low-activity stocks without context
- Don't confuse quantity with quality

---

## Data Retention

- **Stored:** Last 5 days
- **Cleaned:** Automatic daily
- **Indexed:** Fast queries even with 1000s of signals
- **Persistent:** Survives bot restarts

---

## Troubleshooting

### "No signals found"
- Make sure scanners are running
- Check if symbol is in watchlist
- Signals only stored when alerts trigger

### Database errors
- Database auto-creates on first run
- Check bot has write permissions
- Delete `trading_signals.db` to reset

### Slow queries
- Should be fast (indexed)
- If slow, check disk space
- Consider cleanup if database is huge

---

## Example Summary (Annotated)

```
📊 TSLA Trading Summary - Today          ← Date range

Overall Sentiment: 🐂 BULLISH             ← Aggregate direction

💰 Price Action                           ← Price movement today
Current: $245.32
Range: $241.15 - $247.89 (2.8%)

📡 Signal Breakdown                       ← Count by type
🐋 Whale Flow: 2 signals
   └ CALL(2)                              ← Subtypes
📊 Z-Score: 1 signal
   └ BUY_SIGNAL(1)

🕐 Recent Activity                        ← Last 3 signals
🐋 `10:15 AM` WHALE - CALL 🟢
📊 `11:30 AM` ZSCORE - BUY_SIGNAL 🟢
🐋 `02:45 PM` WHALE - CALL 🟢

💡 Key Insights                           ← AI-generated
• 🐋 Whales accumulating - strong buying  ← Pattern detection
• 📊 Hit oversold - bounce opportunity    ← Context
• 🎯 Consistent bullish - strong uptrend  ← Trend analysis

Total Signals: 3 | 3🟢 0🔴 0⚪          ← Summary stats
```

---

## Quick Decision Framework

### High Confidence (Take Trade)
✅ 3+ signals aligned (same direction)  
✅ Multiple signal types agree  
✅ Clear AI insights  
✅ High-quality signals (⭐⭐⭐)

### Medium Confidence (Wait for Confirmation)
⚠️ 1-2 signals  
⚠️ Mixed directions  
⚠️ Low-quality signals (⚠️)  
⚠️ Conflicting insights

### Low Confidence (Skip)
❌ No signals  
❌ Opposite of your thesis  
❌ Low activity stock  
❌ Stale data (5 days old)

---

## API Usage (For Scripting)

```python
from bot.services.signal_storage import get_storage

storage = get_storage()

# Get signals
signals = storage.get_signals('TSLA', days=1)

# Get summary
summary = storage.get_summary('AAPL', days=3)

# Get timeline
timeline = storage.get_stock_activity_timeline('SPY', days=5)
```
