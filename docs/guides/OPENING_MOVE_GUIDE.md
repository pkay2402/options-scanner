# Opening Move Alert - Discord Bot Command

## 🎯 What It Does

Automatically scans your watchlist every **15 minutes during market hours** and alerts on the **top 3 trade opportunities** with:

- Call/Put Walls
- Gamma Flip Levels  
- Stock price & momentum
- Options flow analysis (Put/Call ratios)
- Opportunity score (0-100)
- Specific reasons why each is a top play

## 📊 What It Analyzes

For each stock in your watchlist, it calculates:

### 1. **Opportunity Score (0-100)**
Weighted scoring based on:
- **Price Momentum** (25 pts): Movement > 2%
- **Options Activity** (25 pts): Options volume vs stock volume
- **Put/Call Imbalance** (25 pts): Strong directional flow
- **Near Key Levels** (25 pts): Within 3% of walls/flip levels

### 2. **Key Price Levels**
- **Call Wall**: Highest call OI above current price (resistance)
- **Put Wall**: Highest put OI below current price (support)
- **Gamma Flip Level**: Max gamma exposure strike

### 3. **Direction & Flow**
- Put/Call Ratio
- Total call/put volume
- Bullish vs Bearish sentiment

## 🤖 Discord Commands

### Setup (One-time)
```
/setup_opening_move
```
- Sets current channel for alerts
- Shows watchlist preview
- Explains what will be monitored

### Start Auto-Scanning
```
/start_opening_move
```
- Starts 15-minute loop
- Only runs during market hours (9:30 AM - 4:00 PM ET)
- Skips weekends automatically

### Stop Auto-Scanning
```
/stop_opening_move
```
- Stops the scanner
- Can restart anytime

### Manual Test (Instant)
```
/opening_move_now
```
- Runs analysis immediately
- Great for testing during market hours
- See what the auto-alerts will look like

## 📋 Watchlist Source

Reads from: `/user_preferences.json`
- Currently: **72 symbols** (SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, etc.)
- Analyzes top 30 to avoid rate limits
- Automatically updates when you modify the file

## 🕐 Schedule

**Weekdays Only**
- Start: 9:30 AM ET (market open)
- End: 4:00 PM ET (market close)
- Frequency: Every 15 minutes
- Scans per day: ~26 scans

**Total daily scans**: Up to 26 during market hours

## 📨 Alert Format

Each alert includes **up to 3 opportunities**, showing:

```
📊 Opening Move Report
Top 3 Trade Opportunities • 10:15 AM ET

🟢 #1: NVDA - BULLISH
Price: $145.32 (+3.2%)
Score: 87/100
Put/Call Ratio: 0.65

📈 Call Wall: $150.00
📉 Put Wall: $142.00
⚡ Flip Level: $146.50

Why This Setup:
• Strong momentum: +3.2%
• High options activity: 2.3x stock volume
• Bullish flow: PCR 0.65
• Near gamma flip at $146.50
```

## 🔧 Code Structure

**Main file**: `/discord-bot/bot/commands/opening_move.py`

**Key functions**:
- `_scan_top_opportunities()`: Scans watchlist and scores stocks
- `_calculate_opportunity_score()`: Calculates 0-100 score
- `_analyze_volume_walls()`: Finds call/put walls and flip level
- `_create_opportunity_embed()`: Formats Discord message
- `_scanner_loop()`: 15-minute automation loop

## 🚀 Deployment

### 1. Test Locally (No Discord needed)
```bash
cd /Users/piyushkhaitan/schwab/options
python3 test_opening_move_simple.py
```

### 2. Deploy to Discord
```bash
cd discord-bot
python run_bot.py
```

### 3. Setup in Discord
1. Bot joins your server
2. Go to desired channel
3. Run: `/setup_opening_move`
4. Test: `/opening_move_now`
5. Start: `/start_opening_move`

## 📍 Channel Setup

The bot sends to the **same channel as whale_score** alerts.
- Configure which channel in Discord with `/setup_opening_move`
- Can change channels anytime by running setup again

## 🔄 Integration with Existing Pages

This bot leverages data from your Streamlit pages:
- **Trading Dashboard**: Live watchlist
- **Option Volume Walls**: Call/Put wall logic  
- **Stock Option Finder**: Gamma calculations
- **Whale Flows**: Put/Call ratio analysis

## ⚠️ Rate Limiting

- Analyzes max 30 symbols per scan (prevents API throttling)
- 0.5 second delay between symbols
- Takes ~15-20 seconds per full scan
- Graceful error handling if API fails

## 🎨 Customization

Want to adjust the bot? Edit these values in `opening_move.py`:

```python
self.scan_interval_minutes = 15  # Change scan frequency
strike_count=20  # Options strikes to analyze
min_score = 30  # Minimum opportunity score threshold
```

## 💡 Pro Tips

1. **First 15 minutes**: Bot starts at 9:30 AM, first alert at 9:45 AM
2. **Best times**: Most activity between 9:45-10:30 AM and 3:00-4:00 PM
3. **Manual checks**: Use `/opening_move_now` anytime during market
4. **Monitor quality**: Higher scores (>70) = stronger setups
5. **Compare with whale_score**: Look for alignment between tools

## 🐛 Troubleshooting

**No opportunities found?**
- Market may be slow (low volatility days)
- Threshold is 30/100 - very selective
- Try `/opening_move_now` to see current scores

**Bot not sending alerts?**
- Check if scanner is running: look for "started" message
- Verify market hours (weekday, 9:30 AM - 4:00 PM ET)
- Check channel permissions

**API errors?**
- Schwab tokens may need refresh
- Check `discord-bot/.env` credentials
- Bot auto-retries on transient failures

## 📈 Success Metrics

Track your bot's performance:
- Opportunities found per day
- Average score of top picks
- Hit rate on directional calls
- Timing of best alerts (usually morning/close)

## 🔜 Future Enhancements

Potential additions:
- Entry/exit price suggestions
- Stop-loss recommendations  
- Historical performance tracking
- Alert only for score > 80 (super high confidence)
- Integration with your existing newsletter generator

---

**Status**: ✅ Built and tested
**Next**: Deploy to Discord and monitor during market hours
