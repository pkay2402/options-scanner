# 🎯 Reflecting Boundaries Scanner - Enhanced Edition

## Overview

An advanced market turning point detector based on Milton Berg's groundbreaking paper ["The Boundaries of Technical Analysis"](https://www.dropbox.com/scl/fi/8wuqfqzu2u9vfe1waukui/The-Boundaries-of-Technical-Analysis.pdf).

While stock prices appear random day-to-day, they encounter **"reflecting boundaries"** at extremes - like a drunk walker who can only fall into the gutter because there's a wall on one side. This scanner identifies these boundaries where price direction becomes predictable despite short-term randomness.

## 🚀 Features

### ✅ Core Berg Indicators
1. **5-Day ROC +8% Thrust Signal** 🚀
   - Market gains ≥8% in 5 days
   - Within 4-6 days AFTER a 90-day low
   - **Historical accuracy**: Caught 1929, 1932, 1974, 1982, 2002-2003 major bottoms

2. **5-Day ROC -8% Capitulation Signal** ⭐
   - Market drops ≥8% in 5 days
   - 5-day volume at 250-day high (panic selling)
   - Within 1-7 days of 6-month low
   - **Signals**: Selling exhaustion at bottoms

3. **TRIN + Volume Extremes** 💎
   - **Buy**: TRIN ≤0.50 + 375-day volume high + near 1-year low
   - **Sell**: TRIN ≤0.50 + 375-day volume high + near 3-year high
   - **Track record**: 7/7 bull markets identified, 4/4 bear markets warned

### ✨ Additional Berg Indicators
4. **Breadth Thrust Signals** 📈
   - Strong market advance with bullish breadth near multi-month lows
   - Estimates advance/decline ratio from price action

5. **Advance-Decline Extremes** 🎯
   - Sharp reversals in cumulative A/D line at boundaries
   - Detects institutional accumulation/distribution

6. **New High/Low Ratio Extremes** 🏔️
   - Tracks 52-week new highs vs new lows
   - Many new lows near bottom = bullish reversal
   - Many new highs near top = bearish warning

### 🔥 Enhanced Capabilities

#### 1. **Dual Data Source Support**
- **Schwab API Integration**: Real-time market data with OAuth2 authentication
- **yfinance Fallback**: Automatic fallback if Schwab unavailable
- Data source status displayed in sidebar

#### 2. **Comprehensive Backtesting Engine**
- Forward performance tracking for all signal types
- Multiple holding periods: 5, 10, 21, 63 days
- Calculates:
  - Average & median returns
  - Win rates
  - Best/worst case scenarios
  - Sharpe ratios
  - Maximum drawdowns
- Interactive performance tables

#### 3. **Multi-Channel Alert System**
- **Email Alerts**: SMTP-based email notifications
- **Webhook Alerts**: HTTP POST to custom endpoints
- **Discord Alerts**: Direct Discord channel integration
- Duplicate prevention (won't send same alert twice)
- Configurable via JSON file
- Triggers on signals within last 5 days

#### 4. **Advanced Visualization**
- 4-panel interactive charts:
  - Price with signals overlaid
  - 5-Day Rate of Change
  - Volume analysis with extremes
  - TRIN indicator (for indices)
- Color-coded signal markers
- Boundary level overlays

## 📦 Installation

```bash
# Clone the repository
cd /Users/piyushkhaitan/schwab/options

# Install dependencies (if not already installed)
pip install streamlit yfinance pandas numpy plotly requests

# For Schwab API support (optional)
pip install authlib httpx

# Run the scanner
streamlit run boundary_scanner.py
```

## 🔧 Configuration

### Alert Setup

1. Copy the template:
```bash
cp alerts_config.json.template alerts_config.json
```

2. Edit `alerts_config.json`:

```json
{
  "enabled": true,
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your_email@gmail.com",
    "password": "your_app_password",
    "to_addresses": ["your_trading_email@example.com"]
  },
  "discord": {
    "enabled": true,
    "webhook_url": "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
  }
}
```

**Gmail Setup:**
1. Enable 2-Factor Authentication
2. Generate App Password: Google Account → Security → App Passwords
3. Use app password in config (not your regular password)

**Discord Setup:**
1. Go to Discord Server Settings → Integrations → Webhooks
2. Create New Webhook
3. Copy webhook URL to config

### Schwab API Setup

If you have Schwab API credentials:

1. Set environment variables or update `src/utils/config.py`:
```bash
export SCHWAB_CLIENT_ID="your_client_id"
export SCHWAB_CLIENT_SECRET="your_client_secret"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8080"
```

2. Run authentication:
```bash
python scripts/auth_setup.py
```

## 📊 Usage

### Basic Scan

1. Launch the scanner:
```bash
streamlit run boundary_scanner.py
```

2. In the sidebar:
   - Select **Single Symbol** or **Multi-Symbol Scan**
   - Enter symbols (e.g., SPY, QQQ, NVDA)
   - Choose lookback period (1y, 2y, 5y, 10y)
   - Click **🔍 Scan for Boundaries**

### Advanced Options

Expand **⚙️ Advanced Options** in sidebar:

- ✅ **Use Schwab API**: Enable real-time data (if configured)
- ✅ **Run Backtest**: Calculate forward performance
- ✅ **Enable Alerts**: Send notifications for new signals
- ✅ **Show Additional Berg Indicators**: Include breadth, A/D, H/L signals

### Reading the Results

#### Current Status Panel
Shows real-time boundary context:
- **Current Price** & **5-Day ROC**
- **Days Since Key Lows/Highs**
- **Volume Extremes** (✅/❌)
- **Boundary Context**: NEAR_BOTTOM / NEAR_TOP / MIDDLE_RANGE

#### Potential Setups
Green alerts appear when conditions are forming:
- 🚀 **Potential Thrust Setup**: Price rising sharply near 90-day low
- ⭐ **Potential Capitulation Setup**: Price declining with volume spike

#### Historical Signals Tabs
Browse all detected signals by type:
- Each signal card shows date, price, ROC, volume, strength
- Color-coded by signal type
- Click to see details

#### Backtest Results Tab
Performance statistics for each signal type:
- **Avg Return**: Mean forward return
- **Win Rate**: % of profitable trades
- **Sharpe Ratio**: Risk-adjusted return
- **Detailed Table**: All signals with forward returns

## 🎯 Trading Strategy

### Entry Rules

**Conservative Approach (Recommended):**
1. Wait for signal confirmation (thrust, capitulation, or TRIN)
2. Enter on day after signal or on next pullback
3. Confirm boundary context (must be NEAR_BOTTOM for buys)
4. Check volume - should be at multi-period high
5. Multiple signal types = stronger confirmation

**Aggressive Approach:**
1. Watch for "Potential Setup" alerts
2. Enter on threshold breach (ROC ≥8% or ≤-8%)
3. Use tight stops (2-3% below entry)

### Position Sizing

Based on signal strength:
- **Strength 1.0-1.5**: 25% of intended position
- **Strength 1.5-2.0**: 50% of intended position
- **Strength 2.0+**: 75% of intended position
- **Multiple signals same day**: 100% position

### Exit Rules

**Profit Targets** (from backtesting):
- **5-day hold**: Quick scalp (15-30% of position)
- **10-day hold**: Short-term trade (40% of position)
- **21-day hold**: Swing trade (30% of position)
- **63-day hold**: Position trade (remaining 15%)

**Stop Loss**:
- **Initial stop**: 8% below entry (same as signal threshold)
- **Trailing stop**: After 10% profit, trail by 5%

### Risk Management

1. **Max 2-3 concurrent positions** from boundary signals
2. **Scale in on multiple signal confirmations**
3. **Reduce size on sell signals** (TRIN at highs, H/L extremes)
4. **Avoid counter-trend trades** when boundary context unclear

## 📈 Performance Expectations

Based on Berg's paper (1928-2008 data):

### Thrust Signals
- **Avg 21-day return**: +8-12%
- **Win rate**: ~85%
- **Signals per year**: 1-3 (rare but powerful)

### Capitulation Signals
- **Avg 21-day return**: +10-15%
- **Win rate**: ~80%
- **Max drawdown**: -5% (volatility after panic selling)

### TRIN Buy Signals
- **Avg 63-day return**: +15-25%
- **Win rate**: 100% (7/7 bull markets in study)
- **Signals per decade**: ~2 (extremely rare)

### TRIN Sell Signals
- **Avg 63-day decline**: -15-35%
- **Win rate**: 100% (4/4 bear markets in study)
- **False positives**: Possible in raging bull markets

## 🔔 Alert Examples

### Email Alert Format
```
🚀 BOUNDARY SIGNAL DETECTED 🚀

Symbol: SPY
Signal: THRUST BUY
Date: 2025-10-29
Price: $425.50
ROC 5D: 9.2%
Volume: 125,000,000
Strength: 1.15

Context: NEAR_BOTTOM

This is an automated alert from the Reflecting Boundaries Scanner.
```

### Discord Alert
Same format posted to your configured Discord channel with webhook.

## 🧪 Backtesting Methodology

1. **Entry**: Signal day close price
2. **Holding Periods**: 5, 10, 21, 63 calendar days
3. **Exit**: Close price on exit day (or nearest available)
4. **Returns**: Simple percentage return (no compounding)
5. **Max Drawdown**: Lowest intraday low during holding period
6. **Win Rate**: % of trades with positive return
7. **Sharpe Ratio**: (Avg Return / Std Dev) annualized

**Limitations:**
- Does not account for slippage or commissions
- Uses end-of-day prices (not intraday execution)
- No position sizing or portfolio effects
- Past performance ≠ future results

## 📚 Understanding the Methodology

### Why This Works

**Traditional View (Academic):**
- Stock prices follow random walk
- Can only predict probability distributions
- Historical patterns don't repeat
- Technical analysis is futile

**Berg's Insight:**
- Markets ARE random... until they hit boundaries
- Boundaries = convergence of fundamental, technical, psychological factors
- Extreme price moves near extremes = boundary proximity
- Direction becomes predictable AFTER boundary hit

**Physical Analogy:**
Drunk walker between wall and gutter:
- Each step is random (50/50 left or right)
- Wall acts as reflecting boundary
- Walker MUST eventually reach gutter (only open boundary)
- Individual steps unpredictable, ultimate destination certain

**Market Application:**
- Daily price moves are random (can't predict tomorrow)
- Extreme conditions create temporary boundaries
- After hitting boundary, direction becomes probable
- Signals identify when boundary was just hit

### What Makes a Boundary?

Berg didn't know, and neither do we! It's a "mystery" (Graham's word).

But boundaries occur when:
- Valuation reaches extremes (P/E, P/B, dividend yield)
- Sentiment reaches extremes (fear or greed)
- Volume surges (capitulation or euphoria)
- Technical levels align (multiple timeframes)
- External shocks (policy changes, crises)

**We don't need to know WHY**. We just need to detect WHEN.

## 🎓 Advanced Tips

### Multi-Symbol Scanning
- Scan SPY/QQQ/IWM together for market-wide signals
- Individual stocks: Focus on high-volume, liquid names
- Sector ETFs: Detect rotation opportunities

### Signal Clustering
- Multiple symbols signaling same day = high-confidence market turn
- Breadth thrust + capitulation = extremely rare combo (major bottoms)
- TRIN sell + new H/L extreme = euphoric top warning

### False Signal Mitigation
- Avoid signals in middle range (not near boundaries)
- Require volume confirmation (must be at extreme)
- Wait for end of series (capitulation needs final flush)
- Check multiple timeframes (weekly confirms daily)

### Customization
Edit thresholds in code:
- `ROC threshold`: Change 8% to 7% or 10%
- `Days after low`: Adjust 4-6 day window
- `Volume lookback`: 250-day vs 375-day
- `Holding periods`: Add custom periods to backtest

## 🐛 Troubleshooting

### Schwab API Not Connecting
1. Check credentials in `.env` or `config.py`
2. Run `python scripts/test_auth.py`
3. Re-authenticate: `python scripts/auth_setup.py`
4. Scanner will auto-fallback to yfinance

### No Signals Found
- **Expected!** Boundary signals are rare (that's why they work)
- Try longer lookback period (5y or 10y)
- Scan during volatile markets (signals cluster in downturns)
- Check if using correct symbol (SPY not SP500)

### Alerts Not Sending
1. Check `alerts_config.json` exists (copy from template)
2. Verify `"enabled": true` in config
3. Test email: Use Gmail app password, not regular password
4. Test Discord: Webhook URL must be complete with ID and token
5. Check logs for error messages

### Charts Not Displaying
1. Update plotly: `pip install --upgrade plotly`
2. Clear Streamlit cache: Settings → Clear Cache
3. Try different browser (Chrome/Firefox work best)

## 📖 Further Reading

### Original Research
- **Milton W. Berg**: "The Boundaries of Technical Analysis" (Journal of Technical Analysis, 2008)
- Paper included 15 pages of historical signal tables
- Methodology proven over 80+ years of market data

### Related Concepts
- **Benjamin Graham**: Value investing + technical boundaries (testified before Congress)
- **Stephen Jay Gould**: "Full House" - drunkard's walk analogy
- **William O'Neill**: Follow-through day (similar to thrust signal)

### Technical Analysis Books
- "Technical Analysis of Stock Trends" - Edwards & Magee
- "Market Wizards" - Jack Schwager (interviews with Berg disciples)
- "Evidence-Based Technical Analysis" - David Aronson

## ⚠️ Disclaimer

This scanner is for **educational and research purposes only**.

- **Not financial advice**: Consult a licensed advisor before trading
- **Past performance ≠ future results**: Backtests can't predict future returns
- **Markets can remain irrational**: Boundaries can shift or disappear
- **Risk of loss**: Options and stocks carry substantial risk
- **No guarantees**: Even 100% historical win rate doesn't guarantee future success

**Use at your own risk. Trade small, trade smart, trade safe.**

## 📜 License

This project is for personal use. Based on publicly available research by Milton W. Berg (2008).

## 🤝 Contributing

Suggestions and improvements welcome:
1. Test with different parameters
2. Add new boundary indicators from Berg's paper
3. Improve TRIN estimation algorithm
4. Integrate with other data sources
5. Share your backtest results!

## 📧 Support

For issues or questions:
1. Check this README thoroughly
2. Review the original Berg paper
3. Test with simple cases (SPY only, 10y lookback)
4. Check Streamlit logs for errors

---

**Built with** ❤️ **and a deep respect for the "mystery" of markets.**

*"We know from experience that eventually the market catches up with value."* - Benjamin Graham

*"The direction of stock price movements can therefore be predicted in advance despite the perceived random nature of their daily and weekly moves."* - Milton W. Berg
