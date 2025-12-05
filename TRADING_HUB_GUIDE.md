# 🎯 Trading Hub - Complete Guide

## Overview
The **Trading Hub** is your unified command center for options trading, combining real-time price action, market positioning, live watchlists, and whale flow detection into a single, powerful dashboard.

---

## 🚀 Key Features

### 1. **Live Price Chart** (Center Panel)
Advanced candlestick chart with multiple technical indicators and option levels:

**Technical Indicators:**
- **VWAP** (Volume Weighted Average Price) - Cyan line showing fair value
- **21 EMA** (Exponential Moving Average) - Orange line for trend direction
- **MACD Crossovers** - Green/Red triangles marking bullish/bearish momentum shifts

**Option Levels Overlay:**
- 🟢 **Call Wall** - Strike with highest call volume (potential resistance)
- 🔴 **Put Wall** - Strike with highest put volume (potential support)
- 🟣 **Flip Level** - Critical strike where put/call volume balance shifts (key pivot point)
- 💎 **Max GEX** - Strike with maximum gamma exposure (likely pin point)

**Timeframe Options:**
- **Intraday**: 5-minute candles, last 2 trading days, market hours only (9:30 AM - 4:00 PM ET)
- **Daily**: Daily candles, last 30 days, clean chart without gaps

---

### 2. **Key Metrics Bar**
Real-time metrics displayed above the chart:

| Metric | Description |
|--------|-------------|
| **Price** | Current stock price with daily % change |
| **Flip Level** | Critical pivot strike where sentiment shifts |
| **Call Wall** | Highest call volume strike (resistance) |
| **Put Wall** | Highest put volume strike (support) |
| **P/C Ratio** | Put/Call volume ratio (>1 = bearish, <1 = bullish) |

---

### 3. **Advanced Analytics** (Below Chart)
Three powerful visualizations side-by-side:

#### 💎 Net GEX (Gamma Exposure)
- **Horizontal bar chart** showing net gamma exposure by strike
- **Green bars** = Positive GEX (dealers long gamma, suppresses volatility)
- **Red bars** = Negative GEX (dealers short gamma, amplifies moves)
- **Interpretation**: Largest bars show where market makers have most exposure

#### 📈 Net Volume Profile
- **Net volume = Put Volume - Call Volume** by strike
- **Red bars** = Put-heavy strikes (bearish positioning)
- **Green bars** = Call-heavy strikes (bullish positioning)
- **Use case**: Identify where traders are positioned

#### 💰 Net Premium Flow Heatmap
- **Call premium - Put premium** across strikes and 4 expiries
- **Green** = Net call buying (bullish flow)
- **Red** = Net put buying (bearish flow)
- **Yellow line** = Current price
- **Use case**: Track institutional money flow across time

---

### 4. **Key Levels Summary Table**
Compact table showing all critical strikes:

| Column | Description |
|--------|-------------|
| **Level** | Type of level (Call Wall, Put Wall, Flip, Max GEX) |
| **Strike** | Exact strike price |
| **Distance** | Percentage distance from current price |
| **Volume/GEX** | Trading volume or gamma exposure amount |

---

### 5. **📊 Live Watchlist** (Left Panel)
Auto-refreshing watchlist of 40+ stocks across all major sectors:

**Coverage:**
- Major Indices: SPY, QQQ, IWM, DIA
- Mega Cap Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- High Growth: PLTR, AMD, CRWD, SNOW, DDOG, NET, PANW
- Semiconductors: TSM, AVGO, QCOM, MU, INTC, ASML, NBIS, OKLO
- AI & Cloud: ORCL, CRM, NOW, ADBE
- Financial: JPM, WFC, GS, MS, V, MA, COIN
- Consumer: NFLX, LOW, COST, WMT, HD
- Healthcare: UNH, JNJ, ABBV, LLY

**Display:**
- ▲/▼ **Daily % Change** - Color-coded (green = up, red = down)
- **Volume** - Trading volume (K = thousands, M = millions)
- **Sorted by % change** - Top movers at the top

**Click any stock** to instantly load its chart and analysis!

**Refresh**: Auto-updates every 120 seconds

---

### 6. **🐋 Whale Flows** (Right Panel)
Real-time detection of large institutional options trades:

**Scanning Coverage:**
- **40+ stocks** across all sectors
- **4 weekly expiries** (next 4 Fridays)
- **160 option chains** scanned simultaneously
- **ATM focus**: Strikes within ±5% of current price

**Whale Score Calculation (VALR Formula):**
```
Whale Score = (Leverage Ratio × IV) × (Volume/OI) × (Dollar Volume Ratio) × 1000

Where:
- Leverage Ratio = (Delta × Stock Price) / Option Premium
- IV = Implied Volatility
- Volume/OI = Volume to Open Interest ratio
- Dollar Volume Ratio = Option $ Volume / Stock $ Volume
```

**Filters:**
- Volume/OI must be > 1.5x (unusual activity)
- Whale Score must be > 100 (significant size)

**Card Display:**
- **Symbol & Type** (CALL/PUT)
- **Whale Score** - Higher = more significant
- **Strike Price** - The option strike
- **Volume** - Contracts traded
- **Vol/OI Ratio** - Unusual activity indicator
- **Premium** - Option price
- **Delta** - Price sensitivity
- **Expiry & DTE** - Expiration date and days to expiry

**Color Coding:**
- 🟢 **Green** = CALL flows (bullish)
- 🔴 **Red** = PUT flows (bearish)

**Refresh**: Auto-updates every 120 seconds

---

### 7. **📊 Market Positioning Summary** (Collapsible)
Located at top of right panel - click to expand:

**Metrics:**
- **Market Bias** - Overall sentiment (Bullish/Bearish/Neutral)
- **Call Volume %** - Percentage of call option volume
- **Put Volume %** - Percentage of put option volume
- **P/C Ratio** - Put/Call ratio
- **Net Premium Flow** - Direction of money flow
- **Distance to Flip** - How far to key pivot level

**Interpretation:**
- **Call Volume > 60%** → Bullish bias
- **Put Volume > 60%** → Bearish bias
- **40-60% range** → Neutral

---

## 🎮 How to Use

### Quick Start
1. **Select Symbol** - Use quick buttons (SPY, QQQ, NVDA, etc.) or type manually
2. **Choose Timeframe** - Intraday (5-min) or Daily
3. **Analyze Chart** - Look for MACD crossovers, VWAP tests, option level reactions
4. **Check Metrics** - Review key strikes and P/C ratio
5. **Watch Whale Flows** - Monitor large institutional trades in real-time
6. **Browse Watchlist** - Click any stock for instant analysis

### Trading Strategies

#### **Trend Following**
- Price above VWAP + above 21 EMA = **Bullish trend**
- Price below VWAP + below 21 EMA = **Bearish trend**
- MACD bullish crossover + green triangle = **Entry signal**

#### **Option Level Trading**
- **Call Wall as resistance** - Price often struggles above
- **Put Wall as support** - Price often bounces here
- **Flip Level** - Break above/below can signal trend change
- **Max GEX** - Price tends to gravitate toward this strike

#### **Whale Flow Following**
- High whale score CALL flows = **Follow institutional bullish bets**
- High whale score PUT flows = **Follow institutional bearish bets**
- Volume/OI > 3x = **Extremely unusual, pay attention**
- Check expiry - **closer DTE = more urgent positioning**

#### **Volume Profile Analysis**
- Heavy put volume below current price = **Support zone**
- Heavy call volume above current price = **Resistance zone**
- Net volume flip = **Key pivot area**

#### **GEX Trading**
- Large positive GEX = **Low volatility expected, range-bound**
- Large negative GEX = **High volatility expected, breakout likely**
- Trade toward max GEX strike = **Market maker hedging creates magnet effect**

---

## ⚡ Performance

**Lightning-Fast Data Loading:**
- **Parallel API Fetching** - ThreadPoolExecutor with 20-30 workers
- **Watchlist**: 1-2 seconds (40 stocks)
- **Whale Flows**: 3-5 seconds (160 option chains)
- **Smart Caching**: 90-second TTL reduces redundant API calls
- **Total Refresh**: ~5 seconds for complete dashboard

**Technology:**
- Concurrent execution via `ThreadPoolExecutor`
- Streamlit fragment-based auto-refresh
- Optimized data processing pipeline

---

## 📱 Quick Reference Card

| Feature | What It Shows | How to Use |
|---------|---------------|------------|
| **VWAP** | Fair value line | Buy below, sell above |
| **21 EMA** | Trend direction | Above = bullish, below = bearish |
| **MACD Triangles** | Momentum shifts | Green = buy signal, red = sell signal |
| **Call Wall** | Resistance level | Expect rejection/consolidation |
| **Put Wall** | Support level | Expect bounce/holding |
| **Flip Level** | Sentiment pivot | Break signals trend change |
| **P/C Ratio** | Market sentiment | >1 bearish, <1 bullish |
| **Net GEX** | Volatility forecast | Positive = low vol, negative = high vol |
| **Whale Flows** | Smart money | Follow high-score trades |
| **Volume Profile** | Position clustering | Red = bearish, green = bullish |

---

## 🔄 Auto-Refresh

Both **Watchlist** and **Whale Flows** automatically refresh every **120 seconds** (2 minutes) to keep data current without manual intervention.

---

## 💡 Pro Tips

1. **Start with the big picture** - Check market positioning first
2. **Use multiple timeframes** - Intraday for entries, daily for trend
3. **Confirm with whale flows** - Smart money often knows first
4. **Watch the flip level** - Most critical strike for intraday moves
5. **GEX + Volume = High conviction** - Both pointing same way = strong signal
6. **Check DTE on whale flows** - Shorter expiry = more immediate catalyst expected
7. **Watchlist for ideas** - Top movers show where action is
8. **Option levels are magnets** - Price often tests and reacts
9. **Premium heatmap shows time decay** - Further expiries = less urgency
10. **Volume spikes matter** - Unusual vol/OI > 3x deserves attention

---

## 🆘 Troubleshooting

**Slow loading?**
- Check internet connection
- Market closed? Less data flows faster
- Clear cache with 🔄 button

**Missing data?**
- Verify market hours (9:30 AM - 4:00 PM ET)
- Some tickers may have limited options data
- Try refreshing with 🔄 button

**Whale flows not showing?**
- May indicate low unusual activity (normal)
- Try broader market symbols (SPY, QQQ)
- Check during high volume periods (open, close)

---

## 📊 Symbol Coverage

**Quick Select Symbols:**
SPY | QQQ | NVDA | TSLA | AAPL | PLTR | META | MSFT | AMZN | GOOGL | AMD | NFLX | CRWD | $SPX

**Full Watchlist:** 40+ stocks covering all major sectors

**Custom Input:** Type any ticker to analyze

---

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**. Options trading carries significant risk. Always:
- Do your own research
- Manage position sizing
- Use stop losses
- Never risk more than you can afford to lose
- Consider consulting a financial advisor

Past performance does not guarantee future results.

---

## 🎯 Perfect For

- **Day Traders** - Intraday levels and MACD signals
- **Options Traders** - Whale flows and GEX analysis
- **Swing Traders** - Daily charts and volume profiles
- **Market Makers** - Understanding dealer positioning
- **Institutional Following** - Tracking smart money flows

---

**Built with:** Streamlit + Python + Schwab API + ThreadPoolExecutor
**Update Frequency:** Real-time with 120-second auto-refresh
**Data Source:** Charles Schwab Market Data API

---

*Happy Trading! 🚀📈*
