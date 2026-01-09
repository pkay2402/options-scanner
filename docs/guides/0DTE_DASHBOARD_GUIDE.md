# 0DTE Trading Dashboard Guide

## Overview
The redesigned Index Positioning page is now a professional **0DTE (Zero Days to Expiration) Trading Dashboard** specifically optimized for same-day expiration options trading on SPY, SPX, and QQQ.

## 🚀 Key Features

### 1. **Real-Time Session Tracking**
- Live countdown timer to market close/expiration
- Market session indicator (Pre-Market, Opening Range, Mid-Day, Power Hour, After Hours)
- Visual progress bar showing how much of the trading day has elapsed
- Eastern Time (ET) display for precise timing

### 2. **Hero Metrics Section**
Four critical levels displayed prominently:
- **Current Price** - Real-time underlying price
- **Zero Gamma Level** - The flip point where dealer behavior changes
- **Max Pain** - Strike with maximum option seller pain (magnetic level)
- **Expected Move** - Statistical 1σ intraday range based on ATM IV

### 3. **Gamma Positioning Analysis**
- **Net Gamma Exposure** - Total market gamma (positive = volatility suppression, negative = volatility amplification)
- **Dealer Position** - Whether dealers are long/short gamma
- **Volatility Regime** - Clear indication of expected price action behavior

### 4. **Flow Sentiment Analysis**
- Volume-weighted Put/Call ratios
- Real-time options flow direction (bullish/bearish/neutral)
- Call vs Put volume comparison
- Open Interest analysis

### 5. **Critical Price Levels**
Three categories of key levels:
- **Top 5 Gamma Walls** - Strongest magnetic levels sorted by GEX
- **Put Walls (Support)** - High OI put strikes below current price
- **Call Walls (Resistance)** - High OI call strikes above current price

Each level shows:
- Strike price
- Distance from current price (%)
- Gamma exposure / Open Interest
- Visual color coding

### 6. **Strategy Recommendations**
AI-powered strategy suggestions based on:
- Current gamma regime (long/short)
- Flow sentiment (bullish/bearish/neutral)
- Time to expiration (theta decay rate)
- Position relative to zero gamma and max pain

Recommended strategies include:
- Directional 0DTE calls/puts
- Iron Condors at zero gamma
- Max Pain straddles
- Theta harvesting opportunities

### 7. **Visual Analytics**
- **Net Gamma Profile Chart** - Shows gamma concentration by strike
- **Open Interest Distribution** - Separate call/put OI visualization
- Current price overlay on all charts

### 8. **Detailed Metrics**
- Total Call/Put Open Interest
- P/C Ratio with sentiment indicator
- Net Delta Exposure
- Strike-by-strike breakdown with Greeks

### 9. **Advanced Tables (Expandable)**
Detailed strike data including:
- Strike, OI, Volume
- Notional Gamma (in dollars)
- Delta, Gamma, Vega
- Implied Volatility
- Distance from current price

## 🎯 Trading Use Cases

### Morning Trading (Opening Range)
1. Check overnight gaps and expected move
2. Identify key gamma walls that might act as support/resistance
3. Monitor flow sentiment for directional bias
4. Watch for pin at zero gamma level

### Mid-Day Trading
1. Track price action relative to max pain
2. Monitor changes in gamma positioning
3. Look for breakdown/breakthrough of gamma walls
4. Assess theta decay impact

### Power Hour (Last 60 Minutes)
1. Extreme theta decay accelerates
2. Pin risk increases at major gamma levels
3. Volatility often compresses (if long gamma) or explodes (if short gamma)
4. Max pain becomes increasingly magnetic

## 📊 Key Concepts

### Zero Gamma Level
- The strike where net gamma = 0
- **Above Zero Gamma**: Market is long gamma (dealers short) → Volatility suppression
- **Below Zero Gamma**: Market is short gamma (dealers long) → Volatility amplification

### Max Pain
- Strike where option sellers lose the least money
- Price tends to gravitate toward max pain as expiration approaches
- Strongest effect in final hours of trading

### Gamma Walls
- Strikes with massive gamma exposure
- Act as "magnets" - price tends to pin at these levels
- Harder to break through with size

### Expected Move
- 1 standard deviation intraday range
- Based on ATM implied volatility
- Useful for determining if current move is statistically significant

## ⚙️ Dashboard Settings

### Symbol Selection
- **SPY**: Most liquid, tight spreads, accessible
- **SPX**: Cash-settled, 10x leverage, tax advantages (1256 treatment)
- **QQQ**: Tech-focused, different gamma dynamics

### Strike Range
- Default: 5% from current price
- Narrow range (2-3%) for focused intraday trading
- Wider range (10-15%) to see full positioning landscape

### Auto-Refresh
- Optionally refresh data every 30 seconds during market hours
- Useful for active monitoring without manual refresh

## 🔥 Pro Tips

1. **Follow the Gamma**: If net gamma is negative, expect larger moves. If positive, expect range-bound action.

2. **Zero Gamma Pin**: Price often stalls at the zero gamma level. Watch for consolidation there.

3. **Max Pain Magnetism**: Strongest in final 2 hours. If price is far from max pain, expect drift toward it.

4. **Flow vs Structure**: Strong directional flow + negative gamma = explosive moves. Use caution.

5. **Opening Range**: First 30 minutes set the tone. If price accepts outside yesterday's range, trend likely continues.

6. **Power Hour Dynamics**: Last hour sees most pinning behavior. Gamma walls become extra strong.

7. **Theta Burn**: With <2 hours to expiry, theta decay is extreme. Selling premium becomes high-risk/high-reward.

## ⚠️ Risk Warnings

- 0DTE options are extremely risky and can result in total loss
- Theta decay accelerates exponentially in final hours
- Liquidity can dry up in less popular strikes
- Slippage can be significant with market orders
- This dashboard is for informational purposes only - not financial advice

## 🔧 Technical Notes

### Data Sources
- Schwab API for real-time options chain
- Live quotes for underlying prices
- Fallback to yfinance if API issues occur

### Update Frequency
- Manual refresh: On-demand
- Auto-refresh: 30 seconds (during market hours only)
- Cache: 5 minutes for performance

### Calculations
- **Net GEX Formula**: Σ(Γ × 100 × OI × S² × 0.01) with professional sign convention
- **Max Pain**: Minimizes total intrinsic value for all options
- **Expected Move**: S × IV × √(T) where T is in trading hours
- **Zero Gamma**: Linear interpolation where net gamma crosses zero

## 📱 Mobile Optimization
The dashboard is fully responsive and works on mobile devices for on-the-go monitoring.

---

**Last Updated**: November 4, 2025
**Dashboard Version**: 2.0 (0DTE Specialized)
**Author**: Options Trading Platform Team
