# Multi-Channel Alert Setup Guide

## Overview

The Discord bot now supports **3 separate alert channels** with different types of intelligence:

1. **🐋 Whale Flows Channel** - Individual stock whale flows (score > 300)
2. **📊 0DTE Channel** - SPY/QQQ/SPX wall levels and positioning  
3. **🧠 Market Intelligence Channel** - Advanced SPY/QQQ market analysis (next 10 expiries)

## Setup Commands

### 1. Configure Channels

Run these commands in the channels where you want alerts:

```
/setup_whale_alerts
```
- Sets current channel for whale flow alerts
- Monitors: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, CRM, PLTR, COIN, SNOW, CRWD, APP
- Threshold: Whale score > 300

```
/setup_0dte_alerts
```
- Sets current channel for 0DTE level updates
- Monitors: SPY, QQQ, $SPX
- Shows: Call/Put walls, Max Pain, positioning

```
/setup_market_intel
```
- Sets current channel for market intelligence
- Monitors: SPY, QQQ (next 10 expiries)
- Provides: Directional bias, GEX analysis, flow patterns, trading implications

### 2. Start Alert Service

After configuring at least one channel:

```
/start_alerts
```
- Starts automated scanning (every 15 minutes)
- Only active during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
- Requires administrator permissions

### 3. Check Status

```
/alert_status
```
- Shows service status (running/stopped)
- Lists configured channels
- Shows market hours status
- Cache statistics

### 4. Stop Alerts

```
/stop_alerts
```
- Stops all automated alerts
- Requires administrator permissions

## Market Intelligence Details

### What Makes SPY/QQQ Different?

For major indices like SPY and QQQ, whale scores often won't meet the >300 threshold due to:
- High liquidity spreads flows across many strikes
- Lower implied volatility
- Smaller relative price moves

### Solution: Aggregate Analysis

The **Market Intelligence channel** solves this by analyzing:

#### 1. Net Gamma Exposure (GEX)
- Positive GEX = Dealers stabilize (buy dips, sell rips)
- Negative GEX = Dealers amplify moves (volatility)

#### 2. Put/Call Ratio Analysis
- P/C < 0.7: Heavy call buying (bullish)
- P/C > 1.3: Heavy put buying (bearish)
- Tracks both volume and open interest ratios

#### 3. ATM Flow Direction
- Immediate directional intent
- Call-heavy ATM = bullish positioning
- Put-heavy ATM = bearish/hedging

#### 4. OTM Flow Patterns
- OTM calls = bullish positioning for upside
- OTM puts = protection/bearish positioning

#### 5. Fresh Institutional Flows
- Vol/OI >= 3.0x indicates new positions
- Tracks top strikes with highest fresh activity
- Separates call vs put institutional flows

#### 6. Signal Strength Scoring
- STRONG_BULLISH 🚀: +5 or higher
- BULLISH 🟢: +2 to +4  
- NEUTRAL 🟡: -1 to +1
- BEARISH 🔴: -2 to -4
- STRONG_BEARISH 💀: -5 or lower

### Example Market Intel Alert

```
🚀 SPY Market Intelligence
Signal: STRONG_BULLISH (Strength: +7)
Analyzing next 10 expiries

📊 Current State:
Price: $455.32
Net GEX: +2.4B
P/C Ratio: 0.65

💹 Flow Analysis:
Total Vol: 2.1M calls / 1.4M puts
ATM Flow: 1.8x
OTM Flow: 2.3x

⚡ Fresh Institutional Flows:
Call Flows: 47 strikes
Put Flows: 23 strikes

🎯 Key Signals:
✅ Positive GEX (+2.4B) - dealers buy dips
🟢 Bullish P/C: 0.65 (heavy call buying)
🟢 ATM Flow: 1.8x more calls (bullish)
🚀 OTM Calls: 2.3x > OTM Puts (bullish positioning)
💰 Fresh Call Flows: 47 vs 23 puts

🟢 Top Fresh Call Strikes:
$460.00: 12,450 vol (4.2x Vol/OI)
$465.00: 9,820 vol (3.8x Vol/OI)
$470.00: 7,340 vol (3.5x Vol/OI)

💡 Trading Implication:
✅ Bias: BULLISH - Consider call positions or long exposure
```

## Recommended Channel Structure

Create 3 Discord channels:

1. **#whale-flows** → `/setup_whale_alerts`
   - Individual stock opportunities
   - High-conviction plays
   - Specific strike recommendations

2. **#0dte-levels** → `/setup_0dte_alerts`
   - Intraday SPY/QQQ/SPX levels
   - Support/resistance from options
   - Quick reference for day trading

3. **#market-intelligence** → `/setup_market_intel`
   - Big picture market direction
   - Institutional positioning
   - Strategic bias for portfolio management

## Alert Frequency

- **Scan Interval**: Every 15 minutes
- **Market Hours**: 9:30 AM - 4:00 PM ET (Mon-Fri)
- **Cache Duration**: Clears daily at market close

## Benefits of Separation

### Before (Single Channel)
- Cluttered with mixed signals
- Hard to track specific alert types
- Information overload

### After (Multi-Channel)
- **Whale Flows**: Quick scan for individual stock setups
- **0DTE**: Day traders get clean level updates
- **Market Intel**: Portfolio managers see big picture
- Each channel serves specific trading style
- Easy to mute channels you don't need

## Troubleshooting

### No alerts appearing?
1. Check `/alert_status` to confirm service is running
2. Verify market hours (US Eastern Time)
3. Ensure Schwab API is connected
4. Check bot has permission to send messages in channels

### Too many/few alerts?
- Whale flow threshold is configurable (default: 300)
- Market intel signals are scored, not threshold-based
- 0DTE updates are informational (not filtered)

### Want different symbols?
- Edit `TOP_TECH_STOCKS` in `whale_score.py` for whale flows
- Market intel is fixed to SPY/QQQ (most relevant for market direction)
- 0DTE is fixed to SPY/QQQ/SPX (most liquid options)

## API Usage

All three channels share the same Schwab API client:
- Efficient caching prevents duplicate API calls
- Error handling is channel-specific (one failure doesn't break others)
- Token refresh is automatic
- Rate limiting is managed automatically

## Next Steps

1. Create 3 Discord channels
2. Run setup commands in each channel
3. Execute `/start_alerts` (admin only)
4. Monitor `/alert_status` periodically
5. Adjust based on your trading needs
