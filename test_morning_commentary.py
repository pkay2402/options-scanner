#!/usr/bin/env python3
"""Test morning commentary with actual data from Jan 23, 2026 + Watchlist movers with whale score"""
import os
import sys
import sqlite3
import requests
import json
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not set. Please export GROQ_API_KEY=your_key")
    sys.exit(1)
client = Groq(api_key=api_key)

# === FETCH WATCHLIST MOVERS ===
print("=== Fetching Watchlist Movers from Droplet ===\n")
try:
    response = requests.get("http://138.197.210.166:8000/api/watchlist?order_by=daily_change_pct&limit=100", timeout=10)
    watchlist_data = response.json()['data']
    
    # Top 3 bullish (highest %)
    top_bullish = watchlist_data[:3]
    # Top 3 bearish (lowest %)
    top_bearish = sorted(watchlist_data, key=lambda x: x['daily_change_pct'])[:3]
    
    print("TOP 3 BULLISH MOVERS:")
    for s in top_bullish:
        print(f"  {s['symbol']}: +{s['daily_change_pct']:.2f}% @ ${s['price']:.2f}")
    print("\nTOP 3 BEARISH MOVERS:")
    for s in top_bearish:
        print(f"  {s['symbol']}: {s['daily_change_pct']:.2f}% @ ${s['price']:.2f}")
except Exception as e:
    print(f"Error fetching watchlist: {e}")
    top_bullish = []
    top_bearish = []

# === RUN WHALE SCORE FOR MOVERS ===
print("\n=== Running Whale Score Analysis ===\n")

# Add discord-bot to path and import whale score functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discord-bot'))
from bot.commands.whale_score import scan_stock_whale_flows, get_next_three_fridays

# Get Schwab client
from src.api.schwab_client import SchwabClient
schwab_client = SchwabClient()

expiry_dates = get_next_three_fridays()
movers_symbols = [s['symbol'] for s in top_bullish + top_bearish]

whale_analysis = {}
for symbol in movers_symbols:
    try:
        flows = scan_stock_whale_flows(schwab_client, symbol, expiry_dates, min_whale_score=50)
        if flows:
            calls = [f for f in flows if f['type'] == 'CALL']
            puts = [f for f in flows if f['type'] == 'PUT']
            
            call_vol = sum(f['volume'] for f in calls)
            put_vol = sum(f['volume'] for f in puts)
            pc_ratio = put_vol / call_vol if call_vol > 0 else 0
            
            # Determine sentiment from options
            if call_vol > put_vol * 1.5:
                options_sentiment = "BULLISH"
            elif put_vol > call_vol * 1.5:
                options_sentiment = "BEARISH"
            else:
                options_sentiment = "NEUTRAL"
            
            top_flow = max(flows, key=lambda x: x['whale_score'])
            
            whale_analysis[symbol] = {
                'total_flows': len(flows),
                'calls': len(calls),
                'puts': len(puts),
                'call_volume': call_vol,
                'put_volume': put_vol,
                'pc_ratio': pc_ratio,
                'options_sentiment': options_sentiment,
                'top_strike': top_flow['strike'],
                'top_type': top_flow['type'],
                'top_score': top_flow['whale_score']
            }
            print(f"  {symbol}: {len(flows)} flows, P/C={pc_ratio:.2f}, Sentiment={options_sentiment}")
        else:
            print(f"  {symbol}: No significant whale flows")
    except Exception as e:
        print(f"  {symbol}: Error - {e}")

# === LOAD SIGNAL DATA ===
print("\n=== Loading Jan 23 Morning Signals ===\n")
db_path = "discord-bot/data/signals.db"
conn = sqlite3.connect(db_path)

# Get signals from first hour (9:30-10:30 AM ET = 14:30-15:30 UTC) on Jan 23
cursor = conn.execute("""
    SELECT symbol, signal_subtype, direction, price, data 
    FROM signals 
    WHERE signal_type = 'WHALE' 
    AND timestamp >= '2026-01-23 14:30:00' 
    AND timestamp <= '2026-01-23 15:30:00'
""")
whale_flows = [{"symbol": r[0], "type": r[1], "direction": r[2], "price": r[3], "data": r[4]} for r in cursor]

cursor = conn.execute("""
    SELECT symbol, signal_subtype, direction, price 
    FROM signals 
    WHERE signal_type = 'TOS' 
    AND timestamp >= '2026-01-23 14:30:00' 
    AND timestamp <= '2026-01-23 15:30:00'
""")
tos_alerts = [{"symbol": r[0], "alert_type": r[1], "direction": r[2], "price": r[3]} for r in cursor]

cursor = conn.execute("""
    SELECT symbol, signal_subtype, direction, price, data 
    FROM signals 
    WHERE signal_type = 'ZSCORE' 
    AND timestamp >= '2026-01-23 14:30:00' 
    AND timestamp <= '2026-01-23 15:30:00'
""")
zscore_signals = [{"symbol": r[0], "condition": r[1], "direction": r[2], "price": r[3], "data": r[4]} for r in cursor]

print(f"Whale flows: {len(whale_flows)}")
print(f"TOS alerts: {len(tos_alerts)}")
print(f"Z-Score signals: {len(zscore_signals)}")

# === BUILD PROMPT ===
prompt_parts = []
prompt_parts.append("Market Session: End of Day Recap - January 23, 2026")
prompt_parts.append("")

# Watchlist Movers
prompt_parts.append("WATCHLIST TOP MOVERS:")
prompt_parts.append("  Bullish Leaders:")
for s in top_bullish:
    prompt_parts.append(f"    • {s['symbol']}: +{s['daily_change_pct']:.2f}% @ ${s['price']:.2f}")
prompt_parts.append("  Bearish Laggards:")
for s in top_bearish:
    prompt_parts.append(f"    • {s['symbol']}: {s['daily_change_pct']:.2f}% @ ${s['price']:.2f}")
prompt_parts.append("")

# Options Activity for Movers
if whale_analysis:
    prompt_parts.append("OPTIONS FLOW ANALYSIS FOR TOP MOVERS:")
    for symbol, analysis in whale_analysis.items():
        # Find price change for this symbol
        price_info = next((s for s in top_bullish + top_bearish if s['symbol'] == symbol), None)
        change_pct = price_info['daily_change_pct'] if price_info else 0
        
        prompt_parts.append(f"  {symbol} ({change_pct:+.2f}% today):")
        prompt_parts.append(f"    • {analysis['total_flows']} whale flows ({analysis['calls']} calls, {analysis['puts']} puts)")
        prompt_parts.append(f"    • Call Vol: {analysis['call_volume']:,} | Put Vol: {analysis['put_volume']:,}")
        prompt_parts.append(f"    • P/C Ratio: {analysis['pc_ratio']:.2f} | Options Sentiment: {analysis['options_sentiment']}")
        prompt_parts.append(f"    • Top Flow: {analysis['top_type']} ${analysis['top_strike']} (Score: {analysis['top_score']:.0f})")
    prompt_parts.append("")

# Morning Signals Summary
bullish_whale = [f for f in whale_flows if f["direction"] == "BULLISH"]
bearish_whale = [f for f in whale_flows if f["direction"] == "BEARISH"]

if whale_flows:
    prompt_parts.append(f"MORNING WHALE FLOWS ({len(whale_flows)} detected in first hour):")
    if bullish_whale:
        symbols = list(set([f['symbol'] for f in bullish_whale]))
        prompt_parts.append(f"  Bullish: {', '.join(symbols)}")
    if bearish_whale:
        symbols = list(set([f['symbol'] for f in bearish_whale]))
        prompt_parts.append(f"  Bearish: {', '.join(symbols)}")
    prompt_parts.append("")

if tos_alerts:
    prompt_parts.append(f"MORNING TOS ALERTS ({len(tos_alerts)} signals):")
    for alert in tos_alerts[:6]:
        prompt_parts.append(f"  • {alert['symbol']}: {alert.get('alert_type', 'Signal')} ({alert.get('direction', 'N/A')})")
    prompt_parts.append("")

if zscore_signals:
    prompt_parts.append(f"Z-SCORE EXTREMES ({len(zscore_signals)} signals):")
    for sig in zscore_signals[:5]:
        prompt_parts.append(f"  • {sig['symbol']}: {sig.get('condition', 'Signal')}")
    prompt_parts.append("")

prompt_parts.append("""Based on this data, provide market commentary that:
1. Analyzes the top movers and whether their options flow CONFIRMS or CONTRADICTS the price movement
2. Highlights any divergence between price action and options sentiment
3. Notes key themes from the morning signals
4. Suggests what to watch tomorrow based on the options positioning""")

prompt = "\n".join(prompt_parts)
print("\n=== PROMPT ===")
print(prompt)

# === CALL GROQ ===
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": """You are a professional market analyst specializing in options flow analysis. 
Your style is:
- Concise and actionable (max 350 words)
- Focus on the relationship between price moves and options positioning
- Highlight confirmation or divergence between price and options flow
- Use trader-friendly language with emojis
- Identify potential smart money positioning

Do NOT provide specific trading advice. Focus on analysis and observations."""},
        {"role": "user", "content": prompt}
    ],
    max_tokens=600,
    temperature=0.7
)

print("\n" + "="*60)
print("=== AI MARKET COMMENTARY WITH WHALE ANALYSIS ===")
print("="*60)
print(resp.choices[0].message.content)
print("="*60)
