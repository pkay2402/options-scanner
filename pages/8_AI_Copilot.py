#!/usr/bin/env python3
"""
AI Trading Copilot - Chat Interface
Powered by Groq's free Llama 3.1 API
"""

import streamlit as st
import sys
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai_brain.copilot import TradingCopilot

# Configuration
DROPLET_API_URL = "http://138.197.210.166:8000"

# ==================== MARKET COMMENTARY FUNCTION ====================
def generate_market_commentary(copilot) -> str:
    """
    Generate AI market commentary by aggregating:
    - Watchlist top movers from droplet API
    - Scanner signals from signals.db
    - Options flow analysis for movers
    - Breakout candidates
    """
    data = {
        'watchlist_movers': {'bullish': [], 'bearish': []},
        'whale_flows': [],
        'tos_alerts': [],
        'zscore_signals': [],
        'options_activity': [],
        'breakout_candidates': [],
        # Technical Scanner Data from Droplet
        'macd_signals': {'bullish': [], 'bearish': []},
        'ttm_squeeze': {'active': [], 'fired_bullish': [], 'fired_bearish': []},
        'vpb_signals': {'breakouts': [], 'breakdowns': []}
    }
    
    # Store full watchlist for breakout scanning
    watchlist_data = []
    
    # 1. Fetch watchlist movers from droplet API
    try:
        response = requests.get(
            f"{DROPLET_API_URL}/api/watchlist?order_by=daily_change_pct&limit=100",
            timeout=10
        )
        if response.status_code == 200:
            watchlist_data = response.json().get('data', [])
            data['watchlist_movers']['bullish'] = watchlist_data[:3]
            data['watchlist_movers']['bearish'] = sorted(
                watchlist_data, 
                key=lambda x: x.get('daily_change_pct', 0)
            )[:3]
    except Exception as e:
        st.warning(f"Could not fetch watchlist: {e}")
    
    # 2. Fetch recent signals from signals.db
    signals_db = project_root / "discord-bot" / "data" / "signals.db"
    if signals_db.exists():
        try:
            conn = sqlite3.connect(str(signals_db))
            cutoff_time = datetime.now() - timedelta(hours=6)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Whale flows - using parameterized query to prevent SQL injection
            cursor = conn.execute(
                "SELECT symbol, signal_subtype, direction, price FROM signals WHERE signal_type = 'WHALE' AND timestamp >= ? ORDER BY timestamp DESC LIMIT 10",
                (cutoff_str,)
            )
            data['whale_flows'] = [
                {"symbol": r[0], "type": r[1], "direction": r[2], "price": r[3]} 
                for r in cursor
            ]
            
            # TOS alerts
            cursor = conn.execute(
                "SELECT symbol, signal_subtype, direction, price FROM signals WHERE signal_type = 'TOS' AND timestamp >= ? ORDER BY timestamp DESC LIMIT 10",
                (cutoff_str,)
            )
            data['tos_alerts'] = [
                {"symbol": r[0], "alert_type": r[1], "direction": r[2], "price": r[3]} 
                for r in cursor
            ]
            
            # Z-Score signals
            cursor = conn.execute(
                "SELECT symbol, signal_subtype, direction, price FROM signals WHERE signal_type = 'ZSCORE' AND timestamp >= ? ORDER BY timestamp DESC LIMIT 10",
                (cutoff_str,)
            )
            data['zscore_signals'] = [
                {"symbol": r[0], "condition": r[1], "direction": r[2], "price": r[3]} 
                for r in cursor
            ]
            
            conn.close()
        except Exception as e:
            st.warning(f"Could not fetch signals: {e}")
    
    # 3. Run whale score analysis on top movers
    movers_symbols = [s['symbol'] for s in data['watchlist_movers']['bullish'] + data['watchlist_movers']['bearish']]
    
    if movers_symbols:
        try:
            from src.api.schwab_client import SchwabClient
            sys.path.insert(0, str(project_root / "discord-bot"))
            from bot.commands.whale_score import scan_stock_whale_flows, get_next_three_fridays
            
            client = SchwabClient()
            expiry_dates = get_next_three_fridays()
            
            for symbol in movers_symbols[:6]:  # Limit to 6 to avoid slowness
                try:
                    flows = scan_stock_whale_flows(client, symbol, expiry_dates, min_whale_score=50)
                    if flows:
                        calls = [f for f in flows if f['type'] == 'CALL']
                        puts = [f for f in flows if f['type'] == 'PUT']
                        call_vol = sum(f['volume'] for f in calls)
                        put_vol = sum(f['volume'] for f in puts)
                        pc_ratio = put_vol / call_vol if call_vol > 0 else 0
                        
                        if call_vol > put_vol * 1.5:
                            sentiment = "BULLISH"
                        elif put_vol > call_vol * 1.5:
                            sentiment = "BEARISH"
                        else:
                            sentiment = "NEUTRAL"
                        
                        top_flow = max(flows, key=lambda x: x['whale_score'])
                        
                        data['options_activity'].append({
                            'symbol': symbol,
                            'total_flows': len(flows),
                            'calls': len(calls),
                            'puts': len(puts),
                            'call_volume': call_vol,
                            'put_volume': put_vol,
                            'pc_ratio': pc_ratio,
                            'sentiment': sentiment,
                            'top_strike': top_flow['strike'],
                            'top_type': top_flow['type'],
                            'top_score': top_flow['whale_score'],
                            'underlying_price': top_flow['underlying_price']
                        })
                except Exception as e:
                    continue
        except Exception as e:
            st.warning(f"Could not run options analysis: {e}")
    
    # 4. Scan for breakout candidates
    if watchlist_data:
        try:
            import pandas as pd
            from src.api.schwab_client import SchwabClient
            
            client = SchwabClient()
            
            # Get SPY for RS comparison
            spy_change = 0
            try:
                spy_quote = client.get_quote("SPY")
                if spy_quote and "SPY" in spy_quote:
                    spy_change = spy_quote["SPY"].get("quote", {}).get("netPercentChangeInDouble", 0)
            except:
                pass
            
            # Scan watchlist for breakout setups (limit to 30 for speed)
            watchlist_symbols = [item['symbol'] for item in watchlist_data][:30]
            
            for symbol in watchlist_symbols:
                try:
                    # Get price history
                    history = client.get_price_history(
                        symbol, period_type='month', period=3,
                        frequency_type='daily', frequency=1
                    )
                    
                    if not history or 'candles' not in history or len(history['candles']) < 20:
                        continue
                    
                    df = pd.DataFrame(history['candles'])
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) >= 2 else latest
                    
                    open_price = latest['open']
                    close_price = latest['close']
                    volume = latest['volume']
                    avg_volume_20d = df['volume'].tail(20).mean()
                    high_52w = df['high'].max()
                    prev_close = prev['close']
                    
                    if not all([open_price, close_price, high_52w, volume, avg_volume_20d]):
                        continue
                    
                    # Criteria checks
                    is_green = close_price > open_price
                    volume_ratio = volume / avg_volume_20d if avg_volume_20d > 0 else 0
                    distance_from_high = ((high_52w - close_price) / high_52w) * 100 if high_52w > 0 else 100
                    change_pct = ((close_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                    rs_vs_spy = change_pct - spy_change
                    
                    # Must pass: green candle, volume > 1.2x, RS > 0, within 20% of high
                    if (is_green and 
                        volume_ratio >= 1.2 and 
                        rs_vs_spy > 0 and 
                        distance_from_high <= 20):
                        
                        reasons = []
                        score = 0
                        
                        reasons.append(f"Green candle")
                        reasons.append(f"Volume {volume_ratio:.1f}x avg")
                        score += 1
                        
                        if distance_from_high <= 5:
                            reasons.append(f"🔥 Near 52w high ({distance_from_high:.1f}%)")
                            score += 3
                        else:
                            reasons.append(f"{distance_from_high:.1f}% from high")
                            score += 1
                        
                        reasons.append(f"RS +{rs_vs_spy:.1f}% vs SPY")
                        score += 1
                        
                        data['breakout_candidates'].append({
                            'symbol': symbol,
                            'price': close_price,
                            'change_pct': change_pct,
                            'volume_ratio': volume_ratio,
                            'distance_from_high': distance_from_high,
                            'high_52w': high_52w,
                            'rs_vs_spy': rs_vs_spy,
                            'reasons': reasons,
                            'score': score
                        })
                except Exception:
                    continue
            
            # Sort by score and limit to top 5
            data['breakout_candidates'].sort(key=lambda x: x['score'], reverse=True)
            data['breakout_candidates'] = data['breakout_candidates'][:5]
            
        except Exception as e:
            st.warning(f"Could not scan breakouts: {e}")
    
    # 5. Fetch Technical Scanner Data from Droplet API (MACD, TTM Squeeze, VPB)
    try:
        # MACD Bullish
        response = requests.get(f"{DROPLET_API_URL}/api/macd_scanner?filter=bullish&limit=5", timeout=10)
        if response.status_code == 200:
            macd_data = response.json().get('data', [])
            data['macd_signals']['bullish'] = [
                {'symbol': s['symbol'], 'price': s['price'], 'change_pct': s.get('price_change_pct', 0)}
                for s in macd_data
            ]
        
        # MACD Bearish
        response = requests.get(f"{DROPLET_API_URL}/api/macd_scanner?filter=bearish&limit=5", timeout=10)
        if response.status_code == 200:
            macd_data = response.json().get('data', [])
            data['macd_signals']['bearish'] = [
                {'symbol': s['symbol'], 'price': s['price'], 'change_pct': s.get('price_change_pct', 0)}
                for s in macd_data
            ]
        
        # TTM Squeeze Active
        response = requests.get(f"{DROPLET_API_URL}/api/ttm_squeeze_scanner?filter=active&limit=5", timeout=10)
        if response.status_code == 200:
            ttm_data = response.json().get('data', [])
            data['ttm_squeeze']['active'] = [
                {'symbol': s['symbol'], 'price': s['price'], 'squeeze_duration': s.get('squeeze_duration', 0)}
                for s in ttm_data
            ]
        
        # TTM Squeeze Fired
        response = requests.get(f"{DROPLET_API_URL}/api/ttm_squeeze_scanner?filter=fired&limit=10", timeout=10)
        if response.status_code == 200:
            ttm_data = response.json().get('data', [])
            for s in ttm_data:
                signal = {'symbol': s['symbol'], 'price': s['price'], 'fire_date': s.get('fire_date')}
                if s.get('fire_direction') == 'bullish':
                    data['ttm_squeeze']['fired_bullish'].append(signal)
                else:
                    data['ttm_squeeze']['fired_bearish'].append(signal)
        
        # VPB Breakouts
        response = requests.get(f"{DROPLET_API_URL}/api/vpb_scanner?filter=buy&limit=5", timeout=10)
        if response.status_code == 200:
            vpb_data = response.json().get('data', [])
            data['vpb_signals']['breakouts'] = [
                {'symbol': s['symbol'], 'price': s['price'], 'change_pct': s.get('price_change_pct', 0), 
                 'volume_surge_pct': s.get('volume_surge_pct', 0)}
                for s in vpb_data if s.get('buy_signal')
            ]
        
        # VPB Breakdowns
        response = requests.get(f"{DROPLET_API_URL}/api/vpb_scanner?filter=sell&limit=5", timeout=10)
        if response.status_code == 200:
            vpb_data = response.json().get('data', [])
            data['vpb_signals']['breakdowns'] = [
                {'symbol': s['symbol'], 'price': s['price'], 'change_pct': s.get('price_change_pct', 0),
                 'volume_surge_pct': s.get('volume_surge_pct', 0)}
                for s in vpb_data if s.get('sell_signal')
            ]
    except Exception as e:
        st.warning(f"Could not fetch technical scanner data: {e}")
    
    # 6. Build prompt for AI
    prompt_parts = []
    prompt_parts.append(f"Market Session: {datetime.now().strftime('%B %d, %Y %I:%M %p ET')}")
    prompt_parts.append("")
    
    # Watchlist movers
    if data['watchlist_movers']['bullish'] or data['watchlist_movers']['bearish']:
        prompt_parts.append("WATCHLIST TOP MOVERS:")
        if data['watchlist_movers']['bullish']:
            prompt_parts.append("  Bullish Leaders:")
            for s in data['watchlist_movers']['bullish']:
                prompt_parts.append(f"    • {s['symbol']}: +{s.get('daily_change_pct', 0):.2f}% @ ${s.get('price', 0):.2f}")
        if data['watchlist_movers']['bearish']:
            prompt_parts.append("  Bearish Laggards:")
            for s in data['watchlist_movers']['bearish']:
                prompt_parts.append(f"    • {s['symbol']}: {s.get('daily_change_pct', 0):.2f}% @ ${s.get('price', 0):.2f}")
        prompt_parts.append("")
    
    # Options flow analysis
    if data['options_activity']:
        mover_changes = {}
        for m in data['watchlist_movers']['bullish']:
            mover_changes[m['symbol']] = m.get('daily_change_pct', 0)
        for m in data['watchlist_movers']['bearish']:
            mover_changes[m['symbol']] = m.get('daily_change_pct', 0)
        
        prompt_parts.append(f"OPTIONS FLOW ANALYSIS ({len(data['options_activity'])} stocks scanned):")
        for opt in data['options_activity']:
            symbol = opt['symbol']
            sentiment = opt['sentiment']
            price_change = mover_changes.get(symbol)
            
            divergence = ""
            if price_change is not None:
                if price_change > 1 and sentiment == 'BEARISH':
                    divergence = " ⚠️ DIVERGENCE: Price up but options bearish"
                elif price_change < -1 and sentiment == 'BULLISH':
                    divergence = " ⚠️ DIVERGENCE: Price down but options bullish"
            
            change_str = f" ({price_change:+.2f}% today)" if price_change else ""
            prompt_parts.append(f"  • {symbol}{change_str}: {sentiment} options sentiment{divergence}")
            prompt_parts.append(f"    Calls: {opt['calls']} ({opt['call_volume']:,} vol) | Puts: {opt['puts']} ({opt['put_volume']:,} vol)")
            prompt_parts.append(f"    P/C Ratio: {opt['pc_ratio']:.2f} | Top Flow: {opt['top_type']} ${opt['top_strike']} (Score: {opt['top_score']:.0f})")
        prompt_parts.append("")
    
    # Whale flows
    if data['whale_flows']:
        bullish = [f for f in data['whale_flows'] if f.get('direction') == 'BULLISH']
        bearish = [f for f in data['whale_flows'] if f.get('direction') == 'BEARISH']
        prompt_parts.append(f"RECENT WHALE FLOWS ({len(data['whale_flows'])} detected):")
        if bullish:
            symbols = list(set([f['symbol'] for f in bullish]))[:5]
            prompt_parts.append(f"  Bullish: {', '.join(symbols)}")
        if bearish:
            symbols = list(set([f['symbol'] for f in bearish]))[:5]
            prompt_parts.append(f"  Bearish: {', '.join(symbols)}")
        prompt_parts.append("")
    
    # TOS Alerts
    if data['tos_alerts']:
        prompt_parts.append(f"TOS ALERTS ({len(data['tos_alerts'])} signals):")
        for alert in data['tos_alerts'][:5]:
            prompt_parts.append(f"  • {alert['symbol']}: {alert.get('alert_type', 'Signal')} ({alert.get('direction', 'N/A')})")
        prompt_parts.append("")
    
    # Z-Score signals
    if data['zscore_signals']:
        prompt_parts.append(f"Z-SCORE EXTREMES ({len(data['zscore_signals'])} signals):")
        for sig in data['zscore_signals'][:5]:
            prompt_parts.append(f"  • {sig['symbol']}: {sig.get('condition', 'Signal')}")
        prompt_parts.append("")
    
    # Breakout Candidates
    if data['breakout_candidates']:
        prompt_parts.append(f"🚀 BREAKOUT CANDIDATES ({len(data['breakout_candidates'])} stocks with breakout potential):")
        prompt_parts.append("  Criteria: Green candle, Volume > 1.2x avg, Within 20% of 52w high, Outperforming SPY")
        for candidate in data['breakout_candidates']:
            prompt_parts.append(f"  • {candidate['symbol']}: ${candidate['price']:.2f} ({candidate['change_pct']:+.2f}%)")
            prompt_parts.append(f"    {candidate['distance_from_high']:.1f}% from 52w high (${candidate['high_52w']:.2f})")
            prompt_parts.append(f"    Volume: {candidate['volume_ratio']:.1f}x avg | RS: +{candidate['rs_vs_spy']:.1f}% vs SPY")
        prompt_parts.append("")
    
    # MACD Scanner
    if data['macd_signals']['bullish'] or data['macd_signals']['bearish']:
        prompt_parts.append("📊 MACD CROSSOVERS (Recent):")
        if data['macd_signals']['bullish']:
            prompt_parts.append("  Bullish Crosses:")
            for s in data['macd_signals']['bullish'][:3]:
                prompt_parts.append(f"    🟢 {s['symbol']}: ${s['price']:.2f} ({s['change_pct']:+.2f}%)")
        if data['macd_signals']['bearish']:
            prompt_parts.append("  Bearish Crosses:")
            for s in data['macd_signals']['bearish'][:3]:
                prompt_parts.append(f"    🔴 {s['symbol']}: ${s['price']:.2f} ({s['change_pct']:+.2f}%)")
        prompt_parts.append("")
    
    # TTM Squeeze Scanner
    if data['ttm_squeeze']['active'] or data['ttm_squeeze']['fired_bullish'] or data['ttm_squeeze']['fired_bearish']:
        prompt_parts.append("🔄 TTM SQUEEZE SCANNER:")
        if data['ttm_squeeze']['active']:
            prompt_parts.append(f"  Active Squeezes ({len(data['ttm_squeeze']['active'])} stocks in compression - potential explosion):")
            for s in data['ttm_squeeze']['active'][:3]:
                prompt_parts.append(f"    ⚡ {s['symbol']}: ${s['price']:.2f} (Squeeze: {s['squeeze_duration']} bars)")
        if data['ttm_squeeze']['fired_bullish']:
            prompt_parts.append(f"  🟢 Fired Bullish ({len(data['ttm_squeeze']['fired_bullish'])} - squeeze released upward):")
            for s in data['ttm_squeeze']['fired_bullish'][:3]:
                prompt_parts.append(f"    • {s['symbol']}: ${s['price']:.2f}")
        if data['ttm_squeeze']['fired_bearish']:
            prompt_parts.append(f"  🔴 Fired Bearish ({len(data['ttm_squeeze']['fired_bearish'])} - squeeze released downward):")
            for s in data['ttm_squeeze']['fired_bearish'][:3]:
                prompt_parts.append(f"    • {s['symbol']}: ${s['price']:.2f}")
        prompt_parts.append("")
    
    # VPB Scanner
    if data['vpb_signals']['breakouts'] or data['vpb_signals']['breakdowns']:
        prompt_parts.append("📈 VOLUME-PRICE BREAKOUT SCANNER:")
        if data['vpb_signals']['breakouts']:
            prompt_parts.append("  Breakouts (Price UP + Volume Surge):")
            for s in data['vpb_signals']['breakouts'][:3]:
                prompt_parts.append(f"    🟢 {s['symbol']}: ${s['price']:.2f} ({s['change_pct']:+.2f}%, Vol +{s['volume_surge_pct']:.0f}%)")
        if data['vpb_signals']['breakdowns']:
            prompt_parts.append("  Breakdowns (Price DOWN + Volume Surge):")
            for s in data['vpb_signals']['breakdowns'][:3]:
                prompt_parts.append(f"    🔴 {s['symbol']}: ${s['price']:.2f} ({s['change_pct']:+.2f}%, Vol +{s['volume_surge_pct']:.0f}%)")
        prompt_parts.append("")
    
    prompt_parts.append("""Based on this data, provide market commentary that:
1. Summarizes key market themes and unusual activity
2. Highlights any DIVERGENCE between price action and options flow (this is critical)
3. Notes confirmation when price and options align
4. Highlight BREAKOUT CANDIDATES - these show technical strength with volume confirmation
5. Discuss MACD crossovers, TTM Squeeze firings, and VPB signals that confirm or contradict other signals
6. Suggests what to watch based on options positioning, technical scanners, and breakout setups""")
    
    prompt = "\n".join(prompt_parts)
    
    # 6. Generate AI commentary using copilot's client directly
    system_prompt = """You are a professional market analyst specializing in options flow analysis. 
Your style is:
- Concise and actionable (max 400 words)
- Focus on the relationship between price moves and options positioning
- Highlight confirmation or divergence between price and options flow
- Call out breakout candidates with strong technical setups
- Use trader-friendly language with emojis
- Identify potential smart money positioning

Do NOT provide specific trading advice. Focus on analysis and observations."""
    
    try:
        response = copilot.client.chat.completions.create(
            model=copilot.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating commentary: {str(e)}"


# Page config
st.set_page_config(
    page_title="AI Trading Copilot",
    page_icon="🤖",
    layout="wide"
)

# Get API key from secrets (multiple methods)
import os
api_key = None
discord_webhook = None

# Get Discord webhook from secrets
try:
    discord_webhook = st.secrets["alerts"]["discord_webhook"]
except:
    pass

# Debug: show available secrets keys
# st.write("Available secrets:", list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else "none")

# Method 1: Direct access at top level
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    pass

# Method 2: Check inside [alerts] section
if not api_key:
    try:
        api_key = st.secrets["alerts"]["GROQ_API_KEY"]
    except:
        pass

# Method 3: Environment variable fallback
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

# Initialize copilot with caching to prevent re-initialization on every rerun
@st.cache_resource
def get_copilot(api_key_value):
    """Cache the copilot instance to avoid re-initialization"""
    return TradingCopilot(api_key=api_key_value)

copilot = get_copilot(api_key)

# ==================== PAGE HEADER ====================
st.title("🤖 AI Trading Copilot")
st.markdown("*Your AI-powered market analyst - powered by Llama 3.1 via Groq (FREE)*")

# ==================== NEWS SUMMARY AT TOP ====================
# Cache news summary to prevent re-fetching on every interaction
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_news_summary():
    return copilot.get_news_summary()

# Initialize news loading state
if 'news_loaded' not in st.session_state:
    st.session_state.news_loaded = False

with st.expander("📰 Today's News Summary (Upgrades/Downgrades)", expanded=False):
    # Lazy load - only fetch when user expands
    if st.button("🔄 Load News", key="load_news_btn") or st.session_state.news_loaded:
        st.session_state.news_loaded = True
        with st.spinner("Fetching news from MarketAux + Yahoo Finance..."):
            news_summary = get_cached_news_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔼 Recent Upgrades")
            if news_summary['upgraded_tickers']:
                # Show tickers with upgrades
                for ticker, headlines in list(news_summary['upgraded_tickers'].items())[:8]:
                    st.markdown(f"**{ticker}** - {headlines[0][:60]}...")
            else:
                st.caption("No recent upgrades found")
        
        with col2:
            st.markdown("### 🔽 Recent Downgrades")
            if news_summary['downgraded_tickers']:
                # Show tickers with downgrades
                for ticker, headlines in list(news_summary['downgraded_tickers'].items())[:8]:
                    st.markdown(f"**{ticker}** - {headlines[0][:60]}...")
            else:
                st.caption("No recent downgrades found")
        
        # Quick summary
        st.caption(f"📊 Upgrades: {news_summary['total_upgrades']} tickers | Downgrades: {news_summary['total_downgrades']} tickers (via MarketAux + Yahoo Finance)")
    else:
        st.info("Click 'Load News' to fetch latest upgrades/downgrades")

# ==================== SETTINGS AT TOP ====================
# Check MarketAux API key
marketaux_key = None
try:
    marketaux_key = st.secrets.get("MARKETAUX_API_KEY") or st.secrets.get("alerts", {}).get("MARKETAUX_API_KEY")
except:
    pass
if not marketaux_key:
    marketaux_key = os.environ.get("MARKETAUX_API_KEY")

with st.expander("⚙️ Settings & Connection", expanded=not copilot.is_available()):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Groq AI (Chat & Analysis)**")
        if copilot.is_available():
            st.success("✅ Connected")
            st.caption(f"Model: {copilot.model}")
        else:
            st.warning("🔑 API Key Required")
            st.markdown("""
**Get your FREE Groq API key:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Add to Streamlit secrets as `GROQ_API_KEY`
            """)
            
            manual_key = st.text_input("Or paste API Key here:", type="password")
            if manual_key:
                copilot = TradingCopilot(api_key=manual_key)
                if copilot.is_available():
                    st.success("✅ Connected!")
                    st.rerun()
                else:
                    st.error("Invalid API key")
    
    with col2:
        st.markdown("**📰 News Sources**")
        if marketaux_key:
            st.success("✅ MarketAux Connected")
            st.caption("Primary: MarketAux API with sentiment analysis")
        else:
            st.warning("⚠️ MarketAux not configured")
            st.markdown("""
**Get your FREE MarketAux API key:**
1. Go to [marketaux.com/register](https://www.marketaux.com/register)
2. Sign up (free tier: 100 req/day)
3. Add to Streamlit secrets as `MARKETAUX_API_KEY`
            """)
        st.caption("Fallback: Yahoo Finance (yfinance) - always available")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to send to Discord
def send_to_discord(content: str, title: str = "AI Copilot Analysis"):
    """Send message to Discord webhook"""
    if not discord_webhook:
        return False, "Discord webhook not configured"
    
    # Discord has 2000 char limit per message, split if needed
    # Use embed for nicer formatting
    embed = {
        "title": f"🤖 {title}",
        "description": content[:4000],  # Discord embed description limit
        "color": 5814783,  # Blue color
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "AI Trading Copilot"}
    }
    
    payload = {"embeds": [embed]}
    
    try:
        response = requests.post(discord_webhook, json=payload, timeout=10)
        if response.status_code == 204:
            return True, "Sent to Discord!"
        else:
            return False, f"Discord error: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ==================== MAIN INTERFACE ====================
if copilot.is_available():
    # Chat input at top
    if prompt := st.chat_input("Ask me anything about the market..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = copilot.chat(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Quick Actions Row
    st.markdown("### 🎯 Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔥 Get Trade Recommendations", use_container_width=True, type="primary"):
            st.session_state.messages.append({"role": "user", "content": "Get AI Trade Recommendations"})
            with st.spinner("🔍 Analyzing scanner data for trade setups..."):
                response = copilot.get_ai_trade_recommendations()
            st.session_state.messages.append({"role": "assistant", "content": f"**🎯 AI Trade Recommendations**\n\n{response}"})
            st.rerun()
    
    with action_col2:
        if st.button("📈 Weekly Plays", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me the best weekly options plays"})
            with st.spinner("Loading weekly setups..."):
                weekly = copilot.get_scanner_data("weekly")
                if weekly:
                    plays_text = copilot.format_scanner_plays(weekly, "⚡ Weekly Options Plays")
                    response = copilot.chat(f"Here are the top weekly plays from the scanner:\n\n{plays_text}\n\nProvide brief analysis on the top 3 picks with specific strike and expiry recommendations. Use plain text only.", include_context=False)
                else:
                    response = "No weekly plays found. Run the newsletter scanner first."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with action_col3:
        if st.button("🐻 Bearish Setups", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me bearish setups for puts"})
            with st.spinner("Loading bearish setups..."):
                bearish = copilot.get_scanner_data("bearish")
                if bearish:
                    plays_text = copilot.format_scanner_plays(bearish, "🔴 Bearish Setups")
                    response = copilot.chat(f"Here are bearish setups from the scanner:\n\n{plays_text}\n\nProvide brief analysis on the top 3 for put plays with specific strike and expiry recommendations. Use plain text only.", include_context=False)
                else:
                    response = "No bearish setups found currently."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Market Forecast Section
    st.markdown("---")
    st.markdown("### 🔮 5-Day Market Forecast")
    st.caption("Analyzes SPY & QQQ using live options data, technicals, gamma levels, and VIX")
    
    forecast_col1, forecast_col2 = st.columns([3, 1])
    with forecast_col1:
        st.info("📈 Get a day-by-day forecast for the next 5 trading days with specific price targets and trade ideas")
    with forecast_col2:
        if st.button("🔮 Generate Forecast", use_container_width=True, type="primary"):
            st.session_state.messages.append({"role": "user", "content": "Generate 5-Day Market Forecast"})
            with st.spinner("🔮 Analyzing SPY, QQQ options data, technicals, gamma walls..."):
                response = copilot.get_5_day_market_forecast()
            st.session_state.messages.append({"role": "assistant", "content": f"**🔮 5-Day Market Forecast**\n\n{response}"})
            st.rerun()
    
    # Market Commentary Section (NEW)
    st.markdown("---")
    st.markdown("### 📊 AI Market Commentary")
    st.caption("Aggregates watchlist movers, scanner signals, and options flow analysis with divergence detection")
    
    commentary_col1, commentary_col2 = st.columns([3, 1])
    with commentary_col1:
        st.info("🐋 Combines top movers, TOS alerts, Z-Score signals, and whale flow analysis to generate comprehensive market commentary")
    with commentary_col2:
        if st.button("📊 Generate Commentary", use_container_width=True, type="primary"):
            st.session_state.messages.append({"role": "user", "content": "Generate AI Market Commentary"})
            with st.spinner("📊 Fetching watchlist movers, scanner signals, running options analysis..."):
                response = generate_market_commentary(copilot)
            st.session_state.messages.append({"role": "assistant", "content": f"**📊 AI Market Commentary**\n\n{response}"})
            st.rerun()
    
    st.markdown("---")
    
    # Analyze stock section
    st.markdown("### 🔍 Analyze Stock")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_ticker = st.text_input("Enter ticker symbol", placeholder="NVDA, AMD, AAPL...", label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    
    if analyze_btn and analyze_ticker:
        ticker = analyze_ticker.upper().strip()
        st.session_state.messages.append({"role": "user", "content": f"Analyze {ticker}"})
        with st.spinner(f"🔍 Analyzing {ticker} (fetching live data, options flow, IV, gamma walls, earnings)..."):
            response = copilot.analyze_stock(ticker)
        st.session_state.messages.append({"role": "assistant", "content": f"**🔍 Analysis: {ticker}**\n\n{response}"})
        st.rerun()
    
    st.markdown("---")
    
    # Display chat messages
    if st.session_state.messages:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Add Discord send button for assistant messages
                if message["role"] == "assistant" and discord_webhook:
                    if st.button("📤 Send to Discord", key=f"discord_{idx}"):
                        success, msg = send_to_discord(message["content"])
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Enter a ticker above or ask a question to get started")

else:
    # Not connected - show info
    st.info("👆 Add your Groq API key above to start chatting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
### What can the AI Copilot do?

**📰 Morning Brief**
- Market sentiment overview
- Top 3 trade setups
- Stocks with improving momentum

**🎯 Find Best Setups**
- Confluence detection
- Multi-signal alignment
        """)
    
    with col2:
        st.markdown("""
### Features

**🔍 Stock Analysis**
- Historical score tracking
- Momentum assessment
- Trade thesis generation

**💬 Ask Anything**
- "What's your market view?"
- "Compare NVDA vs AMD"
        """)

# ==================== FOOTER ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    history_file = project_root / "data" / "newsletter_scan_history.json"
    if history_file.exists():
        import json
        with open(history_file) as f:
            history = json.load(f)
        dates = sorted(history.keys(), reverse=True)
        st.caption(f"📊 Newsletter Data: {len(dates)} scan(s)")
    else:
        st.caption("📊 No newsletter data yet")

with col2:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

with col3:
    st.caption("🤖 Llama 3.1 via Groq (FREE)")
