"""
Option Volume Walls & Key Levels
Simplified, reliable version for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

# Page config
st.set_page_config(page_title="Option Volume Walls", page_icon="🧱", layout="wide")

# ============================================
# CACHED DATA FUNCTIONS
# ============================================

@st.cache_data(ttl=120, show_spinner="Fetching market data...")
def fetch_data(symbol: str, expiry_date: str):
    """Fetch all required data with caching"""
    client = SchwabClient()
    
    if not client.authenticate():
        return {"error": "Authentication failed"}
    
    # Get quote
    quote = client.get_quote(symbol)
    if not quote or symbol not in quote:
        return {"error": f"Could not get quote for {symbol}"}
    
    price = quote[symbol].get('quote', {}).get('lastPrice', 0)
    if not price:
        return {"error": "Could not extract price"}
    
    # Get options chain
    options = client.get_options_chain(
        symbol=symbol,
        contract_type='ALL',
        from_date=expiry_date,
        to_date=expiry_date
    )
    
    if not options or 'callExpDateMap' not in options:
        return {"error": f"No options data for {symbol} on {expiry_date}"}
    
    # Get price history (last 2 days)
    now = datetime.now()
    price_history = client.get_price_history(
        symbol=symbol,
        frequency_type='minute',
        frequency=5,
        start_date=int((now - timedelta(hours=48)).timestamp() * 1000),
        end_date=int(now.timestamp() * 1000),
        need_extended_hours=False
    )
    
    return {
        "symbol": symbol,
        "price": price,
        "options": options,
        "price_history": price_history,
        "timestamp": datetime.now().strftime("%I:%M:%S %p")
    }


def calculate_walls(options_data: dict, price: float) -> dict:
    """Calculate call wall, put wall, and flip level from options data"""
    
    call_volumes = {}
    put_volumes = {}
    call_oi = {}
    put_oi = {}
    
    # Process calls
    for exp_date, strikes in options_data.get('callExpDateMap', {}).items():
        for strike_str, contracts in strikes.items():
            if contracts:
                strike = float(strike_str)
                contract = contracts[0]
                call_volumes[strike] = call_volumes.get(strike, 0) + contract.get('totalVolume', 0)
                call_oi[strike] = call_oi.get(strike, 0) + contract.get('openInterest', 0)
    
    # Process puts
    for exp_date, strikes in options_data.get('putExpDateMap', {}).items():
        for strike_str, contracts in strikes.items():
            if contracts:
                strike = float(strike_str)
                contract = contracts[0]
                put_volumes[strike] = put_volumes.get(strike, 0) + contract.get('totalVolume', 0)
                put_oi[strike] = put_oi.get(strike, 0) + contract.get('openInterest', 0)
    
    if not call_volumes or not put_volumes:
        return None
    
    # Calculate net volumes (Put - Call)
    all_strikes = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
    net_volumes = {s: put_volumes.get(s, 0) - call_volumes.get(s, 0) for s in all_strikes}
    
    # Find walls
    call_wall_strike = max(call_volumes, key=call_volumes.get) if call_volumes else None
    put_wall_strike = max(put_volumes, key=put_volumes.get) if put_volumes else None
    
    # Find flip level (where net volume changes sign near price)
    nearby_strikes = [s for s in all_strikes if abs(s - price) < price * 0.05]
    flip_level = None
    for i in range(len(nearby_strikes) - 1):
        s1, s2 = nearby_strikes[i], nearby_strikes[i + 1]
        if net_volumes.get(s1, 0) * net_volumes.get(s2, 0) < 0:
            flip_level = (s1 + s2) / 2
            break
    
    return {
        "call_wall": {"strike": call_wall_strike, "volume": call_volumes.get(call_wall_strike, 0)},
        "put_wall": {"strike": put_wall_strike, "volume": put_volumes.get(put_wall_strike, 0)},
        "flip_level": flip_level,
        "all_strikes": all_strikes,
        "call_volumes": call_volumes,
        "put_volumes": put_volumes,
        "net_volumes": net_volumes,
        "total_call_vol": sum(call_volumes.values()),
        "total_put_vol": sum(put_volumes.values()),
    }


def create_price_chart(price_history: dict, price: float, walls: dict, symbol: str):
    """Create candlestick chart with key levels"""
    
    if not price_history or 'candles' not in price_history:
        return None
    
    df = pd.DataFrame(price_history['candles'])
    if df.empty:
        return None
    
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    
    # Filter to market hours only
    df = df[(df['datetime'].dt.hour >= 10) | 
            ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30))]
    df = df[df['datetime'].dt.hour < 16]
    
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add levels
    x_range = [df['datetime'].iloc[0], df['datetime'].iloc[-1]]
    
    if walls['call_wall']['strike']:
        fig.add_trace(go.Scatter(
            x=x_range, y=[walls['call_wall']['strike']] * 2,
            mode='lines', name=f"Call Wall ${walls['call_wall']['strike']:.0f}",
            line=dict(color='#22c55e', width=3, dash='dot')
        ))
    
    if walls['put_wall']['strike']:
        fig.add_trace(go.Scatter(
            x=x_range, y=[walls['put_wall']['strike']] * 2,
            mode='lines', name=f"Put Wall ${walls['put_wall']['strike']:.0f}",
            line=dict(color='#ef4444', width=3, dash='dot')
        ))
    
    if walls['flip_level']:
        fig.add_trace(go.Scatter(
            x=x_range, y=[walls['flip_level']] * 2,
            mode='lines', name=f"Flip ${walls['flip_level']:.0f}",
            line=dict(color='#a855f7', width=3)
        ))
    
    # Current price line
    fig.add_hline(y=price, line_dash="dash", line_color="gold", 
                  annotation_text=f"${price:.2f}")
    
    fig.update_layout(
        title=f"{symbol} with Option Volume Walls",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_volume_profile(walls: dict, price: float):
    """Create horizontal volume profile chart"""
    
    strikes = walls['all_strikes']
    net_vols = [walls['net_volumes'].get(s, 0) for s in strikes]
    
    # Filter to ±10% of price
    mask = [(price * 0.9 <= s <= price * 1.1) for s in strikes]
    strikes = [s for s, m in zip(strikes, mask) if m]
    net_vols = [v for v, m in zip(net_vols, mask) if m]
    
    if not strikes:
        return None
    
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in net_vols]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[f"${s:.0f}" for s in strikes],
        x=net_vols,
        orientation='h',
        marker_color=colors,
        text=[f"{abs(v):,.0f}" for v in net_vols],
        textposition='auto'
    ))
    
    # Mark current price
    closest_strike = min(strikes, key=lambda s: abs(s - price))
    closest_idx = strikes.index(closest_strike)
    
    fig.add_annotation(
        y=closest_idx, x=0,
        text="💰", showarrow=False,
        font=dict(size=20)
    )
    
    fig.update_layout(
        title="Net Volume Profile (Red=Put Heavy, Green=Call Heavy)",
        xaxis_title="Net Volume (Put - Call)",
        yaxis_title="Strike",
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================
# MAIN UI
# ============================================

st.title("🧱 Option Volume Walls & Key Levels")
st.markdown("Identify support/resistance from option volume concentrations")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Quick symbol buttons
    st.subheader("Quick Select")
    quick_cols = st.columns(4)
    quick_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMZN', 'MSFT', 'META']
    
    selected = st.session_state.get('selected_symbol', 'SPY')
    
    for i, sym in enumerate(quick_symbols):
        with quick_cols[i % 4]:
            if st.button(sym, key=f"quick_{sym}", use_container_width=True,
                        type="primary" if sym == selected else "secondary"):
                st.session_state.selected_symbol = sym
                st.rerun()
    
    st.divider()
    
    # Manual symbol input
    symbol = st.text_input("Symbol", value=st.session_state.get('selected_symbol', 'SPY')).upper()
    st.session_state.selected_symbol = symbol
    
    # Date selection
    today = datetime.now().date()
    weekday = today.weekday()
    
    # Default to today for SPY/QQQ (0DTE), next Friday for others
    if symbol in ['SPY', 'QQQ', '$SPX']:
        default_date = today if weekday < 5 else today + timedelta(days=(7 - weekday))
    else:
        days_to_friday = (4 - weekday) % 7 or 7
        default_date = today + timedelta(days=days_to_friday)
    
    expiry_date = st.date_input("Expiration", value=default_date)
    
    st.divider()
    
    # Analyze button
    analyze = st.button("🔍 Calculate Levels", type="primary", use_container_width=True)
    
    # Clear cache button
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Main content area
if analyze or st.session_state.get('show_results', False):
    st.session_state.show_results = True
    
    # Fetch data
    with st.spinner(f"Analyzing {symbol}..."):
        data = fetch_data(symbol, expiry_date.strftime('%Y-%m-%d'))
    
    # Check for errors
    if "error" in data:
        st.error(f"❌ {data['error']}")
        st.stop()
    
    # Calculate walls
    walls = calculate_walls(data['options'], data['price'])
    
    if not walls:
        st.error("Could not calculate walls - insufficient data")
        st.stop()
    
    # Header metrics
    st.markdown(f"### {symbol} Analysis @ {data['timestamp']}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Price", f"${data['price']:.2f}")
    
    with col2:
        if walls['call_wall']['strike']:
            dist = ((walls['call_wall']['strike'] - data['price']) / data['price']) * 100
            st.metric("📈 Call Wall (Resistance)", 
                     f"${walls['call_wall']['strike']:.0f}",
                     f"{dist:+.1f}%")
    
    with col3:
        if walls['put_wall']['strike']:
            dist = ((walls['put_wall']['strike'] - data['price']) / data['price']) * 100
            st.metric("📉 Put Wall (Support)", 
                     f"${walls['put_wall']['strike']:.0f}",
                     f"{dist:+.1f}%")
    
    with col4:
        if walls['flip_level']:
            st.metric("🔄 Flip Level", f"${walls['flip_level']:.0f}")
        else:
            st.metric("🔄 Flip Level", "N/A")
    
    with col5:
        net_vol = walls['total_put_vol'] - walls['total_call_vol']
        sentiment = "🐻 Bearish" if net_vol > 0 else "🐂 Bullish"
        st.metric("Sentiment", sentiment)
    
    st.divider()
    
    # Charts
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        price_chart = create_price_chart(data['price_history'], data['price'], walls, symbol)
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)
        else:
            st.warning("Price chart not available")
    
    with chart_col2:
        volume_chart = create_volume_profile(walls, data['price'])
        if volume_chart:
            st.plotly_chart(volume_chart, use_container_width=True)
        else:
            st.warning("Volume profile not available")
    
    # Trading levels table
    st.subheader("📊 Key Levels Summary")
    
    levels_data = []
    
    if walls['put_wall']['strike']:
        levels_data.append({
            "Level": "Put Wall (Support)",
            "Strike": f"${walls['put_wall']['strike']:.0f}",
            "Volume": f"{walls['put_volumes'].get(walls['put_wall']['strike'], 0):,}",
            "Distance": f"{((walls['put_wall']['strike'] - data['price']) / data['price'] * 100):+.2f}%"
        })
    
    if walls['flip_level']:
        levels_data.append({
            "Level": "Flip Level",
            "Strike": f"${walls['flip_level']:.0f}",
            "Volume": "-",
            "Distance": f"{((walls['flip_level'] - data['price']) / data['price'] * 100):+.2f}%"
        })
    
    levels_data.append({
        "Level": "Current Price",
        "Strike": f"${data['price']:.2f}",
        "Volume": "-",
        "Distance": "0.00%"
    })
    
    if walls['call_wall']['strike']:
        levels_data.append({
            "Level": "Call Wall (Resistance)",
            "Strike": f"${walls['call_wall']['strike']:.0f}",
            "Volume": f"{walls['call_volumes'].get(walls['call_wall']['strike'], 0):,}",
            "Distance": f"{((walls['call_wall']['strike'] - data['price']) / data['price'] * 100):+.2f}%"
        })
    
    st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)
    
    # Interpretation guide
    with st.expander("📖 How to Use These Levels"):
        st.markdown("""
        **Call Wall (Resistance)** 📈
        - Highest call volume strike = price ceiling
        - Market makers hedge by selling stock here
        - Breaking above with volume = bullish breakout
        
        **Put Wall (Support)** 📉  
        - Highest put volume strike = price floor
        - Market makers hedge by buying stock here
        - Breaking below with volume = bearish breakdown
        
        **Flip Level** 🔄
        - Where sentiment shifts from bullish to bearish
        - Price above = bullish territory
        - Price below = bearish territory
        
        **Best Use Cases:**
        - Most effective on expiration day (gamma highest)
        - Use as targets for day trades
        - Combine with price action confirmation
        """)

else:
    # Welcome screen
    st.info("👈 Select a symbol and click **Calculate Levels** to analyze option volume walls")
    
    with st.expander("🧱 What Are Option Volume Walls?", expanded=True):
        st.markdown("""
        Option volume walls are **key price levels** where massive option activity creates support or resistance.
        
        **How it works:**
        - Market makers are **short options** to customers
        - They must **hedge** by buying/selling the underlying
        - **High volume strikes** = lots of hedging activity
        - This creates **price magnets** or **barriers**
        
        **The Three Key Levels:**
        
        1. **Call Wall** 📈 - Highest call volume strike (resistance)
        2. **Put Wall** 📉 - Highest put volume strike (support)  
        3. **Flip Level** 🔄 - Where sentiment flips bullish/bearish
        
        **Configure settings in the sidebar and click 'Calculate Levels' to start!**
        """)
