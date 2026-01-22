#!/usr/bin/env python3
"""
Multi-Scanner - Technical and Options Strategy Scanners
Combines: EMA Cloud Scanner, IV Mean Reversion, ToS Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client

# Page config
st.set_page_config(
    page_title="Multi-Scanner",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default universes
STOCK_UNIVERSES = {
    'Mega Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'Semiconductors': ['NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'MU', 'TSM'],
    'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK'],
    'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK'],
    'High Beta': ['COIN', 'MSTR', 'HOOD', 'PLTR', 'SOFI', 'SQ', 'PYPL'],
    'Mixed': ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'AMD', 'META', 'AMZN', 'GOOGL', 'COIN']
}


# ==================== DATA FETCHING ====================
@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period='3mo'):
    """Fetch stock price data using yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df['Symbol'] = symbol
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def fetch_options_for_iv(symbol):
    """Fetch options data for IV analysis"""
    try:
        client = get_client()
        if not client:
            return None, 0
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=20)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, 0
        
        return chain, chain.get('underlyingPrice', 0)
    except:
        return None, 0


# ==================== TECHNICAL INDICATORS ====================
def calculate_ema(df, period):
    """Calculate EMA"""
    return df['Close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema_cloud_signal(df):
    """Calculate EMA cloud signal"""
    df['EMA9'] = calculate_ema(df, 9)
    df['EMA21'] = calculate_ema(df, 21)
    df['EMA50'] = calculate_ema(df, 50)
    
    current_price = df['Close'].iloc[-1]
    ema9 = df['EMA9'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    
    # Bullish: Price > EMA9 > EMA21 > EMA50
    bullish = current_price > ema9 > ema21 > ema50
    # Bearish: Price < EMA9 < EMA21 < EMA50
    bearish = current_price < ema9 < ema21 < ema50
    
    if bullish:
        return 'BULLISH', '#10b981'
    elif bearish:
        return 'BEARISH', '#ef4444'
    else:
        return 'NEUTRAL', '#f59e0b'


# ==================== IV ANALYSIS ====================
def calculate_iv_metrics(chain, underlying_price):
    """Calculate IV metrics from options chain"""
    if not chain:
        return None
    
    iv_values = []
    
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        for strike_str, contracts in strikes.items():
            if contracts:
                iv = contracts[0].get('volatility', 0)
                if iv and iv > 0:
                    iv_values.append(iv)
    
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        for strike_str, contracts in strikes.items():
            if contracts:
                iv = contracts[0].get('volatility', 0)
                if iv and iv > 0:
                    iv_values.append(iv)
    
    if not iv_values:
        return None
    
    current_iv = np.mean(iv_values)
    
    return {
        'current_iv': current_iv,
        'iv_percentile': np.percentile(iv_values, 50),  # Simplified
        'iv_min': min(iv_values),
        'iv_max': max(iv_values)
    }


# ==================== EMA CLOUD SCANNER TAB ====================
def render_ema_scanner_tab(symbols):
    """Scan for EMA cloud setups"""
    st.subheader("📈 EMA Cloud Scanner")
    st.caption("Scans for stocks in bullish/bearish EMA alignment")
    
    progress = st.progress(0)
    results = []
    
    for i, symbol in enumerate(symbols):
        df = fetch_stock_data(symbol)
        if df is not None and len(df) > 50:
            signal, color = calculate_ema_cloud_signal(df)
            rsi = calculate_rsi(df).iloc[-1]
            
            results.append({
                'Symbol': symbol,
                'Price': df['Close'].iloc[-1],
                'Signal': signal,
                'Color': color,
                'RSI': rsi,
                'Change%': ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100,
                'EMA9': df['EMA9'].iloc[-1],
                'EMA21': df['EMA21'].iloc[-1],
                'EMA50': df['EMA50'].iloc[-1]
            })
        progress.progress((i + 1) / len(symbols))
    
    progress.empty()
    
    if not results:
        st.warning("No results found")
        return
    
    df_results = pd.DataFrame(results)
    
    # Summary metrics
    bullish_count = len(df_results[df_results['Signal'] == 'BULLISH'])
    bearish_count = len(df_results[df_results['Signal'] == 'BEARISH'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Bullish", bullish_count)
    col2.metric("🔴 Bearish", bearish_count)
    col3.metric("🟡 Neutral", len(results) - bullish_count - bearish_count)
    
    # Filter by signal
    signal_filter = st.selectbox("Filter by Signal", ['All', 'BULLISH', 'BEARISH', 'NEUTRAL'])
    
    if signal_filter != 'All':
        df_results = df_results[df_results['Signal'] == signal_filter]
    
    # Display results
    for _, row in df_results.iterrows():
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
        
        with col1:
            emoji = '🟢' if row['Signal'] == 'BULLISH' else ('🔴' if row['Signal'] == 'BEARISH' else '🟡')
            st.markdown(f"**{emoji} {row['Symbol']}**")
        
        with col2:
            st.metric("Price", f"${row['Price']:.2f}", f"{row['Change%']:+.1f}%")
        
        with col3:
            st.metric("Signal", row['Signal'])
        
        with col4:
            rsi_color = 'green' if row['RSI'] < 30 else ('red' if row['RSI'] > 70 else 'gray')
            st.metric("RSI", f"{row['RSI']:.0f}")
        
        with col5:
            st.caption(f"EMAs: {row['EMA9']:.1f} / {row['EMA21']:.1f} / {row['EMA50']:.1f}")
        
        st.divider()


# ==================== IV MEAN REVERSION TAB ====================
def render_iv_scanner_tab(symbols):
    """Scan for IV mean reversion opportunities"""
    st.subheader("📊 IV Mean Reversion Scanner")
    st.caption("Find stocks with elevated or depressed implied volatility")
    
    progress = st.progress(0)
    results = []
    
    for i, symbol in enumerate(symbols):
        chain, price = fetch_options_for_iv(symbol)
        if chain:
            iv_data = calculate_iv_metrics(chain, price)
            if iv_data:
                results.append({
                    'Symbol': symbol,
                    'Price': price,
                    'IV': iv_data['current_iv'],
                    'IV_Range': f"{iv_data['iv_min']:.0f}-{iv_data['iv_max']:.0f}",
                    'Signal': 'HIGH IV' if iv_data['current_iv'] > 50 else ('LOW IV' if iv_data['current_iv'] < 25 else 'NORMAL')
                })
        progress.progress((i + 1) / len(symbols))
    
    progress.empty()
    
    if not results:
        st.warning("No IV data found")
        return
    
    df_results = pd.DataFrame(results).sort_values('IV', ascending=False)
    
    # Metrics
    high_iv = len(df_results[df_results['Signal'] == 'HIGH IV'])
    low_iv = len(df_results[df_results['Signal'] == 'LOW IV'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High IV (Sell Premium)", high_iv)
    col2.metric("🟢 Low IV (Buy Premium)", low_iv)
    col3.metric("🟡 Normal IV", len(results) - high_iv - low_iv)
    
    # IV Chart
    fig = go.Figure()
    
    colors = ['#ef4444' if s == 'HIGH IV' else ('#10b981' if s == 'LOW IV' else '#f59e0b') 
              for s in df_results['Signal']]
    
    fig.add_trace(go.Bar(
        x=df_results['Symbol'],
        y=df_results['IV'],
        marker_color=colors,
        text=[f"{iv:.0f}%" for iv in df_results['IV']],
        textposition='auto'
    ))
    
    fig.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Low IV Zone")
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="High IV Zone")
    
    fig.update_layout(
        title="IV by Symbol",
        yaxis_title="Implied Volatility (%)",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown("**IV Scan Results**")
    
    for _, row in df_results.iterrows():
        emoji = '🔴' if row['Signal'] == 'HIGH IV' else ('🟢' if row['Signal'] == 'LOW IV' else '🟡')
        strategy = "Sell Premium" if row['Signal'] == 'HIGH IV' else ("Buy Premium" if row['Signal'] == 'LOW IV' else "Hold")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.markdown(f"**{emoji} {row['Symbol']}** @ ${row['Price']:.2f}")
        with col2:
            st.metric("IV", f"{row['IV']:.0f}%")
        with col3:
            st.metric("Range", row['IV_Range'])
        with col4:
            st.markdown(f"**Strategy:** {strategy}")
        
        st.divider()


# ==================== MOMENTUM SCANNER TAB ====================
def render_momentum_scanner_tab(symbols):
    """Scan for momentum setups"""
    st.subheader("🚀 Momentum Scanner")
    st.caption("Find stocks with strong price momentum")
    
    progress = st.progress(0)
    results = []
    
    for i, symbol in enumerate(symbols):
        df = fetch_stock_data(symbol)
        if df is not None and len(df) > 20:
            current_price = df['Close'].iloc[-1]
            price_5d_ago = df['Close'].iloc[-5] if len(df) > 5 else current_price
            price_20d_ago = df['Close'].iloc[-20] if len(df) > 20 else current_price
            
            change_5d = ((current_price / price_5d_ago) - 1) * 100
            change_20d = ((current_price / price_20d_ago) - 1) * 100
            
            avg_vol = df['Volume'].tail(20).mean()
            today_vol = df['Volume'].iloc[-1]
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1
            
            rsi = calculate_rsi(df).iloc[-1]
            
            results.append({
                'Symbol': symbol,
                'Price': current_price,
                'Change_5D': change_5d,
                'Change_20D': change_20d,
                'Volume_Ratio': vol_ratio,
                'RSI': rsi,
                'Momentum': 'STRONG' if change_5d > 5 and vol_ratio > 1.5 else ('WEAK' if change_5d < -5 else 'MODERATE')
            })
        progress.progress((i + 1) / len(symbols))
    
    progress.empty()
    
    if not results:
        st.warning("No results found")
        return
    
    df_results = pd.DataFrame(results).sort_values('Change_5D', ascending=False)
    
    # Summary
    strong = len(df_results[df_results['Momentum'] == 'STRONG'])
    weak = len(df_results[df_results['Momentum'] == 'WEAK'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🚀 Strong Momentum", strong)
    col2.metric("📉 Weak Momentum", weak)
    col3.metric("➡️ Moderate", len(results) - strong - weak)
    
    # Chart
    fig = go.Figure()
    
    colors = ['#10b981' if m == 'STRONG' else ('#ef4444' if m == 'WEAK' else '#f59e0b') 
              for m in df_results['Momentum']]
    
    fig.add_trace(go.Bar(
        x=df_results['Symbol'],
        y=df_results['Change_5D'],
        marker_color=colors,
        text=[f"{c:+.1f}%" for c in df_results['Change_5D']],
        textposition='auto',
        name='5D Change'
    ))
    
    fig.update_layout(
        title="5-Day Price Change",
        yaxis_title="Change (%)",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    for _, row in df_results.iterrows():
        emoji = '🚀' if row['Momentum'] == 'STRONG' else ('📉' if row['Momentum'] == 'WEAK' else '➡️')
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{emoji} {row['Symbol']}** @ ${row['Price']:.2f}")
        with col2:
            st.metric("5D", f"{row['Change_5D']:+.1f}%")
        with col3:
            st.metric("20D", f"{row['Change_20D']:+.1f}%")
        with col4:
            st.metric("Vol Ratio", f"{row['Volume_Ratio']:.1f}x")
        with col5:
            st.metric("RSI", f"{row['RSI']:.0f}")
        
        st.divider()


# ==================== MAIN APP ====================
def main():
    st.title("🔎 Multi-Scanner")
    st.caption("Technical and options-based market scanners")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Scanner Settings")
        
        universe = st.selectbox("Stock Universe", list(STOCK_UNIVERSES.keys()), index=9)
        symbols = STOCK_UNIVERSES[universe]
        
        custom = st.text_input("Add Custom Symbols (comma-separated)")
        if custom:
            symbols = symbols + [s.strip().upper() for s in custom.split(',')]
        
        st.caption(f"Scanning {len(symbols)} symbols")
        
        if st.button("🔄 Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 EMA Cloud", "📊 IV Scanner", "🚀 Momentum"])
    
    with tab1:
        render_ema_scanner_tab(symbols)
    
    with tab2:
        render_iv_scanner_tab(symbols)
    
    with tab3:
        render_momentum_scanner_tab(symbols)


if __name__ == "__main__":
    main()
