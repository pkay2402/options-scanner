#!/usr/bin/env python3
"""
Smart Money Flow - Consolidated Options Flow Analysis
Combines: Whale Flows, Flow Scanner, Options Flow Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client

# Page config
st.set_page_config(
    page_title="Smart Money Flow",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Watchlists
TOP_STOCKS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 
              'AMD', 'INTC', 'PLTR', 'COIN', 'MSTR', 'HOOD', 'SOFI', 'SQ', 'PYPL',
              'XLF', 'XLE', 'XLK', 'XLV', 'IWM', 'TLT', 'GLD', 'SLV', 'USO', 'DIA']

# CSS
st.markdown("""
<style>
    .whale-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
    }
    .bullish-card { border-left: 4px solid #10b981; }
    .bearish-card { border-left: 4px solid #ef4444; }
    .whale-premium { font-size: 1.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==================== DATA FETCHING ====================
@st.cache_data(ttl=120)
def fetch_symbol_flow(symbol):
    """Fetch options flow data for a symbol"""
    try:
        client = get_client()
        if not client:
            return None, 0
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=30)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, 0
        
        underlying_price = chain.get('underlyingPrice', 0)
        return chain, underlying_price
    except Exception as e:
        return None, 0


def analyze_flow(chain, underlying_price, symbol):
    """Analyze options flow for whale activity and sentiment"""
    if not chain:
        return None
    
    calls_flow, puts_flow = [], []
    total_call_vol, total_put_vol = 0, 0
    total_call_premium, total_put_premium = 0, 0
    
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                vol = c.get('totalVolume', 0)
                oi = c.get('openInterest', 0)
                mark = c.get('mark', 0)
                premium = vol * mark * 100
                
                total_call_vol += vol
                total_call_premium += premium
                
                # Whale criteria: premium > $100K or vol/oi > 3
                vol_oi_ratio = vol / oi if oi > 0 else 0
                if premium > 100000 or (vol_oi_ratio > 3 and premium > 25000):
                    calls_flow.append({
                        'type': 'CALL', 'symbol': symbol,
                        'strike': float(strike_str), 'expiry': exp_key,
                        'volume': vol, 'oi': oi, 'premium': premium,
                        'vol_oi': vol_oi_ratio, 'mark': mark,
                        'delta': c.get('delta', 0), 'iv': c.get('volatility', 0)
                    })
    
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                vol = c.get('totalVolume', 0)
                oi = c.get('openInterest', 0)
                mark = c.get('mark', 0)
                premium = vol * mark * 100
                
                total_put_vol += vol
                total_put_premium += premium
                
                vol_oi_ratio = vol / oi if oi > 0 else 0
                if premium > 100000 or (vol_oi_ratio > 3 and premium > 25000):
                    puts_flow.append({
                        'type': 'PUT', 'symbol': symbol,
                        'strike': float(strike_str), 'expiry': exp_key,
                        'volume': vol, 'oi': oi, 'premium': premium,
                        'vol_oi': vol_oi_ratio, 'mark': mark,
                        'delta': c.get('delta', 0), 'iv': c.get('volatility', 0)
                    })
    
    # Calculate metrics
    total_vol = total_call_vol + total_put_vol
    pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    net_premium = total_call_premium - total_put_premium
    
    return {
        'symbol': symbol,
        'price': underlying_price,
        'call_vol': total_call_vol,
        'put_vol': total_put_vol,
        'pc_ratio': pc_ratio,
        'call_premium': total_call_premium,
        'put_premium': total_put_premium,
        'net_premium': net_premium,
        'whale_calls': sorted(calls_flow, key=lambda x: x['premium'], reverse=True)[:10],
        'whale_puts': sorted(puts_flow, key=lambda x: x['premium'], reverse=True)[:10]
    }


# ==================== WHALE SCANNER TAB ====================
def render_whale_scanner_tab(symbols):
    """Scan multiple symbols for whale activity"""
    st.subheader("🐋 Whale Flow Scanner")
    
    progress = st.progress(0)
    all_whales = []
    summaries = []
    
    for i, symbol in enumerate(symbols):
        chain, price = fetch_symbol_flow(symbol)
        if chain:
            flow = analyze_flow(chain, price, symbol)
            if flow:
                summaries.append(flow)
                all_whales.extend(flow['whale_calls'])
                all_whales.extend(flow['whale_puts'])
        progress.progress((i + 1) / len(symbols))
    
    progress.empty()
    
    if not all_whales:
        st.info("No whale trades detected in current scan")
        return
    
    # Sort all whales by premium
    all_whales = sorted(all_whales, key=lambda x: x['premium'], reverse=True)[:30]
    
    # Display whale trades
    st.markdown(f"**Found {len(all_whales)} whale trades across {len(symbols)} symbols**")
    
    for whale in all_whales:
        is_call = whale['type'] == 'CALL'
        card_class = 'bullish-card' if is_call else 'bearish-card'
        emoji = '🟢' if is_call else '🔴'
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{emoji} {whale['symbol']}** {whale['type']}")
            st.caption(f"Strike: ${whale['strike']:.0f} | Exp: {whale['expiry']}")
        
        with col2:
            st.metric("Premium", f"${whale['premium']/1000:.0f}K")
        
        with col3:
            st.metric("Volume", f"{whale['volume']:,}")
            st.caption(f"Vol/OI: {whale['vol_oi']:.1f}x")
        
        with col4:
            st.metric("IV", f"{whale['iv']:.0f}%")
        
        st.divider()


# ==================== FLOW SUMMARY TAB ====================
def render_flow_summary_tab(symbols):
    """Show aggregated flow summary"""
    st.subheader("📊 Market Flow Summary")
    
    progress = st.progress(0)
    summaries = []
    
    for i, symbol in enumerate(symbols):
        chain, price = fetch_symbol_flow(symbol)
        if chain:
            flow = analyze_flow(chain, price, symbol)
            if flow:
                summaries.append(flow)
        progress.progress((i + 1) / len(symbols))
    
    progress.empty()
    
    if not summaries:
        st.info("No flow data available")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame([{
        'Symbol': s['symbol'],
        'Price': s['price'],
        'Call Vol': s['call_vol'],
        'Put Vol': s['put_vol'],
        'P/C Ratio': s['pc_ratio'],
        'Call Premium': s['call_premium'],
        'Put Premium': s['put_premium'],
        'Net Premium': s['net_premium'],
        'Sentiment': 'BULLISH' if s['net_premium'] > 0 else 'BEARISH'
    } for s in summaries])
    
    # Metrics
    total_call_prem = df['Call Premium'].sum()
    total_put_prem = df['Put Premium'].sum()
    net_market = total_call_prem - total_put_prem
    bullish_count = (df['Net Premium'] > 0).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Market Flow", f"${net_market/1e6:.1f}M", 
                "Bullish" if net_market > 0 else "Bearish")
    col2.metric("Total Call Premium", f"${total_call_prem/1e6:.1f}M")
    col3.metric("Total Put Premium", f"${total_put_prem/1e6:.1f}M")
    col4.metric("Bullish Symbols", f"{bullish_count}/{len(df)}")
    
    # Flow Chart
    fig = go.Figure()
    
    df_sorted = df.sort_values('Net Premium', ascending=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['Net Premium']]
    
    fig.add_trace(go.Bar(
        y=df_sorted['Symbol'],
        x=df_sorted['Net Premium'] / 1e6,
        orientation='h',
        marker_color=colors,
        text=[f"${x/1e6:.1f}M" for x in df_sorted['Net Premium']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Net Premium Flow by Symbol",
        xaxis_title="Net Premium (Millions)",
        template='plotly_dark',
        height=max(400, len(df) * 25)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown("**Detailed Flow Data**")
    df_display = df.copy()
    df_display['Call Premium'] = df_display['Call Premium'].apply(lambda x: f"${x/1000:.0f}K")
    df_display['Put Premium'] = df_display['Put Premium'].apply(lambda x: f"${x/1000:.0f}K")
    df_display['Net Premium'] = df_display['Net Premium'].apply(lambda x: f"${x/1000:+.0f}K")
    df_display['P/C Ratio'] = df_display['P/C Ratio'].apply(lambda x: f"{x:.2f}")
    df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ==================== SINGLE SYMBOL TAB ====================
def render_single_symbol_tab():
    """Deep dive into single symbol flow"""
    st.subheader("🔍 Single Symbol Analysis")
    
    symbol = st.text_input("Enter Symbol", value="SPY", key="flow_symbol").upper().strip()
    
    if not symbol:
        return
    
    chain, price = fetch_symbol_flow(symbol)
    
    if not chain:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    flow = analyze_flow(chain, price, symbol)
    
    if not flow:
        st.warning("No flow data available")
        return
    
    # Header
    st.header(f"📈 {symbol} @ ${price:.2f}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Call Volume", f"{flow['call_vol']:,}")
    col2.metric("Put Volume", f"{flow['put_vol']:,}")
    col3.metric("P/C Ratio", f"{flow['pc_ratio']:.2f}")
    col4.metric("Net Premium", f"${flow['net_premium']/1000:+.0f}K",
                "Bullish" if flow['net_premium'] > 0 else "Bearish")
    
    # Premium breakdown
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                        subplot_titles=('Premium Split', 'Volume by Type'))
    
    fig.add_trace(go.Pie(
        values=[flow['call_premium'], flow['put_premium']],
        labels=['Calls', 'Puts'],
        marker_colors=['#10b981', '#ef4444'],
        hole=0.4
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=['Calls', 'Puts'],
        y=[flow['call_vol'], flow['put_vol']],
        marker_color=['#10b981', '#ef4444']
    ), row=1, col=2)
    
    fig.update_layout(template='plotly_dark', height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Whale trades
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Top Call Flow**")
        for whale in flow['whale_calls'][:5]:
            st.markdown(f"""
            **${whale['strike']:.0f} {whale['expiry']}**  
            Premium: ${whale['premium']/1000:.0f}K | Vol: {whale['volume']:,} | IV: {whale['iv']:.0f}%
            """)
    
    with col2:
        st.markdown("**🔴 Top Put Flow**")
        for whale in flow['whale_puts'][:5]:
            st.markdown(f"""
            **${whale['strike']:.0f} {whale['expiry']}**  
            Premium: ${whale['premium']/1000:.0f}K | Vol: {whale['volume']:,} | IV: {whale['iv']:.0f}%
            """)


# ==================== MAIN APP ====================
def main():
    st.title("🐋 Smart Money Flow")
    st.caption("Track institutional and whale options activity in real-time")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        watchlist = st.multiselect(
            "Select Symbols",
            options=TOP_STOCKS,
            default=['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'AMD', 'META'],
            key="flow_watchlist"
        )
        
        custom = st.text_input("Add Custom Symbols (comma-separated)")
        if custom:
            watchlist.extend([s.strip().upper() for s in custom.split(',')])
        
        if st.button("🔄 Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🐋 Whale Scanner", "📊 Flow Summary", "�� Single Symbol"])
    
    with tab1:
        render_whale_scanner_tab(watchlist)
    
    with tab2:
        render_flow_summary_tab(watchlist)
    
    with tab3:
        render_single_symbol_tab()


if __name__ == "__main__":
    main()
