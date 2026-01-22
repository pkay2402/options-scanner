#!/usr/bin/env python3
"""
Gamma Intelligence - GEX Analysis and Market Structure
Combines: GEX Analysis, Z-Score, Volume Walls
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
    page_title="Gamma Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .gex-positive { color: #10b981; font-weight: bold; }
    .gex-negative { color: #ef4444; font-weight: bold; }
    .strike-wall { border-left: 4px solid #f59e0b; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)


# ==================== DATA FETCHING ====================
@st.cache_data(ttl=300)
def fetch_chain_for_gex(symbol):
    """Fetch options chain for GEX calculation"""
    try:
        client = get_client()
        if not client:
            return None, 0
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=60)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, 0
        
        return chain, chain.get('underlyingPrice', 0)
    except Exception as e:
        st.error(f"Error: {e}")
        return None, 0


def calculate_gex_by_strike(chain, underlying_price):
    """Calculate GEX for each strike"""
    if not chain:
        return pd.DataFrame()
    
    gex_data = []
    
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                strike = float(strike_str)
                gamma = c.get('gamma', 0) or 0
                oi = c.get('openInterest', 0) or 0
                
                # Call GEX is positive (dealers are short calls, long gamma)
                call_gex = gamma * oi * underlying_price * 100
                
                gex_data.append({
                    'strike': strike,
                    'expiry': exp_key,
                    'call_gamma': gamma,
                    'call_oi': oi,
                    'call_gex': call_gex,
                    'type': 'call'
                })
    
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                strike = float(strike_str)
                gamma = c.get('gamma', 0) or 0
                oi = c.get('openInterest', 0) or 0
                
                # Put GEX is negative (dealers are short puts, short gamma)
                put_gex = -gamma * oi * underlying_price * 100
                
                gex_data.append({
                    'strike': strike,
                    'expiry': exp_key,
                    'put_gamma': gamma,
                    'put_oi': oi,
                    'put_gex': put_gex,
                    'type': 'put'
                })
    
    if not gex_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(gex_data)
    return df


def aggregate_gex(df, by='strike'):
    """Aggregate GEX by strike or expiry"""
    if df.empty:
        return pd.DataFrame()
    
    calls = df[df['type'] == 'call'].groupby(by).agg({
        'call_gex': 'sum',
        'call_oi': 'sum'
    }).reset_index()
    
    puts = df[df['type'] == 'put'].groupby(by).agg({
        'put_gex': 'sum',
        'put_oi': 'sum'
    }).reset_index()
    
    merged = pd.merge(calls, puts, on=by, how='outer').fillna(0)
    merged['net_gex'] = merged['call_gex'] + merged['put_gex']
    merged['total_oi'] = merged['call_oi'] + merged['put_oi']
    
    return merged


# ==================== GEX PROFILE TAB ====================
def render_gex_profile_tab(symbol, chain, underlying_price):
    """Render GEX profile visualization"""
    st.subheader("⚡ GEX Profile")
    
    if not chain:
        st.warning("No data available")
        return
    
    gex_df = calculate_gex_by_strike(chain, underlying_price)
    
    if gex_df.empty:
        st.warning("Could not calculate GEX")
        return
    
    agg_df = aggregate_gex(gex_df, 'strike')
    
    # Filter to strikes within 15% of price
    price_range = underlying_price * 0.15
    agg_df = agg_df[(agg_df['strike'] >= underlying_price - price_range) & 
                    (agg_df['strike'] <= underlying_price + price_range)]
    
    # Net GEX
    net_gex = agg_df['net_gex'].sum()
    gex_regime = "Positive Gamma (Stable)" if net_gex > 0 else "Negative Gamma (Volatile)"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${underlying_price:.2f}")
    col2.metric("Net GEX", f"${net_gex/1e9:.2f}B", gex_regime)
    col3.metric("GEX Regime", "🟢 Stable" if net_gex > 0 else "🔴 Volatile")
    
    # Find key levels
    max_call_gex = agg_df.loc[agg_df['call_gex'].idxmax()] if not agg_df.empty else None
    max_put_gex = agg_df.loc[agg_df['put_gex'].abs().idxmax()] if not agg_df.empty else None
    
    if max_call_gex is not None:
        st.info(f"**Call Wall (Resistance):** ${max_call_gex['strike']:.0f} - GEX: ${max_call_gex['call_gex']/1e9:.2f}B")
    if max_put_gex is not None:
        st.warning(f"**Put Wall (Support):** ${max_put_gex['strike']:.0f} - GEX: ${abs(max_put_gex['put_gex'])/1e9:.2f}B")
    
    # GEX Chart
    fig = go.Figure()
    
    # Call GEX (positive)
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['call_gex'] / 1e9,
        name='Call GEX',
        marker_color='#10b981'
    ))
    
    # Put GEX (negative)
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['put_gex'] / 1e9,
        name='Put GEX',
        marker_color='#ef4444'
    ))
    
    # Current price line
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow",
                  annotation_text=f"${underlying_price:.2f}")
    
    fig.update_layout(
        title="GEX by Strike",
        xaxis_title="Strike Price",
        yaxis_title="GEX (Billions $)",
        barmode='relative',
        template='plotly_dark',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Net GEX Profile
    st.subheader("Net GEX Profile")
    
    fig2 = go.Figure()
    colors = ['#10b981' if x > 0 else '#ef4444' for x in agg_df['net_gex']]
    
    fig2.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['net_gex'] / 1e9,
        marker_color=colors,
        name='Net GEX'
    ))
    
    fig2.add_vline(x=underlying_price, line_dash="dash", line_color="yellow")
    
    fig2.update_layout(
        xaxis_title="Strike Price",
        yaxis_title="Net GEX (Billions $)",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)


# ==================== VOLUME WALLS TAB ====================
def render_volume_walls_tab(symbol, chain, underlying_price):
    """Render OI volume walls"""
    st.subheader("🧱 Volume Walls (OI Concentrations)")
    
    if not chain:
        st.warning("No data available")
        return
    
    gex_df = calculate_gex_by_strike(chain, underlying_price)
    
    if gex_df.empty:
        st.warning("No data available")
        return
    
    agg_df = aggregate_gex(gex_df, 'strike')
    
    # Filter range
    price_range = underlying_price * 0.15
    agg_df = agg_df[(agg_df['strike'] >= underlying_price - price_range) & 
                    (agg_df['strike'] <= underlying_price + price_range)]
    
    # Find top OI strikes
    top_call_oi = agg_df.nlargest(5, 'call_oi')
    top_put_oi = agg_df.nlargest(5, 'put_oi')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Call Walls (Resistance)**")
        for _, row in top_call_oi.iterrows():
            dist = ((row['strike'] - underlying_price) / underlying_price) * 100
            st.markdown(f"""
            **${row['strike']:.0f}** ({dist:+.1f}%)  
            OI: {row['call_oi']:,.0f} | GEX: ${row['call_gex']/1e6:.1f}M
            """)
    
    with col2:
        st.markdown("**🔴 Put Walls (Support)**")
        for _, row in top_put_oi.iterrows():
            dist = ((row['strike'] - underlying_price) / underlying_price) * 100
            st.markdown(f"""
            **${row['strike']:.0f}** ({dist:+.1f}%)  
            OI: {row['put_oi']:,.0f} | GEX: ${abs(row['put_gex'])/1e6:.1f}M
            """)
    
    # Combined OI Chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['call_oi'],
        name='Call OI',
        marker_color='#10b981',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['put_oi'],
        name='Put OI',
        marker_color='#ef4444',
        opacity=0.7
    ))
    
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow",
                  annotation_text=f"${underlying_price:.2f}")
    
    fig.update_layout(
        title="Open Interest Distribution",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        barmode='group',
        template='plotly_dark',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================== GEX BY EXPIRY TAB ====================
def render_gex_by_expiry_tab(symbol, chain, underlying_price):
    """Show GEX breakdown by expiration"""
    st.subheader("📅 GEX by Expiration")
    
    if not chain:
        st.warning("No data available")
        return
    
    gex_df = calculate_gex_by_strike(chain, underlying_price)
    
    if gex_df.empty:
        st.warning("No data available")
        return
    
    agg_df = aggregate_gex(gex_df, 'expiry')
    agg_df = agg_df.sort_values('expiry')
    
    # Metrics
    total_gex = agg_df['net_gex'].sum()
    nearest_exp = agg_df.iloc[0] if not agg_df.empty else None
    
    col1, col2 = st.columns(2)
    col1.metric("Total GEX", f"${total_gex/1e9:.2f}B")
    if nearest_exp is not None:
        col2.metric(f"Nearest Exp ({nearest_exp['expiry']})", f"${nearest_exp['net_gex']/1e9:.2f}B")
    
    # Chart
    fig = go.Figure()
    
    colors = ['#10b981' if x > 0 else '#ef4444' for x in agg_df['net_gex']]
    
    fig.add_trace(go.Bar(
        x=agg_df['expiry'],
        y=agg_df['net_gex'] / 1e9,
        marker_color=colors,
        text=[f"${x/1e9:.2f}B" for x in agg_df['net_gex']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Net GEX by Expiration Date",
        xaxis_title="Expiration",
        yaxis_title="Net GEX (Billions $)",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown("**GEX Breakdown by Expiry**")
    display_df = agg_df[['expiry', 'call_gex', 'put_gex', 'net_gex', 'total_oi']].copy()
    display_df['call_gex'] = display_df['call_gex'].apply(lambda x: f"${x/1e9:.2f}B")
    display_df['put_gex'] = display_df['put_gex'].apply(lambda x: f"${x/1e9:.2f}B")
    display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"${x/1e9:.2f}B")
    display_df['total_oi'] = display_df['total_oi'].apply(lambda x: f"{x:,.0f}")
    display_df.columns = ['Expiry', 'Call GEX', 'Put GEX', 'Net GEX', 'Total OI']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==================== MAIN APP ====================
def main():
    st.title("⚡ Gamma Intelligence")
    st.caption("GEX analysis, volume walls, and market structure insights")
    
    # Symbol input
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Symbol", value="SPY", key="gex_symbol").upper().strip()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if not symbol:
        st.warning("Enter a symbol to begin")
        return
    
    # Fetch data
    with st.spinner(f"Loading {symbol} GEX data..."):
        chain, underlying_price = fetch_chain_for_gex(symbol)
    
    if not chain:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    st.header(f"📈 {symbol} @ ${underlying_price:.2f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚡ GEX Profile", "🧱 Volume Walls", "📅 By Expiry"])
    
    with tab1:
        render_gex_profile_tab(symbol, chain, underlying_price)
    
    with tab2:
        render_volume_walls_tab(symbol, chain, underlying_price)
    
    with tab3:
        render_gex_by_expiry_tab(symbol, chain, underlying_price)


if __name__ == "__main__":
    main()
