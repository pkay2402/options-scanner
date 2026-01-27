#!/usr/bin/env python3
"""
Gamma Intelligence - GEX Analysis and Market Structure
Redesigned to match professional GEX heatmap layout
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
    initial_sidebar_state="collapsed"
)

# CSS for professional look
st.markdown("""
<style>
    .metric-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 20px;
        font-weight: bold;
    }
    .metric-delta {
        color: #64748b;
        font-size: 12px;
    }
    .positive { color: #10b981 !important; }
    .negative { color: #ef4444 !important; }
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0 4px;
    }
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
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=80)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, 0
        
        return chain, chain.get('underlyingPrice', 0)
    except Exception as e:
        st.error(f"Error: {e}")
        return None, 0


def calculate_gex_matrix(chain, underlying_price):
    """Calculate GEX for each strike x expiry combination"""
    if not chain:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    call_gex_data = {}
    put_gex_data = {}
    net_gex_data = {}
    
    # Process calls
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                strike = float(strike_str)
                gamma = c.get('gamma', 0) or 0
                oi = c.get('openInterest', 0) or 0
                call_gex = gamma * oi * underlying_price * 100
                
                if strike not in call_gex_data:
                    call_gex_data[strike] = {}
                call_gex_data[strike][exp_key] = call_gex
    
    # Process puts
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                strike = float(strike_str)
                gamma = c.get('gamma', 0) or 0
                oi = c.get('openInterest', 0) or 0
                put_gex = -gamma * oi * underlying_price * 100
                
                if strike not in put_gex_data:
                    put_gex_data[strike] = {}
                put_gex_data[strike][exp_key] = put_gex
    
    # Create DataFrames
    call_df = pd.DataFrame(call_gex_data).T.sort_index(ascending=False)
    put_df = pd.DataFrame(put_gex_data).T.sort_index(ascending=False)
    
    # Align columns
    all_expiries = sorted(set(call_df.columns.tolist() + put_df.columns.tolist()))
    call_df = call_df.reindex(columns=all_expiries, fill_value=0)
    put_df = put_df.reindex(columns=all_expiries, fill_value=0)
    
    # Align indices
    all_strikes = sorted(set(call_df.index.tolist() + put_df.index.tolist()), reverse=True)
    call_df = call_df.reindex(all_strikes, fill_value=0)
    put_df = put_df.reindex(all_strikes, fill_value=0)
    
    # Net GEX
    net_df = call_df + put_df
    
    return call_df, put_df, net_df


def calculate_key_levels(net_df, call_df, put_df, underlying_price):
    """Calculate key GEX levels"""
    if net_df.empty:
        return {}
    
    # Sum across expiries for each strike
    strike_totals = net_df.sum(axis=1)
    call_totals = call_df.sum(axis=1)
    put_totals = put_df.sum(axis=1)
    
    # Net GEX
    net_gex = strike_totals.sum()
    
    # Max positive GEX strike
    max_pos_strike = strike_totals.idxmax() if not strike_totals.empty else underlying_price
    max_pos_gex = strike_totals.max()
    
    # Max negative GEX strike
    max_neg_strike = strike_totals.idxmin() if not strike_totals.empty else underlying_price
    max_neg_gex = strike_totals.min()
    
    # Call wall (max call GEX above price)
    above_price = call_totals[call_totals.index >= underlying_price]
    call_wall = above_price.idxmax() if not above_price.empty else underlying_price
    call_wall_gex = above_price.max() if not above_price.empty else 0
    
    # Put wall (max put GEX below price)
    below_price = put_totals[put_totals.index <= underlying_price]
    put_wall = below_price.abs().idxmax() if not below_price.empty else underlying_price
    put_wall_gex = below_price.loc[put_wall] if put_wall in below_price.index else 0
    
    # Gamma flip (where net GEX crosses zero near price)
    gamma_flip = underlying_price
    sorted_strikes = strike_totals.sort_index()
    for i in range(len(sorted_strikes) - 1):
        s1, s2 = sorted_strikes.index[i], sorted_strikes.index[i+1]
        v1, v2 = sorted_strikes.iloc[i], sorted_strikes.iloc[i+1]
        if v1 * v2 < 0 and s1 <= underlying_price <= s2:
            # Linear interpolation
            gamma_flip = s1 + (s2 - s1) * abs(v1) / (abs(v1) + abs(v2))
            break
    
    return {
        'net_gex': net_gex,
        'max_pos_strike': max_pos_strike,
        'max_pos_gex': max_pos_gex,
        'max_neg_strike': max_neg_strike,
        'max_neg_gex': max_neg_gex,
        'call_wall': call_wall,
        'call_wall_gex': call_wall_gex,
        'put_wall': put_wall,
        'put_wall_gex': put_wall_gex,
        'gamma_flip': gamma_flip
    }


def format_gex_value(value):
    """Format GEX value for display"""
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.0f}K"
    else:
        return f"{value:.0f}"


def filter_expiries(df, filter_type, today=None):
    """Filter expiries based on type"""
    if df.empty:
        return df
    
    if today is None:
        today = datetime.now().date()
    
    if filter_type == "All":
        return df
    
    filtered_cols = []
    for col in df.columns:
        try:
            exp_date = datetime.strptime(col, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            
            if filter_type == "0DTE" and days_to_exp == 0:
                filtered_cols.append(col)
            elif filter_type == "Weekly" and days_to_exp <= 7:
                filtered_cols.append(col)
            elif filter_type == "Monthly" and days_to_exp <= 30:
                filtered_cols.append(col)
        except:
            continue
    
    if filtered_cols:
        return df[filtered_cols]
    return df


# ==================== MAIN APP ====================
def main():
    # Title row with symbol input
    col_title, col_symbol, col_refresh = st.columns([3, 2, 1])
    
    with col_title:
        st.markdown("## Gamma Exposure (GEX) Analysis")
    
    with col_symbol:
        symbol = st.text_input("Symbol", value="SPY", key="gex_symbol", label_visibility="collapsed").upper().strip()
    
    with col_refresh:
        refresh = st.button("🔄 Refresh", use_container_width=True)
        if refresh:
            st.cache_data.clear()
            st.rerun()
    
    if not symbol:
        st.warning("Enter a symbol")
        return
    
    # Fetch data
    with st.spinner(f"Loading {symbol} data..."):
        chain, underlying_price = fetch_chain_for_gex(symbol)
    
    if not chain:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    # Calculate GEX matrices
    call_df, put_df, net_df = calculate_gex_matrix(chain, underlying_price)
    
    if net_df.empty:
        st.error("No GEX data available")
        return
    
    # Filter to reasonable strike range (±10%)
    price_range = underlying_price * 0.10
    valid_strikes = [s for s in net_df.index if underlying_price - price_range <= s <= underlying_price + price_range]
    
    call_df = call_df.loc[call_df.index.isin(valid_strikes)]
    put_df = put_df.loc[put_df.index.isin(valid_strikes)]
    net_df = net_df.loc[net_df.index.isin(valid_strikes)]
    
    # Calculate key levels
    levels = calculate_key_levels(net_df, call_df, put_df, underlying_price)
    
    # ==================== METRICS ROW ====================
    st.markdown("---")
    
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${underlying_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        gamma_flip = levels.get('gamma_flip', underlying_price)
        flip_pct = ((gamma_flip - underlying_price) / underlying_price) * 100
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Gamma Flip</div>
            <div class="metric-value">${gamma_flip:.2f}</div>
            <div class="metric-delta">({flip_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        call_wall = levels.get('call_wall', underlying_price)
        cw_pct = ((call_wall - underlying_price) / underlying_price) * 100
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Call Wall</div>
            <div class="metric-value positive">${call_wall:.0f}</div>
            <div class="metric-delta">({cw_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        put_wall = levels.get('put_wall', underlying_price)
        pw_pct = ((put_wall - underlying_price) / underlying_price) * 100
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Put Wall</div>
            <div class="metric-value negative">${put_wall:.0f}</div>
            <div class="metric-delta">({pw_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m5:
        net_gex = levels.get('net_gex', 0)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Net GEX</div>
            <div class="metric-value {'positive' if net_gex > 0 else 'negative'}">${format_gex_value(net_gex)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m6:
        max_pos = levels.get('max_pos_strike', underlying_price)
        max_pos_gex = levels.get('max_pos_gex', 0)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Max +GEX</div>
            <div class="metric-value positive">${max_pos:.0f}</div>
            <div class="metric-delta">${format_gex_value(max_pos_gex)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m7:
        max_neg = levels.get('max_neg_strike', underlying_price)
        max_neg_gex = levels.get('max_neg_gex', 0)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Max -GEX</div>
            <div class="metric-value negative">${max_neg:.0f}</div>
            <div class="metric-delta">-${format_gex_value(abs(max_neg_gex))}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== FILTER BUTTONS ====================
    col_filters, col_view, col_type = st.columns([2, 2, 2])
    
    with col_filters:
        exp_filter = st.radio(
            "Expiry Filter",
            options=["All", "0DTE", "Weekly", "Monthly"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_view:
        view_type = st.radio(
            "View",
            options=["Heatmap", "Bar"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_type:
        gex_type = st.radio(
            "GEX Type",
            options=["Net", "Call", "Put"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Apply filters
    filtered_call = filter_expiries(call_df, exp_filter)
    filtered_put = filter_expiries(put_df, exp_filter)
    filtered_net = filter_expiries(net_df, exp_filter)
    
    # Select data based on type
    if gex_type == "Call":
        display_df = filtered_call
    elif gex_type == "Put":
        display_df = filtered_put
    else:
        display_df = filtered_net
    
    if display_df.empty:
        st.warning("No data for selected filters")
        return
    
    # ==================== VISUALIZATION ====================
    if view_type == "Heatmap":
        # Format column names (expiry dates) to shorter format
        display_df_copy = display_df.copy()
        new_cols = []
        for col in display_df_copy.columns:
            try:
                d = datetime.strptime(col, '%Y-%m-%d')
                new_cols.append(d.strftime('%-m/%-d'))
            except:
                new_cols.append(col)
        display_df_copy.columns = new_cols
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=display_df_copy.values,
            x=display_df_copy.columns,
            y=[f"${s:.0f}" for s in display_df_copy.index],
            colorscale=[
                [0, '#ef4444'],      # Red for negative
                [0.5, '#1e293b'],    # Dark for zero
                [1, '#10b981']       # Green for positive
            ],
            zmid=0,
            text=[[format_gex_value(v) for v in row] for row in display_df_copy.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>GEX: %{text}<extra></extra>",
            colorbar=dict(
                title="GEX",
                tickformat=".0s"
            )
        ))
        
        # Add horizontal line for current price
        price_idx = None
        for i, s in enumerate(display_df_copy.index):
            if s <= underlying_price:
                price_idx = i
                break
        
        fig.update_layout(
            template='plotly_dark',
            height=max(500, len(display_df_copy) * 22),
            xaxis_title="Expiration",
            yaxis_title="Strike",
            yaxis=dict(tickmode='array', tickvals=list(range(len(display_df_copy.index))), 
                      ticktext=[f"${s:.0f}" for s in display_df_copy.index]),
            margin=dict(l=80, r=20, t=20, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Bar chart view
        strike_totals = display_df.sum(axis=1)
        
        fig = go.Figure()
        
        colors = ['#10b981' if x > 0 else '#ef4444' for x in strike_totals]
        
        fig.add_trace(go.Bar(
            x=strike_totals.index,
            y=strike_totals.values / 1e9,
            marker_color=colors,
            text=[format_gex_value(v) for v in strike_totals.values],
            textposition='auto'
        ))
        
        # Current price line
        fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow",
                      annotation_text=f"${underlying_price:.2f}")
        
        fig.update_layout(
            title=f"{gex_type} GEX by Strike",
            xaxis_title="Strike Price",
            yaxis_title="GEX (Billions $)",
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== GEX BY EXPIRY SUMMARY ====================
    with st.expander("📅 GEX by Expiration Summary", expanded=False):
        exp_totals = filtered_net.sum(axis=0)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_exp = go.Figure()
            colors = ['#10b981' if x > 0 else '#ef4444' for x in exp_totals]
            
            fig_exp.add_trace(go.Bar(
                x=exp_totals.index,
                y=exp_totals.values / 1e9,
                marker_color=colors,
                text=[format_gex_value(v) for v in exp_totals.values],
                textposition='auto'
            ))
            
            fig_exp.update_layout(
                xaxis_title="Expiration",
                yaxis_title="Net GEX (Billions)",
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            st.markdown("**GEX by Expiry**")
            for exp, val in exp_totals.items():
                color = "🟢" if val > 0 else "🔴"
                st.markdown(f"{color} **{exp}**: {format_gex_value(val)}")


if __name__ == "__main__":
    main()
