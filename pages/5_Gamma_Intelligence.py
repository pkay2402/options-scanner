#!/usr/bin/env python3
"""
Gamma Intelligence - GEX Analysis and Market Structure
Redesigned with clean heatmap layout (6 expiries default)
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
    .gex-guide {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    .gex-guide h4 { color: #f8fafc; margin-bottom: 8px; }
    .gex-guide p { color: #94a3b8; font-size: 13px; line-height: 1.5; }
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


def calculate_gex_matrix(chain, underlying_price, max_expiries=6, selected_expiry=None):
    """Calculate GEX for each strike x expiry combination"""
    if not chain:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    
    call_gex_data = {}
    put_gex_data = {}
    all_expiries = set()
    
    # Process calls
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        all_expiries.add(exp_key)
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
        all_expiries.add(exp_key)
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
    
    # Sort expiries and limit to next N
    sorted_expiries = sorted(all_expiries)
    
    # If specific expiry selected, only show that one
    if selected_expiry:
        filtered_expiries = [selected_expiry] if selected_expiry in sorted_expiries else sorted_expiries[:max_expiries]
    else:
        filtered_expiries = sorted_expiries[:max_expiries]
    
    # Create DataFrames with only filtered expiries
    call_df = pd.DataFrame(call_gex_data).T.sort_index(ascending=False)
    put_df = pd.DataFrame(put_gex_data).T.sort_index(ascending=False)
    
    # Filter to selected expiries only
    call_df = call_df[[c for c in filtered_expiries if c in call_df.columns]]
    put_df = put_df[[c for c in filtered_expiries if c in put_df.columns]]
    
    # Fill NaN with 0 for selected expiries
    call_df = call_df.fillna(0)
    put_df = put_df.fillna(0)
    
    # Align indices
    all_strikes = sorted(set(call_df.index.tolist() + put_df.index.tolist()), reverse=True)
    call_df = call_df.reindex(all_strikes, fill_value=0)
    put_df = put_df.reindex(all_strikes, fill_value=0)
    
    # Net GEX
    net_df = call_df + put_df
    
    return call_df, put_df, net_df, sorted_expiries


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
    if pd.isna(value) or value == 0:
        return ""
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.0f}K"
    else:
        return f"{value:.0f}"


# ==================== MAIN APP ====================
def main():
    # Title row with symbol input
    col_title, col_symbol, col_refresh = st.columns([3, 2, 1])
    
    with col_title:
        st.markdown("## ⚡ Gamma Exposure (GEX) Analysis")
    
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
    
    # GEX Guide - collapsible at top
    with st.expander("📖 What is GEX and How to Read This", expanded=False):
        st.markdown("""
        <div class="gex-guide">
        <h4>🎯 GEX (Gamma Exposure) Explained</h4>
        <p>GEX measures how much dealers need to hedge when price moves. It predicts <b>volatility regime</b> and <b>key support/resistance levels</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🟢 Positive GEX (Green)**
            - Dealers are **long gamma** → they hedge BY selling rallies & buying dips
            - This **dampens** volatility → expect **range-bound, mean-reverting** price action
            - Price tends to gravitate toward high GEX strikes (magnetic effect)
            
            **🔴 Negative GEX (Red)**  
            - Dealers are **short gamma** → they hedge BY buying rallies & selling dips
            - This **amplifies** moves → expect **trending, volatile** price action
            - Price can move explosively through negative GEX zones
            """)
        
        with col2:
            st.markdown("""
            **📊 Key Levels to Watch**
            - **Call Wall**: Resistance - dealers sell here, hard to break above
            - **Put Wall**: Support - dealers buy here, hard to break below
            - **Gamma Flip**: Where GEX switches sign - volatility regime change
            - **Max +GEX**: Strongest magnetic price level (pin risk)
            - **Max -GEX**: Most volatile zone, avoid or trade momentum
            
            **🎯 Trading Implications**
            - Above gamma flip = sell premium, fade moves
            - Below gamma flip = buy premium, trade momentum
            """)
    
    # Fetch data first to get available expiries
    with st.spinner(f"Loading {symbol} data..."):
        chain, underlying_price = fetch_chain_for_gex(symbol)
    
    if not chain:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    # Get all available expiries for the date picker
    _, _, _, all_expiries = calculate_gex_matrix(chain, underlying_price, max_expiries=100, selected_expiry=None)
    
    st.markdown("---")
    
    # ==================== FILTER ROW ====================
    col_exp_count, col_specific_exp, col_view, col_type = st.columns([1.5, 2, 1.5, 1.5])
    
    with col_exp_count:
        num_expiries = st.selectbox(
            "Expiries to Show",
            options=[6, 8, 10, 12, "All"],
            index=0,
            help="Number of nearest expiries to display"
        )
    
    with col_specific_exp:
        expiry_options = ["All (Next " + str(num_expiries) + ")"] + all_expiries
        selected_expiry = st.selectbox(
            "Or Select Specific Expiry",
            options=expiry_options,
            index=0,
            help="Choose a specific expiration date"
        )
        selected_expiry = None if selected_expiry.startswith("All") else selected_expiry
    
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
    
    # Calculate GEX with selected parameters
    max_exp = 100 if num_expiries == "All" else int(num_expiries)
    call_df, put_df, net_df, _ = calculate_gex_matrix(chain, underlying_price, max_expiries=max_exp, selected_expiry=selected_expiry)
    
    if net_df.empty:
        st.error("No GEX data available")
        return
    
    # Filter to reasonable strike range (±8%)
    price_range = underlying_price * 0.08
    valid_strikes = [s for s in net_df.index if underlying_price - price_range <= s <= underlying_price + price_range]
    
    call_df = call_df.loc[call_df.index.isin(valid_strikes)]
    put_df = put_df.loc[put_df.index.isin(valid_strikes)]
    net_df = net_df.loc[net_df.index.isin(valid_strikes)]
    
    # Calculate key levels
    levels = calculate_key_levels(net_df, call_df, put_df, underlying_price)
    
    # ==================== METRICS ROW ====================
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
            <div class="metric-delta">{format_gex_value(max_pos_gex)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m7:
        max_neg = levels.get('max_neg_strike', underlying_price)
        max_neg_gex = levels.get('max_neg_gex', 0)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Max -GEX</div>
            <div class="metric-value negative">${max_neg:.0f}</div>
            <div class="metric-delta">{format_gex_value(max_neg_gex)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Select data based on type
    if gex_type == "Call":
        display_df = call_df
    elif gex_type == "Put":
        display_df = put_df
    else:
        display_df = net_df
    
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
        
        # Create formatted text matrix (no "nan" or zeros shown as empty)
        text_matrix = []
        for row in display_df_copy.values:
            text_row = []
            for v in row:
                if pd.isna(v) or v == 0:
                    text_row.append("")
                else:
                    text_row.append(format_gex_value(v))
            text_matrix.append(text_row)
        
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
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>GEX: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="GEX",
                tickformat=".0s",
                len=0.5
            )
        ))
        
        # Highlight current price row
        price_y_idx = None
        for i, s in enumerate(display_df_copy.index):
            if s <= underlying_price:
                price_y_idx = i
                break
        
        fig.update_layout(
            template='plotly_dark',
            height=max(450, len(display_df_copy) * 28),
            xaxis_title="Expiration Date",
            yaxis_title="Strike Price",
            margin=dict(l=80, r=40, t=30, b=50),
            xaxis=dict(side='top', tickangle=0),
        )
        
        # Add annotation for current price
        if price_y_idx is not None:
            fig.add_annotation(
                x=-0.08,
                y=price_y_idx,
                xref="paper",
                yref="y",
                text="◄ SPOT",
                showarrow=False,
                font=dict(color="yellow", size=10),
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick interpretation below heatmap
        net_gex = levels.get('net_gex', 0)
        regime = "🟢 **Positive Gamma** - Expect mean-reversion, sell premium strategies favored" if net_gex > 0 else "🔴 **Negative Gamma** - Expect trending/volatile moves, momentum strategies favored"
        
        st.info(f"**Current Regime:** {regime}")
    
    else:
        # Bar chart view
        strike_totals = display_df.sum(axis=1)
        
        fig = go.Figure()
        
        colors = ['#10b981' if x > 0 else '#ef4444' for x in strike_totals]
        
        fig.add_trace(go.Bar(
            x=strike_totals.index,
            y=strike_totals.values / 1e6,
            marker_color=colors,
            text=[format_gex_value(v) for v in strike_totals.values],
            textposition='outside',
            textfont=dict(size=10)
        ))
        
        # Current price line
        fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow",
                      annotation_text=f"${underlying_price:.2f}", annotation_position="top")
        
        # Call wall and put wall markers
        call_wall = levels.get('call_wall', underlying_price)
        put_wall = levels.get('put_wall', underlying_price)
        
        fig.add_vline(x=call_wall, line_dash="dot", line_color="#10b981", 
                      annotation_text="Call Wall", annotation_position="top right")
        fig.add_vline(x=put_wall, line_dash="dot", line_color="#ef4444",
                      annotation_text="Put Wall", annotation_position="top left")
        
        fig.update_layout(
            title=f"{gex_type} GEX by Strike (Next {len(display_df.columns)} Expiries)",
            xaxis_title="Strike Price",
            yaxis_title="GEX (Millions $)",
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== GEX BY EXPIRY SUMMARY ====================
    with st.expander("📅 GEX by Expiration Summary", expanded=False):
        exp_totals = net_df.sum(axis=0)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_exp = go.Figure()
            colors = ['#10b981' if x > 0 else '#ef4444' for x in exp_totals]
            
            fig_exp.add_trace(go.Bar(
                x=exp_totals.index,
                y=exp_totals.values / 1e6,
                marker_color=colors,
                text=[format_gex_value(v) for v in exp_totals.values],
                textposition='auto'
            ))
            
            fig_exp.update_layout(
                xaxis_title="Expiration",
                yaxis_title="Net GEX (Millions)",
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            st.markdown("**GEX by Expiry**")
            for exp, val in exp_totals.items():
                color = "🟢" if val > 0 else "🔴"
                try:
                    d = datetime.strptime(exp, '%Y-%m-%d')
                    exp_display = d.strftime('%m/%d')
                except:
                    exp_display = exp
                st.markdown(f"{color} **{exp_display}**: {format_gex_value(val)}")


if __name__ == "__main__":
    main()
