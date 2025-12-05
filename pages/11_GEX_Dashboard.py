"""
GEX Dashboard - Gamma Exposure Analysis
Visualizes gamma exposure, options inventory, and Net GEX heatmaps
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e2130;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .stMetric {
        background: #1e2130;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'gex_symbol' not in st.session_state:
    st.session_state.gex_symbol = 'SPY'
if 'gex_expiry' not in st.session_state:
    today = datetime.now().date()
    weekday = today.weekday()
    if weekday == 5:
        st.session_state.gex_expiry = today + timedelta(days=2)
    elif weekday == 6:
        st.session_state.gex_expiry = today + timedelta(days=1)
    else:
        st.session_state.gex_expiry = today

@st.cache_data(ttl=60, show_spinner=False)
def get_market_snapshot(symbol: str, expiry_date: str):
    """Fetches complete market data snapshot for a symbol"""
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        query_symbol_quote = symbol
        query_symbol_options = symbol
        
        quote = client.get_quote(query_symbol_quote)
        if not quote:
            return None
        
        underlying_price = quote.get(query_symbol_quote, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            return None
        
        chain_params = {
            'symbol': query_symbol_options,
            'contract_type': 'ALL',
            'from_date': expiry_date,
            'to_date': expiry_date
        }
        
        if symbol in ['$SPX', 'DJX', 'NDX', 'RUT']:
            chain_params['strike_count'] = 50
        
        options = client.get_options_chain(**chain_params)
        
        if not options or 'callExpDateMap' not in options:
            return None
        
        return {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options,
            'fetched_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

def calculate_gex_data(options_data, underlying_price):
    """Calculate gamma exposure data for visualization"""
    try:
        gamma_data = []
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        volume = contract.get('totalVolume', 0) or 0
                        iv = contract.get('volatility', 0) or 0
                        
                        # Positive GEX for calls
                        gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'strike': strike,
                            'type': 'CALL',
                            'gex': gex,
                            'volume': volume,
                            'oi': oi,
                            'gamma': gamma,
                            'iv': iv
                        })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        volume = contract.get('totalVolume', 0) or 0
                        iv = contract.get('volatility', 0) or 0
                        
                        # Negative GEX for puts
                        gex = -gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'strike': strike,
                            'type': 'PUT',
                            'gex': gex,
                            'volume': volume,
                            'oi': oi,
                            'gamma': gamma,
                            'iv': iv
                        })
        
        if not gamma_data:
            return None
        
        df = pd.DataFrame(gamma_data)
        
        # Aggregate by strike
        strike_summary = df.groupby('strike').agg({
            'gex': 'sum',
            'volume': 'sum',
            'oi': 'sum'
        }).reset_index()
        
        # Separate call and put data
        call_data = df[df['type'] == 'CALL'].groupby('strike').agg({
            'gex': 'sum',
            'volume': 'sum',
            'oi': 'sum'
        }).reset_index()
        
        put_data = df[df['type'] == 'PUT'].groupby('strike').agg({
            'gex': 'sum',
            'volume': 'sum',
            'oi': 'sum'
        }).reset_index()
        
        return {
            'strike_summary': strike_summary,
            'call_data': call_data,
            'put_data': put_data,
            'all_data': df
        }
        
    except Exception as e:
        logger.error(f"Error calculating GEX data: {str(e)}")
        return None

def create_gamma_exposure_chart(gex_data, underlying_price, symbol):
    """Create horizontal gamma exposure chart"""
    try:
        df = gex_data['strike_summary'].copy()
        
        # Filter to reasonable range (±5%)
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        df = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
        
        # Sort by strike descending for top-to-bottom display
        df = df.sort_values('strike', ascending=False)
        
        # Create colors based on GEX sign
        colors = ['#26a69a' if x > 0 else '#ef5350' for x in df['gex']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df['strike'],
            x=df['gex'] / 1000,  # Convert to thousands
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            text=[f"${x/1000:.1f}K" for x in df['gex']],
            textposition='outside',
            hovertemplate='<b>Strike:</b> $%{y:.2f}<br><b>GEX:</b> %{x:.1f}K<extra></extra>',
            showlegend=False
        ))
        
        # Add current price line
        fig.add_hline(
            y=underlying_price,
            line=dict(color='#ffd700', width=2, dash='dash'),
            annotation_text=f"${underlying_price:.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=dict(
                text="gamma exposure",
                font=dict(size=14, color='#ffffff'),
                x=0,
                xanchor='left'
            ),
            xaxis=dict(
                title="shares (x move=-200.0K  -150.0K  -100.0K  -50.0K  0  50.0K  100.0K  150.0K  200.0K",
                titlefont=dict(size=10, color='#888888'),
                tickfont=dict(size=9, color='#888888'),
                gridcolor='#2a2e39',
                zerolinecolor='#ffd700',
                zerolinewidth=1
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=9, color='#888888'),
                gridcolor='#2a2e39'
            ),
            height=600,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1e2130',
            margin=dict(l=80, r=80, t=40, b=40),
            hovermode='y'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gamma exposure chart: {str(e)}")
        return None

def create_options_inventory_chart(gex_data, underlying_price):
    """Create options inventory chart showing calls vs puts"""
    try:
        call_df = gex_data['call_data'].copy()
        put_df = gex_data['put_data'].copy()
        
        # Filter to reasonable range
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        
        call_df = call_df[(call_df['strike'] >= min_strike) & (call_df['strike'] <= max_strike)]
        put_df = put_df[(put_df['strike'] >= min_strike) & (put_df['strike'] <= max_strike)]
        
        # Sort descending
        call_df = call_df.sort_values('strike', ascending=False)
        put_df = put_df.sort_values('strike', ascending=False)
        
        fig = go.Figure()
        
        # Add calls (positive, green, on right)
        fig.add_trace(go.Bar(
            y=call_df['strike'],
            x=call_df['oi'],
            orientation='h',
            name='CALLS',
            marker=dict(color='#26a69a', line=dict(width=0)),
            hovertemplate='<b>Strike:</b> $%{y:.2f}<br><b>Call OI:</b> %{x:,}<extra></extra>',
            showlegend=True
        ))
        
        # Add puts (negative, red, on left)
        fig.add_trace(go.Bar(
            y=put_df['strike'],
            x=-put_df['oi'],  # Negative for left side
            orientation='h',
            name='PUTS',
            marker=dict(color='#ef5350', line=dict(width=0)),
            hovertemplate='<b>Strike:</b> $%{y:.2f}<br><b>Put OI:</b> %{x:,}<extra></extra>',
            showlegend=True
        ))
        
        # Add current price line
        fig.add_hline(
            y=underlying_price,
            line=dict(color='#ffd700', width=2, dash='dash'),
            annotation_text=f"${underlying_price:.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=dict(
                text="options inventory",
                font=dict(size=14, color='#ffffff'),
                x=0,
                xanchor='left'
            ),
            xaxis=dict(
                title="",
                titlefont=dict(size=10, color='#888888'),
                tickfont=dict(size=9, color='#888888'),
                gridcolor='#2a2e39',
                zerolinecolor='#ffd700',
                zerolinewidth=1,
                tickformat=','
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=9, color='#888888'),
                gridcolor='#2a2e39'
            ),
            height=600,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1e2130',
            margin=dict(l=80, r=80, t=40, b=40),
            barmode='overlay',
            hovermode='y',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5,
                font=dict(size=10, color='#ffffff')
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating options inventory chart: {str(e)}")
        return None

def create_net_gex_heatmap(options_data, underlying_price, num_expiries=4):
    """Create 3D Net GEX heatmap"""
    try:
        gamma_data = []
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'expiry': exp_date.split(':')[0],
                            'strike': strike,
                            'gex': gex
                        })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        gex = -gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'expiry': exp_date.split(':')[0],
                            'strike': strike,
                            'gex': gex
                        })
        
        if not gamma_data:
            return None
        
        df_gamma = pd.DataFrame(gamma_data)
        
        # Get unique expiries and strikes
        expiries = sorted(df_gamma['expiry'].unique())[:num_expiries]
        all_strikes = sorted(df_gamma['strike'].unique())
        
        # Filter strikes to range
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # Limit strikes for readability
        if len(filtered_strikes) > 15:
            step = len(filtered_strikes) // 15
            filtered_strikes = filtered_strikes[::step]
        
        # Create data matrix
        heat_data = []
        for strike in filtered_strikes:
            row = []
            for exp in expiries:
                gex_value = df_gamma[(df_gamma['strike'] == strike) & (df_gamma['expiry'] == exp)]['gex'].sum()
                row.append(gex_value / 1000)  # Convert to thousands
            heat_data.append(row)
        
        # Create meshgrid for 3D surface
        x = np.arange(len(expiries))
        y = np.array(filtered_strikes)
        z = np.array(heat_data)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[
                [0, '#ef5350'],      # Red for negative
                [0.5, '#ffeb3b'],    # Yellow for near zero
                [1, '#26a69a']       # Green for positive
            ],
            showscale=True,
            hovertemplate='<b>Strike:</b> $%{y:.2f}<br><b>GEX:</b> %{z:.1f}K<extra></extra>',
            colorbar=dict(
                title="GEX (K)",
                titleside="right",
                tickfont=dict(size=9, color='#888888'),
                titlefont=dict(size=10, color='#888888')
            )
        )])
        
        # Add current price line
        price_index = np.searchsorted(filtered_strikes, underlying_price)
        fig.add_trace(go.Scatter3d(
            x=[0, len(expiries)-1],
            y=[underlying_price, underlying_price],
            z=[np.max(z), np.max(z)],
            mode='lines',
            line=dict(color='#ffd700', width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        fig.update_layout(
            title=dict(
                text="Net GEX Heatmap",
                font=dict(size=14, color='#ffffff'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title="Expiry",
                    ticktext=expiry_labels,
                    tickvals=list(range(len(expiries))),
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                yaxis=dict(
                    title="Strike",
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                zaxis=dict(
                    title="GEX (K)",
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.3)
                )
            ),
            height=400,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Net GEX heatmap: {str(e)}")
        return None

def create_net_iv_heatmap(options_data, underlying_price, num_expiries=4):
    """Create 3D Net IV heatmap"""
    try:
        iv_data = []
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        iv = contract.get('volatility', 0) or 0
                        volume = contract.get('totalVolume', 0) or 0
                        
                        if volume > 0:
                            iv_data.append({
                                'expiry': exp_date.split(':')[0],
                                'strike': strike,
                                'iv': iv * 100,  # Convert to percentage
                                'volume': volume,
                                'type': 'CALL'
                            })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        iv = contract.get('volatility', 0) or 0
                        volume = contract.get('totalVolume', 0) or 0
                        
                        if volume > 0:
                            iv_data.append({
                                'expiry': exp_date.split(':')[0],
                                'strike': strike,
                                'iv': iv * 100,
                                'volume': volume,
                                'type': 'PUT'
                            })
        
        if not iv_data:
            return None
        
        df_iv = pd.DataFrame(iv_data)
        
        # Get unique expiries and strikes
        expiries = sorted(df_iv['expiry'].unique())[:num_expiries]
        all_strikes = sorted(df_iv['strike'].unique())
        
        # Filter strikes to range
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # Limit strikes for readability
        if len(filtered_strikes) > 15:
            step = len(filtered_strikes) // 15
            filtered_strikes = filtered_strikes[::step]
        
        # Create data matrix - weighted average IV by volume
        heat_data = []
        for strike in filtered_strikes:
            row = []
            for exp in expiries:
                strike_exp_data = df_iv[(df_iv['strike'] == strike) & (df_iv['expiry'] == exp)]
                if len(strike_exp_data) > 0:
                    # Weighted average IV by volume
                    weighted_iv = (strike_exp_data['iv'] * strike_exp_data['volume']).sum() / strike_exp_data['volume'].sum()
                    row.append(weighted_iv)
                else:
                    row.append(0)
            heat_data.append(row)
        
        # Create meshgrid for 3D surface
        x = np.arange(len(expiries))
        y = np.array(filtered_strikes)
        z = np.array(heat_data)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[
                [0, '#4a148c'],      # Dark purple for low IV
                [0.5, '#ffeb3b'],    # Yellow for mid IV
                [1, '#f44336']       # Red for high IV
            ],
            showscale=True,
            hovertemplate='<b>Strike:</b> $%{y:.2f}<br><b>IV:</b> %{z:.1f}%<extra></extra>',
            colorbar=dict(
                title="IV %",
                titleside="right",
                tickfont=dict(size=9, color='#888888'),
                titlefont=dict(size=10, color='#888888')
            )
        )])
        
        # Add current price line
        price_index = np.searchsorted(filtered_strikes, underlying_price)
        fig.add_trace(go.Scatter3d(
            x=[0, len(expiries)-1],
            y=[underlying_price, underlying_price],
            z=[np.max(z), np.max(z)],
            mode='lines',
            line=dict(color='#ffd700', width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        fig.update_layout(
            title=dict(
                text="Net IV Heatmap",
                font=dict(size=14, color='#ffffff'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title="Expiry",
                    ticktext=expiry_labels,
                    tickvals=list(range(len(expiries))),
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                yaxis=dict(
                    title="Strike",
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                zaxis=dict(
                    title="IV %",
                    titlefont=dict(size=10, color='#888888'),
                    tickfont=dict(size=8, color='#888888'),
                    gridcolor='#2a2e39',
                    backgroundcolor='#1e2130'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.3)
                )
            ),
            height=400,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Net IV heatmap: {str(e)}")
        return None

def calculate_key_metrics(gex_data, underlying_price):
    """Calculate key GEX metrics"""
    try:
        df = gex_data['strike_summary']
        all_data = gex_data['all_data']
        
        # Total GEX
        total_gex = df['gex'].sum()
        
        # Positive and Negative GEX
        pos_gex = df[df['gex'] > 0]['gex'].sum()
        neg_gex = df[df['gex'] < 0]['gex'].sum()
        
        # Zero Gamma level (where GEX crosses zero)
        df_sorted = df.sort_values('strike')
        zero_gamma = None
        for i in range(len(df_sorted) - 1):
            if (df_sorted.iloc[i]['gex'] > 0 and df_sorted.iloc[i+1]['gex'] < 0) or \
               (df_sorted.iloc[i]['gex'] < 0 and df_sorted.iloc[i+1]['gex'] > 0):
                zero_gamma = df_sorted.iloc[i]['strike']
                break
        
        # GEX Ratio (pos / abs(neg))
        gex_ratio = pos_gex / abs(neg_gex) if neg_gex != 0 else 0
        
        # Call and Put OI
        call_oi = gex_data['call_data']['oi'].sum()
        put_oi = gex_data['put_data']['oi'].sum()
        
        # Call and Put Volume
        call_volume = gex_data['call_data']['volume'].sum()
        put_volume = gex_data['put_data']['volume'].sum()
        
        # Flow Ratio (Put Volume / Call Volume)
        flow_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # Net Flow (Put - Call)
        net_flow = put_volume - call_volume
        
        # Calculate weighted average IV for calls and puts near the money (within 5%)
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        
        call_data_atm = all_data[(all_data['type'] == 'CALL') & 
                                  (all_data['strike'] >= min_strike) & 
                                  (all_data['strike'] <= max_strike)]
        put_data_atm = all_data[(all_data['type'] == 'PUT') & 
                                 (all_data['strike'] >= min_strike) & 
                                 (all_data['strike'] <= max_strike)]
        
        # Weighted average IV by volume
        if len(call_data_atm) > 0 and call_data_atm['volume'].sum() > 0:
            call_iv = (call_data_atm['iv'] * call_data_atm['volume']).sum() / call_data_atm['volume'].sum()
        else:
            call_iv = 0
        
        if len(put_data_atm) > 0 and put_data_atm['volume'].sum() > 0:
            put_iv = (put_data_atm['iv'] * put_data_atm['volume']).sum() / put_data_atm['volume'].sum()
        else:
            put_iv = 0
        
        return {
            'gex_ratio': gex_ratio,
            'net_gex': total_gex,
            'flow_ratio': flow_ratio,
            'net_flow': net_flow,
            'pos_gex': pos_gex,
            'neg_gex': neg_gex,
            'zero_gamma': zero_gamma,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'call_iv': call_iv * 100,  # Convert to percentage
            'put_iv': put_iv * 100
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return None

# ===== HEADER =====
col_title, col_controls = st.columns([3, 1])

with col_title:
    st.title("📊 GEX Dashboard")

with col_controls:
    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ===== CONTROLS =====
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    symbol_input = st.text_input("Symbol", value=st.session_state.gex_symbol, key="symbol_input").upper()
    if symbol_input != st.session_state.gex_symbol:
        st.session_state.gex_symbol = symbol_input
        st.rerun()

with col2:
    expiry_input = st.date_input("Expiry Date", value=st.session_state.gex_expiry, key="expiry_input")
    if expiry_input != st.session_state.gex_expiry:
        st.session_state.gex_expiry = expiry_input
        st.rerun()

with col3:
    st.write("")
    st.write("")
    analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ===== MAIN ANALYSIS =====
if analyze or 'gex_data' in st.session_state:
    symbol = st.session_state.gex_symbol
    expiry_date = st.session_state.gex_expiry
    expiry_str = expiry_date.strftime('%Y-%m-%d')
    
    with st.spinner(f"Loading data for {symbol}..."):
        snapshot = get_market_snapshot(symbol, expiry_str)
        
        if not snapshot:
            st.error("Failed to fetch data")
            st.stop()
        
        underlying_price = snapshot['underlying_price']
        
        # Calculate GEX data
        gex_data = calculate_gex_data(snapshot['options_chain'], underlying_price)
        
        if not gex_data:
            st.error("Failed to calculate GEX data")
            st.stop()
        
        # Calculate metrics
        metrics = calculate_key_metrics(gex_data, underlying_price)
        
        # Store in session state
        st.session_state.gex_data = gex_data
        st.session_state.metrics = metrics
        st.session_state.underlying_price = underlying_price
        st.session_state.snapshot = snapshot
    
    # ===== LAYOUT =====
    
    # Header with symbol and expiry
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown(f"### {symbol}")
        st.caption(f"${underlying_price:.2f}")
    with col_h2:
        st.markdown(f"### {expiry_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Main layout: Charts on left, metrics and heatmap on right
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Gamma Exposure Chart
        gamma_chart = create_gamma_exposure_chart(gex_data, underlying_price, symbol)
        if gamma_chart:
            st.plotly_chart(gamma_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Options Inventory Chart
        inventory_chart = create_options_inventory_chart(gex_data, underlying_price)
        if inventory_chart:
            st.plotly_chart(inventory_chart, use_container_width=True)
    
    with col_right:
        # Metrics in styled cards
        st.markdown(f"""
        <div style="background: #1e2130; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">GEX Ratio</div>
                    <div style="color: {'#26a69a' if metrics['gex_ratio'] > 1 else '#ef5350'}; font-size: 24px; font-weight: bold;">{metrics['gex_ratio']:.2f}</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Net GEX</div>
                    <div style="color: {'#26a69a' if metrics['net_gex'] > 0 else '#ef5350'}; font-size: 24px; font-weight: bold;">{metrics['net_gex']/1e9:.2f}B</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Flow Ratio</div>
                    <div style="color: {'#ef5350' if metrics['flow_ratio'] > 1 else '#26a69a'}; font-size: 24px; font-weight: bold;">{metrics['flow_ratio']:.2f}</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Net Flow</div>
                    <div style="color: {'#ef5350' if metrics['net_flow'] > 0 else '#26a69a'}; font-size: 20px; font-weight: bold;">{metrics['net_flow']/1000:.1f}K</div>
                </div>
            </div>
        </div>
        
        <div style="background: #1e2130; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Call OI</div>
                    <div style="color: #26a69a; font-size: 20px; font-weight: bold;">{metrics['call_oi']/1000:.1f}K @ {metrics['zero_gamma']:.0f if metrics['zero_gamma'] else 0:.0f}</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Pos GEX</div>
                    <div style="color: #26a69a; font-size: 20px; font-weight: bold;">{metrics['pos_gex']/1000:.1f}K @ {metrics['zero_gamma']:.0f if metrics['zero_gamma'] else 0:.0f}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Zero Gamma</div>
                    <div style="color: #ffd700; font-size: 20px; font-weight: bold;">{metrics['zero_gamma']:.2f if metrics['zero_gamma'] else 0:.2f}</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Neg GEX</div>
                    <div style="color: #ef5350; font-size: 20px; font-weight: bold;">{metrics['neg_gex']/1000:.1f}K @ {metrics['zero_gamma']:.0f if metrics['zero_gamma'] else 0:.0f}</div>
                </div>
            </div>
        </div>
        
        <div style="background: #1e2130; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="text-align: center; margin-bottom: 10px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Put OI</div>
                <div style="color: #ef5350; font-size: 20px; font-weight: bold;">{metrics['put_oi']/1000:.1f}K @ {metrics['zero_gamma']:.0f if metrics['zero_gamma'] else 0:.0f}</div>
            </div>
            <div style="display: flex; justify-content: space-around;">
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Call IV</div>
                    <div style="color: #26a69a; font-size: 18px; font-weight: bold;">{metrics['call_iv']:.1f}%</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Put IV</div>
                    <div style="color: #ef5350; font-size: 18px; font-weight: bold;">{metrics['put_iv']:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Heatmaps
        st.markdown("### Net GEX Heatmap")
        
        view_mode_gex = st.radio("View", ["3D View", "2D View"], horizontal=True, label_visibility="collapsed", key="gex_view")
        
        if view_mode_gex == "3D View":
            heatmap = create_net_gex_heatmap(snapshot['options_chain'], underlying_price)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
        else:
            # 2D heatmap placeholder
            st.info("2D view coming soon")
        
        st.markdown("---")
        st.markdown("### Net IV Heatmap")
        
        view_mode_iv = st.radio("View", ["3D View", "2D View"], horizontal=True, label_visibility="collapsed", key="iv_view")
        
        if view_mode_iv == "3D View":
            iv_heatmap = create_net_iv_heatmap(snapshot['options_chain'], underlying_price)
            if iv_heatmap:
                st.plotly_chart(iv_heatmap, use_container_width=True)
        else:
            st.info("2D view coming soon")

else:
    st.info("👆 Select a symbol and expiry date, then click Analyze to view GEX dashboard")
