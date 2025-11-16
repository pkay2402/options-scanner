# app.py
import time
import threading
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.api.schwab_client import SchwabClient

# Add this near the top of your app
st.set_page_config(layout="wide", page_title="Live Greeks Dashboard")

# Add these CSS styles
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 1rem;
        }
        .stPlotlyChart {
            background-color: #0e1117;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Schwab client
@st.cache_resource
def get_schwab_client():
    """Initialize and cache Schwab client"""
    client = SchwabClient()
    if not client.ensure_valid_session():
        st.error("Failed to authenticate with Schwab API. Please check your credentials.")
        return None
    return client

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_price = None
    st.session_state.last_update = None
    st.session_state.gamma_data = None
    st.session_state.delta_data = None
    st.session_state.auto_refresh = False

# Helper functions
def calculate_gamma_exposure(options_df, current_price):
    """Calculate gamma exposure by strike"""
    if options_df.empty:
        return pd.DataFrame()
    
    gamma_by_strike = []
    for strike in options_df['strikePrice'].unique():
        strike_data = options_df[options_df['strikePrice'] == strike]
        
        # Calculate gamma exposure for calls and puts
        call_gamma = 0
        put_gamma = 0
        
        for _, opt in strike_data.iterrows():
            oi = opt.get('openInterest', 0)
            gamma = opt.get('gamma', 0)
            
            if opt['putCall'] == 'CALL':
                call_gamma += oi * gamma * 100 * current_price * current_price / 1_000_000
            else:
                put_gamma += oi * gamma * 100 * current_price * current_price / 1_000_000
        
        gamma_by_strike.append({
            'strike': strike,
            'call_gamma': call_gamma,
            'put_gamma': -put_gamma,  # Negative for puts
            'net_gamma': call_gamma - put_gamma
        })
    
    return pd.DataFrame(gamma_by_strike).sort_values('strike')

def calculate_delta_exposure(options_df, current_price):
    """Calculate delta exposure by strike"""
    if options_df.empty:
        return pd.DataFrame()
    
    delta_by_strike = []
    for strike in options_df['strikePrice'].unique():
        strike_data = options_df[options_df['strikePrice'] == strike]
        
        # Calculate delta exposure for calls and puts
        call_delta = 0
        put_delta = 0
        
        for _, opt in strike_data.iterrows():
            oi = opt.get('openInterest', 0)
            delta = opt.get('delta', 0)
            
            if opt['putCall'] == 'CALL':
                call_delta += oi * delta * 100 / 1_000_000
            else:
                put_delta += oi * delta * 100 / 1_000_000
        
        delta_by_strike.append({
            'strike': strike,
            'call_delta': call_delta,
            'put_delta': put_delta,
            'net_delta': call_delta + put_delta
        })
    
    return pd.DataFrame(delta_by_strike).sort_values('strike')

def create_gamma_chart(gamma_df, current_price, symbol):
    """Create gamma exposure chart"""
    if gamma_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
    else:
        fig = go.Figure()
        
        # Add call gamma bars
        fig.add_trace(go.Bar(
            x=gamma_df['strike'],
            y=gamma_df['call_gamma'],
            name='Call Gamma',
            marker_color='#00ff00',
            opacity=0.7
        ))
        
        # Add put gamma bars
        fig.add_trace(go.Bar(
            x=gamma_df['strike'],
            y=gamma_df['put_gamma'],
            name='Put Gamma',
            marker_color='#ff0000',
            opacity=0.7
        ))
        
        # Add current price line
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="cyan",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"{symbol} - Gamma Exposure by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure (Millions)",
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#e8e8e8', size=12),
        height=500,
        barmode='relative',
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(gridcolor='#2d3561', showgrid=True),
        yaxis=dict(gridcolor='#2d3561', showgrid=True)
    )
    
    return fig

def create_delta_chart(delta_df, current_price, symbol):
    """Create delta exposure chart"""
    if delta_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
    else:
        fig = go.Figure()
        
        # Add call delta bars
        fig.add_trace(go.Bar(
            x=delta_df['strike'],
            y=delta_df['call_delta'],
            name='Call Delta',
            marker_color='#00ff00',
            opacity=0.7
        ))
        
        # Add put delta bars
        fig.add_trace(go.Bar(
            x=delta_df['strike'],
            y=delta_df['put_delta'],
            name='Put Delta',
            marker_color='#ff0000',
            opacity=0.7
        ))
        
        # Add current price line
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="cyan",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"{symbol} - Delta Exposure by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Delta Exposure (Millions of Shares)",
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#e8e8e8', size=12),
        height=500,
        barmode='relative',
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(gridcolor='#2d3561', showgrid=True),
        yaxis=dict(gridcolor='#2d3561', showgrid=True)
    )
    
    return fig

def fetch_options_data(client, symbol, expiry_date, strike_range):
    """Fetch options chain data from Schwab API"""
    try:
        # Get current stock price
        quote = client.get_quotes(symbol)
        if not quote or symbol not in quote:
            st.error(f"Could not fetch quote for {symbol}")
            return None, None, None
        
        current_price = quote[symbol]['quote']['lastPrice']
        
        # Get options chain
        options_chain = client.get_options_chain(
            symbol=symbol,
            contract_type="ALL",
            from_date=expiry_date,
            to_date=expiry_date
        )
        
        if not options_chain:
            st.error("Could not fetch options chain")
            return None, None, None
        
        # Parse options data
        options_list = []
        
        # Process call options
        if 'callExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['callExpDateMap'].items():
                for strike_price, contracts in strikes.items():
                    for contract in contracts:
                        contract['putCall'] = 'CALL'
                        options_list.append(contract)
        
        # Process put options
        if 'putExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['putExpDateMap'].items():
                for strike_price, contracts in strikes.items():
                    for contract in contracts:
                        contract['putCall'] = 'PUT'
                        options_list.append(contract)
        
        if not options_list:
            st.warning("No options found for this expiration date")
            return current_price, None, None
        
        options_df = pd.DataFrame(options_list)
        
        # Filter by strike range
        min_strike = current_price * (1 - strike_range / 100)
        max_strike = current_price * (1 + strike_range / 100)
        options_df = options_df[
            (options_df['strikePrice'] >= min_strike) &
            (options_df['strikePrice'] <= max_strike)
        ]
        
        # Calculate exposures
        gamma_df = calculate_gamma_exposure(options_df, current_price)
        delta_df = calculate_delta_exposure(options_df, current_price)
        
        return current_price, gamma_df, delta_df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None, None

# Setup UI
st.title("📊 Live Options Greeks Dashboard")

# Input section
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

with col1:
    symbol = st.text_input("Symbol", value="SPY", key="symbol_input").upper()

with col2:
    expiry_date = st.date_input("Expiry Date", value=datetime.now())
    expiry_str = expiry_date.strftime("%Y-%m-%d")

with col3:
    strike_range = st.number_input("Strike Range %", min_value=1, max_value=50, value=10)

with col4:
    refresh_rate = st.number_input("Refresh (sec)", min_value=5, max_value=300, value=30)

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

with col_btn1:
    if st.button("▶️ Start" if not st.session_state.initialized else "⏸️ Stop", use_container_width=True):
        st.session_state.initialized = not st.session_state.initialized
        if st.session_state.initialized:
            st.session_state.auto_refresh = True
        st.rerun()

with col_btn2:
    if st.button("🔄 Refresh Now", use_container_width=True, disabled=not st.session_state.initialized):
        st.session_state.last_update = None  # Force refresh
        st.rerun()

# Status indicator
if st.session_state.initialized:
    st.success(f"✅ Live - Auto-refreshing every {refresh_rate}s")
    if st.session_state.last_update:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
else:
    st.info("⏸️ Paused - Click Start to begin live tracking")

st.markdown("---")

# Create side-by-side columns for charts
col1, col2 = st.columns(2)

# Create placeholders for charts
with col1:
    gamma_placeholder = st.empty()
with col2:
    delta_placeholder = st.empty()

# Fetch and display data
if st.session_state.initialized:
    # Check if we need to fetch new data
    should_fetch = (
        st.session_state.last_update is None or
        (datetime.now() - st.session_state.last_update).total_seconds() >= refresh_rate
    )
    
    if should_fetch:
        with st.spinner("Fetching live data from Schwab..."):
            client = get_schwab_client()
            if client:
                current_price, gamma_df, delta_df = fetch_options_data(
                    client, symbol, expiry_str, strike_range
                )
                
                if current_price is not None:
                    st.session_state.current_price = current_price
                    st.session_state.gamma_data = gamma_df
                    st.session_state.delta_data = delta_df
                    st.session_state.last_update = datetime.now()

# Display charts (either from fresh data or cached)
if st.session_state.gamma_data is not None and st.session_state.delta_data is not None:
    with gamma_placeholder:
        gamma_fig = create_gamma_chart(
            st.session_state.gamma_data,
            st.session_state.current_price,
            symbol
        )
        st.plotly_chart(gamma_fig, use_container_width=True)
    
    with delta_placeholder:
        delta_fig = create_delta_chart(
            st.session_state.delta_data,
            st.session_state.current_price,
            symbol
        )
        st.plotly_chart(delta_fig, use_container_width=True)
    
    # Display current price info
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Current Price", f"${st.session_state.current_price:.2f}")
    
    with col_info2:
        if not st.session_state.gamma_data.empty:
            max_gamma_strike = st.session_state.gamma_data.loc[
                st.session_state.gamma_data['net_gamma'].abs().idxmax()
            ]['strike']
            st.metric("Max Gamma Strike", f"${max_gamma_strike:.2f}")
    
    with col_info3:
        if not st.session_state.delta_data.empty:
            total_net_delta = st.session_state.delta_data['net_delta'].sum()
            st.metric("Total Net Delta", f"{total_net_delta:.2f}M shares")

else:
    # Show empty charts
    with gamma_placeholder:
        st.plotly_chart(create_gamma_chart(pd.DataFrame(), 0, symbol), use_container_width=True)
    with delta_placeholder:
        st.plotly_chart(create_delta_chart(pd.DataFrame(), 0, symbol), use_container_width=True)

# Auto-refresh logic
if st.session_state.initialized and st.session_state.auto_refresh:
    time.sleep(1)
    st.rerun()