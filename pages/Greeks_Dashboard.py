import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.api.schwab_client import SchwabClient
from datetime import datetime, timedelta

st.set_page_config(page_title="Greeks Exposure Dashboard", layout="wide")
st.title("📊 Options Greeks Exposure Dashboard")

# --- Helper Functions ---
def extract_price(data, symbol):
    """Extract price from Schwab API quote response"""
    price_keys = ['lastPrice', 'mark', 'closePrice', 'bidPrice', 'askPrice']
    
    # Check if response has symbol key
    if symbol in data and isinstance(data[symbol], dict):
        quote_data = data[symbol].get('quote', data[symbol])
    else:
        quote_data = data.get('quote', data)
    
    # Try each price key
    for key in price_keys:
        if key in quote_data and quote_data[key] is not None:
            return float(quote_data[key])
    
    return None

def parse_options_chain(chain, symbol, current_price, expiry_range_weeks=12):
    """Parse Schwab options chain and extract relevant data"""
    strike_min = current_price * 0.90
    strike_max = current_price * 1.10
    expiry_today = datetime.now().date()
    expiry_limit = expiry_today + timedelta(weeks=expiry_range_weeks)
    
    options = []
    
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in chain:
            continue
            
        is_call = (option_type == 'callExpDateMap')
        
        for exp_date_str, strikes_data in chain[option_type].items():
            # Parse expiry date (format: "2024-12-20:45")
            try:
                expiry_date = datetime.strptime(exp_date_str.split(':')[0], "%Y-%m-%d").date()
            except (ValueError, IndexError):
                continue
                
            # Filter by expiry range
            if not (expiry_today <= expiry_date <= expiry_limit):
                continue
            
            # Parse strikes
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                    
                try:
                    strike = float(strike_str)
                except (ValueError, TypeError):
                    continue
                
                # Filter by strike range
                if not (strike_min <= strike <= strike_max):
                    continue
                
                contract = contracts[0]
                
                # Only include options with valid Greek data
                if contract.get('delta') is None:
                    continue
                
                oi = contract.get('openInterest', 0)
                
                options.append({
                    'expiry': expiry_date,
                    'strike': strike,
                    'type': 'CALL' if is_call else 'PUT',
                    'delta': contract.get('delta', 0),
                    'gamma': contract.get('gamma', 0),
                    'theta': contract.get('theta', 0),
                    'vega': contract.get('vega', 0),
                    'openInterest': oi,
                    'volume': contract.get('totalVolume', 0),
                    'impliedVolatility': contract.get('volatility', 0),
                    # Calculate exposures (multiplied by OI and contract multiplier)
                    'gammaExposure': contract.get('gamma', 0) * oi * 100 * current_price * current_price * 0.01,
                    'deltaExposure': contract.get('delta', 0) * oi * 100 * current_price * 0.01,
                    'vannaExposure': contract.get('vega', 0) * oi * 100 * 0.01 if contract.get('vega') else 0,
                    'charmExposure': contract.get('theta', 0) * oi * 100 * 0.01 if contract.get('theta') else 0,
                })
    
    return options

def create_exposure_chart(df, current_price, greek_type='gamma'):
    """Create exposure by strike chart"""
    greek_col_map = {
        'gamma': 'gammaExposure',
        'delta': 'deltaExposure', 
        'vanna': 'vannaExposure',
        'charm': 'charmExposure'
    }
    
    title_map = {
        'gamma': 'Net Gamma Exposure By Strike',
        'delta': 'Net Delta Exposure By Strike',
        'vanna': 'Net Vanna Exposure By Strike',
        'charm': 'Net Charm Exposure By Strike'
    }
    
    exposure_col = greek_col_map[greek_type]
    
    # Aggregate by strike
    call_exp = df[df['type'] == 'CALL'].groupby('strike')[exposure_col].sum()
    put_exp = df[df['type'] == 'PUT'].groupby('strike')[exposure_col].sum()
    
    # For delta, puts are negative from dealer perspective
    if greek_type == 'delta':
        put_exp = -put_exp
    
    # Combine
    all_strikes = sorted(set(call_exp.index) | set(put_exp.index))
    net_exp = []
    
    for strike in all_strikes:
        call_val = call_exp.get(strike, 0)
        put_val = put_exp.get(strike, 0)
        net_exp.append(call_val + put_val)
    
    # Create figure
    fig = go.Figure()
    
    # Net exposure bars
    colors = ['#22c55e' if x > 0 else '#ef4444' for x in net_exp]
    
    fig.add_trace(go.Bar(
        x=all_strikes,
        y=net_exp,
        name='Net Exposure',
        marker_color=colors,
        hovertemplate='Strike: $%{x:.2f}<br>Exposure: %{y:.2f}M<extra></extra>'
    ))
    
    # Add current price line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="cyan",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title=title_map[greek_type],
        xaxis_title="Strike Price",
        yaxis_title="Exposure (Millions)",
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white', size=12),
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#262730',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#262730',
            showgrid=True,
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1
        )
    )
    
    return fig

def create_price_projection_chart(df, current_price, symbol):
    """Create price projection chart based on gamma exposure"""
    # Get gamma exposure by strike
    gamma_by_strike = df.groupby('strike')['gammaExposure'].sum()
    strikes = sorted(gamma_by_strike.index)
    
    # Find max gamma strike (expected pin level)
    max_gamma_strike = gamma_by_strike.abs().idxmax()
    max_gamma_value = gamma_by_strike[max_gamma_strike]
    
    # Calculate resistance and support levels based on call/put walls
    call_walls = df[df['type'] == 'CALL'].groupby('strike')['openInterest'].sum()
    put_walls = df[df['type'] == 'PUT'].groupby('strike')['openInterest'].sum()
    
    resistance = call_walls.idxmax() if not call_walls.empty else current_price * 1.05
    support = put_walls.idxmax() if not put_walls.empty else current_price * 0.95
    
    # Create figure
    fig = go.Figure()
    
    # Current price point
    fig.add_trace(go.Scatter(
        x=[current_price],
        y=[1],
        mode='markers+text',
        marker=dict(size=15, color='cyan', symbol='circle'),
        text=[f"Current: ${current_price:.2f}"],
        textposition='top center',
        name='Current Price',
        showlegend=False
    ))
    
    # Expected pin level (max gamma)
    fig.add_trace(go.Scatter(
        x=[max_gamma_strike],
        y=[2],
        mode='markers+text',
        marker=dict(size=20, color='#ffd700', symbol='star'),
        text=[f"Pin: ${max_gamma_strike:.2f}"],
        textposition='top center',
        name='Max Gamma',
        showlegend=False
    ))
    
    # Resistance level
    fig.add_trace(go.Scatter(
        x=[resistance],
        y=[3],
        mode='markers+text',
        marker=dict(size=15, color='#ef4444', symbol='triangle-down'),
        text=[f"Resistance: ${resistance:.2f}"],
        textposition='top center',
        name='Call Wall',
        showlegend=False
    ))
    
    # Support level
    fig.add_trace(go.Scatter(
        x=[support],
        y=[0],
        mode='markers+text',
        marker=dict(size=15, color='#22c55e', symbol='triangle-up'),
        text=[f"Support: ${support:.2f}"],
        textposition='bottom center',
        name='Put Wall',
        showlegend=False
    ))
    
    # Add projection line from current to pin
    fig.add_trace(go.Scatter(
        x=[current_price, max_gamma_strike],
        y=[1, 2],
        mode='lines',
        line=dict(color='rgba(255,215,0,0.3)', width=3, dash='dash'),
        name='Expected Move',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Projection to Expiry",
        xaxis_title="Price Level",
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white', size=12),
        height=300,
        margin=dict(l=40, r=40, t=60, b=60),
        yaxis=dict(visible=False),
        xaxis=dict(
            gridcolor='#262730',
            showgrid=True,
            range=[min(support, current_price) * 0.99, max(resistance, current_price) * 1.01]
        )
    )
    
    return fig

# --- User Input ---
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    symbol = st.text_input("Stock Symbol", value="SPY").upper()

with col2:
    target_expiry = st.date_input(
        "Target Expiry",
        value=datetime.now().date() + timedelta(days=7),
        help="Primary expiration date to analyze"
    )

with col3:
    strike_range_pct = st.slider(
        "Strike Range %",
        min_value=5,
        max_value=20,
        value=10,
        help="Percentage range around current price"
    ) / 100

# Initialize client
client = SchwabClient()

# --- Fetch Quote ---
with st.spinner("Fetching quote..."):
    try:
        quote = client.get_quote(symbol)
        if not quote:
            st.error("Could not fetch quote. Please check the symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching quote: {str(e)}")
        st.stop()

# Extract current price
current_price = extract_price(quote, symbol)

if current_price is None:
    st.error("Could not extract price from quote data.")
    with st.expander("Debug: Quote Response"):
        st.json(quote)
    st.stop()

# Display current price
st.markdown("---")
price_col1, price_col2, price_col3, price_col4 = st.columns(4)
with price_col1:
    st.metric("💰 Current Price", f"${current_price:.2f}")

# --- Fetch Option Chain ---
with st.spinner("Fetching options chain..."):
    try:
        chain = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not chain or ('callExpDateMap' not in chain and 'putExpDateMap' not in chain):
            st.error("No options data available for this symbol.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error fetching option chain: {str(e)}")
        st.stop()

# --- Parse Options ---
options = parse_options_chain(chain, symbol, current_price)

if not options:
    st.warning(f"No options found in the selected range (±{int(strike_range_pct*100)}% strikes, next 12 weeks).")
    st.stop()

df = pd.DataFrame(options)

# Display summary metrics
with price_col2:
    st.metric("📊 Total Contracts", f"{len(df):,}")
with price_col3:
    total_oi = df['openInterest'].sum()
    st.metric("📈 Total OI", f"{int(total_oi):,}")
with price_col4:
    st.metric("📅 Expiration Dates", df['expiry'].nunique())

st.markdown("---")

# --- Key Levels ---
st.subheader("🎯 Key Levels & Price Projection")

key_col1, key_col2, key_col3 = st.columns(3)

# Max Gamma Exposure
max_gamma_idx = df['gammaExposure'].abs().idxmax()
max_gamma_strike = df.loc[max_gamma_idx, 'strike']
max_gamma_value = df.loc[max_gamma_idx, 'gammaExposure']

with key_col1:
    st.metric(
        "🎯 Max Gamma Strike",
        f"${max_gamma_strike:.2f}",
        f"{max_gamma_value/1e6:.2f}M exposure",
        help="Expected pin risk and high volatility zone"
    )

# Net Delta
net_delta = df['deltaExposure'].sum()
delta_bias = "Bullish 📈" if net_delta > 0 else "Bearish 📉" if net_delta < 0 else "Neutral ➡️"

with key_col2:
    st.metric(
        "📊 Net Delta Bias",
        delta_bias,
        f"{net_delta/1e6:.2f}M exposure",
        help="Market maker positioning direction"
    )

# High OI Strike
high_oi_idx = df.groupby('strike')['openInterest'].sum().idxmax()
high_oi_value = df.groupby('strike')['openInterest'].sum().max()

with key_col3:
    st.metric(
        "🔒 Max OI Strike",
        f"${high_oi_idx:.2f}",
        f"{int(high_oi_value):,} contracts",
        help="Potential support/resistance level"
    )

# Price projection chart
st.markdown("#### 📈 Expected Price Movement to Expiry")
price_proj_chart = create_price_projection_chart(df, current_price, symbol)
st.plotly_chart(price_proj_chart, use_container_width=True)

st.info("💡 **Insight**: Price tends to gravitate towards max gamma strikes as dealers hedge their positions. Support/resistance levels indicate where large option walls may limit price movement.")

st.markdown("---")

# --- Exposure Charts ---
st.subheader("🎯 Net Exposure by Strike")

# Create 2x2 grid of charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    gamma_chart = create_exposure_chart(df, current_price, 'gamma')
    st.plotly_chart(gamma_chart, use_container_width=True)

with chart_col2:
    delta_chart = create_exposure_chart(df, current_price, 'delta')
    st.plotly_chart(delta_chart, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    vanna_chart = create_exposure_chart(df, current_price, 'vanna')
    st.plotly_chart(vanna_chart, use_container_width=True)

with chart_col4:
    charm_chart = create_exposure_chart(df, current_price, 'charm')
    st.plotly_chart(charm_chart, use_container_width=True)

st.markdown("---")

# --- Aggregate Greeks Summary (Collapsible) ---
with st.expander("📋 Greeks Summary - By Expiry & Type", expanded=False):
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown("#### By Expiry Date")
        greek_cols = ['gammaExposure', 'deltaExposure', 'vannaExposure', 'charmExposure']
        expiry_aggs = df.groupby('expiry')[greek_cols].sum() / 1_000_000  # Convert to millions
        expiry_aggs.columns = ['Gamma (M)', 'Delta (M)', 'Vanna (M)', 'Charm (M)']
        expiry_aggs = expiry_aggs.round(2)
        st.dataframe(expiry_aggs, use_container_width=True)

    with summary_col2:
        st.markdown("#### By Option Type")
        type_aggs = df.groupby('type')[greek_cols].sum() / 1_000_000
        type_aggs.columns = ['Gamma (M)', 'Delta (M)', 'Vanna (M)', 'Charm (M)']
        type_aggs = type_aggs.round(2)
        st.dataframe(type_aggs, use_container_width=True)

st.markdown("---")

# --- Download Data ---
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Greeks Data (CSV)",
    data=csv,
    file_name=f"{symbol}_greeks_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    use_container_width=False
)