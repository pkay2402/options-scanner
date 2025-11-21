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
import yfinance as yf

st.set_page_config(page_title="Historical Market Structure", layout="wide", page_icon="📊")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0f1419;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .signal-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d1f3d 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .signal-text {
        color: #ef4444;
        font-size: 1.3rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

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

def parse_options_chain(chain, symbol, current_price, target_expiry=None, expiry_range_weeks=12):
    """Parse Schwab options chain and extract relevant data"""
    strike_min = current_price * 0.90
    strike_max = current_price * 1.10
    expiry_today = datetime.now().date()
    
    # If target_expiry is specified, use a narrower range around it
    if target_expiry:
        expiry_limit = target_expiry + timedelta(days=7)  # Include week after target
        expiry_start = max(expiry_today, target_expiry - timedelta(days=7))  # Include week before
    else:
        expiry_start = expiry_today
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
            if not (expiry_start <= expiry_date <= expiry_limit):
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

def fetch_historical_data(symbol, days=21):
    """Fetch historical options and price data"""
    try:
        # Get historical price data
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist_prices = ticker.history(start=start_date, end=end_date)
        
        if hist_prices.empty:
            return None
        
        # For demo purposes, simulate GEX and max pain data
        # In production, you'd fetch this from your database or API
        dates = hist_prices.index
        prices = hist_prices['Close'].values
        
        # Simulate Net GEX trend (would come from historical calculations)
        gex_base = np.random.randn(len(dates)).cumsum() * 50 + 200
        
        # Simulate gamma flip level (typically around current price ± some %)
        gamma_flip = prices * (1 + np.random.randn(len(dates)) * 0.02)
        
        # Simulate max pain (tends to be near money)
        max_pain = prices * (1 + np.random.randn(len(dates)) * 0.03)
        
        historical_data = pd.DataFrame({
            'date': dates,
            'price': prices,
            'net_gex': gex_base,
            'gamma_flip': gamma_flip,
            'max_pain': max_pain
        })
        
        return historical_data
        
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

def create_historical_structure_chart(historical_df, symbol, current_price):
    """Create the three-panel historical market structure chart"""
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Net Gamma Exposure (GEX) Trend',
            'Market Regime: Price vs. Gamma Flip',
            'Max Pain Trend (Nearest Expiry)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    dates = historical_df['date']
    
    # Panel 1: Net GEX Trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=historical_df['net_gex'],
            name='Total Net GEX',
            line=dict(color='#3b82f6', width=2),
            mode='lines',
            hovertemplate='Date: %{x}<br>GEX: $%{y:.2f}B<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add zero line for GEX
    fig.add_hline(
        y=0, line_dash="solid", line_color="rgba(239, 68, 68, 0.5)",
        line_width=2, row=1, col=1
    )
    
    # Panel 2: Price vs Gamma Flip
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=historical_df['gamma_flip'],
            name='Gamma Flip (Pivot)',
            line=dict(color='#ef4444', width=2, dash='solid'),
            mode='lines',
            hovertemplate='Date: %{x}<br>Flip: $%{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=historical_df['price'],
            name=f'{symbol} Price',
            line=dict(color='#06b6d4', width=2, dash='dot'),
            mode='lines',
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add regime shading
    above_flip = historical_df['price'] > historical_df['gamma_flip']
    
    for i in range(len(dates) - 1):
        if above_flip.iloc[i]:
            fig.add_vrect(
                x0=dates[i], x1=dates[i+1],
                fillcolor='rgba(34, 197, 94, 0.1)',
                line_width=0,
                row=2, col=1,
                layer='below'
            )
        else:
            fig.add_vrect(
                x0=dates[i], x1=dates[i+1],
                fillcolor='rgba(239, 68, 68, 0.1)',
                line_width=0,
                row=2, col=1,
                layer='below'
            )
    
    # Add regime labels
    fig.add_annotation(
        x=dates[int(len(dates)*0.1)],
        y=historical_df['price'].max(),
        text="Bullish Regime",
        showarrow=False,
        font=dict(size=10, color='#22c55e'),
        xref=f'x2', yref=f'y2',
        bgcolor='rgba(34, 197, 94, 0.2)',
        borderpad=3
    )
    
    fig.add_annotation(
        x=dates[int(len(dates)*0.1)],
        y=historical_df['price'].min(),
        text="Bearish Regime",
        showarrow=False,
        font=dict(size=10, color='#ef4444'),
        xref=f'x2', yref=f'y2',
        bgcolor='rgba(239, 68, 68, 0.2)',
        borderpad=3
    )
    
    # Panel 3: Max Pain Trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=historical_df['max_pain'],
            name='Max Pain Strike',
            line=dict(color='#a855f7', width=2),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(168, 85, 247, 0.1)',
            hovertemplate='Date: %{x}<br>Max Pain: $%{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#1f2937',
        color='#9ca3af',
        title_font=dict(size=11)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#1f2937',
        color='#9ca3af',
        title_font=dict(size=11)
    )
    
    # Specific y-axis labels
    fig.update_yaxes(title_text="Total Notional GEX ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price / Strike ($)", row=2, col=1)
    fig.update_yaxes(title_text="Strike Price ($)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Calculate current signal
    current_gex = historical_df['net_gex'].iloc[-1]
    current_distance = current_price - historical_df['gamma_flip'].iloc[-1]
    
    signal = "POSITIVE GAMMA (STABILITY)" if current_gex > 0 else "NEGATIVE GAMMA (VOLATILITY)"
    bias = "BUY DIPS" if current_distance > 0 else "SELL RIPS"
    
    # Update main title with signal
    title_text = f"{symbol} Historical Market Structure ({len(dates)} Days)<br>"
    title_text += f"<span style='color: #ef4444;'>SIGNAL: {signal} | DISTANCE: {current_distance:.2f} | BIAS: {bias}</span>"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color='#e5e7eb'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='#111827',
        paper_bgcolor='#0f1419',
        font=dict(color='#e5e7eb', size=11),
        height=900,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(17, 24, 39, 0.8)',
            bordercolor='#374151',
            borderwidth=1
        ),
        margin=dict(l=80, r=40, t=120, b=60),
        hovermode='x unified'
    )
    
    return fig, signal, current_distance, bias

def create_price_projection_chart(df, current_price, symbol):
    """Create price projection chart based on gamma exposure with multiple levels"""
    # Get gamma exposure by strike
    gamma_by_strike = df.groupby('strike')['gammaExposure'].sum()
    strikes = sorted(gamma_by_strike.index)
    
    # Find top 3 gamma strikes
    top_gamma_strikes = gamma_by_strike.abs().nlargest(3)
    max_gamma_strike = top_gamma_strikes.index[0]
    
    # Calculate call and put walls (top 3 of each)
    call_walls = df[df['type'] == 'CALL'].groupby('strike')['openInterest'].sum()
    put_walls = df[df['type'] == 'PUT'].groupby('strike')['openInterest'].sum()
    
    top_call_walls = call_walls.nlargest(3) if not call_walls.empty else pd.Series()
    top_put_walls = put_walls.nlargest(3) if not put_walls.empty else pd.Series()
    
    # Calculate net delta by strike for flow direction
    call_delta = df[df['type'] == 'CALL'].groupby('strike')['deltaExposure'].sum()
    put_delta = df[df['type'] == 'PUT'].groupby('strike')['deltaExposure'].sum()
    net_delta_by_strike = call_delta.add(-put_delta, fill_value=0)
    
    # Create figure
    fig = go.Figure()
    
    # Add gamma zones as background shapes
    for idx, (strike, gamma_val) in enumerate(top_gamma_strikes.items()):
        if idx < 3:  # Show top 3
            fig.add_vrect(
                x0=strike - 0.5, x1=strike + 0.5,
                fillcolor='rgba(255, 215, 0, 0.1)',
                line_width=0,
                layer='below'
            )
    
    # Current price point
    fig.add_trace(go.Scatter(
        x=[current_price],
        y=[2],
        mode='markers+text',
        marker=dict(size=18, color='cyan', symbol='circle', line=dict(width=2, color='white')),
        text=[f"Current<br>${current_price:.2f}"],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name='Current Price',
        showlegend=False
    ))
    
    # Expected pin level (max gamma)
    fig.add_trace(go.Scatter(
        x=[max_gamma_strike],
        y=[3.5],
        mode='markers+text',
        marker=dict(size=24, color='#ffd700', symbol='star', line=dict(width=2, color='white')),
        text=[f"Max Gamma Pin<br>${max_gamma_strike:.2f}"],
        textposition='top center',
        textfont=dict(size=11, color='#ffd700', family='Arial Black'),
        name='Max Gamma',
        showlegend=False
    ))
    
    # Add secondary gamma levels
    for idx, (strike, gamma_val) in enumerate(list(top_gamma_strikes.items())[1:3]):
        fig.add_trace(go.Scatter(
            x=[strike],
            y=[3 - idx*0.3],
            mode='markers+text',
            marker=dict(size=14, color='rgba(255, 215, 0, 0.6)', symbol='star'),
            text=[f"${strike:.2f}"],
            textposition='top center',
            textfont=dict(size=9, color='rgba(255, 215, 0, 0.8)'),
            showlegend=False
        ))
    
    # Resistance levels (call walls)
    for idx, (strike, oi) in enumerate(top_call_walls.items()):
        alpha = 1 - (idx * 0.3)
        fig.add_trace(go.Scatter(
            x=[strike],
            y=[4 - idx*0.4],
            mode='markers+text',
            marker=dict(size=16-idx*3, color=f'rgba(239, 68, 68, {alpha})', symbol='triangle-down'),
            text=[f"R{idx+1}: ${strike:.2f}<br>{int(oi/1000)}K OI" if idx == 0 else f"${strike:.2f}"],
            textposition='top center',
            textfont=dict(size=10-idx, color=f'rgba(239, 68, 68, {alpha})'),
            name=f'Resistance {idx+1}',
            showlegend=False
        ))
    
    # Support levels (put walls)
    for idx, (strike, oi) in enumerate(top_put_walls.items()):
        alpha = 1 - (idx * 0.3)
        fig.add_trace(go.Scatter(
            x=[strike],
            y=[0 + idx*0.4],
            mode='markers+text',
            marker=dict(size=16-idx*3, color=f'rgba(34, 197, 94, {alpha})', symbol='triangle-up'),
            text=[f"S{idx+1}: ${strike:.2f}<br>{int(oi/1000)}K OI" if idx == 0 else f"${strike:.2f}"],
            textposition='bottom center',
            textfont=dict(size=10-idx, color=f'rgba(34, 197, 94, {alpha})'),
            name=f'Support {idx+1}',
            showlegend=False
        ))
    
    # Add projection line from current to max gamma
    direction = 'up' if max_gamma_strike > current_price else 'down'
    fig.add_trace(go.Scatter(
        x=[current_price, max_gamma_strike],
        y=[2, 3.5],
        mode='lines',
        line=dict(color='rgba(255,215,0,0.4)', width=4, dash='dash'),
        name='Expected Move',
        showlegend=False
    ))
    
    # Add arrow annotation
    fig.add_annotation(
        x=(current_price + max_gamma_strike) / 2,
        y=2.75,
        text=f"{'↗' if direction == 'up' else '↘'} Gamma Pull",
        showarrow=False,
        font=dict(size=12, color='#ffd700'),
        bgcolor='rgba(0,0,0,0.6)',
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} Price Projection to Expiry",
            font=dict(size=16, color='#e8e8e8')
        ),
        xaxis_title="Price Level",
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#e8e8e8', size=12),
        height=350,
        margin=dict(l=40, r=40, t=60, b=60),
        yaxis=dict(visible=False, range=[-0.5, 4.5]),
        xaxis=dict(
            gridcolor='#2d3561',
            showgrid=True,
            range=[
                min(list(top_put_walls.index) + [current_price]) * 0.995,
                max(list(top_call_walls.index) + [current_price]) * 1.005
            ]
        ),
        hoverlabel=dict(
            bgcolor='#2d3561',
            font_size=12
        )
    )
    
    return fig

# Title
st.title("📊 Historical Market Structure Dashboard")

# --- User Input ---
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    symbol = st.text_input("Stock Symbol", value="SPY", help="Enter ticker symbol").upper()

with col2:
    lookback_days = st.selectbox(
        "Lookback Period",
        options=[7, 14, 21, 30, 45, 60],
        index=2,
        help="Historical days to analyze"
    )

with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

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
    st.stop()

# Fetch historical data
with st.spinner(f"Fetching {lookback_days}-day historical data for {symbol}..."):
    historical_df = fetch_historical_data(symbol, days=lookback_days)
    
    if historical_df is None or historical_df.empty:
        st.error("Unable to fetch historical data. Please try a different symbol.")
        st.stop()

st.markdown("---")

# Create and display the main chart
structure_chart, signal, distance, bias = create_historical_structure_chart(
    historical_df, symbol, current_price
)

# Signal header
st.markdown(f"""
<div class="signal-header">
    <div class="signal-text">
        SIGNAL: {signal} | DISTANCE: {distance:.2f} | BIAS: {bias}
    </div>
</div>
""", unsafe_allow_html=True)

# Display the chart
st.plotly_chart(structure_chart, use_container_width=True)

st.markdown("---")

# Key Metrics Row
st.subheader("📊 Current Market Structure Metrics")

met_col1, met_col2, met_col3, met_col4 = st.columns(4)

with met_col1:
    current_gex = historical_df['net_gex'].iloc[-1]
    prev_gex = historical_df['net_gex'].iloc[-2] if len(historical_df) > 1 else current_gex
    st.metric(
        "Net GEX",
        f"${current_gex:.2f}B",
        f"{current_gex - prev_gex:.2f}B",
        delta_color="normal" if current_gex > 0 else "inverse"
    )

with met_col2:
    gamma_flip = historical_df['gamma_flip'].iloc[-1]
    st.metric(
        "Gamma Flip Level",
        f"${gamma_flip:.2f}",
        "Pivot Point"
    )

with met_col3:
    max_pain = historical_df['max_pain'].iloc[-1]
    st.metric(
        "Max Pain Strike",
        f"${max_pain:.2f}",
        f"${abs(current_price - max_pain):.2f} away"
    )

with met_col4:
    regime = "Bullish 📈" if current_price > gamma_flip else "Bearish 📉"
    st.metric(
        "Current Regime",
        regime,
        bias
    )

st.markdown("---")

# Trading Insights
st.subheader("💡 Trading Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.info(f"""
    **Gamma Regime Analysis:**
    - Current GEX: **${current_gex:.2f}B** ({'Positive' if current_gex > 0 else 'Negative'})
    - Price is **${abs(distance):.2f}** {'above' if distance > 0 else 'below'} gamma flip
    - Market makers are {'stabilizing' if current_gex > 0 else 'amplifying'} moves
    
    **Recommended Action:** {bias}
    """)

with insights_col2:
    st.warning(f"""
    **Key Levels to Watch:**
    - **Max Pain:** ${max_pain:.2f} (magnet for expiry)
    - **Gamma Flip:** ${gamma_flip:.2f} (regime change level)
    - **Distance from Flip:** {distance:.2f}
    
    **Risk:** {'Lower volatility expected' if current_gex > 0 else 'Higher volatility possible'}
    """)

st.markdown("---")

# Historical data table
with st.expander("📋 Historical Data Table", expanded=False):
    display_df = historical_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['Date', 'Price', 'Net GEX ($B)', 'Gamma Flip ($)', 'Max Pain ($)']
    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        hide_index=True
    )

# Download data
csv = historical_df.to_csv(index=False)
st.download_button(
    label="📥 Download Historical Data (CSV)",
    data=csv,
    file_name=f"{symbol}_market_structure_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)