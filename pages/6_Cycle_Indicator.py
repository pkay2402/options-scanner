"""
Market Cycle Peak/Bottom Indicator
Identifies market cycles, peaks, and bottoms using Ehlers methodology
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

st.set_page_config(page_title="Cycle Indicator", page_icon="🔄", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert { margin-top: 1rem; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .signal-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.85rem;
        margin: 0.1rem;
    }
    .peak { background: #ff3333; color: white; }
    .bottom { background: #00ff00; color: black; }
    .approaching-peak { background: #ff9933; color: black; }
    .approaching-bottom { background: #90ee90; color: black; }
</style>
""", unsafe_allow_html=True)

# Load scanner results
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_scanner_results():
    """Load latest cycle scanner results"""
    try:
        results_file = Path(__file__).parent.parent / 'data' / 'cycle_signals.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading scanner results: {e}")
    return None

st.title("🔄 Market Cycle Peak/Bottom Indicator")

# Display scanner results at the top
scanner_results = load_scanner_results()
if scanner_results:
    scan_time = datetime.fromisoformat(scanner_results['metadata']['scan_time'])
    time_ago = datetime.now() - scan_time
    hours_ago = time_ago.total_seconds() / 3600
    
    st.success(f"📡 **Live Scanner Results** | Last scan: {scan_time.strftime('%I:%M %p')} ({hours_ago:.1f}h ago)")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Peak Signals", len(scanner_results['peak']))
    with col2:
        st.metric("🟢 Bottom Signals", len(scanner_results['bottom']))
    with col3:
        st.metric("⚠️ Approaching Peak", len(scanner_results['approaching_peak']))
    with col4:
        st.metric("💡 Approaching Bottom", len(scanner_results['approaching_bottom']))
    
    # Display signals in expander
    if any([scanner_results['peak'], scanner_results['bottom'], 
            scanner_results['approaching_peak'], scanner_results['approaching_bottom']]):
        
        with st.expander("📊 View All Cycle Signals", expanded=True):
            # BUY signals (bottom + approaching bottom)
            buy_signals = scanner_results['bottom'] + scanner_results['approaching_bottom']
            if buy_signals:
                st.markdown("### 🟢 BUY OPPORTUNITIES")
                buy_df = pd.DataFrame(buy_signals)
                buy_df = buy_df.sort_values('cycle_value', ascending=True)  # Most negative first
                buy_df['cycle_value'] = buy_df['cycle_value'].apply(lambda x: f"{x:.2f}σ")
                buy_df['price'] = buy_df['price'].apply(lambda x: f"${x:.2f}")
                buy_df['phase'] = buy_df['phase'].apply(lambda x: f"{x:.0f}°")
                st.dataframe(
                    buy_df[['symbol', 'action', 'price', 'cycle_value', 'phase', 'strength']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # SELL signals (peak + approaching peak)
            sell_signals = scanner_results['peak'] + scanner_results['approaching_peak']
            if sell_signals:
                st.markdown("### 🔴 SELL OPPORTUNITIES")
                sell_df = pd.DataFrame(sell_signals)
                sell_df = sell_df.sort_values('cycle_value', ascending=False)  # Most positive first
                sell_df['cycle_value'] = sell_df['cycle_value'].apply(lambda x: f"{x:.2f}σ")
                sell_df['price'] = sell_df['price'].apply(lambda x: f"${x:.2f}")
                sell_df['phase'] = sell_df['phase'].apply(lambda x: f"{x:.0f}°")
                st.dataframe(
                    sell_df[['symbol', 'action', 'price', 'cycle_value', 'phase', 'strength']],
                    use_container_width=True,
                    hide_index=True
                )
    
    st.markdown("---")
st.markdown("*Based on John Ehlers' Dominant Cycle Period Detection*")

# Initialize symbol from sidebar or use default
if 'symbol' not in st.session_state:
    st.session_state.symbol = "SPY"

# Prominent symbol input
col1, col2 = st.columns([3, 1])
with col1:
    main_symbol = st.text_input(
        "🎯 Enter Stock Symbol",
        value=st.session_state.symbol,
        max_chars=10,
        help="Enter any stock ticker (e.g., SPY, AAPL, TSLA, QQQ)",
        key="main_symbol_input"
    ).upper()
    if main_symbol:
        st.session_state.symbol = main_symbol

symbol = st.session_state.symbol

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

st.markdown("---")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("### 📊 Stock Selection")
sidebar_symbol = st.sidebar.text_input("Enter Symbol", value=st.session_state.symbol, max_chars=10, help="Enter any stock ticker (e.g., SPY, AAPL, TSLA)").upper()
if sidebar_symbol and sidebar_symbol != st.session_state.symbol:
    st.session_state.symbol = sidebar_symbol
    st.rerun()

st.sidebar.markdown("### 📅 Data Period")
period_options = {
    "1 Month (~21 trading days)": "1mo",
    "3 Months (~63 days)": "3mo", 
    "6 Months (~126 days)": "6mo",
    "1 Year (~252 days)": "1y",
    "2 Years (~504 days)": "2y"
}
selected_period = st.sidebar.selectbox("Historical Data Range", list(period_options.keys()), index=2)
period = period_options[selected_period]

st.sidebar.markdown("### 🎚️ Algorithm Settings")
smoothing_length = st.sidebar.slider("Smoothing Length", 5, 20, 10, help="Higher = smoother but slower to react")
detrend_period = st.sidebar.slider("Detrend Period", 10, 40, 20, help="Period for trend removal")

# Auto-refresh
st.sidebar.markdown("### 🔄 Auto-Refresh")
auto_refresh = st.sidebar.checkbox("Enable auto-refresh (3 min)", value=False)
if auto_refresh:
    st.sidebar.success("✅ Auto-refreshing every 3 minutes")
    st.markdown("""
        <script>
            setTimeout(function() {
                window.location.reload();
            }, 180000);
        </script>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_data(symbol, period):
    """Fetch price data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")
        
        if df.empty:
            return None
            
        # Ensure timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[['Close']].copy()
        df.columns = ['close']
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def smooth_price(price, length=6):
    """Smooth prices using weighted moving average"""
    weights = np.array([1, 2, 2, 1])
    smoothed = np.convolve(price, weights / weights.sum(), mode='same')
    return smoothed

def calculate_cycle_indicator(df, smoothing_length=10, detrend_period=20):
    """
    Calculate Ehlers cycle indicator with phase and dominant period
    """
    df = df.copy()
    price = df['close'].values
    n = len(price)
    
    # Initialize arrays
    smooth = np.zeros(n)
    inphase = np.zeros(n)
    quadrature = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    re_smooth = np.zeros(n)
    im_smooth = np.zeros(n)
    period = np.zeros(n)
    smooth_period = np.zeros(n)
    phase = np.zeros(n)
    
    # Smooth prices
    for i in range(3, n):
        smooth[i] = (price[i] + 2*price[i-1] + 2*price[i-2] + price[i-3]) / 6
    
    # Calculate Hilbert Transform components
    for i in range(7, n):
        inphase[i] = (smooth[i] - smooth[i-7]) / 2
        
    for i in range(3, n):
        quadrature[i] = (smooth[i] - smooth[i-2] + 2*(smooth[i-1] - smooth[i-3])) / 4
    
    # Calculate Re and Im for period
    for i in range(1, n):
        re[i] = inphase[i] * inphase[i-1] + quadrature[i] * quadrature[i-1]
        im[i] = inphase[i] * quadrature[i-1] - quadrature[i] * inphase[i-1]
    
    # Smooth Re and Im
    for i in range(3, n):
        re_smooth[i] = (re[i] + 2*re[i-1] + 2*re[i-2] + re[i-3]) / 6
        im_smooth[i] = (im[i] + 2*im[i-1] + 2*im[i-2] + im[i-3]) / 6
    
    # Calculate period
    for i in range(3, n):
        if re_smooth[i] != 0 and im_smooth[i] != 0:
            deltaphase = abs(np.arctan(im_smooth[i] / re_smooth[i]) * 180 / np.pi)
        else:
            deltaphase = 0
            
        if deltaphase > 0:
            inst_period = 360 / deltaphase
        else:
            inst_period = period[i-1] if i > 0 else 20
            
        # Constrain period
        period[i] = np.clip(inst_period, 10, 40)
    
    # Smooth period
    for i in range(3, n):
        smooth_period[i] = (period[i] + 2*period[i-1] + 2*period[i-2] + period[i-3]) / 6
    
    # Calculate phase
    for i in range(1, n):
        if inphase[i] != 0:
            raw_phase = np.arctan(quadrature[i] / inphase[i]) * 180 / np.pi
        else:
            raw_phase = 0
            
        # Accumulate phase
        if raw_phase < phase[i-1] - 270:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1] + 360)
        elif raw_phase > phase[i-1] + 270:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1] - 360)
        else:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1])
    
    # Wrap phase to 0-360
    wrapped_phase = phase % 360
    
    # Detrended price (remove trend to show pure cycle)
    detrended = np.zeros(n)
    for i in range(detrend_period, n):
        trend = np.mean(smooth[i-detrend_period:i])
        detrended[i] = smooth[i] - trend
    
    # Normalize detrended price
    normalized_cycle = np.zeros(n)
    for i in range(detrend_period*2, n):
        window = detrended[i-detrend_period:i]
        std = np.std(window)
        if std > 0:
            normalized_cycle[i] = detrended[i] / std
    
    # Calculate momentum
    momentum_period = 5
    momentum = np.zeros(n)
    for i in range(momentum_period, n):
        momentum[i] = smooth[i] - smooth[i-momentum_period]
    
    # Smooth momentum
    momentum_ma = np.convolve(momentum, np.ones(5)/5, mode='same')
    
    # Cycle strength
    cycle_strength = np.abs(normalized_cycle)
    
    # Add to dataframe
    df['smooth'] = smooth
    df['normalized_cycle'] = normalized_cycle
    df['phase'] = wrapped_phase
    df['period'] = smooth_period
    df['momentum'] = momentum_ma
    df['cycle_strength'] = cycle_strength
    
    # Local extrema detection (peaks and troughs in normalized cycle)
    window = 5  # Look 5 bars forward and back
    df['is_local_max'] = False
    df['is_local_min'] = False
    
    for i in range(window, n - window):
        # Local maximum
        if normalized_cycle[i] == max(normalized_cycle[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('is_local_max')] = True
        # Local minimum
        if normalized_cycle[i] == min(normalized_cycle[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('is_local_min')] = True
    
    # Signal detection with strict filters
    # Peak: phase + local max + strong cycle + above threshold
    df['is_peak'] = (
        ((df['phase'] >= 315) | (df['phase'] <= 45)) & 
        (df['normalized_cycle'] > 1.5) & 
        (df['cycle_strength'] > 1.0) &
        df['is_local_max']
    )
    
    # Bottom: phase + local min + strong cycle + below threshold
    # Made more sensitive: lower threshold (-1.3), wider phase range (120-240), lower strength requirement (0.8)
    df['is_bottom'] = (
        ((df['phase'] >= 120) & (df['phase'] <= 240)) & 
        (df['normalized_cycle'] < -1.3) &  # More sensitive threshold
        (df['cycle_strength'] > 0.8) &  # Lower strength requirement
        df['is_local_min'] &
        (df['momentum'] < 0)  # Momentum confirmation: should be declining
    )
    
    # Approaching signals (less strict)
    df['approaching_peak'] = (
        ((df['phase'] >= 270) & (df['phase'] < 315)) & 
        (df['normalized_cycle'] > 1.2) &
        (df['cycle_strength'] > 0.8)
    )
    
    # Approaching bottom - more sensitive to catch opportunities early
    df['approaching_bottom'] = (
        ((df['phase'] >= 75) & (df['phase'] < 120)) &  # Earlier phase range
        (df['normalized_cycle'] < -1.0) &  # Lower threshold
        (df['cycle_strength'] > 0.6) &  # More lenient strength
        (df['momentum'] < 0)  # Momentum should be negative
    )
    
    return df

def create_price_chart(df, symbol):
    """Create price chart with cycle signals overlaid"""
    fig = go.Figure()
    
    # 20-day MA (behind price)
    ma20 = df['close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=ma20,
        name='20-MA',
        line=dict(color='rgba(255,165,0,0.4)', width=2, dash='dot'),
        mode='lines',
        hovertemplate='MA20: $%{y:.2f}<extra></extra>'
    ))
    
    # Price line (on top)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name=f'{symbol} Price',
        line=dict(color='#00D9FF', width=2.5),
        mode='lines',
        hovertemplate='Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Mark peaks on price chart
    peak_df = df[df['is_peak']]
    if not peak_df.empty:
        fig.add_trace(go.Scatter(
            x=peak_df.index,
            y=peak_df['close'],
            name='🔴 Cycle Peak (Sell)',
            mode='markers+text',
            marker=dict(
                color='red', 
                size=16, 
                symbol='triangle-down',
                line=dict(color='white', width=1.5)
            ),
            text=['▼'] * len(peak_df),
            textposition='top center',
            textfont=dict(size=10, color='red'),
            hovertemplate='<b>PEAK SIGNAL</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<br>Action: Consider Selling<extra></extra>'
        ))
    
    # Mark bottoms on price chart
    bottom_df = df[df['is_bottom']]
    if not bottom_df.empty:
        fig.add_trace(go.Scatter(
            x=bottom_df.index,
            y=bottom_df['close'],
            name='🟢 Cycle Bottom (Buy)',
            mode='markers+text',
            marker=dict(
                color='lime', 
                size=16, 
                symbol='triangle-up',
                line=dict(color='white', width=1.5)
            ),
            text=['▲'] * len(bottom_df),
            textposition='bottom center',
            textfont=dict(size=10, color='lime'),
            hovertemplate='<b>BOTTOM SIGNAL</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<br>Action: Consider Buying<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> Price Action with Cycle Turning Points",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=350,  # Reduced height to fit both charts on screen
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.01, 
            y=0.99, 
            bgcolor='rgba(20,20,20,0.85)',
            bordercolor='rgba(0,217,255,0.5)',
            borderwidth=1.5
        ),
        margin=dict(t=50, b=40, l=60, r=20)
    )
    
    return fig

def create_cycle_chart(df):
    """Create interactive Plotly chart for cycle indicator"""
    fig = go.Figure()
    
    # Main cycle oscillator with better visibility
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['normalized_cycle'],
        name='Cycle Oscillator',
        line=dict(color='#00D9FF', width=3),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)',
        hovertemplate='Cycle: %{y:.2f}σ<extra></extra>'
    ))
    
    # Phase indicator (scaled) - hidden by default
    phase_scaled = (df['phase'] - 180) / 90
    fig.add_trace(go.Scatter(
        x=df.index,
        y=phase_scaled,
        name='Phase',
        line=dict(color='rgba(255,165,0,0.3)', width=1, dash='dot'),
        mode='lines',
        visible='legendonly',
        hovertemplate='Phase: %{y:.2f}<extra></extra>'
    ))
    
    # Momentum - hidden by default
    momentum_normalized = df['momentum'] / (df['close'] * 0.01)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=momentum_normalized,
        name='Momentum',
        line=dict(color='rgba(255,0,255,0.3)', width=1, dash='dot'),
        mode='lines',
        visible='legendonly',
        hovertemplate='Momentum: %{y:.2f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)
    
    # Critical threshold lines
    fig.add_hline(y=1.5, line_dash="dash", line_color="rgba(255,100,100,0.8)", line_width=2,
                  annotation_text="Peak Threshold (+1.5σ)", annotation_position="right",
                  annotation=dict(font=dict(size=10, color='rgba(255,100,100,1)')))
    fig.add_hline(y=-1.3, line_dash="dash", line_color="rgba(100,255,100,0.8)", line_width=2,
                  annotation_text="Bottom Threshold (-1.3σ)", annotation_position="right",
                  annotation=dict(font=dict(size=10, color='rgba(100,255,100,1)')))
    
    # Extreme zones at ±2σ
    fig.add_hline(y=2.0, line_dash="dot", line_color="rgba(255,0,0,0.5)", line_width=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="rgba(0,255,0,0.5)", line_width=1)
    
    # Peak signals with enhanced visibility
    peak_df = df[df['is_peak']]
    if not peak_df.empty:
        fig.add_trace(go.Scatter(
            x=peak_df.index,
            y=peak_df['normalized_cycle'],
            name='🔴 Peak (Sell)',
            mode='markers+text',
            marker=dict(
                color='#FF3333', 
                size=18, 
                symbol='circle',
                line=dict(color='white', width=2.5)
            ),
            text=['▼'] * len(peak_df),
            textposition='top center',
            textfont=dict(size=12, color='#FF3333', family='Arial Black'),
            hovertemplate='<b>🔴 PEAK SIGNAL</b><br>Date: %{x|%Y-%m-%d}<br>Cycle: %{y:.2f}σ<br><b>Action: SELL</b><extra></extra>'
        ))
    
    # Bottom signals with enhanced visibility
    bottom_df = df[df['is_bottom']]
    if not bottom_df.empty:
        fig.add_trace(go.Scatter(
            x=bottom_df.index,
            y=bottom_df['normalized_cycle'],
            name='🟢 Bottom (Buy)',
            mode='markers+text',
            marker=dict(
                color='#00FF00', 
                size=18, 
                symbol='circle',
                line=dict(color='white', width=2.5)
            ),
            text=['▲'] * len(bottom_df),
            textposition='bottom center',
            textfont=dict(size=12, color='#00FF00', family='Arial Black'),
            hovertemplate='<b>🟢 BOTTOM SIGNAL</b><br>Date: %{x|%Y-%m-%d}<br>Cycle: %{y:.2f}σ<br><b>Action: BUY</b><extra></extra>'
        ))
    
    # Approaching peak (fewer, more selective)
    approach_peak_df = df[df['approaching_peak']]
    if not approach_peak_df.empty:
        fig.add_trace(go.Scatter(
            x=approach_peak_df.index,
            y=[2.5] * len(approach_peak_df),
            name='⚠️ Peak Warning',
            mode='markers',
            marker=dict(
                color='orange', 
                size=10, 
                symbol='triangle-up',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>APPROACHING PEAK</b><br>Date: %{x|%Y-%m-%d}<br>Prepare to sell<extra></extra>'
        ))
    
    # Approaching bottom
    approach_bottom_df = df[df['approaching_bottom']]
    if not approach_bottom_df.empty:
        fig.add_trace(go.Scatter(
            x=approach_bottom_df.index,
            y=[-2.5] * len(approach_bottom_df),
            name='💡 Bottom Warning',
            mode='markers',
            marker=dict(
                color='lightgreen', 
                size=10, 
                symbol='triangle-down',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>APPROACHING BOTTOM</b><br>Date: %{x|%Y-%m-%d}<br>Prepare to buy<extra></extra>'
        ))
    
    # Shaded extreme zones (beyond ±2σ)
    fig.add_hrect(y0=2, y1=3.5, fillcolor="rgba(255,0,0,0.08)", line_width=0,
                  annotation_text="Extreme Overbought", annotation_position="top right",
                  annotation=dict(font=dict(size=9, color='rgba(255,100,100,0.6)')))
    fig.add_hrect(y0=-3.5, y1=-2, fillcolor="rgba(0,255,0,0.08)", line_width=0,
                  annotation_text="Extreme Oversold", annotation_position="bottom right",
                  annotation=dict(font=dict(size=9, color='rgba(100,255,100,0.6)')))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> Market Cycle Oscillator (Ehlers Method)",
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title="Normalized Cycle (σ)",
            range=[-3.5, 3.5],  # Fixed range for consistent viewing
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=1
        ),
        height=380,  # Reduced height to fit both charts
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.01, 
            y=0.99, 
            bgcolor='rgba(10,10,10,0.85)',
            bordercolor='rgba(0,217,255,0.5)',
            borderwidth=1.5,
            font=dict(size=11, color='white')
        ),
        margin=dict(t=50, b=80, l=70, r=100),  # More bottom margin for range slider
        plot_bgcolor='rgba(15,15,15,1)',
        paper_bgcolor='rgba(15,15,15,1)'
    )
    
    # Add range slider for zoom control
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
        rangeslider_bgcolor='rgba(50,50,50,0.5)'
    )
    
    return fig

def generate_alerts(df):
    """Generate trading alerts from signals"""
    alerts = []
    
    # Recent signals (last 30 days)
    recent = df.tail(30)
    
    for idx, row in recent.iterrows():
        if row['is_peak']:
            alerts.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Signal': 'PEAK',
                'Cycle Value': f"{row['normalized_cycle']:.2f}σ",
                'Phase': f"{row['phase']:.0f}°",
                'Price': f"${row['close']:.2f}",
                'Action': '🔴 SELL / Take Profit',
                'Strength': f"{row['cycle_strength']:.2f}"
            })
        elif row['is_bottom']:
            alerts.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Signal': 'BOTTOM',
                'Cycle Value': f"{row['normalized_cycle']:.2f}σ",
                'Phase': f"{row['phase']:.0f}°",
                'Price': f"${row['close']:.2f}",
                'Action': '🟢 BUY / Enter Long',
                'Strength': f"{row['cycle_strength']:.2f}"
            })
        elif row['approaching_peak']:
            alerts.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Signal': 'APPROACHING PEAK',
                'Cycle Value': f"{row['normalized_cycle']:.2f}σ",
                'Phase': f"{row['phase']:.0f}°",
                'Price': f"${row['close']:.2f}",
                'Action': '🟠 Prepare to Sell',
                'Strength': f"{row['cycle_strength']:.2f}"
            })
        elif row['approaching_bottom']:
            alerts.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Signal': 'APPROACHING BOTTOM',
                'Cycle Value': f"{row['normalized_cycle']:.2f}σ",
                'Phase': f"{row['phase']:.0f}°",
                'Price': f"${row['close']:.2f}",
                'Action': '🟢 Prepare to Buy',
                'Strength': f"{row['cycle_strength']:.2f}"
            })
    
    return pd.DataFrame(alerts)

# Main execution
if symbol:
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_data(symbol, period)
    
    if df is not None and len(df) > 50:
        # Show data info
        st.info(f"📊 **Loaded {len(df)} trading days** | Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} | Symbol: **{symbol}**")
        
        # Calculate indicator
        with st.spinner("Calculating cycle indicator..."):
            df = calculate_cycle_indicator(df, smoothing_length, detrend_period)
        
        # Display current metrics
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cycle Period", f"{latest['period']:.1f} bars")
        with col2:
            phase_color = "🔴" if latest['is_peak'] else "🟢" if latest['is_bottom'] else "⚪"
            st.metric("Phase", f"{phase_color} {latest['phase']:.0f}°")
        with col3:
            st.metric("Cycle Value", f"{latest['normalized_cycle']:.2f}σ")
        with col4:
            strength_emoji = "💪" if latest['cycle_strength'] > 1.5 else "👍" if latest['cycle_strength'] > 0.8 else "👎"
            st.metric("Strength", f"{strength_emoji} {latest['cycle_strength']:.2f}")
        
        # Current signal
        if latest['is_peak']:
            st.error("🔴 **PEAK ZONE DETECTED** - Consider selling or taking profits")
        elif latest['is_bottom']:
            st.success("🟢 **BOTTOM ZONE DETECTED** - Consider buying or entering long")
        elif latest['approaching_peak']:
            st.warning("🟠 **APPROACHING PEAK** - Prepare to sell, tighten stops")
        elif latest['approaching_bottom']:
            st.info("🟢 **APPROACHING BOTTOM** - Prepare to buy, look for entry")
        else:
            st.info("⚪ **MID-CYCLE** - No clear signal, wait for setup")
        
        # Display both charts together
        # Option 1: Vertical compact layout (both visible without scrolling)
        st.markdown("### 📊 Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Price Action")
            st.plotly_chart(create_price_chart(df, symbol), use_container_width=True)
        
        with col2:
            st.markdown("#### 🔄 Cycle Oscillator")
            st.info("💡 **Filters:** Peaks (>+1.5σ, str>1.0) | Bottoms (<-1.3σ, str>0.8, -momentum)")
            st.plotly_chart(create_cycle_chart(df), use_container_width=True)
        
        # Signal summary below charts
        st.markdown("---")
        peaks_count = df['is_peak'].sum()
        bottoms_count = df['is_bottom'].sum()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🔴 Peak Signals", peaks_count, help="Strong sell signals detected")
        with col_b:
            st.metric("🟢 Bottom Signals", bottoms_count, help="Strong buy signals detected")
        with col_c:
            if peaks_count + bottoms_count > 0:
                st.metric("📊 Signal Quality", "High" if latest['cycle_strength'] > 1.5 else "Moderate", 
                         help=f"Based on cycle strength: {latest['cycle_strength']:.2f}")
        
        # Interpretation guide
        with st.expander("📖 How to Interpret"):
            st.markdown("""
            ### Signal Guide
            - **Red Dots**: Peak detected → Consider selling/taking profits
            - **Green Dots**: Bottom detected → Consider buying/entering long
            - **Orange Triangles**: Peak approaching → Prepare to sell
            - **Light Green Triangles**: Bottom approaching → Prepare to buy
            
            ### Phase Cycle
            - **0°/360°**: Peak/Top → SELL ZONE
            - **90°**: Declining phase
            - **180°**: Bottom/Trough → BUY ZONE
            - **270°**: Rising phase
            
            ### Cycle Value
            - **> +2.0σ**: Extreme overbought, peak imminent
            - **< -2.0σ**: Extreme oversold, bottom imminent
            - **Cross above 0**: Bullish momentum
            - **Cross below 0**: Bearish momentum
            
            ### Strength
            - **> 1.5**: Strong, reliable cycle
            - **0.8-1.5**: Moderate cycle
            - **< 0.8**: Weak cycle, use caution
            
            ### Best Trades
            ✅ **Strong Buy**: Bottom + Cycle <-2 + Strength >1.2 + Phase 135-225°  
            ✅ **Strong Sell**: Peak + Cycle >+2 + Strength >1.2 + Phase 315-45°
            """)
        
        # Alerts table
        st.subheader("📊 Recent Signals")
        alerts_df = generate_alerts(df)
        
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            
            # CSV download
            csv = alerts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Alerts CSV",
                data=csv,
                file_name=f"{symbol}_cycle_alerts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No signals detected in the last 30 days")
        
        # Statistics
        st.subheader("📈 Cycle Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_period = df['period'].tail(30).mean()
            st.metric("Avg Cycle Period (30d)", f"{avg_period:.1f} bars")
        with col2:
            avg_strength = df['cycle_strength'].tail(30).mean()
            st.metric("Avg Strength (30d)", f"{avg_strength:.2f}")
        with col3:
            peaks_count = df['is_peak'].tail(60).sum()
            bottoms_count = df['is_bottom'].tail(60).sum()
            st.metric("Peaks/Bottoms (60d)", f"{peaks_count}/{bottoms_count}")
        
    else:
        st.error("Insufficient data. Try a different symbol or period.")
else:
    st.info("Enter a symbol in the sidebar to begin.")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: Yahoo Finance")
