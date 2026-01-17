"""
ES Futures (S&P 500 Futures) - Comprehensive Data Dashboard
Real-time futures data, charts, and analysis
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
from src.utils.cached_client import get_client

st.set_page_config(
    page_title="ES Futures",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ES Futures - S&P 500 Futures Analysis")

# Initialize session state
if 'auto_refresh_futures' not in st.session_state:
    st.session_state.auto_refresh_futures = True
if 'last_refresh_futures' not in st.session_state:
    st.session_state.last_refresh_futures = datetime.now()

# Futures symbols
FUTURES_SYMBOLS = {
    '/ES': 'E-mini S&P 500',
    '/MES': 'Micro E-mini S&P 500',
    '/NQ': 'E-mini NASDAQ-100',
    '/MNQ': 'Micro E-mini NASDAQ-100',
    '/YM': 'E-mini Dow',
    '/RTY': 'E-mini Russell 2000'
}

@st.cache_data(ttl=180, show_spinner=False)
def get_futures_quote(symbol: str):
    """Get futures quote data"""
    try:
        client = get_client()
        if not client:
            return None
        
        quote = client.get_quote(symbol)
        
        if quote:
            # Try different key formats that Schwab API might return
            if symbol in quote:
                return quote[symbol]
            # Sometimes API returns without the forward slash
            elif symbol.replace('/', '') in quote:
                return quote[symbol.replace('/', '')]
            # Or try with URL encoded slash
            elif '%2F' + symbol[1:] in quote:
                return quote['%2F' + symbol[1:]]
            # Return first available key if any
            elif quote:
                first_key = list(quote.keys())[0]
                return quote[first_key]
        
        return None
    except Exception as e:
        logger.error(f"Error fetching futures quote for {symbol}: {e}")
        return None

@st.cache_data(ttl=180, show_spinner=False)
def get_futures_history(symbol: str, period_type: str = "day", period: int = 1):
    """Get futures price history"""
    try:
        client = get_client()
        if not client:
            return None
        
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        
        if period_type == "day":
            start_time = int((now - timedelta(days=period)).timestamp() * 1000)
            frequency_type = 'minute'
            frequency = 5
        else:
            start_time = int((now - timedelta(days=period * 7)).timestamp() * 1000)
            frequency_type = 'daily'
            frequency = 1
        
        history = client.get_price_history(
            symbol=symbol,
            frequency_type=frequency_type,
            frequency=frequency,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=True
        )
        
        return history
    except Exception as e:
        logger.error(f"Error fetching futures history for {symbol}: {e}")
        return None

def create_futures_chart(history_data, quote_data, symbol):
    """Create comprehensive futures chart"""
    try:
        if not history_data or 'candles' not in history_data:
            return None
        
        candles = history_data['candles']
        if not candles:
            return None
        
        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} - {FUTURES_SYMBOLS.get(symbol, symbol)}', 'Volume')
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ), row=1, col=1)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#3b82f6', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#f59e0b', width=1)
        ), row=1, col=1)
        
        # Current price line
        if quote_data and 'quote' in quote_data:
            current_price = quote_data['quote'].get('lastPrice', 0)
            fig.add_hline(
                y=current_price,
                line=dict(color='#8b5cf6', width=2, dash='dash'),
                annotation_text=f"Current: ${current_price:.2f}",
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#22c55e' if close >= open else '#ef4444' 
                  for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            font=dict(size=10)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    except Exception as e:
        logger.error(f"Error creating futures chart: {e}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    selected_symbol = st.selectbox(
        "Select Futures Contract",
        options=list(FUTURES_SYMBOLS.keys()),
        format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x]}"
    )
    
    time_period = st.selectbox(
        "Chart Period",
        options=[
            ("Intraday", "day", 1),
            ("5 Days", "day", 5),
            ("1 Month", "week", 4),
            ("3 Months", "week", 12)
        ],
        format_func=lambda x: x[0]
    )
    
    auto_refresh = st.checkbox("Auto Refresh (3m)", value=st.session_state.auto_refresh_futures)
    st.session_state.auto_refresh_futures = auto_refresh
    
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# Auto refresh
if st.session_state.auto_refresh_futures:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh_futures).seconds
    if time_since_refresh >= 180:
        st.session_state.last_refresh_futures = datetime.now()
        st.rerun()

# Main content
with st.spinner(f"Loading {selected_symbol} data..."):
    quote_data = get_futures_quote(selected_symbol)
    history_data = get_futures_history(selected_symbol, time_period[1], time_period[2])

if quote_data and 'quote' in quote_data:
    quote = quote_data['quote']
    
    # Header metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    last_price = quote.get('lastPrice', 0)
    net_change = quote.get('netChange', 0)
    net_percent = quote.get('netPercentChange', 0)
    
    with col1:
        st.metric(
            label="Last Price",
            value=f"${last_price:.2f}",
            delta=f"{net_change:+.2f} ({net_percent:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Bid",
            value=f"${quote.get('bidPrice', 0):.2f}",
            delta=f"x{quote.get('bidSize', 0):,}"
        )
    
    with col3:
        st.metric(
            label="Ask",
            value=f"${quote.get('askPrice', 0):.2f}",
            delta=f"x{quote.get('askSize', 0):,}"
        )
    
    with col4:
        st.metric(
            label="Volume",
            value=f"{quote.get('totalVolume', 0):,.0f}"
        )
    
    with col5:
        st.metric(
            label="Open Interest",
            value=f"{quote.get('openInterest', 0):,.0f}"
        )
    
    with col6:
        tick = quote.get('tick', 0)
        tick_color = "🟢" if tick > 0 else "🔴" if tick < 0 else "⚪"
        st.metric(
            label="Tick",
            value=f"{tick_color} {tick:+.2f}"
        )
    
    # Price data table
    st.markdown("### 📊 Price Data")
    col1, col2 = st.columns(2)
    
    with col1:
        price_data = {
            "Open": f"${quote.get('openPrice', 0):.2f}",
            "High": f"${quote.get('highPrice', 0):.2f}",
            "Low": f"${quote.get('lowPrice', 0):.2f}",
            "Close (Prev)": f"${quote.get('closePrice', 0):.2f}",
            "52W High": f"${quote.get('52WkHigh', 0):.2f}",
            "52W Low": f"${quote.get('52WkLow', 0):.2f}"
        }
        
        df_price = pd.DataFrame(list(price_data.items()), columns=['Metric', 'Value'])
        st.dataframe(df_price, hide_index=True, use_container_width=True)
    
    with col2:
        trading_data = {
            "Mark": f"${quote.get('mark', 0):.2f}",
            "Mark Change": f"{quote.get('markChange', 0):+.2f}",
            "Mark % Change": f"{quote.get('markPercentChange', 0):+.2f}%",
            "Exchange": quote.get('exchangeName', 'N/A'),
            "Security Status": quote.get('securityStatus', 'N/A'),
            "Last Trade Time": datetime.fromtimestamp(quote.get('quoteTime', 0)/1000).strftime('%Y-%m-%d %H:%M:%S') if quote.get('quoteTime') else 'N/A'
        }
        
        df_trading = pd.DataFrame(list(trading_data.items()), columns=['Metric', 'Value'])
        st.dataframe(df_trading, hide_index=True, use_container_width=True)
    
    # Chart
    st.markdown("### 📈 Price Chart")
    chart = create_futures_chart(history_data, quote_data, selected_symbol)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.error("Unable to load chart data")
    
    # Technical indicators
    if history_data and 'candles' in history_data:
        st.markdown("### 📊 Technical Analysis")
        
        candles = history_data['candles']
        df = pd.DataFrame(candles)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            rsi_color = "🟢" if current_rsi < 30 else "🔴" if current_rsi > 70 else "🟡"
            st.metric(
                label=f"{rsi_color} RSI (14)",
                value=f"{current_rsi:.2f}",
                delta="Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
            )
        
        with col2:
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            current_histogram = histogram.iloc[-1]
            hist_color = "🟢" if current_histogram > 0 else "🔴"
            
            st.metric(
                label=f"{hist_color} MACD Histogram",
                value=f"{current_histogram:.2f}",
                delta="Bullish" if current_histogram > 0 else "Bearish"
            )
        
        with col3:
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            current_price = df['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            bb_position = (current_price - current_lower) / (current_upper - current_lower) * 100
            bb_color = "🔴" if bb_position > 80 else "🟢" if bb_position < 20 else "🟡"
            
            st.metric(
                label=f"{bb_color} Bollinger Band %",
                value=f"{bb_position:.1f}%",
                delta="Upper" if bb_position > 80 else "Lower" if bb_position < 20 else "Mid"
            )
    
    # Related futures comparison
    st.markdown("### 🔄 Related Futures Comparison")
    
    comparison_symbols = ['/ES', '/MES', '/NQ', '/YM', '/RTY']
    comparison_data = []
    
    for sym in comparison_symbols:
        data = get_futures_quote(sym)
        if data and sym in data and 'quote' in data[sym]:
            q = data[sym]['quote']
            comparison_data.append({
                'Symbol': sym,
                'Name': FUTURES_SYMBOLS.get(sym, sym),
                'Last': f"${q.get('lastPrice', 0):.2f}",
                'Change': f"{q.get('netChange', 0):+.2f}",
                '% Change': f"{q.get('netPercentChange', 0):+.2f}%",
                'Volume': f"{q.get('totalVolume', 0):,.0f}",
                'Open Interest': f"{q.get('openInterest', 0):,.0f}"
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)
    
    # Market hours info
    st.markdown("---")
    st.markdown("""
    **📅 Futures Trading Hours** (Eastern Time)
    - **Sunday - Friday**: 6:00 PM - 5:00 PM (next day)
    - **Daily Break**: 5:00 PM - 6:00 PM
    - **Nearly 24/5 Trading**: 23 hours/day, 5 days/week
    
    **ℹ️ Contract Specifications**
    - **Tick Size**: 0.25 index points = $12.50
    - **Contract Multiplier**: $50 per index point
    - **E-mini /ES**: Full contract size
    - **Micro /MES**: 1/10th the size of /ES ($5 per point)
    """)

else:
    st.error(f"Unable to fetch data for {selected_symbol}")
    st.info("Please check your API connection and try again.")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data provided by Schwab API")
