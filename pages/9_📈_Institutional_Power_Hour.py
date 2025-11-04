"""
Institutional Power Hour Analysis
Tracks institutional activity in the last 1-2 hours of trading to predict next-day movement
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import asyncio
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="Institutional Power Hour",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .bullish-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        font-weight: bold;
    }
    .bearish-signal {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        font-weight: bold;
    }
    .neutral-signal {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Institutional Power Hour Analysis")
st.markdown("**Track smart money moves in the final trading hours to predict tomorrow's direction**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    symbols_input = st.text_input(
        "Stock Symbols (comma-separated)",
        value="SPY,QQQ,AAPL,TSLA,NVDA",
        help="Enter stock symbols to analyze"
    )
    
    lookback_days = st.slider(
        "Historical Days",
        min_value=1,
        max_value=10,
        value=5,
        help="Days of historical data for pattern analysis"
    )
    
    power_hour_start = st.selectbox(
        "Power Hour Start Time",
        options=["2:00 PM", "2:30 PM", "3:00 PM"],
        index=2,
        help="When to start tracking institutional activity"
    )
    
    analyze_button = st.button("🔍 Analyze Power Hour Activity", type="primary")

def get_power_hour_start_hour(selection):
    """Convert selection to hour for filtering"""
    mapping = {
        "2:00 PM": 14,
        "2:30 PM": 14,
        "3:00 PM": 15
    }
    return mapping[selection]

def analyze_intraday_momentum(candles_df, power_hour_start_hour=15):
    """
    Analyze intraday momentum focusing on power hour
    Returns momentum metrics and signals
    """
    try:
        if candles_df.empty:
            return None
        
        # Convert datetime column to Eastern Time (US market hours)
        candles_df['datetime'] = pd.to_datetime(candles_df['datetime'], unit='ms', utc=True)
        candles_df['datetime'] = candles_df['datetime'].dt.tz_convert('America/New_York')
        candles_df['hour'] = candles_df['datetime'].dt.hour
        candles_df['minute'] = candles_df['datetime'].dt.minute
        candles_df['date'] = candles_df['datetime'].dt.date
        
        # Filter to regular market hours only (9:30 AM - 4:00 PM ET)
        candles_df = candles_df[
            ((candles_df['hour'] == 9) & (candles_df['minute'] >= 30)) |
            ((candles_df['hour'] >= 10) & (candles_df['hour'] < 16))
        ].copy()
        
        # Get only the most recent trading day
        if not candles_df.empty:
            latest_date = candles_df['date'].max()
            candles_df = candles_df[candles_df['date'] == latest_date].copy()
        
        if candles_df.empty:
            return None
        
        # Separate regular hours and power hour
        power_hour = candles_df[candles_df['hour'] >= power_hour_start_hour].copy()
        morning_session = candles_df[candles_df['hour'] < 12].copy()
        midday_session = candles_df[(candles_df['hour'] >= 12) & (candles_df['hour'] < power_hour_start_hour)].copy()
        
        if power_hour.empty:
            return None
        
        # Calculate metrics
        open_price = candles_df.iloc[0]['open'] if not candles_df.empty else 0
        close_price = candles_df.iloc[-1]['close'] if not candles_df.empty else 0
        
        power_hour_open = power_hour.iloc[0]['open'] if not power_hour.empty else 0
        power_hour_close = power_hour.iloc[-1]['close'] if not power_hour.empty else 0
        power_hour_high = power_hour['high'].max() if not power_hour.empty else 0
        power_hour_low = power_hour['low'].min() if not power_hour.empty else 0
        
        # Calculate volume metrics
        total_volume = candles_df['volume'].sum()
        power_hour_volume = power_hour['volume'].sum()
        morning_volume = morning_session['volume'].sum() if not morning_session.empty else 0
        midday_volume = midday_session['volume'].sum() if not midday_session.empty else 0
        
        # Volume concentration in power hour
        volume_concentration = (power_hour_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Price movement analysis
        day_change = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0
        power_hour_change = ((power_hour_close - power_hour_open) / power_hour_open * 100) if power_hour_open > 0 else 0
        
        # Calculate momentum score
        # Positive factors: strong power hour move, high volume concentration
        momentum_score = 0
        
        if abs(power_hour_change) > abs(day_change) * 0.5:
            momentum_score += 2  # Power hour driving the day
        
        if volume_concentration > 35:
            momentum_score += 2  # High institutional activity
        elif volume_concentration > 25:
            momentum_score += 1
        
        if power_hour_change > 0.5:
            momentum_score += 1  # Bullish power hour
        elif power_hour_change < -0.5:
            momentum_score -= 1  # Bearish power hour
        
        # VWAP analysis for power hour
        power_hour['vwap'] = (power_hour['volume'] * (power_hour['high'] + power_hour['low'] + power_hour['close']) / 3).cumsum() / power_hour['volume'].cumsum()
        
        if not power_hour.empty and 'vwap' in power_hour.columns:
            final_vwap = power_hour.iloc[-1]['vwap']
            price_vs_vwap = ((power_hour_close - final_vwap) / final_vwap * 100) if final_vwap > 0 else 0
            
            if price_vs_vwap > 0.2:
                momentum_score += 1  # Closing above VWAP = bullish
            elif price_vs_vwap < -0.2:
                momentum_score -= 1  # Closing below VWAP = bearish
        
        # Determine signal
        if momentum_score >= 3:
            signal = "STRONG_BULLISH"
            signal_emoji = "🚀"
        elif momentum_score >= 1:
            signal = "BULLISH"
            signal_emoji = "📈"
        elif momentum_score <= -3:
            signal = "STRONG_BEARISH"
            signal_emoji = "💥"
        elif momentum_score <= -1:
            signal = "BEARISH"
            signal_emoji = "📉"
        else:
            signal = "NEUTRAL"
            signal_emoji = "➡️"
        
        # Get trading date info
        trading_date = candles_df['datetime'].iloc[0].strftime('%Y-%m-%d') if not candles_df.empty else 'Unknown'
        trading_day_name = candles_df['datetime'].iloc[0].strftime('%A, %B %d, %Y') if not candles_df.empty else 'Unknown'
        
        return {
            'open_price': open_price,
            'close_price': close_price,
            'day_change_pct': day_change,
            'power_hour_open': power_hour_open,
            'power_hour_close': power_hour_close,
            'power_hour_change_pct': power_hour_change,
            'power_hour_high': power_hour_high,
            'power_hour_low': power_hour_low,
            'total_volume': total_volume,
            'power_hour_volume': power_hour_volume,
            'morning_volume': morning_volume,
            'midday_volume': midday_volume,
            'volume_concentration': volume_concentration,
            'momentum_score': momentum_score,
            'signal': signal,
            'signal_emoji': signal_emoji,
            'candles_df': candles_df,
            'power_hour_df': power_hour,
            'trading_date': trading_date,
            'trading_day_name': trading_day_name
        }
        
    except Exception as e:
        st.error(f"Error analyzing momentum: {str(e)}")
        return None

def create_intraday_chart(analysis_data, symbol):
    """Create interactive intraday chart with power hour highlighted"""
    try:
        df = analysis_data['candles_df']
        power_hour_df = analysis_data['power_hour_df']
        
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#3498db', width=2),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        # Highlight power hour
        if not power_hour_df.empty:
            fig.add_trace(go.Scatter(
                x=power_hour_df['datetime'],
                y=power_hour_df['close'],
                mode='lines',
                name='Power Hour',
                line=dict(color='#e74c3c', width=3),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.1)',
                hovertemplate='Power Hour: %{y:.2f}<extra></extra>'
            ))
        
        # Volume bars (secondary y-axis)
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(158, 158, 158, 0.3)',
            yaxis='y2',
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{symbol} Intraday Activity - Power Hour Highlighted",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def format_volume(volume):
    """Format volume for display"""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    else:
        return f"{volume:,.0f}"

# Main analysis logic
if analyze_button:
    symbols = [s.strip().upper() for s in symbols_input.split(",")]
    
    with st.spinner("🔄 Fetching institutional activity data..."):
        try:
            client = SchwabClient()
            
            if not client.authenticate():
                st.error("Failed to authenticate with Schwab API")
                st.stop()
            
            power_hour_start_hour = get_power_hour_start_hour(power_hour_start)
            
            # Analyze each symbol
            for symbol in symbols:
                st.markdown("---")
                st.header(f"📊 {symbol}")
                
                # Get intraday data (1-minute candles for most recent trading day)
                # Use timestamp range to ensure we get latest data
                now = datetime.now()
                
                # Set to current time to get most recent data
                end_time = int(now.timestamp() * 1000)
                # Start from 24 hours ago to capture full trading day
                start_time = int((now - timedelta(hours=24)).timestamp() * 1000)
                
                try:
                    history = client.get_price_history(
                        symbol=symbol,
                        frequency_type='minute',
                        frequency=1,
                        start_date=start_time,
                        end_date=end_time,
                        need_extended_hours=True
                    )
                    
                    if 'candles' in history and history['candles']:
                        candles_df = pd.DataFrame(history['candles'])
                        
                        # Analyze momentum
                        analysis = analyze_intraday_momentum(candles_df, power_hour_start_hour)
                        
                        if analysis:
                            # Display trading date
                            st.info(f"📅 **Trading Date:** {analysis['trading_day_name']}")
                            
                            # Display signal card
                            signal = analysis['signal']
                            if 'BULLISH' in signal:
                                signal_class = 'bullish-signal'
                            elif 'BEARISH' in signal:
                                signal_class = 'bearish-signal'
                            else:
                                signal_class = 'neutral-signal'
                            
                            st.markdown(f"""
                            <div class="{signal_class}">
                                {analysis['signal_emoji']} {signal.replace('_', ' ')} - Momentum Score: {analysis['momentum_score']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Day Change",
                                    f"{analysis['day_change_pct']:.2f}%",
                                    delta=f"${analysis['close_price'] - analysis['open_price']:.2f}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Power Hour Change",
                                    f"{analysis['power_hour_change_pct']:.2f}%",
                                    delta=f"${analysis['power_hour_close'] - analysis['power_hour_open']:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Volume Concentration",
                                    f"{analysis['volume_concentration']:.1f}%",
                                    help="% of total volume in power hour"
                                )
                            
                            with col4:
                                st.metric(
                                    "Power Hour Volume",
                                    format_volume(analysis['power_hour_volume']),
                                    delta=f"{format_volume(analysis['total_volume'])} total"
                                )
                            
                            # Chart
                            chart = create_intraday_chart(analysis, symbol)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            
                            # Detailed breakdown
                            with st.expander("📋 Detailed Analysis"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### Session Breakdown")
                                    st.write(f"**Morning Volume:** {format_volume(analysis['morning_volume'])}")
                                    st.write(f"**Midday Volume:** {format_volume(analysis['midday_volume'])}")
                                    st.write(f"**Power Hour Volume:** {format_volume(analysis['power_hour_volume'])}")
                                    st.write(f"**Total Volume:** {format_volume(analysis['total_volume'])}")
                                
                                with col2:
                                    st.markdown("### Price Action")
                                    st.write(f"**Day Open:** ${analysis['open_price']:.2f}")
                                    st.write(f"**Day Close:** ${analysis['close_price']:.2f}")
                                    st.write(f"**Power Hour High:** ${analysis['power_hour_high']:.2f}")
                                    st.write(f"**Power Hour Low:** ${analysis['power_hour_low']:.2f}")
                                
                                st.markdown("### 💡 Interpretation")
                                if analysis['volume_concentration'] > 35:
                                    st.success("🎯 High institutional activity detected - positions being built/unwound")
                                
                                if abs(analysis['power_hour_change_pct']) > abs(analysis['day_change_pct']) * 0.5:
                                    st.info("⚡ Power hour is driving the day's price action")
                                
                                if analysis['momentum_score'] >= 3:
                                    st.success("🚀 Strong bullish setup - institutions buying into close")
                                elif analysis['momentum_score'] <= -3:
                                    st.error("💥 Strong bearish setup - institutions selling into close")
                        else:
                            st.warning(f"Could not analyze power hour data for {symbol}")
                    else:
                        st.warning(f"No intraday data available for {symbol}")
                        
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    # Instructions
    st.info("""
    ### 🎯 How This Works
    
    **The "Power Hour" (last 1-2 hours of trading) is when institutions make their moves:**
    
    1. **Volume Concentration** - If >30% of daily volume occurs in power hour, institutions are active
    2. **Price vs VWAP** - Closing above VWAP = bullish, below = bearish
    3. **Momentum Score** - Combines multiple factors to predict next-day direction
    4. **Signal Strength** - Strong signals (3+ score) have higher predictive value
    
    ### 📊 What To Look For
    
    - 🚀 **Strong Bullish**: Heavy buying into close, high volume, closing near highs
    - 📈 **Bullish**: Moderate buying pressure, positive momentum
    - ➡️ **Neutral**: No clear institutional direction
    - 📉 **Bearish**: Moderate selling pressure, negative momentum  
    - 💥 **Strong Bearish**: Heavy selling into close, high volume, closing near lows
    
    ### 💡 Trading Ideas
    
    - **Gap Up Expected**: Strong bullish power hour → Consider calls at open
    - **Gap Down Expected**: Strong bearish power hour → Consider puts at open
    - **Reversal Setup**: Power hour contradicts day's trend → Potential reversal
    
    **Note**: This is experimental analysis. Always do your own research and manage risk!
    """)
    
    st.markdown("---")
    st.markdown("**👈 Configure settings in the sidebar and click 'Analyze' to start**")
