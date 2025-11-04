"""
Option Volume Walls & Levels (NetSPY Indicator)
Visualizes key support/resistance levels based on option volume analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="Option Volume Walls",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Option Volume Walls & Key Levels")
st.markdown("**Identify support/resistance levels based on massive option volume concentrations**")

# Settings at the top
st.markdown("## ⚙️ Settings")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    symbol = st.text_input("Symbol", value="SPY").upper()

with col2:
    expiry_date = st.date_input(
        "Expiration Date",
        value=datetime.now() + timedelta(days=7),
        help="Select the options expiration date to analyze"
    )

with col3:
    strike_spacing = st.number_input(
        "Strike Spacing",
        min_value=0.5,
        max_value=10.0,
        value=5.0 if symbol in ['SPY', 'QQQ'] else 5.0,
        step=0.5,
        help="Distance between strikes (e.g., 5 for SPY)"
    )

with col4:
    num_strikes = st.slider(
        "Number of Strikes (each side)",
        min_value=10,
        max_value=30,
        value=20,
        help="How many strikes above/below current price to analyze"
    )

analyze_button = st.button("🔍 Calculate Levels", type="primary", use_container_width=True)

def calculate_option_walls(options_data, underlying_price, strike_spacing, num_strikes):
    """
    Calculate key levels based on option volume
    Returns: call wall, put wall, net call wall, net put wall, flip level
    """
    try:
        # Round underlying price to nearest 10
        base_strike = np.floor(underlying_price / 10) * 10
        
        # Generate strike range
        strikes_above = [base_strike + strike_spacing * i for i in range(num_strikes + 1)]
        strikes_below = [base_strike - strike_spacing * i for i in range(1, num_strikes + 1)]
        all_strikes = sorted(strikes_below + strikes_above)
        
        # Extract volumes from options data
        call_volumes = {}
        put_volumes = {}
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if strike in all_strikes and contracts:
                        volume = contracts[0].get('totalVolume', 0)
                        call_volumes[strike] = call_volumes.get(strike, 0) + volume
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if strike in all_strikes and contracts:
                        volume = contracts[0].get('totalVolume', 0)
                        put_volumes[strike] = put_volumes.get(strike, 0) + volume
        
        # Calculate net volumes (Put - Call, positive = bearish, negative = bullish)
        net_volumes = {}
        for strike in all_strikes:
            call_vol = call_volumes.get(strike, 0)
            put_vol = put_volumes.get(strike, 0)
            net_volumes[strike] = put_vol - call_vol
        
        # Find walls (max volumes)
        call_wall_strike = max(call_volumes.items(), key=lambda x: x[1])[0] if call_volumes else None
        call_wall_volume = call_volumes.get(call_wall_strike, 0) if call_wall_strike else 0
        
        put_wall_strike = max(put_volumes.items(), key=lambda x: x[1])[0] if put_volumes else None
        put_wall_volume = put_volumes.get(put_wall_strike, 0) if put_wall_strike else 0
        
        # Find net walls (max absolute net volumes)
        bullish_strikes = {k: abs(v) for k, v in net_volumes.items() if v < 0}  # negative = call dominant
        bearish_strikes = {k: abs(v) for k, v in net_volumes.items() if v > 0}  # positive = put dominant
        
        net_call_wall_strike = max(bullish_strikes.items(), key=lambda x: x[1])[0] if bullish_strikes else None
        net_call_wall_volume = net_volumes.get(net_call_wall_strike, 0) if net_call_wall_strike else 0
        
        net_put_wall_strike = max(bearish_strikes.items(), key=lambda x: x[1])[0] if bearish_strikes else None
        net_put_wall_volume = net_volumes.get(net_put_wall_strike, 0) if net_put_wall_strike else 0
        
        # Find flip level (where net volume changes sign near current price)
        strikes_near_price = [s for s in all_strikes if abs(s - underlying_price) < strike_spacing * 5]
        flip_strike = None
        for i in range(len(strikes_near_price) - 1):
            s1, s2 = strikes_near_price[i], strikes_near_price[i + 1]
            if net_volumes.get(s1, 0) * net_volumes.get(s2, 0) < 0:  # Sign change
                flip_strike = s1 if abs(s1 - underlying_price) < abs(s2 - underlying_price) else s2
                break
        
        # Calculate totals
        total_call_vol = sum(call_volumes.values())
        total_put_vol = sum(put_volumes.values())
        total_net_vol = total_put_vol - total_call_vol
        
        return {
            'all_strikes': all_strikes,
            'call_volumes': call_volumes,
            'put_volumes': put_volumes,
            'net_volumes': net_volumes,
            'call_wall': {'strike': call_wall_strike, 'volume': call_wall_volume},
            'put_wall': {'strike': put_wall_strike, 'volume': put_wall_volume},
            'net_call_wall': {'strike': net_call_wall_strike, 'volume': net_call_wall_volume},
            'net_put_wall': {'strike': net_put_wall_strike, 'volume': net_put_wall_volume},
            'flip_level': flip_strike,
            'totals': {
                'call_vol': total_call_vol,
                'put_vol': total_put_vol,
                'net_vol': total_net_vol
            }
        }
        
    except Exception as e:
        st.error(f"Error calculating walls: {str(e)}")
        return None

def create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol):
    """Create intraday chart with key levels overlaid"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        # Filter to today's regular market hours
        df = df[
            ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
            ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
        ]
        
        if df.empty:
            return None
        
        fig = go.Figure()
        
        # Calculate VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Price candlesticks
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add VWAP line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='blue', width=2),
            hovertemplate='<b>VWAP</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add level lines
        if levels['call_wall']['strike']:
            fig.add_hline(
                y=levels['call_wall']['strike'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"📈 Call Wall ${levels['call_wall']['strike']:.2f} ({levels['call_wall']['volume']:,})",
                annotation_position="right"
            )
        
        if levels['put_wall']['strike']:
            fig.add_hline(
                y=levels['put_wall']['strike'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"📉 Put Wall ${levels['put_wall']['strike']:.2f} ({levels['put_wall']['volume']:,})",
                annotation_position="right"
            )
        
        if levels['net_call_wall']['strike']:
            fig.add_hline(
                y=levels['net_call_wall']['strike'],
                line_dash="dot",
                line_color="darkgreen",
                line_width=3,
                annotation_text=f"💚 Net Call Wall ${levels['net_call_wall']['strike']:.2f}",
                annotation_position="left"
            )
        
        if levels['net_put_wall']['strike']:
            fig.add_hline(
                y=levels['net_put_wall']['strike'],
                line_dash="dot",
                line_color="darkred",
                line_width=3,
                annotation_text=f"❤️ Net Put Wall ${levels['net_put_wall']['strike']:.2f}",
                annotation_position="left"
            )
        
        if levels['flip_level']:
            fig.add_hline(
                y=levels['flip_level'],
                line_dash="solid",
                line_color="purple",
                line_width=2,
                annotation_text=f"🔄 Flip Level ${levels['flip_level']:.2f}",
                annotation_position="bottom right"
            )
        
        fig.update_layout(
            title=f"{symbol} Intraday with Option Volume Walls",
            xaxis_title="Time (ET)",
            yaxis_title="Price ($)",
            height=600,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_volume_profile_chart(levels):
    """Create horizontal volume profile showing net volumes by strike"""
    try:
        strikes = levels['all_strikes']
        net_vols = [levels['net_volumes'].get(s, 0) for s in strikes]
        call_vols = [levels['call_volumes'].get(s, 0) for s in strikes]
        put_vols = [levels['put_volumes'].get(s, 0) for s in strikes]
        
        fig = go.Figure()
        
        # Net volume bars (horizontal)
        colors = ['red' if v > 0 else 'green' for v in net_vols]
        fig.add_trace(go.Bar(
            y=strikes,
            x=net_vols,
            orientation='h',
            name='Net Volume (Put - Call)',
            marker_color=colors,
            text=[f"{abs(v):,.0f}" for v in net_vols],
            textposition='outside',
            hovertemplate='<b>$%{y:.2f}</b><br>Net: %{x:,.0f}<extra></extra>'
        ))
        
        # Mark key levels
        annotations = []
        
        if levels['net_call_wall']['strike']:
            annotations.append(dict(
                y=levels['net_call_wall']['strike'],
                x=levels['net_call_wall']['volume'],
                text="💚 Net Call Wall",
                showarrow=True,
                arrowhead=2,
                arrowcolor="darkgreen"
            ))
        
        if levels['net_put_wall']['strike']:
            annotations.append(dict(
                y=levels['net_put_wall']['strike'],
                x=levels['net_put_wall']['volume'],
                text="❤️ Net Put Wall",
                showarrow=True,
                arrowhead=2,
                arrowcolor="darkred"
            ))
        
        if levels['flip_level']:
            fig.add_hline(
                y=levels['flip_level'],
                line_dash="dash",
                line_color="purple",
                annotation_text="🔄 Flip",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Net Option Volume Profile by Strike",
            xaxis_title="Net Volume (Put - Call)",
            yaxis_title="Strike Price ($)",
            height=700,
            template='plotly_white',
            annotations=annotations,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating volume profile: {str(e)}")
        return None

# Main analysis
if analyze_button:
    with st.spinner(f"🔄 Analyzing option volumes for {symbol}..."):
        try:
            client = SchwabClient()
            if not client.authenticate():
                st.error("Failed to authenticate with Schwab API")
                st.stop()
            
            # Get current price
            quote = client.get_quote(symbol)
            
            if not quote:
                st.error("Failed to get quote from API")
                st.stop()
            
            # Extract price from nested structure: quote[symbol]['quote']['lastPrice']
            underlying_price = quote.get(symbol, {}).get('quote', {}).get('lastPrice', 0)
            
            if not underlying_price:
                st.error(f"Could not get current price for {symbol}")
                st.stop()
            
            st.info(f"💰 Current Price: **${underlying_price:.2f}**")
            
            # Get options chain
            exp_date_str = expiry_date.strftime('%Y-%m-%d')
            options = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                from_date=exp_date_str,
                to_date=exp_date_str
            )
            
            if not options or 'callExpDateMap' not in options:
                st.error("No options data available")
                st.stop()
            
            # Calculate levels
            levels = calculate_option_walls(options, underlying_price, strike_spacing, num_strikes)
            
            if not levels:
                st.error("Failed to calculate levels")
                st.stop()
            
            # Display key levels
            st.markdown("## 🎯 Key Levels")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Volume Totals")
                st.metric("Total Call Volume", f"{levels['totals']['call_vol']:,}")
                st.metric("Total Put Volume", f"{levels['totals']['put_vol']:,}")
                net_vol = levels['totals']['net_vol']
                st.metric(
                    "Net Volume (Put - Call)",
                    f"{net_vol:,}",
                    delta="Bearish" if net_vol > 0 else "Bullish"
                )
            
            with col2:
                st.markdown("### 🧱 Volume Walls")
                if levels['call_wall']['strike']:
                    st.metric(
                        "📈 Call Wall (Resistance)",
                        f"${levels['call_wall']['strike']:.2f}",
                        delta=f"{levels['call_wall']['volume']:,} contracts"
                    )
                
                if levels['put_wall']['strike']:
                    st.metric(
                        "📉 Put Wall (Support)",
                        f"${levels['put_wall']['strike']:.2f}",
                        delta=f"{levels['put_wall']['volume']:,} contracts"
                    )
            
            with col3:
                st.markdown("### 💎 Net Walls (Key Levels)")
                if levels['net_call_wall']['strike']:
                    st.metric(
                        "💚 Net Call Wall",
                        f"${levels['net_call_wall']['strike']:.2f}",
                        delta=f"Bullish barrier"
                    )
                
                if levels['net_put_wall']['strike']:
                    st.metric(
                        "❤️ Net Put Wall",
                        f"${levels['net_put_wall']['strike']:.2f}",
                        delta=f"Bearish barrier"
                    )
                
                if levels['flip_level']:
                    st.metric(
                        "🔄 Flip Level",
                        f"${levels['flip_level']:.2f}",
                        delta="Sentiment pivot"
                    )
            
            # Get intraday data
            now = datetime.now()
            end_time = int(now.timestamp() * 1000)
            start_time = int((now - timedelta(hours=24)).timestamp() * 1000)
            
            price_history = client.get_price_history(
                symbol=symbol,
                frequency_type='minute',
                frequency=5,
                start_date=start_time,
                end_date=end_time,
                need_extended_hours=False
            )
            
            # Intraday chart with levels
            chart = create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Volume profile chart
            profile_chart = create_volume_profile_chart(levels)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
            
            # Interpretation
            st.markdown("---")
            
            with st.expander("💡 How to Read This", expanded=False):
                st.markdown("""
                ### 🧱 Volume Walls (Simple)
                - **Call Wall** 📈: Strike with highest call volume = **Resistance** (price ceiling)
                - **Put Wall** 📉: Strike with highest put volume = **Support** (price floor)
                
                ### 💎 Net Walls (Advanced)
                - **Net Call Wall** 💚: Strike with highest *net call dominance* = **Strong upside magnet**
                - **Net Put Wall** ❤️: Strike with highest *net put dominance* = **Strong downside magnet**
                
                ### 🔄 Flip Level (Critical)
                - Where net volume flips from **bearish → bullish** (or vice versa)
                - Above flip = bullish territory | Below flip = bearish territory
                - Breaking flip level often triggers momentum
                
                ### 📊 Trading Strategy
                1. **Breakout Play**: Price breaking through Call Wall with volume = bullish breakout
                2. **Bounce Play**: Price bouncing off Put Wall = support holding
                3. **Flip Trade**: Crossing flip level = sentiment change, momentum shift
                4. **Pin Risk**: Price gravitates toward max pain (highest volume strikes)
                
                ### ⚠️ Important Notes
                - These levels are **dynamic** - recalculate as volume changes
                - Most effective on **expiration day** when gamma is highest
                - Use with **price action** confirmation, not in isolation
                """)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())

else:
    with st.expander("🧱 What Are Option Volume Walls?", expanded=False):
        st.markdown("""
        Option volume walls are **key price levels** where massive option activity creates support or resistance.
        
        **Think of it like this:**
        - Market makers are **short options** to customers
        - They must **hedge** by buying/selling the underlying
        - **High volume strikes** = lots of hedging activity
        - This creates **price magnets** or **barriers**
        
        ### 📊 The Five Key Levels
        
        1. **Call Wall** 📈 - Highest call volume strike (typical resistance)
        2. **Put Wall** 📉 - Highest put volume strike (typical support)
        3. **Net Call Wall** 💚 - Where calls dominate most (strong bullish level)
        4. **Net Put Wall** ❤️ - Where puts dominate most (strong bearish level)
        5. **Flip Level** 🔄 - Where sentiment flips between bullish/bearish
        
        ### 🎯 Why This Matters
        
        - **Pin Risk**: Price often pins to high volume strikes at expiration
        - **Breakout Targets**: Breaking call wall = next leg up
        - **Support Levels**: Put walls act as price floors
        - **Sentiment Gauge**: Net volumes show true directional bias
        
        **Configure settings and click 'Calculate Levels' to start analyzing!**
        """)
