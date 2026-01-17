"""
Quick 0DTE Comparison - SPY vs QQQ vs $SPX
Side-by-side view of the three major indices with 0DTE expiry
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="0DTE",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state for auto-refresh
if 'auto_refresh_0dte' not in st.session_state:
    st.session_state.auto_refresh_0dte = True
if 'last_refresh_0dte' not in st.session_state:
    st.session_state.last_refresh_0dte = datetime.now()

# ===== COPY ESSENTIAL FUNCTIONS FROM OPTION VOLUME WALLS =====

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
        
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(hours=48)).timestamp() * 1000)
        
        price_history = client.get_price_history(
            symbol=query_symbol_quote,
            frequency_type='minute',
            frequency=5,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        return {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options,
            'price_history': price_history,
            'fetched_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

def calculate_option_walls(options_data, underlying_price, strike_spacing, num_strikes):
    """Calculate key levels based on option volume"""
    try:
        base_strike = np.floor(underlying_price / 10) * 10
        strikes_above = [base_strike + strike_spacing * i for i in range(num_strikes + 1)]
        strikes_below = [base_strike - strike_spacing * i for i in range(1, num_strikes + 1)]
        all_strikes = sorted(strikes_below + strikes_above)
        
        call_volumes = {}
        put_volumes = {}
        call_oi = {}
        put_oi = {}
        call_gamma = {}
        put_gamma = {}
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        vol = contract.get('totalVolume', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        gamma = contract.get('gamma', 0) or 0
                        
                        call_volumes[strike] = call_volumes.get(strike, 0) + vol
                        call_oi[strike] = call_oi.get(strike, 0) + oi
                        call_gamma[strike] = call_gamma.get(strike, 0) + gamma
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        vol = contract.get('totalVolume', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        gamma = contract.get('gamma', 0) or 0
                        
                        put_volumes[strike] = put_volumes.get(strike, 0) + vol
                        put_oi[strike] = put_oi.get(strike, 0) + oi
                        put_gamma[strike] = put_gamma.get(strike, 0) + gamma
        
        all_strikes_with_data = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
        all_strikes = all_strikes_with_data
        
        net_volumes = {}
        for strike in all_strikes:
            net_volumes[strike] = put_volumes.get(strike, 0) - call_volumes.get(strike, 0)
        
        gex_by_strike = {}
        for strike in all_strikes:
            call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01
            put_gex = -1 * put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01
            gex_by_strike[strike] = call_gex + put_gex
        
        call_wall_strike = max(call_volumes.items(), key=lambda x: x[1])[0] if call_volumes else None
        call_wall_volume = call_volumes.get(call_wall_strike, 0) if call_wall_strike else 0
        call_wall_oi = call_oi.get(call_wall_strike, 0) if call_wall_strike else 0
        call_wall_gex = gex_by_strike.get(call_wall_strike, 0) if call_wall_strike else 0
        
        put_wall_strike = max(put_volumes.items(), key=lambda x: x[1])[0] if put_volumes else None
        put_wall_volume = put_volumes.get(put_wall_strike, 0) if put_wall_strike else 0
        put_wall_oi = put_oi.get(put_wall_strike, 0) if put_wall_strike else 0
        put_wall_gex = gex_by_strike.get(put_wall_strike, 0) if put_wall_strike else 0
        
        bullish_strikes = {k: abs(v) for k, v in net_volumes.items() if v < 0}
        bearish_strikes = {k: abs(v) for k, v in net_volumes.items() if v > 0}
        
        net_call_wall_strike = max(bullish_strikes.items(), key=lambda x: x[1])[0] if bullish_strikes else None
        net_put_wall_strike = max(bearish_strikes.items(), key=lambda x: x[1])[0] if bearish_strikes else None
        
        strikes_near_price = sorted([s for s in all_strikes if abs(s - underlying_price) < strike_spacing * 5])
        flip_strike = None
        for i in range(len(strikes_near_price) - 1):
            curr_net = net_volumes.get(strikes_near_price[i], 0)
            next_net = net_volumes.get(strikes_near_price[i + 1], 0)
            if (curr_net > 0 and next_net < 0) or (curr_net < 0 and next_net > 0):
                flip_strike = strikes_near_price[i]
                break
        
        total_call_vol = sum(call_volumes.values())
        total_put_vol = sum(put_volumes.values())
        total_net_vol = total_put_vol - total_call_vol
        
        # Build strike_data DataFrame for hot strikes analysis
        strike_data_list = []
        for strike in all_strikes:
            strike_data_list.append({
                'strike': strike,
                'call_vol': call_volumes.get(strike, 0),
                'put_vol': put_volumes.get(strike, 0),
                'call_oi': call_oi.get(strike, 0),
                'put_oi': put_oi.get(strike, 0)
            })
        strike_data_df = pd.DataFrame(strike_data_list)
        
        return {
            'all_strikes': all_strikes,
            'call_volumes': call_volumes,
            'put_volumes': put_volumes,
            'net_volumes': net_volumes,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'gex_by_strike': gex_by_strike,
            'call_wall': {
                'strike': call_wall_strike,
                'volume': call_wall_volume,
                'oi': call_wall_oi,
                'gex': call_wall_gex
            },
            'put_wall': {
                'strike': put_wall_strike,
                'volume': put_wall_volume,
                'oi': put_wall_oi,
                'gex': put_wall_gex
            },
            'flip_level': flip_strike,
            'strike_data': strike_data_df,
            'totals': {
                'call_vol': total_call_vol,
                'put_vol': total_put_vol,
                'net_vol': total_net_vol,
                'total_gex': sum(gex_by_strike.values())
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating walls: {str(e)}")
        return None

def calculate_most_valuable_strike(options_data, underlying_price):
    """Calculate the most valuable strike based on GEX, proximity, and activity"""
    try:
        strikes_analysis = []
        
        if 'callExpDateMap' not in options_data or 'putExpDateMap' not in options_data:
            return None
        
        all_strikes = set()
        call_data = {}
        put_data = {}
        
        for exp_date, strikes in options_data['callExpDateMap'].items():
            for strike_str, contracts in strikes.items():
                strike = float(strike_str)
                all_strikes.add(strike)
                for contract in contracts:
                    if strike not in call_data:
                        call_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                    call_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                    call_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                    call_data[strike]['gamma'] += contract.get('gamma', 0) or 0
        
        for exp_date, strikes in options_data['putExpDateMap'].items():
            for strike_str, contracts in strikes.items():
                strike = float(strike_str)
                all_strikes.add(strike)
                for contract in contracts:
                    if strike not in put_data:
                        put_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                    put_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                    put_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                    put_data[strike]['gamma'] += contract.get('gamma', 0) or 0
        
        for strike in all_strikes:
            call = call_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0})
            put = put_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0})
            
            call_gex = call['gamma'] * call['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = put['gamma'] * put['oi'] * 100 * underlying_price * underlying_price * 0.01
            net_gex = abs(call_gex - put_gex)
            
            distance_pct = abs((strike - underlying_price) / underlying_price) * 100
            proximity_score = 1 / (1 + distance_pct)
            
            total_volume = call['volume'] + put['volume']
            total_oi = call['oi'] + put['oi']
            activity_score = (total_volume / 1000) + (total_oi / 10000)
            
            value_score = (net_gex / 1e6) * proximity_score * (1 + activity_score)
            
            strikes_analysis.append({
                'strike': strike,
                'value_score': value_score,
                'net_gex': net_gex,
                'distance_pct': distance_pct,
                'total_volume': total_volume,
                'total_oi': total_oi
            })
        
        if not strikes_analysis:
            return None
        
        strikes_df = pd.DataFrame(strikes_analysis).sort_values('value_score', ascending=False)
        most_valuable = strikes_df.iloc[0]
        
        return {
            'strike': most_valuable['strike'],
            'value_score': most_valuable['value_score'],
            'net_gex': most_valuable['net_gex'],
            'distance_pct': (most_valuable['strike'] - underlying_price) / underlying_price * 100,
            'total_volume': most_valuable['total_volume'],
            'total_oi': most_valuable['total_oi']
        }
        
    except Exception as e:
        logger.error(f"Error calculating most valuable strike: {str(e)}")
        return None

def create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol, most_valuable_strike=None):
    """Create intraday chart with key levels overlaid"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        df['date'] = df['datetime'].dt.date
        
        df = df[
            ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
            ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
        ].copy()
        
        if df.empty:
            return None
        
        unique_dates = sorted(df['date'].unique(), reverse=True)
        target_dates = unique_dates[:2] if len(unique_dates) >= 2 else unique_dates
        df = df[df['date'].isin(target_dates)].copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        fig = go.Figure()
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # VWAP line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#00bcd4', width=2)
        ))
        
        # 21 EMA
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ema21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='#ff9800', width=2)
        ))
        
        # Key levels with smart positioning to avoid overlaps
        level_data = []
        
        if levels['call_wall']['strike']:
            level_data.append({
                'price': levels['call_wall']['strike'],
                'label': f"Call Wall ${levels['call_wall']['strike']:.2f}",
                'color': "#f44336",
                'dash': "solid",
                'priority': 1
            })
        
        if levels['put_wall']['strike']:
            level_data.append({
                'price': levels['put_wall']['strike'],
                'label': f"Put Wall ${levels['put_wall']['strike']:.2f}",
                'color': "#4caf50",
                'dash': "solid",
                'priority': 1
            })
        
        if levels['flip_level']:
            level_data.append({
                'price': levels['flip_level'],
                'label': f"Flip ${levels['flip_level']:.2f}",
                'color': "#9c27b0",
                'dash': "dash",
                'priority': 2
            })
        
        if most_valuable_strike:
            level_data.append({
                'price': most_valuable_strike,
                'label': f"MVS ${most_valuable_strike:.2f}",
                'color': "#ff9800",
                'dash': "dot",
                'priority': 3
            })
        
        # Sort levels by price
        level_data.sort(key=lambda x: x['price'])
        
        # Smarter positioning algorithm
        min_spacing_pct = 0.005  # 0.5% minimum spacing
        position_cycle = ["right", "top right", "bottom right", "top left", "bottom left"]
        
        for i, level in enumerate(level_data):
            position = "right"
            
            # Check distance to all previous levels
            conflicts = 0
            for j in range(i):
                prev_level = level_data[j]
                price_diff_pct = abs(level['price'] - prev_level['price']) / underlying_price
                
                if price_diff_pct < min_spacing_pct:
                    conflicts += 1
            
            # Use cycling positions based on conflict count
            if conflicts > 0:
                position = position_cycle[min(conflicts, len(position_cycle) - 1)]
            
            # Add the horizontal line
            fig.add_hline(
                y=level['price'],
                line_dash=level['dash'],
                line_color=level['color'],
                line_width=2,
                annotation_text=level['label'],
                annotation_position=position,
                annotation=dict(font=dict(size=10))
            )
        
        fig.update_layout(
            title=dict(text=f"{symbol}", font=dict(size=14)),
            xaxis_title="Time (ET)",
            yaxis_title="Price ($)",
            height=500,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                type='date',
                tickformat='%I:%M %p\n%b %d',
                dtick=3600000,
                tickangle=0,
                rangebreaks=[
                    dict(bounds=[16, 9.5], pattern="hour"),  # Hide hours between 4 PM and 9:30 AM
                ],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=80, r=120, l=60, b=60)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

def get_gex_by_strike(options_data, underlying_price, expiry_date):
    """Extract GEX values by strike"""
    try:
        gamma_data = []
        
        if hasattr(expiry_date, 'strftime'):
            exp_date_str = expiry_date.strftime('%Y-%m-%d')
        else:
            exp_date_str = str(expiry_date)
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                if exp_date_str not in exp_date:
                    continue
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        notional_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        gamma_data.append({
                            'strike': strike,
                            'signed_notional_gamma': notional_gamma
                        })
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                if exp_date_str not in exp_date:
                    continue
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        gamma = contract.get('gamma', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        notional_gamma = -1 * gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        gamma_data.append({
                            'strike': strike,
                            'signed_notional_gamma': notional_gamma
                        })
        
        if not gamma_data:
            return None
        
        df_gamma = pd.DataFrame(gamma_data)
        strike_gex = df_gamma.groupby('strike')['signed_notional_gamma'].sum().reset_index()
        strike_gex.columns = ['strike', 'net_gex']
        
        min_strike = underlying_price * 0.90
        max_strike = underlying_price * 1.10
        strike_gex = strike_gex[(strike_gex['strike'] >= min_strike) & (strike_gex['strike'] <= max_strike)]
        
        if len(strike_gex) == 0:
            return None
        
        return strike_gex.sort_values('strike')
        
    except Exception as e:
        logger.error(f"Error calculating GEX by strike: {str(e)}")
        return None

# ===== MAIN PAGE =====

st.title("⚡ 0DTE")

# Auto-refresh controls at top
col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 2, 3])

with col_refresh1:
    # Fragment-based auto-refresh (non-blocking)
    @st.fragment(run_every="180s")
    def auto_refresh_fragment():
        st.cache_data.clear()
        st.session_state.last_refresh_0dte = datetime.now()
    
    if st.checkbox(
        "🔄 Auto-Refresh (3 min)",
        value=st.session_state.get('auto_refresh_0dte', True),
        key="auto_refresh_0dte_checkbox",
        help="Automatically refresh data every 3 minutes"
    ):
        auto_refresh_fragment()

with col_refresh2:
    if st.button("🔃 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh_0dte = datetime.now()
        st.rerun()

with col_refresh3:
    st.caption(f"Last updated: {st.session_state.last_refresh_0dte.strftime('%I:%M:%S %p')}")

st.markdown("---")

# Get default date for 0DTE
today = datetime.now().date()
weekday = today.weekday()

if weekday == 5:
    default_expiry = today + timedelta(days=2)
elif weekday == 6:
    default_expiry = today + timedelta(days=1)
else:
    default_expiry = today

# Get default date for 0DTE
today = datetime.now().date()
weekday = today.weekday()

if weekday == 5:
    default_expiry = today + timedelta(days=2)
elif weekday == 6:
    default_expiry = today + timedelta(days=1)
else:
    default_expiry = today

# Calculate time to expiry
now = datetime.now()
expiry_datetime = datetime.combine(default_expiry, datetime.strptime("16:00", "%H:%M").time())
time_to_expiry = expiry_datetime - now
hours_remaining = int(time_to_expiry.total_seconds() / 3600)
minutes_remaining = int((time_to_expiry.total_seconds() % 3600) / 60)
time_status = "🔴 EXPIRED" if time_to_expiry.total_seconds() < 0 else f"⏰ {hours_remaining}h {minutes_remaining}m"
time_color = "#ef5350" if time_to_expiry.total_seconds() < 0 else "#ff9800"

header_col1, header_col2, header_col3 = st.columns([3, 1.5, 1])
with header_col1:
    # Show last refresh time and time to expiry
    refresh_time = st.session_state.last_refresh_0dte.strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <span style="font-weight: 600;">SPY • QQQ • $SPX expiration comparison</span>
        <span style="font-size: 12px; color: #666;">Last refresh: {refresh_time}</span>
        <span style="background: {time_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
            {time_status}
        </span>
    </div>
    """, unsafe_allow_html=True)
with header_col2:
    expiry_date = st.date_input(
        "Expiry Date",
        value=default_expiry,
        min_value=today,
        max_value=today + timedelta(days=365),
        label_visibility="collapsed"
    )
with header_col3:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.session_state.last_refresh_0dte = datetime.now()
        st.cache_data.clear()
        st.rerun()

# Fixed settings
strike_spacing = 5.0
num_strikes = 15
chart_height = 500

st.markdown("---")

# Symbols to compare
symbols = ['SPY', 'QQQ', '$SPX']

# Store results for divergence analysis
results = {}

# Create 3 columns
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

for idx, symbol in enumerate(symbols):
    with columns[idx]:
        st.markdown(f"### {symbol}")
        
        with st.spinner(f"Loading {symbol}..."):
            try:
                exp_date_str = expiry_date.strftime('%Y-%m-%d')
                snapshot = get_market_snapshot(symbol, exp_date_str)
                
                if not snapshot:
                    st.error(f"Failed to fetch {symbol} data")
                    continue
                
                underlying_price = snapshot['underlying_price']
                options = snapshot['options_chain']
                price_history = snapshot['price_history']
                quote_data = snapshot['quote']
                
                # Calculate daily change
                prev_close = quote_data.get(symbol.replace('$', ''), {}).get('quote', {}).get('closePrice', underlying_price)
                daily_change = underlying_price - prev_close
                daily_change_pct = (daily_change / prev_close * 100) if prev_close else 0
                change_color = "#26a69a" if daily_change >= 0 else "#ef5350"
                change_arrow = "▲" if daily_change >= 0 else "▼"
                
                # Compact price + sentiment in single line
                net_vol_temp = 0
                levels = calculate_option_walls(options, underlying_price, strike_spacing, num_strikes)
                if levels:
                    net_vol_temp = levels['totals']['net_vol']
                
                sentiment_icon = "🐻" if net_vol_temp > 0 else "🐂"
                sentiment_color = "#ef5350" if net_vol_temp > 0 else "#26a69a"
                
                # Calculate Put/Call Ratio
                total_call_vol = levels['totals']['call_vol']
                total_put_vol = levels['totals']['put_vol']
                pc_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else 0
                
                # Store results for divergence analysis
                results[symbol] = {
                    'price': underlying_price,
                    'daily_change_pct': daily_change_pct,
                    'flip_level': levels.get('flip_level'),
                    'above_flip': underlying_price > levels.get('flip_level', underlying_price) if levels.get('flip_level') else None,
                    'pc_ratio': pc_ratio,
                    'net_vol': net_vol_temp,
                    'call_wall': levels.get('call_wall', {}).get('strike'),
                    'put_wall': levels.get('put_wall', {}).get('strike')
                }
                
                # Determine bullish/bearish based on flip level
                flip_bias = ""
                flip_bias_color = ""
                distance_to_flip = 0
                if levels and levels.get('flip_level'):
                    distance_to_flip = ((underlying_price - levels['flip_level']) / underlying_price * 100)
                    if underlying_price > levels['flip_level']:
                        flip_bias = "BULLISH 🐂"
                        flip_bias_color = "#26a69a"
                    else:
                        flip_bias = "BEARISH 🐻"
                        flip_bias_color = "#ef5350"
                
                # Display ticker with bias label and enhanced metrics
                if flip_bias:
                    alert_badge = ""
                    # Check proximity to key levels (within 0.5%)
                    if levels.get('call_wall', {}).get('strike'):
                        dist_to_call_wall = abs((underlying_price - levels['call_wall']['strike']) / underlying_price * 100)
                        if dist_to_call_wall < 0.5:
                            alert_badge = " 🔴 AT CALL WALL"
                    if levels.get('put_wall', {}).get('strike'):
                        dist_to_put_wall = abs((underlying_price - levels['put_wall']['strike']) / underlying_price * 100)
                        if dist_to_put_wall < 0.5:
                            alert_badge = " 🟢 AT PUT WALL"
                    if abs(distance_to_flip) < 0.3:
                        alert_badge = " ⚡ NEAR FLIP"
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 4px;">
                        <span style="background: {flip_bias_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 900;">
                            {flip_bias}{alert_badge}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 6px; margin-bottom: 8px;">
                    <div style="color: white;">
                        <div style="font-size: 11px; opacity: 0.9;">CURRENT</div>
                        <div style="font-size: 22px; font-weight: 900;">${underlying_price:.2f}</div>
                        <div style="font-size: 11px; color: {change_color}; margin-top: 2px;">
                            {change_arrow} ${abs(daily_change):.2f} ({daily_change_pct:+.2f}%)
                        </div>
                    </div>
                    <div style="text-align: right; color: white;">
                        <div style="font-size: 11px; opacity: 0.9;">FLOW</div>
                        <div style="font-size: 18px; font-weight: 900;">{sentiment_icon} {abs(net_vol_temp):,.0f}</div>
                        <div style="font-size: 11px; margin-top: 2px;">
                            P/C: {pc_ratio:.2f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hot Strikes Section
                if levels and 'strike_data' in levels:
                    df = levels['strike_data']
                    # Filter to strikes within ±5% of current price
                    df_near = df[abs((df['strike'] - underlying_price) / underlying_price) <= 0.05].copy()
                    df_near['total_vol'] = df_near['call_vol'] + df_near['put_vol']
                    top_strikes = df_near.nlargest(3, 'total_vol')
                    
                    if len(top_strikes) > 0:
                        hot_strikes_items = []
                        for _, row in top_strikes.iterrows():
                            strike = row['strike']
                            total_vol = row['total_vol']
                            call_vol = row['call_vol']
                            put_vol = row['put_vol']
                            
                            # Determine if calls or puts dominate
                            dominant = "C" if call_vol > put_vol else "P"
                            vol_color = "#22c55e" if dominant == "C" else "#ef4444"
                            
                            # Distance from current price
                            dist_pct = ((strike - underlying_price) / underlying_price) * 100
                            dist_str = f"{dist_pct:+.1f}%"
                            
                            rgba_color = '34, 197, 94' if dominant == 'C' else '239, 68, 68'
                            hot_strikes_items.append(
                                f'<div style="background: rgba(255,255,255,0.08); padding: 5px 8px; border-radius: 4px; margin-bottom: 4px; border: 1px solid rgba(255,255,255,0.1);">'
                                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">'
                                f'<span style="font-weight: 700; font-size: 13px;">${strike:.0f}</span>'
                                f'<span style="font-size: 9px; font-weight: 700; color: {vol_color}; background: rgba({rgba_color}, 0.2); padding: 2px 5px; border-radius: 3px;">{dominant}</span>'
                                f'</div>'
                                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                                f'<span style="font-size: 8px; opacity: 0.6;">{dist_str}</span>'
                                f'<span style="font-size: 9px; opacity: 0.75; font-weight: 600;">{total_vol:,.0f} vol</span>'
                                f'</div>'
                                f'</div>'
                            )
                        
                        if hot_strikes_items:
                            hot_strikes_html = (
                                '<div style="background: rgba(102, 126, 234, 0.1); padding: 8px; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(102, 126, 234, 0.3);">'
                                '<div style="font-size: 9px; opacity: 0.8; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #667eea;">🔥 Hot Strikes (Top 3)</div>'
                                + ''.join(hot_strikes_items) +
                                '</div>'
                            )
                            st.markdown(hot_strikes_html, unsafe_allow_html=True)
                
                if not levels:
                    st.error(f"Failed to calculate levels for {symbol}")
                    continue
                
                mvs = calculate_most_valuable_strike(options, underlying_price)
                
                # Get GEX for display
                strike_gex = get_gex_by_strike(options, underlying_price, expiry_date)
                
                # Compact GEX summary below price card
                if strike_gex is not None and len(strike_gex) > 0:
                    # Find key GEX levels
                    top_positive = strike_gex[strike_gex['net_gex'] > 0].nlargest(3, 'net_gex')
                    top_negative = strike_gex[strike_gex['net_gex'] < 0].nsmallest(3, 'net_gex')
                    
                    gex_summary = []
                    for _, row in top_positive.iterrows():
                        gex_summary.append(f"🔵 ${row['strike']:.0f}: ${row['net_gex']/1e6:.1f}M")
                    for _, row in top_negative.iterrows():
                        gex_summary.append(f"🔴 ${row['strike']:.0f}: ${row['net_gex']/1e6:.1f}M")
                    
                    if gex_summary:
                        st.markdown(f"""
                        <div style="padding: 6px; background: #f5f5f5; border-radius: 4px; font-size: 9px; margin-bottom: 8px;">
                            <strong>Top GEX:</strong> {' • '.join(gex_summary[:6])}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create chart
                mvs_strike = mvs['strike'] if mvs else None
                chart = create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol, mvs_strike)
                
                if chart:
                    chart.update_layout(height=chart_height)
                    st.plotly_chart(chart, use_container_width=True, key=f"chart_{symbol}")
                else:
                    st.error("Failed to create chart")
                
            except Exception as e:
                st.error(f"Error loading {symbol}: {str(e)}")
                logger.error(f"Error in {symbol}: {str(e)}")

st.markdown("---")

# Divergence Detection Section
if len(results) >= 2:
    st.markdown("### 🔍 Divergence Analysis")
    
    divergences = []
    
    # Check sentiment divergence (flip level positioning)
    sentiments = {sym: data['above_flip'] for sym, data in results.items() if data['above_flip'] is not None}
    if len(set(sentiments.values())) > 1:
        bullish = [sym for sym, is_bull in sentiments.items() if is_bull]
        bearish = [sym for sym, is_bull in sentiments.items() if not is_bull]
        divergences.append(f"**Sentiment Split:** {', '.join(bullish)} bullish • {', '.join(bearish)} bearish")
    
    # Check P/C ratio divergence
    pc_ratios = {sym: data['pc_ratio'] for sym, data in results.items()}
    if pc_ratios:
        max_pc = max(pc_ratios.items(), key=lambda x: x[1])
        min_pc = min(pc_ratios.items(), key=lambda x: x[1])
        if max_pc[1] / min_pc[1] > 1.5:  # 50% difference
            divergences.append(f"**P/C Ratio Divergence:** {max_pc[0]} ({max_pc[1]:.2f}) vs {min_pc[0]} ({min_pc[1]:.2f})")
    
    # Check daily performance divergence
    performances = {sym: data['daily_change_pct'] for sym, data in results.items()}
    if len(performances) >= 3:
        sorted_perf = sorted(performances.items(), key=lambda x: x[1])
        if sorted_perf[-1][1] - sorted_perf[0][1] > 1.0:  # >1% spread
            divergences.append(f"**Performance Spread:** {sorted_perf[-1][0]} ({sorted_perf[-1][1]:+.2f}%) leading, {sorted_perf[0][0]} ({sorted_perf[0][1]:+.2f}%) lagging")
    
    if divergences:
        for div in divergences:
            st.markdown(f"""
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0; border-radius: 4px;">
                ⚠️ {div}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #d4edda; border-left: 4px solid #28a745; padding: 10px; border-radius: 4px;">
            ✅ No significant divergences detected - indices moving in sync
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

with st.expander("💡 How to Use", expanded=False):
    st.markdown("""
    ### 0DTE Analysis
    
    Side-by-side comparison of SPY, QQQ, and $SPX with same-day expiration.
    
    - **Call Wall** (🔴): Resistance level marked on chart
    - **Put Wall** (🟢): Support level marked on chart
    - **Flip Level** (🔄): Sentiment pivot marked on chart
    - **MVS**: Most Valuable Strike (orange dotted line)
    - **Top GEX**: Key gamma exposure levels (🔵 support, 🔴 acceleration)
    - **🔄 Auto-Refresh**: Enable streaming mode (refreshes every 3 minutes)
    
    Compare all three to spot divergences and opportunities!
    """)
