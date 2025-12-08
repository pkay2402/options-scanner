"""
Trading Hub - Unified Command Center
Combines chart with volume walls, live watchlist, and whale flows in a single view
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
import re
from html import unescape

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.utils.droplet_api import DropletAPI, fetch_watchlist, fetch_whale_flows

# Import GEX heatmap function from Stock Option Finder
import importlib.util
spec = importlib.util.spec_from_file_location("stock_option_finder", str(Path(__file__).parent / "3_Stock_Option_Finder.py"))
stock_option_finder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stock_option_finder)
create_professional_netgex_heatmap = stock_option_finder.create_professional_netgex_heatmap
calculate_gamma_strikes = stock_option_finder.calculate_gamma_strikes

st.set_page_config(
    page_title="Trading Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact, professional layout
st.markdown("""
<style>
    /* Remove padding for max space utilization */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Compact cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Watchlist styling */
    .watchlist-item {
        background: white;
        border-radius: 6px;
        padding: 8px;
        margin-bottom: 6px;
        border-left: 3px solid;
        font-size: 12px;
    }
    
    .watchlist-item.bullish {
        border-left-color: #22c55e;
    }
    
    .watchlist-item.bearish {
        border-left-color: #ef4444;
    }
    
    /* Whale flow cards */
    .whale-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    
    .whale-card.call {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    }
    
    .whale-card.put {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Compact table styling */
    .dataframe {
        font-size: 11px !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 14px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 4px;
    }
    
    /* Timeframe toggle */
    .timeframe-btn {
        display: inline-block;
        padding: 6px 16px;
        margin: 2px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_hub_symbol' not in st.session_state:
    st.session_state.trading_hub_symbol = 'SPY'
if 'trading_hub_timeframe' not in st.session_state:
    st.session_state.trading_hub_timeframe = 'intraday'  # 'intraday' or 'daily'
if 'trading_hub_expiry' not in st.session_state:
    st.session_state.trading_hub_expiry = None

def get_default_expiry(symbol):
    """Get default expiry: 0DTE for indices, next Friday for stocks"""
    today = datetime.now().date()
    weekday = today.weekday()
    
    if symbol in ['SPY', '$SPX', 'QQQ']:
        # 0DTE for indices (unless weekend)
        if weekday == 5:  # Saturday
            return today + timedelta(days=2)
        elif weekday == 6:  # Sunday
            return today + timedelta(days=1)
        return today
    else:
        # Next Friday for stocks
        days_to_friday = (4 - weekday) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        return today + timedelta(days=days_to_friday)

def get_next_friday():
    """Get next Friday for weekly options"""
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def get_previous_trading_day(reference_date=None):
    """Get the previous trading day (skips weekends)"""
    if reference_date is None:
        reference_date = datetime.now().date()
    elif isinstance(reference_date, datetime):
        reference_date = reference_date.date()
    
    # Go back one day
    prev_day = reference_date - timedelta(days=1)
    
    # Skip weekends
    while prev_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        prev_day = prev_day - timedelta(days=1)
    
    return prev_day

def get_next_n_fridays(n=4):
    """Get next N Fridays for multiple weekly expiries"""
    fridays = []
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    
    for i in range(n):
        friday = today + timedelta(days=days_ahead + (i * 7))
        fridays.append(friday)
    
    return fridays

@st.cache_data(ttl=300, show_spinner=False)
def get_market_snapshot(symbol: str, expiry_date: str, timeframe: str = 'intraday'):
    """Fetch complete market data with price history based on timeframe"""
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        # Handle symbols with $ prefix (like $SPX) - keep as-is for API
        query_symbol_quote = symbol
        query_symbol_options = symbol
        
        logger.info(f"Fetching market snapshot for {symbol}, expiry: {expiry_date}, timeframe: {timeframe}")
        
        # Get quote
        quote = client.get_quote(query_symbol_quote)
        if not quote:
            logger.error(f"Failed to get quote for {symbol}")
            return None
        
        underlying_price = quote.get(query_symbol_quote, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            logger.error(f"No underlying price found in quote for {symbol}")
            return None
        
        logger.info(f"Got underlying price for {symbol}: ${underlying_price}")
        
        # Get options chain
        chain_params = {
            'symbol': query_symbol_options,
            'contract_type': 'ALL',
            'from_date': expiry_date,
            'to_date': expiry_date
        }
        
        if symbol in ['$SPX', 'DJX', 'NDX', 'RUT']:
            chain_params['strike_count'] = 50
        
        logger.info(f"Fetching options chain with params: {chain_params}")
        options = client.get_options_chain(**chain_params)
        
        if not options or 'callExpDateMap' not in options:
            logger.error(f"No options chain data for {symbol}")
            return None
        
        logger.info(f"Got options chain for {symbol}")
        
        # Get price history based on timeframe
        now = datetime.now()
        
        if timeframe == 'intraday':
            # Get 5 days of 5-minute data to ensure we have previous trading day
            # This covers weekends and most holiday scenarios
            end_time = int(now.timestamp() * 1000)
            start_time = int((now - timedelta(days=5)).timestamp() * 1000)
            
            price_history = client.get_price_history(
                symbol=query_symbol_quote,
                frequency_type='minute',
                frequency=5,
                start_date=start_time,
                end_date=end_time,
                need_extended_hours=False
            )
        else:
            # 30 days of daily data
            try:
                import yfinance as yf
                # For yfinance, use symbol without $ prefix
                yf_symbol = symbol.replace('$', '^') if symbol.startswith('$') else symbol
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period="1mo", interval="1d")
                
                if not hist.empty:
                    candles = []
                    for idx, row in hist.iterrows():
                        candles.append({
                            'datetime': int(idx.timestamp() * 1000),
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        })
                    price_history = {'candles': candles}
                else:
                    price_history = None
            except Exception as e:
                logger.error(f"Error fetching daily history: {e}")
                price_history = None
        
        return {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options,
            'price_history': price_history,
            'fetched_at': datetime.now(),
            'timeframe': timeframe
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}", exc_info=True)
        return None

def calculate_option_levels(options_data, underlying_price):
    """Calculate key option levels: walls, flip level, max GEX"""
    try:
        call_data = {}
        put_data = {}
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in call_data:
                            call_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0}
                        call_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        call_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        call_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        call_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in put_data:
                            put_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0}
                        put_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        put_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        put_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        put_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        # Calculate metrics by strike
        all_strikes = sorted(set(call_data.keys()) | set(put_data.keys()))
        strike_analysis = []
        
        for strike in all_strikes:
            call = call_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0})
            put = put_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0})
            
            # GEX calculation
            call_gex = call['gamma'] * call['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = -put['gamma'] * put['oi'] * 100 * underlying_price * underlying_price * 0.01
            net_gex = call_gex + put_gex
            
            # Net volume
            net_volume = put['volume'] - call['volume']
            
            # Distance from current price
            distance_pct = abs(strike - underlying_price) / underlying_price * 100
            
            strike_analysis.append({
                'strike': strike,
                'call_vol': call['volume'],
                'put_vol': put['volume'],
                'net_vol': net_volume,
                'call_oi': call['oi'],
                'put_oi': put['oi'],
                'call_gex': call_gex,
                'put_gex': put_gex,
                'net_gex': net_gex,
                'call_premium': call['premium'],
                'put_premium': put['premium'],
                'distance_pct': distance_pct
            })
        
        df = pd.DataFrame(strike_analysis)
        
        # Find key levels
        call_wall = df.loc[df['call_vol'].idxmax()] if len(df) > 0 else None
        put_wall = df.loc[df['put_vol'].idxmax()] if len(df) > 0 else None
        max_gex = df.loc[df['net_gex'].abs().idxmax()] if len(df) > 0 else None
        
        # Find flip level (where net volume crosses zero)
        nearby = df[df['distance_pct'] < 2.0].sort_values('strike')
        flip_level = None
        for i in range(len(nearby) - 1):
            if (nearby.iloc[i]['net_vol'] > 0 and nearby.iloc[i+1]['net_vol'] < 0) or \
               (nearby.iloc[i]['net_vol'] < 0 and nearby.iloc[i+1]['net_vol'] > 0):
                flip_level = nearby.iloc[i]['strike']
                break
        
        # Calculate P/C ratio
        total_call_vol = df['call_vol'].sum()
        total_put_vol = df['put_vol'].sum()
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        return {
            'call_wall': call_wall,
            'put_wall': put_wall,
            'max_gex': max_gex,
            'flip_level': flip_level,
            'pc_ratio': pc_ratio,
            'total_call_vol': total_call_vol,
            'total_put_vol': total_put_vol,
            'strike_data': df
        }
        
    except Exception as e:
        logger.error(f"Error calculating levels: {e}")
        return None

def create_trading_chart(price_history, levels, underlying_price, symbol, timeframe='intraday'):
    """Create comprehensive trading chart with all levels"""
    try:
        if not price_history or 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        if timeframe == 'intraday':
            # Filter to market hours
            df['date'] = df['datetime'].dt.date
            df = df[
                (
                    ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
                    ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
                )
            ].copy()
            
            # Get last 2 trading days
            unique_dates = sorted(df['date'].unique(), reverse=True)
            if len(unique_dates) >= 2:
                target_dates = unique_dates[:2]
                df = df[df['date'].isin(target_dates)].copy()
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        if df.empty:
            return None
        
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ))
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#00bcd4', width=2.5),
            hovertemplate='<b>VWAP</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # 21 EMA
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ema21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='#ff9800', width=2),
            hovertemplate='<b>21 EMA</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # MACD crossovers
        try:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            df['macd'] = macd
            df['signal'] = signal
            df['macd_prev'] = df['macd'].shift(1)
            df['signal_prev'] = df['signal'].shift(1)
            
            up_cross = (df['macd'] > df['signal']) & (df['macd_prev'] <= df['signal_prev'])
            down_cross = (df['macd'] < df['signal']) & (df['macd_prev'] >= df['signal_prev'])
            
            if up_cross.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[up_cross, 'datetime'],
                    y=df.loc[up_cross, 'low'] * 0.997,
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='#22c55e', size=12),
                    name='Bull Cross',
                    showlegend=False
                ))
            
            if down_cross.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[down_cross, 'datetime'],
                    y=df.loc[down_cross, 'high'] * 1.003,
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='#ef4444', size=12),
                    name='Bear Cross',
                    showlegend=False
                ))
        except:
            pass
        
        # Add option levels with overlap detection
        if levels:
            # Collect all levels with their prices
            level_annotations = []
            
            if levels['call_wall'] is not None and not levels['call_wall'].empty:
                call_strike = levels['call_wall']['strike']
                level_annotations.append({
                    'price': call_strike,
                    'label': f"Call Wall ${call_strike:.2f}",
                    'color': '#22c55e',
                    'dash': 'dot',
                    'width': 3
                })
            
            if levels['put_wall'] is not None and not levels['put_wall'].empty:
                put_strike = levels['put_wall']['strike']
                level_annotations.append({
                    'price': put_strike,
                    'label': f"Put Wall ${put_strike:.2f}",
                    'color': '#ef4444',
                    'dash': 'dot',
                    'width': 3
                })
            
            if levels['flip_level']:
                level_annotations.append({
                    'price': levels['flip_level'],
                    'label': f"Flip ${levels['flip_level']:.2f}",
                    'color': '#a855f7',
                    'dash': 'solid',
                    'width': 3.5
                })
            
            if levels['max_gex'] is not None and not levels['max_gex'].empty:
                gex_strike = levels['max_gex']['strike']
                level_annotations.append({
                    'price': gex_strike,
                    'label': f"Max GEX ${gex_strike:.2f}",
                    'color': '#9c27b0',
                    'dash': 'dashdot',
                    'width': 3
                })
            
            # Detect overlaps and adjust y-positions
            # Group by similar prices (within 0.5% threshold)
            if level_annotations:
                sorted_levels = sorted(level_annotations, key=lambda x: x['price'])
                overlap_threshold = sorted_levels[0]['price'] * 0.005  # 0.5% threshold
                
                y_offsets = {}
                for i, level in enumerate(sorted_levels):
                    # Check if this level overlaps with previous ones
                    offset = 0
                    for j in range(i):
                        prev_level = sorted_levels[j]
                        if abs(level['price'] - prev_level['price']) < overlap_threshold:
                            offset += 1
                    y_offsets[i] = offset
                
                # Add horizontal lines with adjusted annotation positions
                for i, level in enumerate(sorted_levels):
                    fig.add_hline(
                        y=level['price'],
                        line_dash=level['dash'],
                        line_color=level['color'],
                        line_width=level['width'],
                        annotation_text=level['label'],
                        annotation_position="right",
                        annotation=dict(
                            yshift=y_offsets[i] * 20  # Stack overlapping labels vertically
                        )
                    )
        
        # Add Previous Day High/Low
        try:
            if timeframe == 'intraday' and len(df) > 0:
                df['date'] = df['datetime'].dt.date
                
                # Get today's date and calculate previous trading day
                today = datetime.now().date()
                prev_trading_day = get_previous_trading_day(today)
                
                # Get data for the previous trading day
                prev_day_data = df[df['date'] == prev_trading_day]
                
                if not prev_day_data.empty:
                    prev_high = prev_day_data['high'].max()
                    prev_low = prev_day_data['low'].min()
                    
                    # Previous Day High
                    fig.add_hline(
                        y=prev_high,
                        line_dash="dash",
                        line_color="#8b5cf6",
                        line_width=2,
                        annotation_text=f"PDH ${prev_high:.2f}",
                        annotation_position="left",
                        annotation=dict(font=dict(size=10, color="#8b5cf6"))
                    )
                    
                    # Previous Day Low
                    fig.add_hline(
                        y=prev_low,
                        line_dash="dash",
                        line_color="#8b5cf6",
                        line_width=2,
                        annotation_text=f"PDL ${prev_low:.2f}",
                        annotation_position="left",
                        annotation=dict(font=dict(size=10, color="#8b5cf6"))
                    )
        except Exception as e:
            logger.error(f"Error adding prev day levels: {e}")
        
        # Add Opening Range High/Low (first 30 minutes of current day)
        try:
            if timeframe == 'intraday' and len(df) > 0:
                df['date'] = df['datetime'].dt.date
                df['time'] = df['datetime'].dt.time
                
                # Get current day (last day in data)
                current_day = df['date'].max()
                current_day_data = df[df['date'] == current_day].copy()
                
                if not current_day_data.empty:
                    # Market open is typically 9:30 AM
                    # First 30 minutes = 9:30 AM to 10:00 AM
                    from datetime import time as dt_time
                    market_open = dt_time(9, 30)
                    or_end = dt_time(10, 0)
                    
                    # Filter for opening range (first 30 minutes)
                    or_data = current_day_data[
                        (current_day_data['time'] >= market_open) & 
                        (current_day_data['time'] < or_end)
                    ]
                    
                    if not or_data.empty:
                        or_high = or_data['high'].max()
                        or_low = or_data['low'].min()
                        
                        # Opening Range High (ORH)
                        fig.add_hline(
                            y=or_high,
                            line_dash="dot",
                            line_color="#3b82f6",
                            line_width=2,
                            annotation_text=f"ORH ${or_high:.2f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=10, color="#3b82f6"))
                        )
                        
                        # Opening Range Low (ORL)
                        fig.add_hline(
                            y=or_low,
                            line_dash="dot",
                            line_color="#3b82f6",
                            line_width=2,
                            annotation_text=f"ORL ${or_low:.2f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=10, color="#3b82f6"))
                        )
        except Exception as e:
            logger.error(f"Error adding opening range levels: {e}")
        
        # Layout with rangebreaks to remove gaps
        chart_title = f"{symbol} - {timeframe.title()} Chart"
        
        # Calculate intelligent Y-axis range
        all_prices = [df['high'].max(), df['low'].min(), underlying_price]
        if levels:
            if levels['call_wall'] is not None and not levels['call_wall'].empty:
                all_prices.append(levels['call_wall']['strike'])
            if levels['put_wall'] is not None and not levels['put_wall'].empty:
                all_prices.append(levels['put_wall']['strike'])
            if levels['flip_level']:
                all_prices.append(levels['flip_level'])
            if levels['max_gex'] is not None and not levels['max_gex'].empty:
                all_prices.append(levels['max_gex']['strike'])
        
        price_range = max(all_prices) - min(all_prices)
        y_padding = price_range * 0.12
        y_min = min(all_prices) - y_padding
        y_max = max(all_prices) + y_padding
        
        # Extend x-axis for breathing room
        time_range = df['datetime'].max() - df['datetime'].min()
        x_axis_end = df['datetime'].max() + time_range * 0.08
        
        # Configure xaxis based on timeframe
        if timeframe == 'intraday':
            xaxis_config = dict(
                type='date',
                tickformat='%I:%M %p\n%b %d',
                dtick=3600000,  # Tick every hour
                tickangle=0,
                range=[df['datetime'].min(), x_axis_end],
                rangebreaks=[
                    dict(bounds=[16, 9.5], pattern="hour"),  # Hide after-hours
                ],
                gridcolor='rgba(0,0,0,0.05)'
            )
            xaxis_title = "Time (ET)"
        else:
            xaxis_config = dict(
                type='date',
                tickformat='%b %d',
                tickangle=0,
                range=[df['datetime'].min(), x_axis_end],
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # Hide weekends
                ],
                gridcolor='rgba(0,0,0,0.05)'
            )
            xaxis_title = "Date"
        
        fig.update_layout(
            title=dict(
                text=chart_title, 
                x=0.5, 
                xanchor='center', 
                font=dict(size=18, color='#1f2937')
            ),
            xaxis_title=xaxis_title,
            yaxis_title="Price ($)",
            height=600,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,  # Hide rangeslider to prevent MACD subplot
            showlegend=True,
            xaxis=xaxis_config,
            yaxis=dict(
                range=[y_min, y_max],
                tickformat='$.2f',
                gridcolor='rgba(0,0,0,0.08)',
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(t=80, r=120, l=80, b=80),
            plot_bgcolor='rgba(250, 250, 250, 0.5)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return None

def create_net_gex_chart(levels, underlying_price, symbol):
    """Create Net GEX bar chart showing gamma exposure by strike"""
    try:
        if not levels or 'strike_data' not in levels:
            return None
        
        df = levels['strike_data']
        
        # Filter to ±10% range
        min_strike = underlying_price * 0.90
        max_strike = underlying_price * 1.10
        df_filtered = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)].copy()
        
        if df_filtered.empty:
            return None
        
        # Sort by strike and take top activity
        df_filtered = df_filtered.sort_values('net_gex', key=abs, ascending=False).head(12)
        df_filtered = df_filtered.sort_values('strike')
        
        # Create figure
        fig = go.Figure()
        
        # Color based on GEX sign
        colors = ['#22c55e' if v > 0 else '#ef4444' for v in df_filtered['net_gex']]
        
        fig.add_trace(go.Bar(
            x=df_filtered['net_gex'],
            y=[f"${s:.2f}" for s in df_filtered['strike']],
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.2)', width=1)),
            text=[f"${abs(v)/1e6:.1f}M" if abs(v) >= 1e6 else f"${abs(v)/1e3:.0f}K" for v in df_filtered['net_gex']],
            textposition='auto',
            textfont=dict(size=9),
            hovertemplate='<b>Strike: %{y}</b><br>Net GEX: $%{x:,.0f}<extra></extra>',
            showlegend=False
        ))
        
        # Add zero line
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
        
        # Mark current price
        closest_strike = min(df_filtered['strike'], key=lambda x: abs(x - underlying_price))
        
        fig.update_layout(
            title=dict(text=f"Net GEX ($)<br><sub>{symbol}</sub>", font=dict(size=14), y=0.98),
            xaxis_title="Net GEX ($)",
            yaxis_title="Strike ($)",
            height=600,
            template='plotly_white',
            xaxis=dict(tickformat='$,.0s', gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=60, r=40, t=80, b=60),
            font=dict(size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating GEX chart: {e}")
        return None

def create_volume_profile_chart(levels, underlying_price, symbol):
    """Create horizontal volume profile showing net volumes by strike"""
    try:
        if not levels or 'strike_data' not in levels:
            return None
        
        df = levels['strike_data']
        
        # Filter to ±10% range
        min_strike = underlying_price * 0.90
        max_strike = underlying_price * 1.10
        df_filtered = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)].copy()
        
        if df_filtered.empty:
            return None
        
        # Sort and limit
        df_filtered = df_filtered.sort_values('net_vol', key=abs, ascending=False).head(12)
        df_filtered = df_filtered.sort_values('strike')
        
        # Create figure
        fig = go.Figure()
        
        # Colors: red for put-heavy, green for call-heavy
        colors = ['#ef5350' if v > 0 else '#26a69a' for v in df_filtered['net_vol']]
        
        fig.add_trace(go.Bar(
            x=df_filtered['net_vol'],
            y=[f"${s:.2f}" for s in df_filtered['strike']],
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.1)', width=1)),
            text=[f"{abs(v):,.0f}" for v in df_filtered['net_vol']],
            textposition='auto',
            textfont=dict(size=9),
            hovertemplate='<b>Strike: %{y}</b><br>Net Volume: %{x:,.0f}<extra></extra>',
            showlegend=False
        ))
        
        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
        
        fig.update_layout(
            title=dict(text=f"Net Volume Profile<br><sub>{symbol}</sub>", font=dict(size=14), y=0.98),
            xaxis_title="Net Volume (Put - Call)",
            yaxis_title="Strike ($)",
            height=600,
            template='plotly_white',
            xaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=True),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=60, r=40, t=80, b=60),
            font=dict(size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating volume profile: {e}")
        return None

def create_net_premium_heatmap(options_data, underlying_price, num_expiries=4):
    """Create Net Premium heatmap showing call premium - put premium across strikes and expirations"""
    try:
        # Create a dictionary to aggregate premiums by strike and expiry
        premium_matrix = {}  # {(strike, expiry): {'call': X, 'put': Y}}
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        mark_price = contract.get('mark', 0)
                        
                        # Calculate notional premium
                        notional_premium = volume * mark_price * 100
                        
                        key = (strike, expiry)
                        if key not in premium_matrix:
                            premium_matrix[key] = {'call': 0, 'put': 0}
                        premium_matrix[key]['call'] += notional_premium
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        mark_price = contract.get('mark', 0)
                        
                        # Calculate notional premium
                        notional_premium = volume * mark_price * 100
                        
                        key = (strike, expiry)
                        if key not in premium_matrix:
                            premium_matrix[key] = {'call': 0, 'put': 0}
                        premium_matrix[key]['put'] += notional_premium
        
        if not premium_matrix:
            return None
        
        # Get unique expiries and strikes
        all_strikes = sorted(set(k[0] for k in premium_matrix.keys()))
        all_expiries = sorted(set(k[1] for k in premium_matrix.keys()))
        
        if not all_strikes or not all_expiries:
            return None
        
        # Limit to nearest expiries
        expiries = all_expiries[:min(num_expiries, len(all_expiries))]
        
        # Filter strikes to ±5% range
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        strikes_in_range = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # Get top 12 strikes by activity
        strike_activity = {}
        for strike in strikes_in_range:
            total = sum(abs(premium_matrix.get((strike, exp), {}).get('call', 0)) + 
                       abs(premium_matrix.get((strike, exp), {}).get('put', 0)) 
                       for exp in expiries)
            if total > 0:
                strike_activity[strike] = total
        
        if strike_activity:
            top_strikes = sorted(strike_activity.items(), key=lambda x: x[1], reverse=True)[:12]
            filtered_strikes = sorted([s[0] for s in top_strikes])
        else:
            filtered_strikes = sorted(strikes_in_range, key=lambda x: abs(x - underlying_price))[:12]
        
        if not filtered_strikes:
            return None
        
        # Create data matrix
        heat_data = []
        for strike in filtered_strikes:
            row = []
            for expiry in expiries:
                key = (strike, expiry)
                if key in premium_matrix:
                    net_premium = premium_matrix[key]['call'] - premium_matrix[key]['put']
                    row.append(net_premium)
                else:
                    row.append(0)
            heat_data.append(row)
        
        # Create labels
        strike_labels = [f"${s:.0f}" for s in filtered_strikes]
        expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        # Custom colorscale
        custom_colorscale = [
            [0.0, '#d32f2f'], [0.25, '#ef5350'], [0.4, '#ffcdd2'],
            [0.5, '#ffffff'],
            [0.6, '#c8e6c9'], [0.75, '#66bb6a'], [1.0, '#2e7d32']
        ]
        
        # Create text annotations
        text_annotations = []
        for row in heat_data:
            row_text = []
            for val in row:
                if abs(val) >= 1e6:
                    row_text.append(f"${val/1e6:.1f}M")
                elif abs(val) >= 1e3:
                    row_text.append(f"${val/1e3:.0f}K")
                else:
                    row_text.append("")
            text_annotations.append(row_text)
        
        # Create formatted hover text with M/B format
        hover_text = []
        for row in heat_data:
            hover_row = []
            for val in row:
                if abs(val) >= 1e9:
                    formatted = f"${val/1e9:.2f}B"
                elif abs(val) >= 1e6:
                    formatted = f"${val/1e6:.2f}M"
                elif abs(val) >= 1e3:
                    formatted = f"${val/1e3:.1f}K"
                else:
                    formatted = f"${val:.0f}"
                hover_row.append(formatted)
            hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heat_data,
            x=expiry_labels,
            y=strike_labels,
            customdata=hover_text,
            colorscale=custom_colorscale,
            zmid=0,
            showscale=True,
            colorbar=dict(title="Net ($)", tickformat='$,.0s', len=0.6, thickness=15),
            hovertemplate='<b>Strike: %{y}</b><br>Expiry: %{x}<br>Net Premium: %{customdata}<extra></extra>',
            text=text_annotations,
            texttemplate='%{text}',
            textfont=dict(size=9)
        ))
        
        # Mark current price
        closest_strike = min(filtered_strikes, key=lambda x: abs(x - underlying_price))
        try:
            current_price_idx = filtered_strikes.index(closest_strike)
            fig.add_hline(
                y=current_price_idx,
                line=dict(color="yellow", width=3, dash="dash"),
                annotation_text=f"  ${underlying_price:.2f}",
                annotation_position="right",
                annotation=dict(font_size=11, font_color="yellow", bgcolor="rgba(0,0,0,0.7)")
            )
        except (ValueError, IndexError):
            pass
        
        fig.update_layout(
            title=dict(text=f"Net Premium Flow<br><sub>Current: ${underlying_price:.2f}</sub>", 
                      font=dict(size=14), y=0.98),
            xaxis=dict(title="Expiration", tickfont=dict(size=9)),
            yaxis=dict(title="Strike", tickfont=dict(size=9), dtick=1),
            height=600,
            template='plotly_white',
            margin=dict(l=60, r=80, t=80, b=60),
            font=dict(size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating premium heatmap: {e}")
        return None

def create_market_positioning_summary(levels, underlying_price, symbol):
    """Create market positioning summary card"""
    try:
        if not levels:
            return None
        
        # Calculate market bias
        total_call_vol = levels.get('total_call_vol', 0)
        total_put_vol = levels.get('total_put_vol', 0)
        total_vol = total_call_vol + total_put_vol
        
        if total_vol == 0:
            return None
        
        call_pct = (total_call_vol / total_vol) * 100
        put_pct = (total_put_vol / total_vol) * 100
        
        # Determine bias
        if call_pct > 60:
            bias = "🟢 BULLISH"
            bias_color = "#22c55e"
        elif put_pct > 60:
            bias = "🔴 BEARISH"
            bias_color = "#ef4444"
        else:
            bias = "🟡 NEUTRAL"
            bias_color = "#f59e0b"
        
        # Calculate premium flow direction
        strike_data = levels.get('strike_data')
        if strike_data is not None and not strike_data.empty:
            total_call_premium = strike_data['call_premium'].sum()
            total_put_premium = strike_data['put_premium'].sum()
            net_premium = total_call_premium - total_put_premium
            
            if abs(net_premium) >= 1e6:
                net_premium_str = f"${net_premium/1e6:.1f}M"
            else:
                net_premium_str = f"${net_premium/1e3:.0f}K"
            
            premium_direction = "➚ Calls" if net_premium > 0 else "➘ Puts"
            premium_color = "#22c55e" if net_premium > 0 else "#ef4444"
        else:
            net_premium_str = "N/A"
            premium_direction = "N/A"
            premium_color = "#6b7280"
        
        # Distance to key levels
        flip_distance = "N/A"
        if levels.get('flip_level'):
            flip_dist_pct = ((levels['flip_level'] - underlying_price) / underlying_price) * 100
            flip_distance = f"{flip_dist_pct:+.2f}%"
        
        html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 12px; padding: 16px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); height: 100%;">
            <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; 
                        border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 8px;">
                📊 SPY Positioning
            </div>
            
            <div style="margin-bottom: 10px;">
                <div style="font-size: 11px; opacity: 0.9; margin-bottom: 3px;">Market Bias</div>
                <div style="font-size: 20px; font-weight: 800; color: {bias_color};">
                    {bias}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 10px; opacity: 0.85;">Call Volume</div>
                    <div style="font-size: 16px; font-weight: 700;">{call_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 10px; opacity: 0.85;">Put Volume</div>
                    <div style="font-size: 16px; font-weight: 700;">{put_pct:.1f}%</div>
                </div>
            </div>
            
            <div style="margin-bottom: 10px;">
                <div style="font-size: 10px; opacity: 0.85;">P/C Ratio</div>
                <div style="font-size: 16px; font-weight: 700;">{levels.get('pc_ratio', 0):.2f}</div>
            </div>
            
            <div style="margin-bottom: 10px;">
                <div style="font-size: 10px; opacity: 0.85;">Net Premium Flow</div>
                <div style="font-size: 16px; font-weight: 700; color: {premium_color};">
                    {net_premium_str} {premium_direction}
                </div>
            </div>
            
            <div>
                <div style="font-size: 10px; opacity: 0.85;">Distance to Flip</div>
                <div style="font-size: 14px; font-weight: 700;">{flip_distance}</div>
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error creating positioning summary: {e}")
        return None

def create_key_levels_table(levels, underlying_price):
    """Create compact key levels summary table"""
    try:
        if not levels:
            return None
        
        table_data = []
        
        # Call Wall
        if levels.get('call_wall') is not None and not levels['call_wall'].empty:
            call_strike = levels['call_wall']['strike']
            call_dist = ((call_strike - underlying_price) / underlying_price) * 100
            call_vol = levels['call_wall']['call_vol']
            table_data.append({
                'Level': '🟢 Call Wall',
                'Strike': f"${call_strike:.2f}",
                'Distance': f"{call_dist:+.2f}%",
                'Volume': f"{int(call_vol):,}"
            })
        
        # Put Wall
        if levels.get('put_wall') is not None and not levels['put_wall'].empty:
            put_strike = levels['put_wall']['strike']
            put_dist = ((put_strike - underlying_price) / underlying_price) * 100
            put_vol = levels['put_wall']['put_vol']
            table_data.append({
                'Level': '🔴 Put Wall',
                'Strike': f"${put_strike:.2f}",
                'Distance': f"{put_dist:+.2f}%",
                'Volume': f"{int(put_vol):,}"
            })
        
        # Flip Level
        if levels.get('flip_level'):
            flip_strike = levels['flip_level']
            flip_dist = ((flip_strike - underlying_price) / underlying_price) * 100
            table_data.append({
                'Level': '🟣 Flip Level',
                'Strike': f"${flip_strike:.2f}",
                'Distance': f"{flip_dist:+.2f}%",
                'Volume': '-'
            })
        
        # Max GEX
        if levels.get('max_gex') is not None and not levels['max_gex'].empty:
            gex_strike = levels['max_gex']['strike']
            gex_dist = ((gex_strike - underlying_price) / underlying_price) * 100
            gex_val = levels['max_gex']['net_gex']
            if abs(gex_val) >= 1e6:
                gex_str = f"${gex_val/1e6:.1f}M"
            else:
                gex_str = f"${gex_val/1e3:.0f}K"
            table_data.append({
                'Level': '💎 Max GEX',
                'Strike': f"${gex_strike:.2f}",
                'Distance': f"{gex_dist:+.2f}%",
                'Volume': gex_str
            })
        
        if not table_data:
            return None
        
        df = pd.DataFrame(table_data)
        return df
        
    except Exception as e:
        logger.error(f"Error creating key levels table: {e}")
        return None

@st.fragment(run_every="300s")
@st.fragment(run_every="180s")  # Auto-refresh every 3 minutes
def live_watchlist():
    """Auto-refreshing watchlist widget - fetches from droplet API"""
    st.markdown('<div class="section-header">📊 LIVE WATCHLIST</div>', unsafe_allow_html=True)
    
    # Initialize filter preference in session state
    if 'watchlist_filter' not in st.session_state:
        st.session_state.watchlist_filter = 'all'
    
    # Initialize view mode (stocks vs ETFs)
    if 'watchlist_view_mode' not in st.session_state:
        st.session_state.watchlist_view_mode = 'stocks'
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # View mode toggle (Stocks vs ETFs)
        view_mode = st.radio(
            "View:",
            options=['stocks', 'etfs'],
            format_func=lambda x: '📈' if x == 'stocks' else '🏦',
            horizontal=True,
            key='watchlist_view_mode_selector',
            index=0 if st.session_state.watchlist_view_mode == 'stocks' else 1
        )
        if view_mode != st.session_state.watchlist_view_mode:
            st.session_state.watchlist_view_mode = view_mode
            st.rerun()
    
    with col2:
        # Primary filter toggle
        filter_option = st.radio(
            "Filter:",
            options=['all', 'bull', 'bear'],
            format_func=lambda x: '📊' if x == 'all' else ('🟢' if x == 'bull' else '🔴'),
            horizontal=True,
            key='watchlist_filter_selector',
            index=0 if st.session_state.watchlist_filter == 'all' else (1 if st.session_state.watchlist_filter == 'bull' else 2)
        )
        if filter_option != st.session_state.watchlist_filter:
            st.session_state.watchlist_filter = filter_option
            st.rerun()
    
    with col3:
        # Advanced filter toggle (only show for stocks, not ETFs)
        if st.session_state.watchlist_view_mode == 'stocks':
            if 'watchlist_advanced_filter' not in st.session_state:
                st.session_state.watchlist_advanced_filter = 'none'
            
            advanced_filter = st.radio(
                "Show only:",
                options=['none', 'whale', 'flow', 'premarket', 'news'],
                format_func=lambda x: {
                    'none': '✨',
                    'whale': '🐋',
                    'flow': '📞',
                    'premarket': '🌅',
                    'news': '📰'
                }[x],
                horizontal=True,
                key='watchlist_advanced_filter_selector',
                index=['none', 'whale', 'flow', 'premarket', 'news'].index(st.session_state.watchlist_advanced_filter)
            )
            if advanced_filter != st.session_state.watchlist_advanced_filter:
                st.session_state.watchlist_advanced_filter = advanced_filter
                st.rerun()
        else:
            # ETF mode - show empty space or message
            st.write("")
    
    # Build filter description
    filter_desc_parts = []
    
    # Add primary filter description
    if st.session_state.watchlist_filter == 'bull':
        filter_desc_parts.append("Bullish stocks (positive daily change)")
    elif st.session_state.watchlist_filter == 'bear':
        filter_desc_parts.append("Bearish stocks (negative daily change)")
    
    # Add advanced filter description
    if st.session_state.watchlist_advanced_filter == 'whale':
        filter_desc_parts.append("with 2+ whale flows (last 6h)")
    elif st.session_state.watchlist_advanced_filter == 'flow':
        filter_desc_parts.append("with strong options flow (>$50k net premium)")
    elif st.session_state.watchlist_advanced_filter == 'premarket':
        filter_desc_parts.append("with >1% premarket move")
    elif st.session_state.watchlist_advanced_filter == 'news':
        filter_desc_parts.append("with analyst upgrades/downgrades")
    
    # Display caption with filter description
    caption_text = f"🔄 Auto-updates every 3min • {datetime.now().strftime('%H:%M:%S')}"
    if filter_desc_parts:
        caption_text += f"\n📌 Showing: {' '.join(filter_desc_parts)}"
    
    st.caption(caption_text)
    
    # Handle ETF view mode
    if st.session_state.watchlist_view_mode == 'etfs':
        # Display sector ETFs
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication Services',
            'SMH': 'Semiconductors',
            'XRT': 'Retail',
            'XHB': 'Homebuilders',
            'XME': 'Metals & Mining',
            'XOP': 'Oil & Gas',
            'IBB': 'Biotech',
            'ITB': 'Housing',
            'KBE': 'Banking',
            'KRE': 'Regional Banks'
        }
        
        # Fetch ETF data using Schwab client
        try:
            from src.api.schwab_client import SchwabClient
            client = SchwabClient()
            client.authenticate()
            
            etf_data = []
            for symbol, sector in sector_etfs.items():
                try:
                    quote = client.get_quote(symbol)
                    if quote and symbol in quote:
                        q = quote[symbol]['quote']
                        price = q.get('lastPrice', 0)
                        net_change = q.get('netChange', 0)
                        net_pct_change = q.get('netPercentChange', 0)
                        volume = q.get('totalVolume', 0)
                        
                        etf_data.append({
                            'symbol': symbol,
                            'sector': sector,
                            'price': price,
                            'daily_change': net_change,
                            'daily_change_pct': net_pct_change,
                            'volume': volume
                        })
                except:
                    continue
            
            # Sort by % change
            etf_data = sorted(etf_data, key=lambda x: x['daily_change_pct'], reverse=True)
            
            # Apply bull/bear filter
            if st.session_state.watchlist_filter == 'bull':
                etf_data = [item for item in etf_data if item['daily_change_pct'] >= 0]
            elif st.session_state.watchlist_filter == 'bear':
                etf_data = [item for item in etf_data if item['daily_change_pct'] < 0]
            
            # Display ETFs
            for item in etf_data:
                symbol = item['symbol']
                sector = item['sector']
                price = item['price']
                daily_change = item['daily_change']
                daily_change_pct = item['daily_change_pct']
                volume = item['volume']
                
                sentiment = 'bullish' if daily_change_pct >= 0 else 'bearish'
                change_color = '#22c55e' if daily_change_pct >= 0 else '#ef4444'
                change_symbol = '▲' if daily_change_pct >= 0 else '▼'
                vol_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"
                
                html = f"""
                <div class="watchlist-item {sentiment}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 14px;">{symbol}</strong>
                            <span style="font-size: 11px; color: #6b7280; margin-left: 6px;">{sector}</span>
                            <span style="color: {change_color}; margin-left: 8px;">
                                {change_symbol} ${abs(daily_change):.2f} ({abs(daily_change_pct):.2f}%)
                            </span>
                        </div>
                        <div style="text-align: right; font-size: 13px;">
                            <strong>${price:.2f}</strong>
                        </div>
                    </div>
                    <div style="margin-top: 4px; font-size: 10px; color: #6b7280;">
                        Vol: {vol_str}
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                
                # Make ETF clickable
                if st.button(f"📈 Trade {symbol}", key=f"etf_{symbol}", type="secondary", use_container_width=True):
                    st.session_state.trading_hub_symbol = symbol
                    st.session_state.trading_hub_expiry = get_default_expiry(symbol)
                    st.rerun()
        
        except Exception as e:
            st.error(f"Unable to load ETF data: {str(e)}")
        
        return  # Exit early for ETF view
    
    # Continue with stocks view
    # Fetch scanner signals
    scanner_signals = {}
    try:
        import requests
        
        # Fetch MACD signals
        macd_response = requests.get('http://138.197.210.166:8000/api/macd_scanner?filter=all&limit=150', timeout=3)
        if macd_response.status_code == 200:
            macd_data = macd_response.json().get('data', [])
            for item in macd_data:
                symbol = item['symbol']
                if symbol not in scanner_signals:
                    scanner_signals[symbol] = []
                if item.get('bullish_cross'):
                    scanner_signals[symbol].append('macd_bull')
                elif item.get('bearish_cross'):
                    scanner_signals[symbol].append('macd_bear')
        
        # Fetch VPB signals
        vpb_response = requests.get('http://138.197.210.166:8000/api/vpb_scanner?filter=all&limit=150', timeout=3)
        if vpb_response.status_code == 200:
            vpb_data = vpb_response.json().get('data', [])
            for item in vpb_data:
                symbol = item['symbol']
                if symbol not in scanner_signals:
                    scanner_signals[symbol] = []
                if item.get('buy_signal'):
                    scanner_signals[symbol].append('vpb_bull')
                elif item.get('sell_signal'):
                    scanner_signals[symbol].append('vpb_bear')
    except:
        pass  # Silently fail if scanner data unavailable
    
    # Fetch whale flows for UOA indicator and options flow sentiment
    whale_data = {}
    try:
        whale_flows = fetch_whale_flows(sort_by='time', limit=100, hours=6)
        for flow in whale_flows:
            symbol = flow['symbol']
            if symbol not in whale_data:
                whale_data[symbol] = {'count': 0, 'call_premium': 0, 'put_premium': 0}
            whale_data[symbol]['count'] += 1
            if flow['type'] == 'CALL':
                whale_data[symbol]['call_premium'] += flow['premium'] * flow['volume']
            else:
                whale_data[symbol]['put_premium'] += flow['premium'] * flow['volume']
    except:
        pass
    
    # Fetch news/alerts for upgrades/downgrades
    news_symbols = {}
    try:
        news_feed = fetch_google_alerts("https://news.google.com/rss/search?q=stock+upgrade+OR+downgrade&hl=en-US&gl=US&ceid=US:en")
        if news_feed:
            for alert in news_feed[:50]:  # Check recent 50 alerts
                title_lower = alert.get('title', '').lower()
                for symbol in [item['symbol'] for item in fetch_watchlist(order_by='daily_change_pct', limit=150)]:
                    if symbol.lower() in title_lower:
                        if 'upgrade' in title_lower:
                            news_symbols[symbol] = 'upgrade'
                        elif 'downgrade' in title_lower:
                            news_symbols[symbol] = 'downgrade'
                        break
    except:
        pass
    
    # Fetch from droplet API (cached data) - fetch all 150 stocks from database
    watchlist_data = fetch_watchlist(order_by='daily_change_pct', limit=150)
    
    # Filter based on selection
    if st.session_state.watchlist_filter == 'bull':
        watchlist_data = [item for item in watchlist_data if item['daily_change_pct'] >= 0]
    elif st.session_state.watchlist_filter == 'bear':
        watchlist_data = [item for item in watchlist_data if item['daily_change_pct'] < 0]
        # Sort bears by most negative (most down)
        watchlist_data = sorted(watchlist_data, key=lambda x: x['daily_change_pct'])
    
    # Apply advanced filters
    if st.session_state.watchlist_advanced_filter != 'none':
        filtered_data = []
        for item in watchlist_data:
            symbol = item['symbol']
            include = False
            
            if st.session_state.watchlist_advanced_filter == 'whale':
                # Show stocks with 2+ whale flows
                if symbol in whale_data and whale_data[symbol]['count'] >= 2:
                    include = True
            
            elif st.session_state.watchlist_advanced_filter == 'flow':
                # Show stocks with significant options flow (>$50k net premium)
                if symbol in whale_data:
                    net_premium = whale_data[symbol]['call_premium'] - whale_data[symbol]['put_premium']
                    if abs(net_premium) > 50000:
                        include = True
            
            elif st.session_state.watchlist_advanced_filter == 'premarket':
                # Show stocks with >1% premarket move
                premarket_change = item.get('premarket_change_pct', 0)
                if abs(premarket_change) > 1.0:
                    include = True
            
            elif st.session_state.watchlist_advanced_filter == 'news':
                # Show stocks with recent upgrades/downgrades
                if symbol in news_symbols:
                    include = True
            
            if include:
                filtered_data.append(item)
        
        watchlist_data = filtered_data
    
    # Display sorted watchlist
    for item in watchlist_data:
        symbol = item['symbol']
        price = item['price']
        daily_change = item.get('daily_change', 0)
        daily_change_pct = item['daily_change_pct']
        volume = item['volume']
        
        # Determine sentiment based on price change
        sentiment = 'bullish' if daily_change_pct >= 0 else 'bearish'
        
        # Create compact watchlist item
        change_color = '#22c55e' if daily_change_pct >= 0 else '#ef4444'
        change_symbol = '▲' if daily_change_pct >= 0 else '▼'
        
        # Format volume
        vol_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"
        
        # Build scanner icons
        scanner_icons = ""
        if symbol in scanner_signals:
            signals = scanner_signals[symbol]
            if 'macd_bull' in signals:
                scanner_icons += '<span title="MACD Bullish Cross" style="margin-left: 4px;">📈</span>'
            if 'macd_bear' in signals:
                scanner_icons += '<span title="MACD Bearish Cross" style="margin-left: 4px;">📉</span>'
            if 'vpb_bull' in signals:
                scanner_icons += '<span title="Volume Breakout" style="margin-left: 4px;">🚀</span>'
            if 'vpb_bear' in signals:
                scanner_icons += '<span title="Volume Breakdown" style="margin-left: 4px;">💥</span>'
        
        # Build additional indicators
        indicators = []
        
        # Whale/UOA indicator
        if symbol in whale_data and whale_data[symbol]['count'] >= 2:
            indicators.append(f'<span title="{whale_data[symbol]["count"]} whale flows detected" style="color: #8b5cf6;">🐋 Whale</span>')
        
        # Options flow sentiment
        if symbol in whale_data:
            call_prem = whale_data[symbol]['call_premium']
            put_prem = whale_data[symbol]['put_premium']
            net_premium = call_prem - put_prem
            if abs(net_premium) > 50000:  # Significant flow > $50k
                if net_premium > 0:
                    indicators.append(f'<span title="${call_prem/1000:.0f}k calls vs ${put_prem/1000:.0f}k puts" style="color: #22c55e;">📞 ${net_premium/1000:.0f}k</span>')
                else:
                    indicators.append(f'<span title="${call_prem/1000:.0f}k calls vs ${put_prem/1000:.0f}k puts" style="color: #ef4444;">📍 ${abs(net_premium)/1000:.0f}k</span>')
        
        # Premarket change (if available in data)
        premarket_change = item.get('premarket_change_pct', 0)
        if abs(premarket_change) > 1.0:  # Show if > 1% premarket move
            pm_color = '#22c55e' if premarket_change > 0 else '#ef4444'
            indicators.append(f'<span title="Premarket change" style="color: {pm_color};">🌅 {premarket_change:+.1f}%</span>')
        
        # News upgrades/downgrades
        if symbol in news_symbols:
            if news_symbols[symbol] == 'upgrade':
                indicators.append('<span title="Recent upgrade" style="color: #22c55e;">⬆️ Upgrade</span>')
            else:
                indicators.append('<span title="Recent downgrade" style="color: #ef4444;">⬇️ Downgrade</span>')
        
        indicators_html = ""
        if indicators:
            indicators_html = f'<div style="margin-top: 4px; font-size: 9px; display: flex; gap: 8px; flex-wrap: wrap;">{" • ".join(indicators)}</div>'
        
        html = f"""
        <div class="watchlist-item {sentiment}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="font-size: 14px;">{symbol}</strong>{scanner_icons}
                    <span style="color: {change_color}; margin-left: 8px;">
                        {change_symbol} ${abs(daily_change):.2f} ({abs(daily_change_pct):.2f}%)
                    </span>
                </div>
                <div style="text-align: right; font-size: 13px;">
                    <strong>${price:.2f}</strong>
                </div>
            </div>
            <div style="margin-top: 4px; font-size: 10px; color: #6b7280;">
                Vol: {vol_str}
            </div>
            {indicators_html}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
        # Make symbol clickable
        if st.button(f"📈 Trade {symbol}", key=f"watch_{symbol}", type="secondary", use_container_width=True):
            st.session_state.trading_hub_symbol = symbol
            st.session_state.trading_hub_expiry = get_default_expiry(symbol)
            st.rerun()

@st.fragment(run_every="180s")  # Auto-refresh every 3 minutes
def whale_flows_feed():
    """Auto-refreshing whale flows feed with sort toggle - fetches from droplet API"""
    
    # Initialize sort preference in session state
    if 'whale_sort_by' not in st.session_state:
        st.session_state.whale_sort_by = 'time'  # Default to recent flows
    
    # Header with prominent sort toggle
    st.markdown('<div class="section-header">🐋 WHALE FLOWS</div>', unsafe_allow_html=True)
    
    # Initialize whale filter state
    if 'whale_filter' not in st.session_state:
        st.session_state.whale_filter = 'all'
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radio buttons for sort - more visible
        sort_option = st.radio(
            "Sort by:",
            options=['time', 'score'],
            format_func=lambda x: '🕐 Most Recent' if x == 'time' else '🏆 Highest Score',
            horizontal=True,
            key='whale_sort_selector',
            index=0 if st.session_state.whale_sort_by == 'time' else 1
        )
        if sort_option != st.session_state.whale_sort_by:
            st.session_state.whale_sort_by = sort_option
            st.rerun()
    
    with col2:
        # Filter by current symbol or all
        filter_option = st.radio(
            "Show flows:",
            options=['all', 'symbol'],
            format_func=lambda x: '📊 All Stocks' if x == 'all' else f'🎯 {st.session_state.trading_hub_symbol} Only',
            horizontal=True,
            key='whale_filter_selector',
            index=0 if st.session_state.whale_filter == 'all' else 1
        )
        if filter_option != st.session_state.whale_filter:
            st.session_state.whale_filter = filter_option
            st.rerun()
    
    st.caption(f"🔄 Auto-updates every 3min • From droplet cache • {datetime.now().strftime('%H:%M:%S')}")

    
    # Fetch from droplet API (cached data) - increased to 20 flows
    whale_flows = fetch_whale_flows(
        sort_by=st.session_state.whale_sort_by,
        limit=100 if st.session_state.whale_filter == 'symbol' else 20,  # Fetch more if filtering by symbol
        hours=6  # Show flows from last 6 hours
    )
    
    # Filter by symbol if selected
    if st.session_state.whale_filter == 'symbol' and whale_flows:
        whale_flows = [flow for flow in whale_flows if flow['symbol'] == st.session_state.trading_hub_symbol]
        whale_flows = whale_flows[:20]  # Limit to 20 after filtering
    
    if whale_flows:
        for flow in whale_flows:
            card_class = 'call' if flow['type'] == 'CALL' else 'put'
            # Format whale score with comma separator for readability
            whale_score_formatted = f"{int(flow['whale_score']):,}"
            
            # Parse expiry date (comes as string from API)
            try:
                from datetime import datetime as dt
                if isinstance(flow['expiry'], str):
                    expiry_date = dt.strptime(flow['expiry'], '%Y-%m-%d').date()
                else:
                    expiry_date = flow['expiry']
                dte = (expiry_date - datetime.now().date()).days
                expiry_display = f"{expiry_date.strftime('%m/%d')} ({dte}DTE)"
            except:
                expiry_display = str(flow.get('expiry', 'N/A'))
            
            # Show different info based on sort
            if st.session_state.whale_sort_by == 'time':
                # Parse detected_at timestamp from API
                try:
                    if isinstance(flow.get('detected_at'), str):
                        detected_at = dt.strptime(flow['detected_at'], '%Y-%m-%d %H:%M:%S')
                    else:
                        detected_at = flow.get('detected_at', datetime.now())
                    time_diff = (datetime.now() - detected_at).total_seconds()
                    if time_diff < 60:
                        time_ago = f"{int(time_diff)}s ago"
                    elif time_diff < 3600:
                        time_ago = f"{int(time_diff / 60)}m ago"
                    else:
                        time_ago = f"{int(time_diff / 3600)}h ago"
                    score_display = f"Score: {whale_score_formatted} • {time_ago}"
                except:
                    score_display = f"Score: {whale_score_formatted}"
            else:
                score_display = f"Score: {whale_score_formatted}"
            
            html = f"""
            <div class="whale-card {card_class}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="font-size: 14px;">{flow['symbol']} {flow['type']}</strong>
                    <span style="font-size: 12px;">{score_display}</span>
                </div>
                <div style="font-size: 11px; opacity: 0.95;">
                    Strike: ${flow['strike']:.2f} | Vol: {int(flow['volume']):,} | Vol/OI: {flow['vol_oi']:.1f}x
                </div>
                <div style="font-size: 11px; opacity: 0.95; margin-top: 2px;">
                    Premium: ${flow['premium']:.2f} | Delta: {abs(flow['delta']):.3f} | Exp: {expiry_display}
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
    else:
        if st.session_state.whale_filter == 'symbol':
            st.info(f"No whale flows detected for {st.session_state.trading_hub_symbol} in the last 6 hours.")
        else:
            st.info("No recent whale flows detected. Worker may be starting up...")

# ===== MAIN PAGE LAYOUT =====

# Title
st.title("🎯 Trading Hub")

# News Alerts Section (Collapsible)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_google_alerts(rss_url):
    """Fetch and parse Google Alerts RSS feed with formatting"""
    try:
        feed = feedparser.parse(rss_url)
        alerts = []
        for entry in feed.entries[:5]:  # Latest 5
            # Clean up title - remove HTML tags and decode entities
            title = entry.title
            title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
            title = unescape(title)  # Decode HTML entities
            title = title.replace('&nbsp;', ' ').strip()
            
            # Parse published date to more readable format
            published = entry.get('published', '')
            try:
                if published:
                    dt = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
                    published = dt.strftime('%b %d, %I:%M %p')
            except:
                pass
            
            # Extract ticker symbols (if present in title)
            tickers = re.findall(r'\b[A-Z]{2,5}\b', title)
            
            alerts.append({
                'title': title,
                'link': entry.link,
                'published': published,
                'tickers': tickers[:3] if tickers else []  # Limit to first 3
            })
        return alerts
    except Exception as e:
        logger.error(f"Error fetching RSS: {e}")
        return []

with st.expander("📰 Market News & Alerts", expanded=False):
    # Split into two halves: News on left, Scanner on right
    news_section, scanner_section = st.columns(2)
    
    with news_section:
        st.markdown("#### Market News")
        news_col1, news_col2 = st.columns(2)
        
        # Replace these with your actual Google Alert RSS URLs
        rss_feeds = {
            'Stock Upgrade': 'https://www.google.com/alerts/feeds/17914089297795458845/3554285287301408399',
            'Stock Downgrade': 'https://www.google.com/alerts/feeds/17914089297795458845/14042214614423891721'
        }
        
        with news_col1:
            st.markdown(f"**🔼 {list(rss_feeds.keys())[0]}**")
            alerts = fetch_google_alerts(list(rss_feeds.values())[0])
            if alerts:
                for alert in alerts:
                    # Show tickers as badges if found
                    ticker_badges = ' '.join([f'`{t}`' for t in alert['tickers']]) if alert['tickers'] else ''
                    st.markdown(f"**[{alert['title']}]({alert['link']})**")
                    
                    # Show tickers and timestamp on same line
                    info_line = []
                    if ticker_badges:
                        info_line.append(ticker_badges)
                    if alert['published']:
                        info_line.append(f"🕐 {alert['published']}")
                    
                    if info_line:
                        st.caption(' • '.join(info_line))
                    st.divider()
            else:
                st.info("No recent alerts")
        
        with news_col2:
            st.markdown(f"**🔽 {list(rss_feeds.keys())[1]}**")
            alerts = fetch_google_alerts(list(rss_feeds.values())[1])
            if alerts:
                for alert in alerts:
                    ticker_badges = ' '.join([f'`{t}`' for t in alert['tickers']]) if alert['tickers'] else ''
                    st.markdown(f"**[{alert['title']}]({alert['link']})**")
                    
                    info_line = []
                    if ticker_badges:
                        info_line.append(ticker_badges)
                    if alert['published']:
                        info_line.append(f"🕐 {alert['published']}")
                    
                    if info_line:
                        st.caption(' • '.join(info_line))
                    st.divider()
            else:
                st.info("No recent alerts")
    
    with scanner_section:
        st.markdown("#### 📊 MACD Scanner")
        
        # Filter toggle
        scanner_filter = st.radio(
            "",
            options=['bullish', 'bearish'],
            format_func=lambda x: '🟢 Bullish Crosses' if x == 'bullish' else '🔴 Bearish Crosses',
            horizontal=True,
            key='macd_scanner_filter'
        )
        
        # Fetch MACD scanner data from API
        try:
            import requests
            response = requests.get(
                'http://138.197.210.166:8000/api/macd_scanner',
                params={'filter': scanner_filter, 'limit': 10},
                timeout=5
            )
            
            if response.status_code == 200:
                scanner_data = response.json().get('data', [])
                
                if scanner_data:
                    # Create compact table
                    table_data = []
                    for item in scanner_data:
                        table_data.append({
                            'Symbol': item['symbol'],
                            'Price': f"${item['price']:.2f}",
                            'Change %': f"{item['price_change_pct']:+.2f}%",
                            'MACD': f"{item['macd']:.2f}",
                            'Trend': '🟢' if item['trend'] == 'bullish' else '🔴'
                        })
                    
                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )
                    
                    st.caption(f"🔄 Updated: {scanner_data[0].get('scanned_at', 'N/A')[:16] if scanner_data else 'N/A'}")
                else:
                    st.info(f"No {scanner_filter} MACD crosses detected")
            else:
                st.warning("⚠️ Scanner API unavailable - Check droplet services")
        except requests.exceptions.ConnectionError:
            st.warning("🔴 Scanner service offline - Run: `systemctl start macd-scanner api-server`")
        except requests.exceptions.Timeout:
            st.warning("⏱️ Scanner request timed out - Service may be busy")
        except Exception as e:
            st.warning(f"⚠️ Scanner unavailable: Connection refused")
        
        st.markdown("---")
        
        st.markdown("#### 📈 Volume-Price Break Scanner")
        
        # Filter toggle
        vpb_filter = st.radio(
            "",
            options=['bullish', 'bearish'],
            format_func=lambda x: '🟢 Bullish Breakouts' if x == 'bullish' else '🔴 Bearish Breakdowns',
            horizontal=True,
            key='vpb_scanner_filter'
        )
        
        # Fetch VPB scanner data from API
        try:
            import requests
            response = requests.get(
                'http://138.197.210.166:8000/api/vpb_scanner',
                params={'filter': vpb_filter, 'limit': 10},
                timeout=5
            )
            
            if response.status_code == 200:
                scanner_data = response.json().get('data', [])
                
                if scanner_data:
                    # Create compact table
                    table_data = []
                    for item in scanner_data:
                        table_data.append({
                            'Symbol': item['symbol'],
                            'Price': f"${item['price']:.2f}",
                            'Change %': f"{item['price_change_pct']:+.2f}%",
                            'Vol Surge': f"+{item['volume_surge_pct']:.1f}%",
                            'Signal': '🟢 BO' if item['buy_signal'] else '🔴 BD'
                        })
                    
                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )
                    
                    st.caption(f"🔄 Updated: {scanner_data[0].get('scanned_at', 'N/A')[:16] if scanner_data else 'N/A'}")
                else:
                    st.info(f"No {vpb_filter} signals detected")
            else:
                st.warning("⚠️ Scanner API unavailable - Check droplet services")
        except requests.exceptions.ConnectionError:
            st.warning("🔴 Scanner service offline - Run: `systemctl start vpb-scanner api-server`")
        except requests.exceptions.Timeout:
            st.warning("⏱️ Scanner request timed out - Service may be busy")
        except Exception as e:
            st.warning(f"⚠️ Scanner unavailable: Connection refused")

# Top controls - Symbol selection, timeframe, and expiry
control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns([2.5, 1, 1.5, 1.2, 0.5])

# Initialize tracking for last quick symbol clicked
if 'last_quick_symbol' not in st.session_state:
    st.session_state.last_quick_symbol = None

with control_col1:
    # Quick symbol buttons - more stocks now fit
    quick_symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 
                     'AAPL', 'PLTR', 'META', 'MSFT', 
                     'AMZN', 'GOOGL','AMD','NFLX','CRWD','$SPX']
    selected_symbol = st.segmented_control(
        "Symbol",
        options=quick_symbols,
        default=st.session_state.trading_hub_symbol if st.session_state.trading_hub_symbol in quick_symbols else None,
        key='symbol_selector'
    )
    # Only trigger if actually clicked (different from last)
    if selected_symbol and selected_symbol != st.session_state.last_quick_symbol:
        st.session_state.last_quick_symbol = selected_symbol
        st.session_state.trading_hub_symbol = selected_symbol
        st.session_state.trading_hub_expiry = get_default_expiry(selected_symbol)
        st.rerun()

with control_col2:
    # Custom symbol input - visible and prominent
    def load_custom_symbol():
        symbol = st.session_state.custom_symbol_input.upper().strip()
        if symbol and symbol != st.session_state.trading_hub_symbol:
            st.session_state.trading_hub_symbol = symbol
            st.session_state.trading_hub_expiry = get_default_expiry(symbol)
    
    symbol_input = st.text_input(
        "Custom Symbol", 
        value="", 
        placeholder="Enter ticker...",
        key="custom_symbol_input",
        on_change=load_custom_symbol,
        help="Type any stock ticker and press Enter to analyze"
    )

with control_col3:
    # Timeframe toggle
    timeframe = st.radio(
        "Timeframe",
        options=['intraday', 'daily'],
        index=0 if st.session_state.trading_hub_timeframe == 'intraday' else 1,
        horizontal=True,
        key='timeframe_selector'
    )
    if timeframe != st.session_state.trading_hub_timeframe:
        st.session_state.trading_hub_timeframe = timeframe
        st.rerun()

with control_col4:
    # Expiry date selector
    # Get next 8 Fridays for expiry options
    available_expiries = get_next_n_fridays(8)
    
    # Create display options
    expiry_options = {}
    today = datetime.now().date()
    for exp_date in available_expiries:
        if exp_date == today:
            expiry_options[exp_date] = "0DTE (Today)"
        else:
            days_out = (exp_date - today).days
            expiry_options[exp_date] = f"{exp_date.strftime('%b %d')} ({days_out}d)"
    
    # Ensure current expiry is in available options
    if st.session_state.trading_hub_expiry not in expiry_options:
        st.session_state.trading_hub_expiry = available_expiries[0]
    
    selected_expiry = st.selectbox(
        "Expiry",
        options=list(expiry_options.keys()),
        format_func=lambda x: expiry_options[x],
        index=list(expiry_options.keys()).index(st.session_state.trading_hub_expiry) if st.session_state.trading_hub_expiry in expiry_options else 0,
        key='expiry_selector',
        help="Select options expiration date for analysis"
    )
    
    if selected_expiry != st.session_state.trading_hub_expiry:
        st.session_state.trading_hub_expiry = selected_expiry
        st.rerun()

with control_col5:
    if st.button("🔄", type="primary", use_container_width=True, help="Refresh data"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Set expiry if not set
if st.session_state.trading_hub_expiry is None:
    st.session_state.trading_hub_expiry = get_default_expiry(st.session_state.trading_hub_symbol)

# Main layout: Chart (center), Watchlist (left), Whale Flows (right)
left_col, center_col, right_col = st.columns([1.2, 3, 1.2])

with right_col:
    # Market Positioning Summary at the top - collapsible
    symbol = st.session_state.trading_hub_symbol
    timeframe = st.session_state.trading_hub_timeframe
    expiry = st.session_state.trading_hub_expiry
    
    snap_for_positioning = get_market_snapshot(symbol, expiry.strftime('%Y-%m-%d'), timeframe)
    if snap_for_positioning and snap_for_positioning.get('underlying_price'):
        price_for_positioning = snap_for_positioning['underlying_price']
        levels_for_positioning = calculate_option_levels(snap_for_positioning['options_chain'], price_for_positioning)
        
        if levels_for_positioning:
            # Create metrics display
            total_call_vol = levels_for_positioning.get('total_call_vol', 0)
            total_put_vol = levels_for_positioning.get('total_put_vol', 0)
            total_vol = total_call_vol + total_put_vol
            
            if total_vol > 0:
                call_pct = (total_call_vol / total_vol) * 100
                put_pct = (total_put_vol / total_vol) * 100
                
                # Determine bias for compact display
                if call_pct > 60:
                    bias_emoji = "🟢"
                    bias_text = "BULLISH"
                elif put_pct > 60:
                    bias_emoji = "🔴"
                    bias_text = "BEARISH"
                else:
                    bias_emoji = "🟡"
                    bias_text = "NEUTRAL"
                
                # Collapsible expander (collapsed by default)
                with st.expander(f"📊 {symbol}: {bias_emoji} {bias_text}", expanded=False):
                    # Compact metrics in 2 columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Calls", f"{call_pct:.0f}%", label_visibility="visible")
                        st.metric("P/C", f"{levels_for_positioning.get('pc_ratio', 0):.2f}", label_visibility="visible")
                    with col2:
                        st.metric("Puts", f"{put_pct:.0f}%", label_visibility="visible")
                        
                        # Distance to flip
                        if levels_for_positioning.get('flip_level'):
                            flip_dist_pct = ((levels_for_positioning['flip_level'] - price_for_positioning) / price_for_positioning) * 100
                            st.metric("Flip", f"{flip_dist_pct:+.2f}%", label_visibility="visible")
    
    # Whale flows feed (no separator needed now)
    whale_flows_feed()

with center_col:
    # Main chart
    symbol = st.session_state.trading_hub_symbol
    timeframe = st.session_state.trading_hub_timeframe
    expiry = st.session_state.trading_hub_expiry
    
    with st.spinner(f"Loading {symbol} data..."):
        snap = get_market_snapshot(symbol, expiry.strftime('%Y-%m-%d'), timeframe)
        
        if snap and snap.get('underlying_price'):
            price = snap['underlying_price']
            
            # Calculate option levels
            levels = calculate_option_levels(snap['options_chain'], price)
            
            # Display chart first
            if snap.get('price_history'):
                chart = create_trading_chart(snap['price_history'], levels, price, symbol, timeframe)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Unable to create chart")
            else:
                st.warning("Price history not available")
            
            # Display key metrics below chart
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                quote_data = snap['quote'].get(symbol, {}).get('quote', {})
                prev_close = quote_data.get('closePrice', price)
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0
                st.metric("Price", f"${price:.2f}", f"{change_pct:+.2f}%")
            
            with metric_col2:
                if levels and levels['flip_level']:
                    st.metric("Flip Level", f"${levels['flip_level']:.2f}")
                else:
                    st.metric("Flip Level", "N/A")
            
            with metric_col3:
                if levels and levels['call_wall'] is not None:
                    st.metric("Call Wall", f"${levels['call_wall']['strike']:.2f}")
                else:
                    st.metric("Call Wall", "N/A")
            
            with metric_col4:
                if levels and levels['put_wall'] is not None:
                    st.metric("Put Wall", f"${levels['put_wall']['strike']:.2f}")
                else:
                    st.metric("Put Wall", "N/A")
            
            with metric_col5:
                if levels:
                    st.metric("P/C Ratio", f"{levels['pc_ratio']:.2f}")
                else:
                    st.metric("P/C Ratio", "N/A")
            
            # GEX HeatMap button
            if st.button("🔥 GEX HeatMap", type="secondary", use_container_width=False):
                st.session_state.show_gex_heatmap = not st.session_state.get('show_gex_heatmap', False)
            
            # Display GEX HeatMap if toggled
            if st.session_state.get('show_gex_heatmap', False):
                with st.spinner("Generating GEX HeatMap..."):
                    # Fetch options chain with multiple expiries for heatmap
                    try:
                        client = SchwabClient()
                        query_symbol = symbol.replace('$', '%24')
                        
                        # Get options for next 30 days to capture multiple expiries
                        from_date = datetime.now().strftime('%Y-%m-%d')
                        to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                        
                        chain_params = {
                            'symbol': query_symbol,
                            'contract_type': 'ALL',
                            'from_date': from_date,
                            'to_date': to_date
                        }
                        
                        if symbol in ['$SPX', 'DJX', 'NDX', 'RUT']:
                            chain_params['strike_count'] = 50
                        
                        options_multi_expiry = client.get_options_chain(**chain_params)
                        
                        if options_multi_expiry and 'callExpDateMap' in options_multi_expiry:
                            df_gamma = calculate_gamma_strikes(options_multi_expiry, price, num_expiries=4)
                        else:
                            df_gamma = pd.DataFrame()
                    except Exception as e:
                        logger.error(f"Error fetching multi-expiry options: {e}")
                        df_gamma = pd.DataFrame()
                    
                    if not df_gamma.empty:
                            # Create table format similar to the image
                            st.markdown("---")
                            st.markdown(f"### 📊 {symbol} Gamma Exposure | Last Price: {price:.1f}")
                            
                            # Get unique expiries (limit to 4)
                            expiries = sorted(df_gamma['expiry'].unique())[:4]
                            
                            # Create short expiry labels (MM-DD format)
                            expiry_labels = {}
                            for exp in expiries:
                                try:
                                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                                    expiry_labels[exp] = exp_date.strftime('%m-%d')
                                except:
                                    expiry_labels[exp] = exp[-5:]  # Last 5 chars (MM-DD)
                            
                            # Get all strikes in reasonable range
                            min_strike = price * 0.92  # 8% below
                            max_strike = price * 1.08  # 8% above
                            all_strikes = sorted(df_gamma['strike'].unique())
                            filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
                            
                            if not filtered_strikes:
                                filtered_strikes = sorted(all_strikes, key=lambda x: abs(x - price))[:30]
                            
                            # Create pivot table for display
                            pivot_data = []
                            for strike in filtered_strikes:
                                row = {'Strike': int(strike)}
                                
                                # Get gamma for each expiry
                                for exp in expiries:
                                    mask = (df_gamma['strike'] == strike) & (df_gamma['expiry'] == exp)
                                    exp_data = df_gamma[mask]
                                    
                                    if not exp_data.empty:
                                        # Sum signed notional gamma for all options at this strike/expiry
                                        gamma_sum = exp_data['signed_notional_gamma'].sum()
                                        # Format: show in thousands (K)
                                        if abs(gamma_sum) >= 1000:
                                            row[expiry_labels[exp]] = f"{gamma_sum/1000:.1f}K"
                                        else:
                                            row[expiry_labels[exp]] = f"{gamma_sum:.0f}"
                                    else:
                                        row[expiry_labels[exp]] = "0"
                                
                                pivot_data.append(row)
                            
                            # Create DataFrame
                            df_display = pd.DataFrame(pivot_data)
                            
                            # Create styled display with color coding
                            def color_gamma(val):
                                """Apply color based on gamma value"""
                                if isinstance(val, str):
                                    # Parse the value
                                    val_str = val.replace('K', '').replace('M', '')
                                    try:
                                        num_val = float(val_str)
                                        if 'K' in val:
                                            num_val *= 1000
                                        elif 'M' in val:
                                            num_val *= 1000000
                                    except:
                                        return ''
                                    
                                    # Color logic: positive = green (support), negative = red (resistance)
                                    if num_val > 50000:
                                        return 'background-color: #00ff00; color: black; font-weight: bold'
                                    elif num_val > 10000:
                                        return 'background-color: #90EE90; color: black'
                                    elif num_val < -50000:
                                        return 'background-color: #ff0000; color: white; font-weight: bold'
                                    elif num_val < -10000:
                                        return 'background-color: #ff6b6b; color: white'
                                    elif abs(num_val) < 1000:
                                        return 'background-color: #2b2b2b; color: #888'
                                return ''
                            
                            def highlight_current_price(row):
                                """Highlight row closest to current price"""
                                strike = row['Strike']
                                if abs(strike - price) <= price * 0.01:  # Within 1%
                                    return ['background-color: yellow; color: black; font-weight: bold'] * len(row)
                                return [''] * len(row)
                            
                            # Apply styling
                            styled_df = df_display.style.applymap(
                                color_gamma, 
                                subset=[col for col in df_display.columns if col != 'Strike']
                            ).apply(highlight_current_price, axis=1)
                            
                            # Create column config with fixed widths
                            col_config = {
                                'Strike': st.column_config.NumberColumn(
                                    'Strike', 
                                    format="%d",
                                    width='small'
                                )
                            }
                            # Add config for each expiry column with compact width
                            for label in expiry_labels.values():
                                col_config[label] = st.column_config.TextColumn(
                                    label,
                                    width='small'
                                )
                            
                            # Display the table
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=600,
                                column_config=col_config
                            )
                            
                            st.caption("🟢 Green: Positive → Support | 🔴 Red: Negative → Resistance | 🟡 Yellow: Current Price")
                    else:
                        st.warning("No gamma data available for heatmap")
            
            # Expiry info
            is_0dte = expiry == datetime.now().date()
            expiry_label = "0DTE" if is_0dte else f"Weekly ({expiry.strftime('%b %d')})"
            st.caption(f"📅 Expiration: {expiry_label} | 🕐 Last updated: {snap['fetched_at'].strftime('%H:%M:%S')}")
            
            # ===== THREE VISUALIZATIONS BELOW CHART =====
            st.markdown("---")
            st.markdown("### 📊 Advanced Analytics")
            
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                st.markdown("#### 💎 Net GEX ($)")
                gex_chart = create_net_gex_chart(levels, price, symbol)
                if gex_chart:
                    st.plotly_chart(gex_chart, use_container_width=True, key="net_gex_chart")
                else:
                    st.info("GEX data not available")
            
            with viz_col2:
                st.markdown("#### 📈 Net Volume Profile")
                volume_chart = create_volume_profile_chart(levels, price, symbol)
                if volume_chart:
                    st.plotly_chart(volume_chart, use_container_width=True, key="volume_profile_chart")
                else:
                    st.info("Volume profile not available")
            
            with viz_col3:
                st.markdown("#### 💰 Net Premium Flow")
                premium_chart = create_net_premium_heatmap(snap['options_chain'], price, num_expiries=4)
                if premium_chart:
                    st.plotly_chart(premium_chart, use_container_width=True, key="premium_flow_chart")
                else:
                    st.info("Premium flow not available")
        else:
            st.error(f"Unable to load data for {symbol}")

with left_col:
    # Live watchlist
    live_watchlist()
