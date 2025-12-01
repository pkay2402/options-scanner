#!/usr/bin/env python3
"""
Command Center - 30 Stock Watchlist Dashboard
Actionable intelligence combining gamma, flows, dark pool, and technical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import concurrent.futures  # <--- Added for parallel execution
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient
from src.utils.dark_pool import get_7day_dark_pool_sentiment

# Page config
st.set_page_config(
    page_title="Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stock-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .stock-card.bullish {
        border-left-color: #22c55e;
        background: linear-gradient(to right, #f0fdf4 0%, white 100%);
    }
    
    .stock-card.neutral {
        border-left-color: #fbbf24;
        background: linear-gradient(to right, #fffbeb 0%, white 100%);
    }
    
    .stock-card.bearish {
        border-left-color: #ef4444;
        background: linear-gradient(to right, #fef2f2 0%, white 100%);
    }
    
    .score-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .score-bull {
        background: #22c55e;
        color: white;
    }
    
    .score-neutral {
        background: #fbbf24;
        color: white;
    }
    
    .score-bear {
        background: #ef4444;
        color: white;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    
    .metric-item {
        padding: 8px;
        background: #f9fafb;
        border-radius: 5px;
        font-size: 0.85em;
    }
    
    .action-box {
        background: #eff6ff;
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    
    .summary-metric {
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        'PLTR', 'COIN', 'SNOW', 'CRWD', 'APP',
        'SPY', 'QQQ', 'IWM', 'DIA', 'XOM',
        'JPM', 'BAC', 'GS', 'V', 'MA',
        'UNH', 'JNJ', 'PG', 'WMT', 'HD'
    ]

if 'watchlist_data' not in st.session_state:
    st.session_state.watchlist_data = {}

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Logger
logger = logging.getLogger(__name__)

def get_next_friday():
    """Get next Friday for weekly options expiry"""
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.cache_data(ttl=300)
def get_stock_price_history(symbol):
    """Fetch 30-day daily price history for chart"""
    try:
        client = SchwabClient()
        
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(days=30)).timestamp() * 1000)
        
        price_history = client.get_price_history(
            symbol=symbol,
            frequency_type='daily',
            frequency=1,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        return price_history
        
    except Exception as e:
        logger.error(f"Error fetching price history for {symbol}: {e}")
        return None

def create_compact_intraday_chart(price_history, underlying_price, symbol, call_wall=None, put_wall=None, max_gex=None):
    """Create 30-day daily chart with MACD crossovers and key levels"""
    try:
        if not price_history or 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        if df.empty:
            return None
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
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
        
        # Calculate 21 EMA
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ema21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='#ff9800', width=2),
            showlegend=False
        ))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        df['macd'] = macd
        df['signal'] = signal
        df['macd_prev'] = df['macd'].shift(1)
        df['signal_prev'] = df['signal'].shift(1)
        
        # Detect crossovers
        bullish_cross = (df['macd'] > df['signal']) & (df['macd_prev'] <= df['signal_prev'])
        bearish_cross = (df['macd'] < df['signal']) & (df['macd_prev'] >= df['signal_prev'])
        
        # Add bullish crossover markers
        if bullish_cross.any():
            bull_dates = df.loc[bullish_cross, 'datetime']
            bull_prices = df.loc[bullish_cross, 'low'] * 0.995
            fig.add_trace(go.Scatter(
                x=bull_dates,
                y=bull_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', color='#22c55e', size=14, line=dict(color='white', width=1)),
                name='MACD Bull Cross',
                showlegend=False,
                hovertemplate='<b>MACD Bullish Cross</b><br>%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Add bearish crossover markers
        if bearish_cross.any():
            bear_dates = df.loc[bearish_cross, 'datetime']
            bear_prices = df.loc[bearish_cross, 'high'] * 1.005
            fig.add_trace(go.Scatter(
                x=bear_dates,
                y=bear_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', color='#ef4444', size=14, line=dict(color='white', width=1)),
                name='MACD Bear Cross',
                showlegend=False,
                hovertemplate='<b>MACD Bearish Cross</b><br>%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Add level lines
        if call_wall:
            fig.add_hline(
                y=call_wall,
                line_dash="dot",
                line_color="#22c55e",
                line_width=2,
                annotation_text=f"Call ${call_wall:.0f}",
                annotation_position="right",
                annotation=dict(font_size=9)
            )
        
        if put_wall:
            fig.add_hline(
                y=put_wall,
                line_dash="dot",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f"Put ${put_wall:.0f}",
                annotation_position="right",
                annotation=dict(font_size=9)
            )
        
        if max_gex:
            fig.add_hline(
                y=max_gex,
                line_dash="solid",
                line_color="#a855f7",
                line_width=2,
                annotation_text=f"GEX ${max_gex:.0f}",
                annotation_position="right",
                annotation=dict(font_size=9)
            )
        
        # Current price line
        fig.add_hline(
            y=underlying_price,
            line_dash="dash",
            line_color="#ffd700",
            line_width=2,
            annotation_text=f"${underlying_price:.2f}",
            annotation_position="left",
            annotation=dict(font_size=10, bgcolor="rgba(255,215,0,0.8)")
        )
        
        fig.update_layout(
            height=350,
            template='plotly_white',
            margin=dict(t=10, r=50, l=40, b=20),
            xaxis=dict(
                type='date',
                tickformat='%b %d',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis=dict(
                tickformat='$.2f',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_stock_gamma_data(symbol):
    """Fetch gamma levels and walls for a stock"""
    try:
        client = SchwabClient()
        
        # Get quote
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
        
        underlying_price = quote[symbol]['quote']['lastPrice']
        volume = quote[symbol]['quote'].get('totalVolume', 0)
        
        # Get options chain
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=30
        )
        
        if not options_data:
            return None
        
        # Calculate gamma exposure
        gamma_strikes = []
        call_volumes = {}
        put_volumes = {}
        
        for option_type in ['callExpDateMap', 'putExpDateMap']:
            if option_type not in options_data:
                continue
            
            for exp_date, strikes_data in options_data[option_type].items():
                for strike_str, contracts in strikes_data.items():
                    if not contracts:
                        continue
                    
                    strike = float(strike_str)
                    contract = contracts[0]
                    
                    gamma = contract.get('gamma', 0)
                    oi = contract.get('openInterest', 0)
                    vol = contract.get('totalVolume', 0)
                    
                    if gamma > 0 and oi > 0:
                        gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_strikes.append({
                            'strike': strike,
                            'type': 'CALL' if 'call' in option_type else 'PUT',
                            'gamma': gamma,
                            'gex': abs(gex),
                            'oi': oi,
                            'volume': vol
                        })
                    
                    # Track volumes for walls
                    if 'call' in option_type:
                        call_volumes[strike] = call_volumes.get(strike, 0) + vol
                    else:
                        put_volumes[strike] = put_volumes.get(strike, 0) + vol
        
        if not gamma_strikes:
            return None
        
        # Find max GEX
        gamma_df = pd.DataFrame(gamma_strikes)
        max_gex_row = gamma_df.loc[gamma_df['gex'].idxmax()]
        
        # Find walls
        call_wall = max(call_volumes.items(), key=lambda x: x[1]) if call_volumes else (None, 0)
        put_wall = max(put_volumes.items(), key=lambda x: x[1]) if put_volumes else (None, 0)
        
        return {
            'price': underlying_price,
            'volume': volume,
            'max_gex_strike': max_gex_row['strike'],
            'max_gex_value': max_gex_row['gex'],
            'call_wall_strike': call_wall[0],
            'call_wall_volume': call_wall[1],
            'put_wall_strike': put_wall[0],
            'put_wall_volume': put_wall[1],
            'total_call_volume': sum(call_volumes.values()),
            'total_put_volume': sum(put_volumes.values())
        }
        
    except Exception as e:
        logger.error(f"Error fetching gamma data for {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def get_stock_flow_data(symbol):
    """Fetch whale flow and fresh OI data"""
    try:
        client = SchwabClient()
        expiry = get_next_friday()
        
        # Get quote
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
        
        underlying_price = quote[symbol]['quote']['lastPrice']
        underlying_volume = quote[symbol]['quote'].get('totalVolume', 0)
        
        if underlying_volume == 0:
            return None
        
        # Get options
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=expiry.strftime('%Y-%m-%d'),
            to_date=expiry.strftime('%Y-%m-%d')
        )
        
        if not options_data:
            return None
        
        call_whale_scores = []
        put_whale_scores = []
        call_oi_flows = []
        put_oi_flows = []
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if not contracts:
                        continue
                    
                    strike = float(strike_str)
                    contract = contracts[0]
                    
                    volume = contract.get('totalVolume', 0)
                    oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                    mark = contract.get('mark', contract.get('last', 1))
                    delta = contract.get('delta', 0)
                    ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                    
                    if volume == 0 or mark == 0:
                        continue
                    
                    # Whale score (ATM ±5%)
                    if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                        leverage = delta * underlying_price
                        leverage_ratio = leverage / mark
                        valr = leverage_ratio * ivol
                        vol_oi = volume / oi
                        dvolume_opt = volume * mark * 100
                        dvolume_und = underlying_price * underlying_volume
                        dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                        whale_score = valr * vol_oi * dvolume_ratio * 1000
                        
                        call_whale_scores.append(whale_score)
                    
                    # OI flow (±10%, Vol/OI >= 3.0)
                    if abs(strike - underlying_price) / underlying_price <= 0.10:
                        vol_oi_ratio = volume / oi
                        if vol_oi_ratio >= 3.0:
                            notional = volume * mark * 100
                            call_oi_flows.append({
                                'strike': strike,
                                'vol_oi': vol_oi_ratio,
                                'notional': notional
                            })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if not contracts:
                        continue
                    
                    strike = float(strike_str)
                    contract = contracts[0]
                    
                    volume = contract.get('totalVolume', 0)
                    oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                    mark = contract.get('mark', contract.get('last', 1))
                    delta = contract.get('delta', 0)
                    ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                    
                    if volume == 0 or mark == 0:
                        continue
                    
                    # Whale score (ATM ±5%)
                    if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                        leverage = abs(delta) * underlying_price
                        leverage_ratio = leverage / mark
                        valr = leverage_ratio * ivol
                        vol_oi = volume / oi
                        dvolume_opt = volume * mark * 100
                        dvolume_und = underlying_price * underlying_volume
                        dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                        whale_score = valr * vol_oi * dvolume_ratio * 1000
                        
                        put_whale_scores.append(whale_score)
                    
                    # OI flow (±10%, Vol/OI >= 3.0)
                    if abs(strike - underlying_price) / underlying_price <= 0.10:
                        vol_oi_ratio = volume / oi
                        if vol_oi_ratio >= 3.0:
                            notional = volume * mark * 100
                            put_oi_flows.append({
                                'strike': strike,
                                'vol_oi': vol_oi_ratio,
                                'notional': notional
                            })
        
        return {
            'call_whale_score': max(call_whale_scores) if call_whale_scores else 0,
            'put_whale_score': max(put_whale_scores) if put_whale_scores else 0,
            'call_oi_flows': call_oi_flows,
            'put_oi_flows': put_oi_flows,
            'max_call_vol_oi': max([f['vol_oi'] for f in call_oi_flows]) if call_oi_flows else 0,
            'max_put_vol_oi': max([f['vol_oi'] for f in put_oi_flows]) if put_oi_flows else 0,
            'call_notional': sum([f['notional'] for f in call_oi_flows]),
            'put_notional': sum([f['notional'] for f in put_oi_flows])
        }
        
    except Exception as e:
        logger.error(f"Error fetching flow data for {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def get_stock_technicals(symbol):
    """Get EMA and technical data using yfinance"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None
        
        # Calculate EMAs
        ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
        ema_21 = hist['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
        ema_50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        
        current_price = hist['Close'].iloc[-1]
        
        # Determine trend
        if current_price > ema_8 > ema_21 > ema_50:
            trend = "Strong Uptrend"
        elif current_price < ema_8 < ema_21 < ema_50:
            trend = "Strong Downtrend"
        elif current_price > ema_50 > ema_200:
            trend = "Uptrend"
        elif current_price < ema_50 < ema_200:
            trend = "Downtrend"
        else:
            trend = "Mixed/Ranging"
        
        return {
            'ema_8': ema_8,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'ema_200': ema_200,
            'trend': trend
        }
        
    except Exception as e:
        logger.error(f"Error fetching technicals for {symbol}: {e}")
        return None

def calculate_bull_bear_score(stock_data):
    """Calculate 0-100 bull/bear score"""
    score = 50  # Start neutral
    
    price = stock_data.get('price', 0)
    max_gex = stock_data.get('max_gex_strike', price)
    call_wall = stock_data.get('call_wall_strike', price)
    put_wall = stock_data.get('put_wall_strike', price)
    
    # Price vs Max GEX (+15/-15)
    if price > max_gex:
        score += 15
    else:
        score -= 15
    
    # Fresh OI Direction (+20/-20)
    call_notional = stock_data.get('call_notional', 0)
    put_notional = stock_data.get('put_notional', 0)
    
    if call_notional > put_notional * 1.5:
        score += 20
    elif put_notional > call_notional * 1.5:
        score -= 20
    
    # Whale Score Direction (+10/-10)
    call_whale = stock_data.get('call_whale_score', 0)
    put_whale = stock_data.get('put_whale_score', 0)
    
    if call_whale > put_whale:
        score += 10
    else:
        score -= 10
    
    # Dark Pool Sentiment (+10/-10)
    dark_pool_ratio = stock_data.get('dark_pool_ratio', 0.5)
    if dark_pool_ratio > 0.55:
        score += 10
    elif dark_pool_ratio < 0.45:
        score -= 10
    
    # P/C Ratio (+10/-10)
    pc_ratio = stock_data.get('pc_ratio', 1.0)
    if pc_ratio < 0.8:
        score += 10
    elif pc_ratio > 1.2:
        score -= 10
    
    # EMA Positioning (+15/-15)
    ema_8 = stock_data.get('ema_8', 0)
    ema_21 = stock_data.get('ema_21', 0)
    ema_50 = stock_data.get('ema_50', 0)
    
    if price > ema_8 > ema_21 > ema_50:
        score += 15
    elif price < ema_8 < ema_21 < ema_50:
        score -= 15
    
    # Distance from Walls (+10/-10)
    if call_wall and put_wall:
        call_dist = abs(price - call_wall)
        put_dist = abs(price - put_wall)
        
        if call_dist < put_dist:
            score -= 10  # Closer to resistance
        else:
            score += 10  # Closer to support
    
    # Vol/OI Extreme (+10)
    max_vol_oi = max(
        stock_data.get('max_call_vol_oi', 0),
        stock_data.get('max_put_vol_oi', 0)
    )
    if max_vol_oi > 5.0:
        score += 10
    
    return max(0, min(100, score))

def generate_action_recommendation(stock_data, score):
    """Generate actionable trading recommendation"""
    symbol = stock_data.get('symbol', '')
    price = stock_data.get('price', 0)
    max_gex = stock_data.get('max_gex_strike', price)
    call_wall = stock_data.get('call_wall_strike', price)
    put_wall = stock_data.get('put_wall_strike', price)
    
    if score >= 70:
        # Bullish setup
        target = call_wall if call_wall and call_wall > price else price * 1.03
        stop = max(put_wall if put_wall and put_wall < price else price * 0.97, max_gex * 0.98)
        entry_strike = (price + target) / 2
        
        risk_reward = (target - price) / (price - stop) if stop < price else 1.0
        
        return {
            'bias': 'BULLISH',
            'action': f'BUY ${entry_strike:.2f} CALLS',
            'logic': f'Above Max GEX (${max_gex:.2f}), fresh call flows, weak resistance',
            'target': f'${target:.2f}',
            'stop': f'${stop:.2f}',
            'risk_reward': f'{risk_reward:.1f}:1'
        }
    
    elif score <= 30:
        # Bearish setup
        target = put_wall if put_wall and put_wall < price else price * 0.97
        stop = min(call_wall if call_wall and call_wall > price else price * 1.03, max_gex * 1.02)
        entry_strike = (price + target) / 2
        
        risk_reward = (price - target) / (stop - price) if stop > price else 1.0
        
        return {
            'bias': 'BEARISH',
            'action': f'BUY ${entry_strike:.2f} PUTS',
            'logic': f'Below Max GEX (${max_gex:.2f}), fresh put flows, weak support',
            'target': f'${target:.2f}',
            'stop': f'${stop:.2f}',
            'risk_reward': f'{risk_reward:.1f}:1'
        }
    
    else:
        # Neutral - range trade
        return {
            'bias': 'NEUTRAL',
            'action': 'RANGE TRADE or WAIT',
            'logic': f'Price between walls, pin at Max GEX (${max_gex:.2f})',
            'target': 'N/A',
            'stop': 'N/A',
            'risk_reward': 'N/A'
        }

def display_stock_card(symbol, data, score):
    """Display comprehensive stock card using Streamlit components"""
    
    # Determine color and emoji
    if score >= 70:
        emoji = "🟢"
        score_color = "green"
        container_type = "success"
    elif score <= 30:
        emoji = "🔴"
        score_color = "red"
        container_type = "error"
    else:
        emoji = "🟡"
        score_color = "orange"
        container_type = "warning"
    
    # Get action recommendation
    action = generate_action_recommendation(data, score)
    
    # Calculate price change
    price = data.get('price', 0)
    price_change = ((price - data.get('ema_21', price)) / data.get('ema_21', price) * 100) if data.get('ema_21') else 0
    
    # Create container with appropriate styling
    with st.container():
        # Header row
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {emoji} {symbol} ${price:.2f} ({price_change:+.1f}%)")
        
        with col2:
            st.markdown(f"**:{score_color}[{score}/100]**")
        
        # Main content: Chart on left, Metrics on right
        chart_col, metrics_col = st.columns([2, 1])
        
        with chart_col:
            # Create and display chart
            price_history = data.get('price_history')
            if price_history:
                chart = create_compact_intraday_chart(
                    price_history,
                    price,
                    symbol,
                    call_wall=data.get('call_wall_strike'),
                    put_wall=data.get('put_wall_strike'),
                    max_gex=data.get('max_gex_strike')
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"chart_{symbol}")
                else:
                    st.caption("📊 Chart unavailable")
            else:
                st.caption("📊 No price history available")
        
        with metrics_col:
            # Metrics in vertical layout
            st.metric(
                "Max GEX", 
                f"${data.get('max_gex_strike', 0):.2f}",
                delta="Above" if price > data.get('max_gex_strike', 0) else "Below",
                delta_color="normal" if price > data.get('max_gex_strike', 0) else "inverse"
            )
            
            st.metric(
                "Call Wall",
                f"${data.get('call_wall_strike', 0):.2f}",
                f"{data.get('call_wall_volume', 0):,.0f} vol"
            )
            
            st.metric(
                "Put Wall",
                f"${data.get('put_wall_strike', 0):.2f}",
                f"{data.get('put_wall_volume', 0):,.0f} vol"
            )
            
            pc_ratio = data.get('pc_ratio', 0)
            pc_sentiment = 'Bullish' if pc_ratio < 0.8 else 'Bearish' if pc_ratio > 1.2 else 'Neutral'
            st.metric("P/C Ratio", f"{pc_ratio:.2f}", pc_sentiment)
            
            dark_pool_ratio = data.get('dark_pool_ratio', 0.5)
            dp_sentiment = 'Bullish' if dark_pool_ratio > 0.55 else 'Bearish' if dark_pool_ratio < 0.45 else 'Neutral'
            st.metric("Dark Pool", f"{dark_pool_ratio*100:.0f}%", dp_sentiment)
            
            max_flow = max(data.get('max_call_vol_oi', 0), data.get('max_put_vol_oi', 0))
            flow_type = "Call" if data.get('max_call_vol_oi', 0) > data.get('max_put_vol_oi', 0) else "Put"
            st.metric("Max Flow", f"{max_flow:.1f}x", flow_type)
        
        # Additional info below
        st.caption(f"**Trend:** {data.get('trend', 'Unknown')} | **Whale Scores:** Calls {data.get('call_whale_score', 0):.0f} / Puts {data.get('put_whale_score', 0):.0f}")
        
        # Action recommendation
        st.info(f"""
**💡 {action['bias']} SETUP**

**Action:** {action['action']}  
**Logic:** {action['logic']}  
**Target:** {action['target']} | **Stop:** {action['stop']} | **R/R:** {action['risk_reward']}
        """)
        
        st.divider()

def process_stock_data(symbol):
    """Helper function to process a single stock for threading"""
    try:
        # Fetch all data
        gamma_data = get_stock_gamma_data(symbol)
        
        # Return None if Gamma data failed (critical dependency)
        if not gamma_data:
            return None
            
        flow_data = get_stock_flow_data(symbol)
        tech_data = get_stock_technicals(symbol)
        price_history = get_stock_price_history(symbol)  # Add price history
        
        # Get dark pool
        try:
            dark_pool = get_7day_dark_pool_sentiment(symbol)
            dark_pool_ratio = dark_pool['ratio']
        except:
            dark_pool_ratio = 0.5
        
        # Combine data
        stock_data = {
            'symbol': symbol,
            'price': gamma_data['price'],
            'max_gex_strike': gamma_data['max_gex_strike'],
            'max_gex_value': gamma_data['max_gex_value'],
            'call_wall_strike': gamma_data['call_wall_strike'],
            'call_wall_volume': gamma_data['call_wall_volume'],
            'put_wall_strike': gamma_data['put_wall_strike'],
            'put_wall_volume': gamma_data['put_wall_volume'],
            'total_call_volume': gamma_data['total_call_volume'],
            'total_put_volume': gamma_data['total_put_volume'],
            'pc_ratio': gamma_data['total_put_volume'] / max(gamma_data['total_call_volume'], 1),
            'price_history': price_history  # Add price history
        }
        
        if flow_data:
            stock_data.update({
                'call_whale_score': flow_data['call_whale_score'],
                'put_whale_score': flow_data['put_whale_score'],
                'max_call_vol_oi': flow_data['max_call_vol_oi'],
                'max_put_vol_oi': flow_data['max_put_vol_oi'],
                'call_notional': flow_data['call_notional'],
                'put_notional': flow_data['put_notional']
            })
        
        if tech_data:
            stock_data.update({
                'ema_8': tech_data['ema_8'],
                'ema_21': tech_data['ema_21'],
                'ema_50': tech_data['ema_50'],
                'ema_200': tech_data['ema_200'],
                'trend': tech_data['trend']
            })
        
        stock_data['dark_pool_ratio'] = dark_pool_ratio
        
        # Calculate score
        score = calculate_bull_bear_score(stock_data)
        stock_data['score'] = score
        
        return stock_data
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def scan_watchlist(symbols):
    """Scan all symbols and compile data using parallel execution"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    # Adjust max_workers if you hit API rate limits (8 is usually safe for Schwab)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(process_stock_data, symbol): symbol for symbol in symbols}
        
        completed = 0
        total = len(symbols)
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            
            # Update UI
            status_text.text(f"Scanning {symbol} (completed {completed}/{total})...")
            progress_bar.progress(completed / total)
            
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"Thread error for {symbol}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

# Main app
st.title("🎯 Command Center - 30 Stock Watchlist")
st.caption(f"Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'Never'}")

# Settings at the top
st.markdown("## ⚙️ Settings")

col_settings1, col_settings2 = st.columns([2, 1])

with col_settings1:
    # Watchlist editor
    st.markdown("### 📝 Watchlist Manager")
    new_watchlist = st.text_area(
        "Edit Watchlist (comma-separated symbols)",
        value=", ".join(st.session_state.watchlist),
        height=150,
        help="Add or remove stock symbols. Separate with commas."
    )
    
    col_update, col_scan = st.columns(2)
    
    with col_update:
        if st.button("💾 Update Watchlist", use_container_width=True):
            st.session_state.watchlist = [s.strip().upper() for s in new_watchlist.split(',')]
            st.success(f"Updated watchlist: {len(st.session_state.watchlist)} symbols")
    
    with col_scan:
        if st.button("🔄 Scan Watchlist", type="primary", use_container_width=True):
            with st.spinner("Scanning all stocks..."):
                st.session_state.watchlist_data = scan_watchlist(st.session_state.watchlist)
                st.session_state.last_update = datetime.now()
                st.success(f"Scanned {len(st.session_state.watchlist_data)} stocks!")
                st.rerun()

with col_settings2:
    # Individual stock scanner
    st.markdown("### 🔍 Scan Individual Stock")
    individual_symbol = st.text_input(
        "Enter Symbol",
        placeholder="e.g., AAPL",
        help="Scan a single stock without adding to watchlist"
    ).upper()
    
    if st.button("🎯 Scan Stock", type="secondary", use_container_width=True):
        if individual_symbol:
            with st.spinner(f"Scanning {individual_symbol}..."):
                stock_data = process_stock_data(individual_symbol)
                if stock_data:
                    st.success(f"Scanned {individual_symbol}!")
                    # Add to temporary display
                    if not st.session_state.watchlist_data:
                        st.session_state.watchlist_data = [stock_data]
                    else:
                        # Check if already exists, replace if so
                        existing_symbols = [s['symbol'] for s in st.session_state.watchlist_data]
                        if individual_symbol in existing_symbols:
                            st.session_state.watchlist_data = [
                                s for s in st.session_state.watchlist_data if s['symbol'] != individual_symbol
                            ]
                        st.session_state.watchlist_data.append(stock_data)
                    st.session_state.last_update = datetime.now()
                    st.rerun()
                else:
                    st.error(f"Failed to scan {individual_symbol}. Check symbol or try again.")
        else:
            st.warning("Please enter a symbol")

st.markdown("---")

# Filter controls at the top
st.markdown("### 🔍 Filters & Sorting")

col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    bias_filter = st.selectbox(
        "Bias Filter",
        ["All", "Bullish (70+)", "Neutral (40-69)", "Bearish (<40)"]
    )

with col_filter2:
    min_flow = st.slider(
        "Min Vol/OI Ratio",
        0.0, 10.0, 0.0, 0.5
    )

with col_filter3:
    sort_by = st.selectbox(
        "Sort By",
        ["Score (High to Low)", "Score (Low to High)", "Symbol (A-Z)", "Price", "Flow Activity"]
    )

st.markdown("---")

# Main content
if not st.session_state.watchlist_data:
    st.info("👆 Click 'Scan Watchlist' in the sidebar to begin analysis")
    st.stop()

# Convert to dataframe for easier filtering
df = pd.DataFrame(st.session_state.watchlist_data)

# Apply filters
if bias_filter == "Bullish (70+)":
    df = df[df['score'] >= 70]
elif bias_filter == "Neutral (40-69)":
    df = df[(df['score'] >= 40) & (df['score'] < 70)]
elif bias_filter == "Bearish (<40)":
    df = df[df['score'] < 40]

if min_flow > 0:
    df = df[
        (df['max_call_vol_oi'] >= min_flow) | 
        (df['max_put_vol_oi'] >= min_flow)
    ]

# Apply sorting
if sort_by == "Score (High to Low)":
    df = df.sort_values('score', ascending=False)
elif sort_by == "Score (Low to High)":
    df = df.sort_values('score', ascending=True)
elif sort_by == "Symbol (A-Z)":
    df = df.sort_values('symbol')
elif sort_by == "Price":
    df = df.sort_values('price', ascending=False)
elif sort_by == "Flow Activity":
    df['max_flow'] = df[['max_call_vol_oi', 'max_put_vol_oi']].max(axis=1)
    df = df.sort_values('max_flow', ascending=False)

# Summary metrics
st.markdown("## 📊 Summary Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

bullish_count = len(df[df['score'] >= 70])
neutral_count = len(df[(df['score'] >= 40) & (df['score'] < 70)])
bearish_count = len(df[df['score'] < 40])
avg_score = df['score'].mean() if not df.empty else 0
total_call_flow = df['call_notional'].sum() if 'call_notional' in df.columns else 0
total_put_flow = df['put_notional'].sum() if 'put_notional' in df.columns else 0

with col1:
    st.metric("🟢 Bullish", bullish_count, help="Stocks with score >= 70")

with col2:
    st.metric("🟡 Neutral", neutral_count, help="Stocks with score 40-69")

with col3:
    st.metric("🔴 Bearish", bearish_count, help="Stocks with score < 40")

with col4:
    st.metric("Avg Score", f"{avg_score:.0f}/100", help="Average bull/bear score")

with col5:
    st.metric("Call Flow", f"${total_call_flow/1e6:.1f}M", help="Total call notional")

with col6:
    st.metric("Put Flow", f"${total_put_flow/1e6:.1f}M", help="Total put notional")

st.markdown("---")

# Display stocks
st.markdown(f"## 📈 Stocks ({len(df)} shown)")

if df.empty:
    st.warning("No stocks match your filter criteria")
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Stocks", 
        "🎯 Top Opportunities", 
        "🟢 Bullish Only",
        "🔴 Bearish Only"
    ])
    
    with tab1:
        for _, row in df.iterrows():
            display_stock_card(row['symbol'], row.to_dict(), row['score'])
    
    with tab2:
        st.markdown("### 🚀 Top 10 Opportunities (Highest Scores)")
        top_10 = df.nlargest(10, 'score')
        for _, row in top_10.iterrows():
            display_stock_card(row['symbol'], row.to_dict(), row['score'])
    
    with tab3:
        bullish = df[df['score'] >= 70]
        if bullish.empty:
            st.info("No bullish setups found")
        else:
            for _, row in bullish.iterrows():
                display_stock_card(row['symbol'], row.to_dict(), row['score'])
    
    with tab4:
        bearish = df[df['score'] < 40]
        if bearish.empty:
            st.info("No bearish setups found")
        else:
            for _, row in bearish.iterrows():
                display_stock_card(row['symbol'], row.to_dict(), row['score'])

# Export option
st.markdown("---")
if st.button("📥 Export to CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"command_center_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def main():
    """Main execution function"""
    pass  # All logic is in module-level code above


if __name__ == "__main__":
    main()
