"""
0DTE by Index - Modern, Data-Dense Layout
Ultra-fast decision making with key metrics at a glance
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
import time

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="0DTE by Index",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-metric {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    .metric-label {
        font-size: 11px;
        opacity: 0.9;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 900;
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 12px;
        opacity: 0.85;
    }
    .level-card {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid;
    }
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }
    .compact-table {
        font-size: 11px;
        width: 100%;
    }
    .compact-table td {
        padding: 4px 8px;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'auto_refresh_byindex' not in st.session_state:
    st.session_state.auto_refresh_byindex = True
if 'last_refresh_byindex' not in st.session_state:
    st.session_state.last_refresh_byindex = datetime.now()
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'SPY'

def get_next_friday():
    """Get next Friday for weekly options expiry"""
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.fragment(run_every="120s")
def live_watchlist_table():
    """Auto-refreshing table showing multiple symbols - updates every 120 seconds"""
    watchlist = ['SPY', 'QQQ', 'PLTR', 'CRWD', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'NBIS', 'MSFT', 'GOOGL', 'NFLX', 'OKLO', 'GS','TEM','COIN']
    next_friday = get_next_friday()
    exp_date_str = next_friday.strftime('%Y-%m-%d')
    
    st.markdown(f"### 📊 Live Watchlist - Weekly Expiry ({next_friday.strftime('%b %d')})")
    st.caption(f"🔄 Auto-updates every 120s • Last: {datetime.now().strftime('%H:%M:%S')}")
    
    table_data = []
    
    for symbol in watchlist:
        try:
            snap = get_market_snapshot(symbol, exp_date_str)
            if snap and snap.get('underlying_price'):
                price = snap['underlying_price']
                quote_data = snap['quote']
                
                # Get daily change
                prev_close = quote_data.get(symbol.replace('$', ''), {}).get('quote', {}).get('closePrice', price)
                daily_change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                
                # Calculate analysis
                ana = calculate_comprehensive_analysis(snap['options_chain'], price)
                
                if ana:
                    table_data.append({
                        'Symbol': symbol,
                        'Price': f"${price:.2f}",
                        '% Change': daily_change_pct,  # Keep as number for sorting
                        'Flip Level': f"${ana['flip_level']:.2f}" if ana['flip_level'] else "-",
                        'Call Wall': f"${ana['call_wall']['strike']:.2f}" if ana['call_wall'] is not None else "-",
                        'Put Wall': f"${ana['put_wall']['strike']:.2f}" if ana['put_wall'] is not None else "-",
                        'Max GEX': f"${ana['max_gex']['strike']:.2f}" if ana['max_gex'] is not None else "-",
                        'P/C': f"{ana['pc_ratio']:.2f}"
                    })
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue
    
    if table_data:
        df = pd.DataFrame(table_data)
        # Sort by % Change (high to low)
        df = df.sort_values('% Change', ascending=False)
        # Format % Change after sorting
        df['% Change'] = df['% Change'].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.warning("Unable to load watchlist data")

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

def calculate_comprehensive_analysis(options_data, underlying_price, strike_spacing=5.0):
    """Calculate all key metrics in one pass"""
    try:
        call_data = {}
        put_data = {}
        
        # Process all options
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in call_data:
                            call_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'delta': 0, 'premium': 0}
                        call_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        call_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        call_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        call_data[strike]['delta'] += contract.get('delta', 0) or 0
                        call_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in put_data:
                            put_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'delta': 0, 'premium': 0}
                        put_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        put_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        put_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        put_data[strike]['delta'] += abs(contract.get('delta', 0) or 0)
                        put_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        # Calculate metrics by strike
        all_strikes = sorted(set(call_data.keys()) | set(put_data.keys()))
        strike_analysis = []
        
        for strike in all_strikes:
            call = call_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'delta': 0, 'premium': 0})
            put = put_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'delta': 0, 'premium': 0})
            
            # GEX calculation
            call_gex = call['gamma'] * call['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = -put['gamma'] * put['oi'] * 100 * underlying_price * underlying_price * 0.01
            net_gex = call_gex + put_gex
            
            # Net volume
            net_volume = put['volume'] - call['volume']
            
            # Distance from current price
            distance = abs(strike - underlying_price)
            distance_pct = (distance / underlying_price) * 100
            
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
                'distance': distance,
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
        
        # Totals
        total_call_vol = df['call_vol'].sum()
        total_put_vol = df['put_vol'].sum()
        total_call_premium = df['call_premium'].sum()
        total_put_premium = df['put_premium'].sum()
        
        return {
            'df': df,
            'call_wall': call_wall,
            'put_wall': put_wall,
            'max_gex': max_gex,
            'flip_level': flip_level,
            'total_call_vol': total_call_vol,
            'total_put_vol': total_put_vol,
            'total_call_premium': total_call_premium,
            'total_put_premium': total_put_premium,
            'pc_ratio': total_put_vol / total_call_vol if total_call_vol > 0 else 0,
            'premium_ratio': total_put_premium / total_call_premium if total_call_premium > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return None

def create_unified_chart(price_history, analysis, underlying_price, symbol):
    """Create a unified chart with price + volume + GEX"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        # Filter to market hours only
        df = df[
            ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
            ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
        ].copy()
        
        if df.empty:
            return None
        
        # Get last 2 days
        df['date'] = df['datetime'].dt.date
        unique_dates = sorted(df['date'].unique(), reverse=True)
        target_dates = unique_dates[:2] if len(unique_dates) >= 2 else unique_dates
        df = df[df['date'].isin(target_dates)].copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create subplots: price chart + volume bar
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} - Price & Key Levels", "Volume Profile"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # EMA
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Price candlesticks
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
        ), row=1, col=1)
        
        # VWAP
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#00bcd4', width=2)
        ), row=1, col=1)
        
        # 21 EMA
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ema21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='#ff9800', width=2)
        ), row=1, col=1)
        
        # --- Calculate MACD on price data and plot crossover arrows on price chart ---
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
                up_times = df.loc[up_cross, 'datetime']
                up_prices = df.loc[up_cross, 'low'] * 0.997  # slightly below low
                fig.add_trace(go.Scatter(
                    x=up_times,
                    y=up_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='#22c55e', size=12),
                    name='MACD Bull Cross',
                    hovertemplate='<b>MACD Bull Cross</b><br>%{x|%I:%M %p}<br>Price: $%{y:.2f}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)

            if down_cross.any():
                down_times = df.loc[down_cross, 'datetime']
                down_prices = df.loc[down_cross, 'high'] * 1.003  # slightly above high
                fig.add_trace(go.Scatter(
                    x=down_times,
                    y=down_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='#ef4444', size=12),
                    name='MACD Bear Cross',
                    hovertemplate='<b>MACD Bear Cross</b><br>%{x|%I:%M %p}<br>Price: $%{y:.2f}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)
        except Exception:
            logger.exception('Error adding MACD cross markers to price chart')
        
        # Add key levels with smart positioning
        levels = []
        if analysis['call_wall'] is not None:
            levels.append({
                'price': analysis['call_wall']['strike'],
                'label': f"CW ${analysis['call_wall']['strike']:.0f}",
                'color': "#f44336"
            })
        if analysis['put_wall'] is not None:
            levels.append({
                'price': analysis['put_wall']['strike'],
                'label': f"PW ${analysis['put_wall']['strike']:.0f}",
                'color': "#4caf50"
            })
        if analysis['flip_level']:
            levels.append({
                'price': analysis['flip_level'],
                'label': f"Flip ${analysis['flip_level']:.0f}",
                'color': "#9c27b0"
            })
        
        # Sort and add levels
        levels.sort(key=lambda x: x['price'])
        position_cycle = ["right", "top right", "bottom right"]
        
        for i, level in enumerate(levels):
            conflicts = sum(1 for j in range(i) if abs(level['price'] - levels[j]['price']) / underlying_price < 0.005)
            position = position_cycle[min(conflicts, len(position_cycle) - 1)]
            
            fig.add_hline(
                y=level['price'],
                line_color=level['color'],
                line_width=2,
                line_dash="solid",
                annotation_text=level['label'],
                annotation_position=position,
                annotation=dict(font=dict(size=10)),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#26a69a' if close >= open else '#ef5350' 
                  for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False,
            opacity=0.7
        ), row=2, col=1)
        
        # Layout
        fig.update_xaxes(
            type='date',
            tickformat='%H:%M',
            rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            row=1, col=1
        )
        
        fig.update_xaxes(
            type='date',
            tickformat='%H:%M',
            rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        
        fig.update_layout(
            height=700,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, r=20, l=60, b=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

# ===== HEADER =====
col_title, col_refresh = st.columns([4, 1])

with col_title:
    st.markdown("# 🎯 0DTE by Index")
    
with col_refresh:
    if st.button("🔄 REFRESH", type="primary", width="stretch"):
        st.cache_data.clear()
        st.session_state.last_refresh_byindex = datetime.now()
        st.rerun()

# Quick stats bar
today = datetime.now().date()
weekday = today.weekday()
if weekday == 5:
    default_expiry = today + timedelta(days=2)
elif weekday == 6:
    default_expiry = today + timedelta(days=1)
else:
    default_expiry = today

now = datetime.now()
expiry_datetime = datetime.combine(default_expiry, datetime.strptime("16:00", "%H:%M").time())
time_to_expiry = expiry_datetime - now
hours_remaining = int(time_to_expiry.total_seconds() / 3600)
minutes_remaining = int((time_to_expiry.total_seconds() % 3600) / 60)

# Quick stats bar - compressed
col_time1, col_time2, col_time3, col_time4 = st.columns(4)
with col_time1:
    st.metric("⏰ To Expiry", f"{hours_remaining}h {minutes_remaining}m", label_visibility="visible")
with col_time2:
    st.metric("📅 Expiry", default_expiry.strftime('%b %d'), label_visibility="visible")
with col_time3:
    refresh_time = st.session_state.last_refresh_byindex.strftime('%H:%M:%S')
    st.metric("🔄 Updated", refresh_time, label_visibility="visible")
with col_time4:
    auto_refresh = st.checkbox("Auto (3min)", value=st.session_state.auto_refresh_byindex)
    st.session_state.auto_refresh_byindex = auto_refresh

st.markdown("---")

# ===== COMPACT SYMBOL SELECTOR + QUICK COMPARE =====
sel_col1, sel_col2, sel_col3, comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1, 1.2, 1.2, 1.2])

with sel_col1:
    if st.button("📊 SPY", type="primary" if st.session_state.selected_symbol == 'SPY' else "secondary", use_container_width=True):
        st.session_state.selected_symbol = 'SPY'
        st.rerun()
with sel_col2:
    if st.button("💻 QQQ", type="primary" if st.session_state.selected_symbol == 'QQQ' else "secondary", use_container_width=True):
        st.session_state.selected_symbol = 'QQQ'
        st.rerun()
with sel_col3:
    if st.button("📈 $SPX", type="primary" if st.session_state.selected_symbol == '$SPX' else "secondary", use_container_width=True):
        st.session_state.selected_symbol = '$SPX'
        st.rerun()

# Quick compare in remaining columns
exp_date_str = default_expiry.strftime('%Y-%m-%d')
for idx, sym in enumerate(['SPY', 'QQQ', '$SPX']):
    with [comp_col1, comp_col2, comp_col3][idx]:
        try:
            snap = get_market_snapshot(sym, exp_date_str)
            if snap:
                price = snap['underlying_price']
                ana = calculate_comprehensive_analysis(snap['options_chain'], price)
                if ana:
                    pc = ana['pc_ratio']
                    net = ana['total_put_vol'] - ana['total_call_vol']
                    st.markdown(f"""
                    <div style="padding: 8px; background: #f5f5f5; border-radius: 6px; text-align: center;">
                        <div style="font-size: 11px; font-weight: 600; opacity: 0.7;">{sym}</div>
                        <div style="font-size: 18px; font-weight: 900; margin: 2px 0;">${price:.2f}</div>
                        <div style="font-size: 9px;">P/C: {pc:.2f} | Net: {int(net/1000):.0f}K</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption(f"{sym}: Loading...")
        except:
            st.caption(f"{sym}: Error")

st.markdown("---")

# ===== MAIN CONTENT =====
selected = st.session_state.selected_symbol

with st.spinner(f"Loading {selected} data..."):
    try:
        exp_date_str = default_expiry.strftime('%Y-%m-%d')
        snapshot = get_market_snapshot(selected, exp_date_str)
        
        if not snapshot:
            st.error(f"Failed to fetch {selected} data")
            st.stop()
        
        underlying_price = snapshot['underlying_price']
        quote_data = snapshot['quote']
        
        # Get daily change
        prev_close = quote_data.get(selected.replace('$', ''), {}).get('quote', {}).get('closePrice', underlying_price)
        daily_change = underlying_price - prev_close
        daily_change_pct = (daily_change / prev_close * 100) if prev_close else 0
        
        # Calculate analysis
        analysis = calculate_comprehensive_analysis(snapshot['options_chain'], underlying_price)
        
        if not analysis:
            st.error("Failed to calculate analysis")
            st.stop()
        
        # Calculate trading signal first
        signal = "NEUTRAL"
        signal_color = "#9e9e9e"
        signal_reason = []
        
        if analysis['flip_level']:
            if underlying_price > analysis['flip_level']:
                signal = "BULLISH"
                signal_color = "#26a69a"
                signal_reason.append("Above flip")
            else:
                signal = "BEARISH"
                signal_color = "#ef5350"
                signal_reason.append("Below flip")
        
        if analysis['pc_ratio'] > 1.2:
            signal_reason.append("High P/C")
        elif analysis['pc_ratio'] < 0.8:
            signal_reason.append("Low P/C")
        
        # ===== TOP METRICS ROW WITH SIGNAL (COLLAPSIBLE) =====
        with st.expander("📊 Key Metrics (Available on Chart)", expanded=False):
            col_m1, col_m2, col_m3, col_m4, col_m5, col_signal = st.columns([1, 1, 1, 1, 1, 1.2])
            
            with col_m1:
                change_color = "#26a69a" if daily_change >= 0 else "#ef5350"
                st.markdown(f"""
                <div class="main-metric">
                    <div class="metric-label">CURRENT PRICE</div>
                    <div class="metric-value">${underlying_price:.2f}</div>
                    <div class="metric-sub" style="color: {change_color}">
                        {daily_change:+.2f} ({daily_change_pct:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                pc_color = "#ef5350" if analysis['pc_ratio'] > 1.0 else "#26a69a"
                st.markdown(f"""
                <div class="main-metric">
                    <div class="metric-label">PUT/CALL RATIO</div>
                    <div class="metric-value" style="color: {pc_color}">{analysis['pc_ratio']:.2f}</div>
                    <div class="metric-sub">Vol: {int(analysis['total_put_vol']):,} / {int(analysis['total_call_vol']):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                net_vol = analysis['total_put_vol'] - analysis['total_call_vol']
                net_color = "#ef5350" if net_vol > 0 else "#26a69a"
                net_icon = "🐻" if net_vol > 0 else "🐂"
                st.markdown(f"""
                <div class="main-metric">
                    <div class="metric-label">NET FLOW</div>
                    <div class="metric-value" style="color: {net_color}">{net_icon} {abs(net_vol):,.0f}</div>
                    <div class="metric-sub">{'Bearish' if net_vol > 0 else 'Bullish'} Bias</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                premium_ratio = analysis['premium_ratio']
                prem_color = "#ef5350" if premium_ratio > 1.0 else "#26a69a"
                st.markdown(f"""
                <div class="main-metric">
                    <div class="metric-label">PREMIUM RATIO</div>
                    <div class="metric-value" style="color: {prem_color}">{premium_ratio:.2f}</div>
                    <div class="metric-sub">${analysis['total_put_premium']/1e6:.1f}M / ${analysis['total_call_premium']/1e6:.1f}M</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m5:
                if analysis['flip_level']:
                    flip_status = "ABOVE" if underlying_price > analysis['flip_level'] else "BELOW"
                    flip_color = "#26a69a" if underlying_price > analysis['flip_level'] else "#ef5350"
                    flip_dist = abs(underlying_price - analysis['flip_level'])
                    st.markdown(f"""
                    <div class="main-metric">
                        <div class="metric-label">FLIP LEVEL</div>
                        <div class="metric-value">${analysis['flip_level']:.2f}</div>
                        <div class="metric-sub" style="color: {flip_color}">{flip_status} (${flip_dist:.2f})</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="main-metric">
                        <div class="metric-label">FLIP LEVEL</div>
                        <div class="metric-value">-</div>
                        <div class="metric-sub">No flip detected</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_signal:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px 10px; background: {signal_color}; border-radius: 10px; color: white; height: 100%;">
                    <div style="font-size: 11px; opacity: 0.9; font-weight: 600; letter-spacing: 0.5px;">SIGNAL</div>
                    <div style="font-size: 28px; font-weight: 900; margin: 5px 0; line-height: 1;">{signal}</div>
                    <div style="font-size: 10px; opacity: 0.85;">{' • '.join(signal_reason) if signal_reason else 'Neutral'}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ===== MAG 7 SENTIMENT PANEL (ABOVE CHART) =====
        st.markdown("### 🌟 Mag 7 Sentiment")
        
        # Get next Friday for weekly expiry
        next_friday = get_next_friday()
        weekly_exp_str = next_friday.strftime('%Y-%m-%d')
        
        mag7_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        mag7_cols = st.columns(7)
        
        for idx, mag_symbol in enumerate(mag7_symbols):
            with mag7_cols[idx]:
                try:
                    mag_snap = get_market_snapshot(mag_symbol, weekly_exp_str)
                    if mag_snap and mag_snap.get('underlying_price'):
                        mag_price = mag_snap['underlying_price']
                        mag_quote = mag_snap['quote'].get(mag_symbol, {}).get('quote', {})
                        mag_prev_close = mag_quote.get('closePrice', mag_price)
                        mag_change_pct = ((mag_price - mag_prev_close) / mag_prev_close * 100) if mag_prev_close else 0
                        
                        # Quick analysis
                        mag_analysis = calculate_comprehensive_analysis(mag_snap['options_chain'], mag_price)
                        
                        if mag_analysis:
                            # Determine sentiment
                            sentiment = "🟢"  # Bullish default
                            sentiment_text = "BULL"
                            bg_color = "#e8f5e9"
                            border_color = "#4caf50"
                            
                            if mag_analysis['flip_level']:
                                if mag_price < mag_analysis['flip_level']:
                                    sentiment = "🔴"
                                    sentiment_text = "BEAR"
                                    bg_color = "#ffebee"
                                    border_color = "#f44336"
                            
                            # Build compact card (reduced height)
                            cw = f"${mag_analysis['call_wall']['strike']:.0f}" if mag_analysis['call_wall'] is not None else "N/A"
                            pw = f"${mag_analysis['put_wall']['strike']:.0f}" if mag_analysis['put_wall'] is not None else "N/A"
                            flip = f"${mag_analysis['flip_level']:.0f}" if mag_analysis['flip_level'] else "N/A"
                            
                            pc_ratio = mag_analysis['pc_ratio']
                            pc_color = "#f44336" if pc_ratio > 1.0 else "#4caf50"
                            
                            # Build hot strikes section
                            hot_strikes_html = ""
                            if mag_analysis and 'df' in mag_analysis:
                                df = mag_analysis['df']
                                # Filter to strikes within ±5% of current price
                                df_near = df[abs((df['strike'] - mag_price) / mag_price) <= 0.05].copy()
                                df_near['total_vol'] = df_near['call_vol'] + df_near['put_vol']
                                top_strikes = df_near.nlargest(3, 'total_vol')
                                
                                if len(top_strikes) > 0:
                                    hot_items = []
                                    for _, row in top_strikes.iterrows():
                                        strike = row['strike']
                                        total_vol = row['total_vol']
                                        call_vol = row['call_vol']
                                        put_vol = row['put_vol']
                                        
                                        # Determine if calls or puts dominate
                                        dominant = "C" if call_vol > put_vol else "P"
                                        dom_color = "#4caf50" if dominant == "C" else "#f44336"
                                        
                                        # Distance from current price
                                        dist_pct = ((strike - mag_price) / mag_price) * 100
                                        dist_str = f"{dist_pct:+.1f}%"
                                        
                                        hot_items.append(
                                            f'<div style="display: flex; justify-content: space-between; font-size: 7px; margin: 2px 0; padding: 2px; background: rgba(0,0,0,0.03); border-radius: 2px;">'
                                            f'<span style="font-weight: 700;">${strike:.0f}</span>'
                                            f'<span style="background: {dom_color}; color: white; padding: 0px 3px; border-radius: 2px; font-weight: 700;">{dominant}</span>'
                                            f'<span style="opacity: 0.6;">{dist_str}</span>'
                                            f'<span style="opacity: 0.8;">{total_vol:,.0f}</span>'
                                            f'</div>'
                                        )
                                    
                                    if hot_items:
                                        hot_strikes_html = '<div style="margin-top: 4px; padding: 3px; background: rgba(255,165,0,0.1); border-radius: 3px; border: 1px solid rgba(255,165,0,0.3);"><div style="font-size: 7px; font-weight: 700; opacity: 0.8; margin-bottom: 2px; text-transform: uppercase;">🔥 Hot Strikes</div>' + ''.join(hot_items) + '</div>'
                            
                            card_html = f"""
                            <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; padding: 6px; height: 200px;">
                                <div style="text-align: center; margin-bottom: 4px;">
                                    <div style="font-size: 14px; font-weight: 800;">{sentiment} {mag_symbol}</div>
                                    <div style="font-size: 13px; font-weight: 700; color: {'#4caf50' if mag_change_pct >= 0 else '#f44336'};">
                                        ${mag_price:.2f}
                                    </div>
                                    <div style="font-size: 9px; color: {'#4caf50' if mag_change_pct >= 0 else '#f44336'};">
                                        {mag_change_pct:+.2f}%
                                    </div>
                                </div>
                                <div style="font-size: 8px; line-height: 1.3; margin-top: 4px;">
                                    <div style="display: flex; justify-content: space-between; margin: 1px 0;">
                                        <span style="opacity: 0.7;">CW:</span>
                                        <strong>{cw}</strong>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin: 1px 0;">
                                        <span style="opacity: 0.7;">PW:</span>
                                        <strong>{pw}</strong>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin: 1px 0;">
                                        <span style="opacity: 0.7;">Flip:</span>
                                        <strong>{flip}</strong>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin: 1px 0;">
                                        <span style="opacity: 0.7;">P/C:</span>
                                        <strong style="color: {pc_color};">{pc_ratio:.2f}</strong>
                                    </div>
                                </div>
                                <div style="text-align: center; margin-top: 4px; padding: 3px; background: rgba(0,0,0,0.05); border-radius: 3px;">
                                    <span style="font-size: 10px; font-weight: 700;">{sentiment_text}</span>
                                </div>
                                """ + hot_strikes_html + """
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.caption(f"{mag_symbol}\nAnalysis N/A")
                    else:
                        st.caption(f"{mag_symbol}\nData N/A")
                except Exception as e:
                    st.caption(f"{mag_symbol}\nError")
        
        st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
        
        # ===== CHART + KEY LEVELS + STOCKS TABLE =====
        col_chart, col_levels, col_stocks = st.columns([3, 0.7, 1.3])
        
        with col_chart:
            chart = create_unified_chart(snapshot['price_history'], analysis, underlying_price, selected)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error("Failed to create chart")
        
        with col_levels:
            st.markdown("### 🎯 Key Levels")
            
            # Call Wall
            if analysis['call_wall'] is not None:
                cw = analysis['call_wall']
                dist_to_cw = ((cw['strike'] - underlying_price) / underlying_price * 100)
                st.markdown(f"""
                <div class="level-card" style="border-color: #f44336; background: rgba(244, 67, 54, 0.05);">
                    <strong style="color: #f44336;">🔴 CALL WALL</strong><br>
                    <span style="font-size: 20px; font-weight: 900;">${cw['strike']:.2f}</span>
                    <span style="font-size: 11px; color: #666;"> ({dist_to_cw:+.2f}%)</span><br>
                    <span style="font-size: 10px;">Vol: {int(cw['call_vol']):,} | OI: {int(cw['call_oi']):,}</span><br>
                    <span style="font-size: 10px;">GEX: ${cw['call_gex']/1e6:.1f}M</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Put Wall
            if analysis['put_wall'] is not None:
                pw = analysis['put_wall']
                dist_to_pw = ((pw['strike'] - underlying_price) / underlying_price * 100)
                st.markdown(f"""
                <div class="level-card" style="border-color: #4caf50; background: rgba(76, 175, 80, 0.05);">
                    <strong style="color: #4caf50;">🟢 PUT WALL</strong><br>
                    <span style="font-size: 20px; font-weight: 900;">${pw['strike']:.2f}</span>
                    <span style="font-size: 11px; color: #666;"> ({dist_to_pw:+.2f}%)</span><br>
                    <span style="font-size: 10px;">Vol: {int(pw['put_vol']):,} | OI: {int(pw['put_oi']):,}</span><br>
                    <span style="font-size: 10px;">GEX: ${abs(pw['put_gex'])/1e6:.1f}M</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Max GEX
            if analysis['max_gex'] is not None:
                mg = analysis['max_gex']
                dist_to_mg = ((mg['strike'] - underlying_price) / underlying_price * 100)
                gex_color = "#f44336" if mg['net_gex'] < 0 else "#2196f3"
                st.markdown(f"""
                <div class="level-card" style="border-color: {gex_color}; background: rgba(33, 150, 243, 0.05);">
                    <strong style="color: {gex_color};">⚡ MAX GEX</strong><br>
                    <span style="font-size: 20px; font-weight: 900;">${mg['strike']:.2f}</span>
                    <span style="font-size: 11px; color: #666;"> ({dist_to_mg:+.2f}%)</span><br>
                    <span style="font-size: 10px;">Net GEX: ${mg['net_gex']/1e6:.1f}M</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Market Pulse - Additional insights
            st.markdown("### 📊 Market Pulse")
            
            # Calculate additional metrics
            net_vol = analysis['total_put_vol'] - analysis['total_call_vol']
            net_premium = analysis['total_put_premium'] - analysis['total_call_premium']
            
            # Flow momentum indicator
            flow_direction = "🐻 BEARISH" if net_vol > 0 else "🐂 BULLISH"
            flow_color = "#ef5350" if net_vol > 0 else "#26a69a"
            flow_strength = min(abs(net_vol) / (analysis['total_call_vol'] + analysis['total_put_vol']) * 200, 100)
            
            # Distance to walls
            dist_to_call_wall = "N/A"
            dist_to_put_wall = "N/A"
            if analysis['call_wall'] is not None:
                dist_pct = ((analysis['call_wall']['strike'] - underlying_price) / underlying_price * 100)
                dist_to_call_wall = f"{dist_pct:+.2f}%"
            if analysis['put_wall'] is not None:
                dist_pct = ((analysis['put_wall']['strike'] - underlying_price) / underlying_price * 100)
                dist_to_put_wall = f"{dist_pct:+.2f}%"
            
            market_pulse_html = f"""
            <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid {flow_color};">
                <div style="margin-bottom: 8px;">
                    <strong style="color: {flow_color};">Flow Direction: {flow_direction}</strong>
                    <div style="background: #e0e0e0; height: 6px; border-radius: 3px; margin-top: 4px;">
                        <div style="background: {flow_color}; width: {flow_strength}%; height: 6px; border-radius: 3px;"></div>
                    </div>
                    <span style="font-size: 10px; color: #666;">Strength: {flow_strength:.0f}%</span>
                </div>
                <div style="font-size: 11px; margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span>💰 Net Premium:</span>
                        <strong>${abs(net_premium)/1e6:.1f}M {' PUT' if net_premium > 0 else ' CALL'}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span>🔴 To Call Wall:</span>
                        <strong>{dist_to_call_wall}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span>🟢 To Put Wall:</span>
                        <strong>{dist_to_put_wall}</strong>
                    </div>
                </div>
            </div>
            """
            st.markdown(market_pulse_html, unsafe_allow_html=True)
        
        with col_stocks:
            # Use the auto-refreshing watchlist table
            live_watchlist_table()
        
        st.markdown("---")
        
        # ===== STRIKE DATA TABLE =====
        st.markdown("### 📊 Strike Analysis")
        
        # Filter to relevant strikes (within 5% of current price)
        df_display = analysis['df'][analysis['df']['distance_pct'] < 5.0].copy()
        df_display = df_display.sort_values('distance', ascending=True).head(20)
        
        # Format for display
        df_display['Strike'] = df_display['strike'].apply(lambda x: f"${x:.2f}")
        df_display['Call Vol'] = df_display['call_vol'].apply(lambda x: f"{int(x):,}")
        df_display['Put Vol'] = df_display['put_vol'].apply(lambda x: f"{int(x):,}")
        df_display['Net Vol'] = df_display['net_vol'].apply(lambda x: f"{int(x):+,}")
        df_display['Net GEX'] = df_display['net_gex'].apply(lambda x: f"${x/1e6:.1f}M")
        df_display['Distance'] = df_display['distance'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            df_display[['Strike', 'Call Vol', 'Put Vol', 'Net Vol', 'Net GEX', 'Distance']],
            width="stretch",
            height=400
        )
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error: {str(e)}")

# Auto-refresh logic
if st.session_state.auto_refresh_byindex:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh_byindex).seconds
    if time_since_refresh >= 180:  # 3 minutes
        st.cache_data.clear()
        st.session_state.last_refresh_byindex = datetime.now()
        st.rerun()
    else:
        # Update timer every 60 seconds to minimize CPU usage and page reloads
        time.sleep(60)
        st.rerun()
