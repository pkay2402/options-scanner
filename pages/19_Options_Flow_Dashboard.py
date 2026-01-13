"""
Options Flow Dashboard - Multi-Expiry Analysis with Interactive Charts
Colorful watchlist showing key levels and multi-expiry volume analysis
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

st.set_page_config(
    page_title="Options Flow Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for colorful design
st.markdown("""
<style>
    .flow-table {
        font-size: 12px;
        width: 100%;
    }
    .flow-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 8px 8px 0 0;
        font-weight: 700;
        text-align: center;
    }
    .symbol-cell {
        font-size: 16px;
        font-weight: 900;
        padding: 12px;
    }
    .bullish {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4caf50;
    }
    .bearish {
        background: rgba(244, 67, 54, 0.15);
        border-left: 4px solid #f44336;
    }
    .neutral {
        background: rgba(158, 158, 158, 0.1);
        border-left: 4px solid #9e9e9e;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        margin: 2px;
    }
    .chart-btn {
        cursor: pointer;
        font-size: 20px;
        transition: transform 0.2s;
    }
    .chart-btn:hover {
        transform: scale(1.2);
    }
</style>
""", unsafe_allow_html=True)

# Session state for chart expansion
if 'expanded_symbol' not in st.session_state:
    st.session_state.expanded_symbol = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def get_next_expiries(count=4):
    """Get next N Friday expiries"""
    expiries = []
    today = datetime.now().date()
    current = today
    
    while len(expiries) < count:
        days_ahead = 4 - current.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_friday = current + timedelta(days=days_ahead)
        expiries.append(next_friday)
        current = next_friday + timedelta(days=1)
    
    return expiries

@st.cache_data(ttl=120, show_spinner=False)
def get_multi_expiry_data(symbol: str, expiries: list):
    """Fetch options data for multiple expiries"""
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        quote = client.get_quote(symbol)
        if not quote:
            return None
        
        underlying_price = quote.get(symbol, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            return None
        
        expiry_data = {}
        
        for expiry in expiries:
            exp_str = expiry.strftime('%Y-%m-%d')
            
            chain_params = {
                'symbol': symbol,
                'contract_type': 'ALL',
                'from_date': exp_str,
                'to_date': exp_str
            }
            
            options = client.get_options_chain(**chain_params)
            
            if options and 'callExpDateMap' in options:
                expiry_data[exp_str] = options
        
        # Get price history for chart
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(days=5)).timestamp() * 1000)
        
        price_history = client.get_price_history(
            symbol=symbol,
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
            'expiry_data': expiry_data,
            'price_history': price_history,
            'fetched_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None

def analyze_expiry(options_data, underlying_price):
    """Analyze single expiry options data"""
    try:
        call_data = {}
        put_data = {}
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in call_data:
                            call_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                        call_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        call_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        call_data[strike]['gamma'] += contract.get('gamma', 0) or 0
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in put_data:
                            put_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                        put_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        put_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        put_data[strike]['gamma'] += abs(contract.get('gamma', 0) or 0)
        
        # Find key levels
        call_wall_strike = max(call_data.keys(), key=lambda k: call_data[k]['volume']) if call_data else None
        put_wall_strike = max(put_data.keys(), key=lambda k: put_data[k]['volume']) if put_data else None
        
        # Calculate gamma exposure
        max_gex_strike = None
        max_gex_value = 0
        
        for strike in set(call_data.keys()) | set(put_data.keys()):
            call = call_data.get(strike, {'gamma': 0, 'oi': 0})
            put = put_data.get(strike, {'gamma': 0, 'oi': 0})
            
            call_gex = call['gamma'] * call['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = -put['gamma'] * put['oi'] * 100 * underlying_price * underlying_price * 0.01
            net_gex = abs(call_gex + put_gex)
            
            if net_gex > max_gex_value:
                max_gex_value = net_gex
                max_gex_strike = strike
        
        total_call_vol = sum(d['volume'] for d in call_data.values())
        total_put_vol = sum(d['volume'] for d in put_data.values())
        
        return {
            'call_wall': call_wall_strike,
            'put_wall': put_wall_strike,
            'max_gex': max_gex_strike,
            'total_call_vol': total_call_vol,
            'total_put_vol': total_put_vol,
            'pc_ratio': total_put_vol / total_call_vol if total_call_vol > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error analyzing expiry: {str(e)}")
        return None

def create_symbol_chart(price_history, underlying_price, symbol, analysis):
    """Create interactive price chart with levels"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        # Filter to last 2 days market hours
        df = df[
            ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
            ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
        ].copy()
        
        if df.empty:
            return None
        
        df['date'] = df['datetime'].dt.date
        unique_dates = sorted(df['date'].unique(), reverse=True)
        target_dates = unique_dates[:2] if len(unique_dates) >= 2 else unique_dates
        df = df[df['date'].isin(target_dates)].copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} Price Action", "Volume"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#00bcd4', width=2)
        ), row=1, col=1)
        
        # Add key levels if available
        if analysis:
            if analysis['call_wall']:
                fig.add_hline(
                    y=analysis['call_wall'],
                    line_color='#f44336',
                    line_width=2,
                    line_dash="dash",
                    annotation_text=f"CW ${analysis['call_wall']:.0f}",
                    annotation_position="right",
                    row=1, col=1
                )
            
            if analysis['put_wall']:
                fig.add_hline(
                    y=analysis['put_wall'],
                    line_color='#4caf50',
                    line_width=2,
                    line_dash="dash",
                    annotation_text=f"PW ${analysis['put_wall']:.0f}",
                    annotation_position="right",
                    row=1, col=1
                )
            
            if analysis['max_gex']:
                fig.add_hline(
                    y=analysis['max_gex'],
                    line_color='#9c27b0',
                    line_width=2,
                    line_dash="dot",
                    annotation_text=f"GEX ${analysis['max_gex']:.0f}",
                    annotation_position="right",
                    row=1, col=1
                )
        
        # Volume bars
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ), row=2, col=1)
        
        # Layout
        fig.update_xaxes(
            type='date',
            tickformat='%H:%M',
            rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
            row=1, col=1
        )
        
        fig.update_xaxes(
            type='date',
            tickformat='%H:%M',
            rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(t=50, r=20, l=60, b=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

# ===== HEADER =====
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown("# 📈 Options Flow Dashboard")
    st.caption("Multi-Expiry Analysis with Interactive Charts")

with col2:
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

with col3:
    st.metric("Last Update", st.session_state.last_refresh.strftime('%H:%M:%S'))

st.markdown("---")

# Get next 4 expiries
expiries = get_next_expiries(4)
st.info(f"📅 Analyzing expiries: {', '.join([e.strftime('%b %d') for e in expiries])}")

# Watchlist
watchlist = ['SPY', 'QQQ', 'PLTR', 'CRWD', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'NBIS', 'MSFT', 'GOOGL', 'NFLX', 'OKLO', 'GS', 'TEM', 'COIN']

# ===== MAIN WATCHLIST TABLE =====
with st.expander("📊 Options Flow Watchlist", expanded=True):
    st.markdown("*Click 📊 to view chart • Weekly expiry levels + 4-week volume analysis*")
    
    table_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(watchlist):
        status_text.text(f"Loading {symbol}... ({idx+1}/{len(watchlist)})")
        progress_bar.progress((idx + 1) / len(watchlist))
        
        try:
            data = get_multi_expiry_data(symbol, expiries)
            
            if not data:
                continue
            
            price = data['underlying_price']
            quote_data = data['quote'].get(symbol, {}).get('quote', {})
            prev_close = quote_data.get('closePrice', price)
            daily_change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            # Analyze weekly expiry (first one)
            weekly_exp_str = expiries[0].strftime('%Y-%m-%d')
            weekly_analysis = None
            
            if weekly_exp_str in data['expiry_data']:
                weekly_analysis = analyze_expiry(data['expiry_data'][weekly_exp_str], price)
            
            # Aggregate volume across all 4 expiries
            total_call_vol_all = 0
            total_put_vol_all = 0
            
            for exp_str, exp_data in data['expiry_data'].items():
                exp_analysis = analyze_expiry(exp_data, price)
                if exp_analysis:
                    total_call_vol_all += exp_analysis['total_call_vol']
                    total_put_vol_all += exp_analysis['total_put_vol']
            
            # Determine sentiment
            sentiment = "NEUTRAL"
            sentiment_class = "neutral"
            
            if weekly_analysis:
                if weekly_analysis['pc_ratio'] > 1.2:
                    sentiment = "BEARISH"
                    sentiment_class = "bearish"
                elif weekly_analysis['pc_ratio'] < 0.8:
                    sentiment = "BULLISH"
                    sentiment_class = "bullish"
            
            table_data.append({
                'symbol': symbol,
                'price': price,
                'change_pct': daily_change_pct,
                'sentiment': sentiment,
                'sentiment_class': sentiment_class,
                'call_wall': weekly_analysis['call_wall'] if weekly_analysis else None,
                'put_wall': weekly_analysis['put_wall'] if weekly_analysis else None,
                'max_gex': weekly_analysis['max_gex'] if weekly_analysis else None,
                'pc_ratio': weekly_analysis['pc_ratio'] if weekly_analysis else 0,
                'weekly_call_vol': weekly_analysis['total_call_vol'] if weekly_analysis else 0,
                'weekly_put_vol': weekly_analysis['total_put_vol'] if weekly_analysis else 0,
                'total_call_vol_4w': total_call_vol_all,
                'total_put_vol_4w': total_put_vol_all,
                'data': data,
                'weekly_analysis': weekly_analysis
            })
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Build HTML table
    if table_data:
        table_html = """
        <table class="flow-table" style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <th style="padding: 12px; text-align: left;">Chart</th>
                    <th style="padding: 12px; text-align: left;">Symbol</th>
                    <th style="padding: 12px; text-align: right;">Price</th>
                    <th style="padding: 12px; text-align: center;">Sentiment</th>
                    <th style="padding: 12px; text-align: right;">Call Wall</th>
                    <th style="padding: 12px; text-align: right;">Put Wall</th>
                    <th style="padding: 12px; text-align: right;">Max GEX</th>
                    <th style="padding: 12px; text-align: center;">P/C Ratio</th>
                    <th style="padding: 12px; text-align: right;">Weekly C/P</th>
                    <th style="padding: 12px; text-align: right;">4-Week C/P</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for row in table_data:
            change_color = "#4caf50" if row['change_pct'] >= 0 else "#f44336"
            pc_color = "#f44336" if row['pc_ratio'] > 1.0 else "#4caf50"
            
            sentiment_colors = {
                'bullish': '#4caf50',
                'bearish': '#f44336',
                'neutral': '#9e9e9e'
            }
            sentiment_color = sentiment_colors.get(row['sentiment_class'], '#9e9e9e')
            
            table_html += f"""
            <tr class="{row['sentiment_class']}" style="border-bottom: 1px solid #eee;">
                <td style="padding: 12px; text-align: center;">
                    <span class="chart-btn" onclick="alert('Chart for {row['symbol']}')">📊</span>
                </td>
                <td class="symbol-cell">{row['symbol']}</td>
                <td style="padding: 12px; text-align: right;">
                    <strong style="font-size: 14px;">${row['price']:.2f}</strong><br>
                    <span style="font-size: 11px; color: {change_color};">{row['change_pct']:+.2f}%</span>
                </td>
                <td style="padding: 12px; text-align: center;">
                    <span class="metric-badge" style="background: {sentiment_color}; color: white;">{row['sentiment']}</span>
                </td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #f44336;">
                    {'$' + f"{row['call_wall']:.0f}" if row['call_wall'] else 'N/A'}
                </td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #4caf50;">
                    {'$' + f"{row['put_wall']:.0f}" if row['put_wall'] else 'N/A'}
                </td>
                <td style="padding: 12px; text-align: right; font-weight: 700; color: #9c27b0;">
                    {'$' + f"{row['max_gex']:.0f}" if row['max_gex'] else 'N/A'}
                </td>
                <td style="padding: 12px; text-align: center;">
                    <strong style="color: {pc_color}; font-size: 14px;">{row['pc_ratio']:.2f}</strong>
                </td>
                <td style="padding: 12px; text-align: right; font-size: 11px;">
                    <span style="color: #4caf50;">📞 {int(row['weekly_call_vol']/1000):.0f}K</span><br>
                    <span style="color: #f44336;">📉 {int(row['weekly_put_vol']/1000):.0f}K</span>
                </td>
                <td style="padding: 12px; text-align: right; font-size: 11px;">
                    <span style="color: #4caf50;">📞 {int(row['total_call_vol_4w']/1000):.0f}K</span><br>
                    <span style="color: #f44336;">📉 {int(row['total_put_vol_4w']/1000):.0f}K</span>
                </td>
            </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Chart selection buttons
        st.markdown("### 📊 View Chart")
        cols = st.columns(6)
        
        for idx, row in enumerate(table_data):
            with cols[idx % 6]:
                if st.button(f"📊 {row['symbol']}", key=f"chart_{row['symbol']}", use_container_width=True):
                    st.session_state.expanded_symbol = row['symbol']
        
        # Display chart if symbol selected
        if st.session_state.expanded_symbol:
            selected_data = next((r for r in table_data if r['symbol'] == st.session_state.expanded_symbol), None)
            
            if selected_data:
                st.markdown(f"### 📈 {selected_data['symbol']} - Detailed Chart")
                
                col_chart, col_stats = st.columns([3, 1])
                
                with col_chart:
                    chart = create_symbol_chart(
                        selected_data['data']['price_history'],
                        selected_data['price'],
                        selected_data['symbol'],
                        selected_data['weekly_analysis']
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.error("Failed to create chart")
                
                with col_stats:
                    st.markdown("#### 📊 Weekly Stats")
                    if selected_data['weekly_analysis']:
                        wa = selected_data['weekly_analysis']
                        
                        st.metric("P/C Ratio", f"{wa['pc_ratio']:.2f}")
                        st.metric("Call Wall", f"${wa['call_wall']:.0f}" if wa['call_wall'] else "N/A")
                        st.metric("Put Wall", f"${wa['put_wall']:.0f}" if wa['put_wall'] else "N/A")
                        st.metric("Max GEX", f"${wa['max_gex']:.0f}" if wa['max_gex'] else "N/A")
                        
                        st.markdown("---")
                        st.markdown("#### 📈 4-Week Volume")
                        st.metric("Total Calls", f"{int(selected_data['total_call_vol_4w']/1000):.0f}K")
                        st.metric("Total Puts", f"{int(selected_data['total_put_vol_4w']/1000):.0f}K")
                        
                        net_vol = selected_data['total_put_vol_4w'] - selected_data['total_call_vol_4w']
                        net_direction = "🐻 Bearish" if net_vol > 0 else "🐂 Bullish"
                        st.metric("Net Flow", f"{abs(net_vol)/1000:.0f}K {net_direction}")
                
                if st.button("✖️ Close Chart", type="secondary"):
                    st.session_state.expanded_symbol = None
                    st.rerun()
    
    else:
        st.warning("No data available. Please refresh.")
