#!/usr/bin/env python3
"""
Options Flow Chart
Interactive stock chart with integrated options data analysis - TrendSpider style
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient
from src.utils.cached_client import get_client

# Page config
st.set_page_config(
    page_title="Options Flow Chart",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Dark theme styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #2d3748 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #374151;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.85em;
        font-weight: 500;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    .metric-value.bullish {
        color: #10b981;
    }
    
    .metric-value.bearish {
        color: #ef4444;
    }
    
    /* Sentiment badge */
    .sentiment-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
        margin: 10px 0;
    }
    
    .sentiment-badge.bullish {
        background-color: #10b981;
        color: white;
    }
    
    .sentiment-badge.bearish {
        background-color: #ef4444;
        color: white;
    }
    
    .sentiment-badge.neutral {
        background-color: #6b7280;
        color: white;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9em;
    }
    
    .strike-highlight {
        background-color: #fef3c7 !important;
        font-weight: bold;
    }
    
    /* Header styling */
    .chart-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .chart-header h1 {
        margin: 0;
        font-size: 2em;
    }
    
    .chart-header .price {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .alert-box.info {
        background-color: #1e40af22;
        border-left-color: #3b82f6;
    }
    
    .alert-box.success {
        background-color: #16a34a22;
        border-left-color: #10b981;
    }
    
    .alert-box.warning {
        background-color: #ca8a0422;
        border-left-color: #f59e0b;
    }
    
    /* Quick stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def get_price_history(symbol, period_type='day', period=1, frequency_type='minute', frequency=30):
    """Get price history for charting"""
    try:
        client = get_client()
        if not client:
            return None
        
        # Clean symbol for price history (remove $)
        query_symbol = symbol.replace('$', '')
        
        price_data = client.get_price_history(
            symbol=query_symbol,
            period_type=period_type,
            period=period,
            frequency_type=frequency_type,
            frequency=frequency
        )
        
        if not price_data or 'candles' not in price_data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        
        return df
    except Exception as e:
        st.error(f"Error fetching price history: {e}")
        return None


@st.cache_data(ttl=60)
def get_options_summary(symbol):
    """Get comprehensive options data summary"""
    try:
        client = get_client()
        if not client:
            return None
        
        # Get quote
        quote_data = client.get_quote(symbol)
        if not quote_data or symbol not in quote_data:
            return None
        
        # Extract price from nested quote structure
        quote = quote_data[symbol].get('quote', {})
        underlying_price = quote.get('lastPrice', 0)
        if underlying_price == 0:
            underlying_price = quote.get('mark', 0)
        if underlying_price == 0:
            underlying_price = quote.get('bidPrice', 0)
        if underlying_price == 0:
            underlying_price = quote.get('askPrice', 0)
        
        # Get options chain
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not options_data:
            return None
        
        # Calculate comprehensive metrics
        summary = analyze_options_data(options_data, underlying_price)
        summary['underlying_price'] = underlying_price
        summary['quote'] = quote_data[symbol]
        
        return summary
        
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return None


def analyze_options_data(options_data, underlying_price):
    """Analyze options data to extract key metrics"""
    
    # Initialize summary
    summary = {
        'total_call_volume': 0,
        'total_put_volume': 0,
        'total_call_premium': 0,
        'total_put_premium': 0,
        'expiries': {},
        'top_strikes': [],
        'sentiment': 'NEUTRAL'
    }
    
    # Process calls
    if 'callExpDateMap' in options_data:
        for exp_date, strikes in options_data['callExpDateMap'].items():
            exp_key = exp_date.split(':')[0]  # Get date part
            
            if exp_key not in summary['expiries']:
                summary['expiries'][exp_key] = {
                    'call_volume': 0,
                    'put_volume': 0,
                    'call_premium': 0,
                    'put_premium': 0,
                    'dte': 0
                }
            
            for strike_str, contracts in strikes.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                contract = contracts[0]
                
                volume = contract.get('totalVolume', 0)
                oi = contract.get('openInterest', 0)
                mark = contract.get('mark', 0)
                
                # Calculate premium (volume * mark * 100)
                premium = volume * mark * 100
                
                summary['total_call_volume'] += volume
                summary['total_call_premium'] += premium
                summary['expiries'][exp_key]['call_volume'] += volume
                summary['expiries'][exp_key]['call_premium'] += premium
                
                # Calculate DTE
                try:
                    exp_datetime = datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                    summary['expiries'][exp_key]['dte'] = dte
                except:
                    pass
                
                # Track top strikes
                summary['top_strikes'].append({
                    'strike': strike,
                    'type': 'CALL',
                    'volume': volume,
                    'oi': oi,
                    'premium': premium,
                    'expiry': exp_key,
                    'dte': summary['expiries'][exp_key]['dte'],
                    'mark': mark,
                    'itm': strike < underlying_price
                })
    
    # Process puts
    if 'putExpDateMap' in options_data:
        for exp_date, strikes in options_data['putExpDateMap'].items():
            exp_key = exp_date.split(':')[0]
            
            if exp_key not in summary['expiries']:
                summary['expiries'][exp_key] = {
                    'call_volume': 0,
                    'put_volume': 0,
                    'call_premium': 0,
                    'put_premium': 0,
                    'dte': 0
                }
            
            for strike_str, contracts in strikes.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                contract = contracts[0]
                
                volume = contract.get('totalVolume', 0)
                oi = contract.get('openInterest', 0)
                mark = contract.get('mark', 0)
                
                premium = volume * mark * 100
                
                summary['total_put_volume'] += volume
                summary['total_put_premium'] += premium
                summary['expiries'][exp_key]['put_volume'] += volume
                summary['expiries'][exp_key]['put_premium'] += premium
                
                # Calculate DTE
                try:
                    exp_datetime = datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                    summary['expiries'][exp_key]['dte'] = dte
                except:
                    pass
                
                summary['top_strikes'].append({
                    'strike': strike,
                    'type': 'PUT',
                    'volume': volume,
                    'oi': oi,
                    'premium': premium,
                    'expiry': exp_key,
                    'dte': summary['expiries'][exp_key]['dte'],
                    'mark': mark,
                    'itm': strike > underlying_price
                })
    
    # Calculate P/C ratios
    summary['pc_volume'] = summary['total_put_volume'] / summary['total_call_volume'] if summary['total_call_volume'] > 0 else 0
    summary['pc_premium'] = summary['total_put_premium'] / summary['total_call_premium'] if summary['total_call_premium'] > 0 else 0
    
    # Determine sentiment
    if summary['pc_premium'] < 0.7:
        summary['sentiment'] = 'BULLISH'
    elif summary['pc_premium'] > 1.3:
        summary['sentiment'] = 'BEARISH'
    else:
        summary['sentiment'] = 'NEUTRAL'
    
    # Sort top strikes by premium
    summary['top_strikes'] = sorted(summary['top_strikes'], key=lambda x: x['premium'], reverse=True)[:30]
    
    # Calculate max pain
    summary['max_pain'] = calculate_max_pain(options_data, underlying_price)
    
    # Detect unusual activity
    summary['unusual_activity'] = detect_unusual_activity(summary['top_strikes'])
    
    return summary


def calculate_max_pain(options_data, underlying_price):
    """Calculate max pain strike - where most options expire worthless"""
    strike_pain = {}
    
    # Collect all strikes with OI
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
        
        for exp_date, strikes in options_data[option_type].items():
            for strike_str, contracts in strikes.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                oi = contracts[0].get('openInterest', 0)
                
                if strike not in strike_pain:
                    strike_pain[strike] = {'call_oi': 0, 'put_oi': 0}
                
                if 'call' in option_type:
                    strike_pain[strike]['call_oi'] += oi
                else:
                    strike_pain[strike]['put_oi'] += oi
    
    # Calculate pain for each strike
    max_pain_strike = underlying_price
    min_pain = float('inf')
    
    for test_strike in strike_pain.keys():
        total_pain = 0
        
        for strike, oi_data in strike_pain.items():
            # Call pain if price closes above strike
            if test_strike > strike:
                total_pain += (test_strike - strike) * oi_data['call_oi'] * 100
            
            # Put pain if price closes below strike
            if test_strike < strike:
                total_pain += (strike - test_strike) * oi_data['put_oi'] * 100
        
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike
    
    return max_pain_strike


def detect_unusual_activity(top_strikes):
    """Detect unusual options activity"""
    unusual = []
    
    for strike in top_strikes[:15]:
        volume = strike.get('volume', 0)
        oi = strike.get('oi', 0)
        
        # Flag unusual volume (vol > 3x open interest)
        if oi > 0 and volume > oi * 3:
            unusual.append({
                'type': 'High Volume',
                'strike': strike['strike'],
                'option_type': strike['type'],
                'volume': volume,
                'oi': oi,
                'ratio': volume / oi if oi > 0 else 0,
                'description': f"Volume {volume/oi:.1f}x Open Interest"
            })
        
        # Flag large premium trades
        premium = strike.get('premium', 0)
        if premium > 500000:  # $500K+
            unusual.append({
                'type': 'Large Premium',
                'strike': strike['strike'],
                'option_type': strike['type'],
                'premium': premium,
                'description': f"${premium/1000000:.2f}M in premium"
            })
    
    return unusual[:10]  # Top 10 unusual activities


def create_price_chart(price_df, symbol, key_strikes=None, current_price=None):
    """Create interactive candlestick chart with key strike levels"""
    
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=price_df['datetime'],
        open=price_df['open'],
        high=price_df['high'],
        low=price_df['low'],
        close=price_df['close'],
        name=symbol,
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444'
    ))
    
    # Add key strike levels as horizontal lines
    if key_strikes and current_price:
        for strike_info in key_strikes[:5]:  # Top 5 strikes
            strike = strike_info['strike']
            strike_type = strike_info['type']
            
            # Only show strikes within reasonable range
            if abs(strike - current_price) / current_price < 0.15:  # Within 15%
                color = '#10b981' if strike_type == 'CALL' else '#ef4444'
                fig.add_hline(
                    y=strike,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.5,
                    annotation_text=f"${strike:.0f} ({strike_type})",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color=color
                )
    
    # Add volume as bar chart
    fig.add_trace(go.Bar(
        x=price_df['datetime'],
        y=price_df['volume'],
        name='Volume',
        marker_color='rgba(100, 116, 139, 0.3)',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Price & Key Strike Levels',
            font=dict(size=20, color='white')
        ),
        xaxis=dict(
            title='Time',
            gridcolor='#374151',
            showgrid=True
        ),
        yaxis=dict(
            title='Price',
            gridcolor='#374151',
            showgrid=True
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def display_sentiment_metrics(summary, underlying_price):
    """Display sentiment and key metrics"""
    
    sentiment = summary['sentiment']
    sentiment_class = sentiment.lower()
    
    st.markdown(f"""
    <div class="chart-header">
        <div class="sentiment-badge {sentiment_class}">{sentiment}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns - now 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pc_vol_color = "bearish" if summary['pc_volume'] > 1 else "bullish"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">P/C (Vol)</div>
            <div class="metric-value {pc_vol_color}">{summary['pc_volume']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pc_prem_color = "bearish" if summary['pc_premium'] > 1 else "bullish"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">P/C (Prem)</div>
            <div class="metric-value {pc_prem_color}">{summary['pc_premium']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        call_vol_k = summary['total_call_volume'] / 1000
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calls</div>
            <div class="metric-value bullish">{call_vol_k:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        put_vol_k = summary['total_put_volume'] / 1000
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Puts</div>
            <div class="metric-value bearish">{put_vol_k:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        max_pain = summary.get('max_pain', underlying_price)
        distance = ((max_pain - underlying_price) / underlying_price) * 100
        pain_color = "bearish" if distance < -2 else ("bullish" if distance > 2 else "")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Pain</div>
            <div class="metric-value {pain_color}">${max_pain:.0f}</div>
            <div style="font-size: 0.7em; color: #9ca3af; margin-top: 5px;">
                {distance:+.1f}% from current
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_expiries_table(summary):
    """Display top expiries by premium"""
    
    st.subheader("📅 Top Expiries (by Premium)")
    
    # Convert expiries to DataFrame
    expiries_data = []
    for exp_date, data in summary['expiries'].items():
        total_vol = data['call_volume'] + data['put_volume']
        if total_vol > 0:  # Only show expiries with volume
            pc_ratio = data['put_volume'] / data['call_volume'] if data['call_volume'] > 0 else 999
            total_prem = data['call_premium'] + data['put_premium']
            
            expiries_data.append({
                'DTE': data['dte'],
                'Vol': f"{total_vol/1000:.1f}K",
                'Calls': f"{data['call_volume']/1000:.1f}K",
                'Puts': f"{data['put_volume']/1000:.1f}K",
                'P/C': f"{pc_ratio:.2f}",
                'Premium': f"${total_prem/1000:.0f}K"
            })
    
    df = pd.DataFrame(expiries_data)
    
    # Sort by DTE
    if not df.empty:
        df = df.sort_values('DTE')
        
        # Color code P/C ratio
        def color_pc(val):
            try:
                num = float(val)
                if num < 0.7:
                    return 'background-color: #10b981; color: white'
                elif num > 1.3:
                    return 'background-color: #ef4444; color: white'
                return ''
            except:
                return ''
        
        styled_df = df.style.applymap(color_pc, subset=['P/C'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_top_strikes_table(summary, underlying_price):
    """Display top strikes by premium"""
    
    st.subheader("🎯 Top Strikes (by Premium)")
    
    # Convert to DataFrame
    df = pd.DataFrame(summary['top_strikes'][:20])  # Top 20
    
    if df.empty:
        st.warning("No strike data available")
        return
    
    # Format columns
    display_df = pd.DataFrame({
        'Strike': df['strike'].apply(lambda x: f"${x:.0f}"),
        'Status': df.apply(lambda row: f"{'ITM' if row['itm'] else 'OTM'} {row['type']}", axis=1),
        'Vol': df['volume'].apply(lambda x: f"{x/1000:.1f}K"),
        'Calls': df.apply(lambda row: f"{row['volume']/1000:.1f}K" if row['type'] == 'CALL' else '0', axis=1),
        'Puts': df.apply(lambda row: f"{row['volume']/1000:.1f}K" if row['type'] == 'PUT' else '0', axis=1),
        'P/C': df.apply(lambda row: '999' if row['type'] == 'PUT' else '0.01', axis=1),
        'Dom Exp (DTE)': df.apply(lambda row: f"{row['dte']}d", axis=1),
        'Premium': df['premium'].apply(lambda x: f"${x/1000:.1f}K")
    })
    
    # Highlight strikes near current price
    def highlight_atm(row):
        try:
            strike = float(row['Strike'].replace('$', ''))
            if abs(strike - underlying_price) < underlying_price * 0.02:  # Within 2%
                return ['background-color: #fef3c7'] * len(row)
        except:
            pass
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_atm, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def create_strike_heatmap(summary, underlying_price):
    """Create visual heatmap of strike prices by volume/OI"""
    
    strikes_data = summary['top_strikes'][:20]
    
    if not strikes_data:
        return None
    
    # Separate calls and puts
    calls = [s for s in strikes_data if s['type'] == 'CALL']
    puts = [s for s in strikes_data if s['type'] == 'PUT']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Volume by Strike', 'Put Volume by Strike'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Call volumes
    if calls:
        call_strikes = [s['strike'] for s in calls]
        call_volumes = [s['volume'] for s in calls]
        
        fig.add_trace(
            go.Bar(
                x=call_volumes,
                y=call_strikes,
                orientation='h',
                name='Calls',
                marker=dict(
                    color=call_volumes,
                    colorscale='Greens',
                    showscale=False
                ),
                text=[f"${v/1000:.0f}K" for v in call_volumes],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # Put volumes
    if puts:
        put_strikes = [s['strike'] for s in puts]
        put_volumes = [s['volume'] for s in puts]
        
        fig.add_trace(
            go.Bar(
                x=put_volumes,
                y=put_strikes,
                orientation='h',
                name='Puts',
                marker=dict(
                    color=put_volumes,
                    colorscale='Reds',
                    showscale=False
                ),
                text=[f"${v/1000:.0f}K" for v in put_volumes],
                textposition='outside'
            ),
            row=1, col=2
        )
    
    # Add current price line
    fig.add_hline(
        y=underlying_price,
        line_dash="dash",
        line_color="yellow",
        opacity=0.7,
        annotation_text=f"Current: ${underlying_price:.2f}",
        row=1, col=1
    )
    fig.add_hline(
        y=underlying_price,
        line_dash="dash",
        line_color="yellow",
        opacity=0.7,
        annotation_text=f"Current: ${underlying_price:.2f}",
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    
    fig.update_xaxes(title_text="Volume", row=1, col=1)
    fig.update_xaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="Strike Price", row=1, col=1)
    fig.update_yaxes(title_text="Strike Price", row=1, col=2)
    
    return fig


def display_unusual_activity(summary):
    """Display unusual activity alerts"""
    
    unusual = summary.get('unusual_activity', [])
    
    if not unusual:
        st.info("No unusual activity detected")
        return
    
    st.subheader("🚨 Unusual Activity Detected")
    
    for activity in unusual[:5]:  # Top 5
        activity_type = activity['type']
        strike = activity['strike']
        option_type = activity['option_type']
        desc = activity['description']
        
        if activity_type == 'High Volume':
            icon = "📊"
            color = "#6366f1"  # Indigo for high volume
            bg_color = "#eef2ff"  # Light indigo background
        else:  # Large Premium
            icon = "💰"
            color = "#f59e0b"  # Amber/orange for large premium
            bg_color = "#fef3c7"  # Light amber background
        
        st.markdown(f"""
        <div style="background: {bg_color};
                    border-left: 5px solid {color};
                    padding: 15px 20px;
                    margin: 10px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                <strong style="color: #1f2937; font-size: 1.1em;">{activity_type}</strong>
            </div>
            <div style="color: #4b5563; font-size: 0.95em; margin-left: 32px;">
                ${strike:.0f} {option_type} - {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Options Flow Chart")
        
        # Symbol input
        default_symbol = st.session_state.get('last_symbol', 'LYFT')
        symbol = st.text_input(
            "Enter Symbol",
            value=default_symbol,
            help="Enter stock symbol (e.g., AAPL, SPY, TSLA)"
        ).upper().strip()
        
        if symbol:
            st.session_state['last_symbol'] = symbol
        
        # Timeframe selection
        st.subheader("⏰ Chart Timeframe")
        timeframe = st.selectbox(
            "Select Timeframe",
            ["30min", "1hour", "4hour", "Daily"],
            index=0
        )
        
        # Map timeframe to API parameters
        timeframe_map = {
            "30min": ('day', 1, 'minute', 30),
            "1hour": ('day', 1, 'minute', 60),
            "4hour": ('day', 5, 'minute', 240),
            "Daily": ('month', 1, 'daily', 1)
        }
        
        period_type, period, freq_type, frequency = timeframe_map[timeframe]
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Info
        st.info("""
        **Features:**
        - Live price chart
        - Options sentiment analysis
        - P/C ratios
        - Top strikes by premium
        - Expiry breakdown
        """)
    
    # Main content
    if not symbol:
        st.warning("⬅️ Enter a symbol to begin")
        return
    
    # Fetch data
    with st.spinner(f"Loading data for {symbol}..."):
        price_data = get_price_history(symbol, period_type, period, freq_type, frequency)
        options_summary = get_options_summary(symbol)
    
    if price_data is None or options_summary is None:
        st.error("Failed to load data. Please check the symbol and try again.")
        return
    
    # Display header with current price
    current_price = options_summary['underlying_price']
    quote = options_summary['quote'].get('quote', {})
    
    change = quote.get('netChange', 0)
    change_pct = quote.get('netPercentChange', 0)
    change_color = "🟢" if change >= 0 else "🔴"
    
    st.markdown(f"""
    <div class="chart-header">
        <h1>{symbol}</h1>
        <div class="price">
            ${current_price:.2f}
            <span style="font-size: 0.6em; margin-left: 20px;">
                {change_color} {change:+.2f} ({change_pct:+.2f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sentiment metrics
    display_sentiment_metrics(options_summary, current_price)
    
    # Unusual activity alert (if any)
    if options_summary.get('unusual_activity'):
        with st.expander("🚨 Unusual Activity Detected", expanded=True):
            display_unusual_activity(options_summary)
    
    # Price chart with key strikes
    st.plotly_chart(
        create_price_chart(price_data, symbol, options_summary['top_strikes'], current_price),
        use_container_width=True
    )
    
    # Strike heatmap
    st.subheader("🔥 Volume Distribution by Strike")
    heatmap = create_strike_heatmap(options_summary, current_price)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Two columns for tables
    col1, col2 = st.columns(2)
    
    with col1:
        display_expiries_table(options_summary)
    
    with col2:
        display_top_strikes_table(options_summary, current_price)
    
    # Additional insights
    st.divider()
    st.subheader("💡 Market Insights")
    
    # Generate insights based on data
    col_insight1, col_insight2 = st.columns([2, 1])
    
    with col_insight1:
        insights = []
        
        if options_summary['pc_premium'] < 0.7:
            insights.append("🔵 **Bullish sentiment** - Call premium significantly outweighs put premium")
        elif options_summary['pc_premium'] > 1.3:
            insights.append("🔴 **Bearish sentiment** - Put premium significantly outweighs call premium")
        else:
            insights.append("⚪ **Neutral sentiment** - Balanced call and put activity")
        
        # Max pain analysis
        max_pain = options_summary.get('max_pain', current_price)
        if max_pain != current_price:
            pain_diff = ((max_pain - current_price) / current_price) * 100
            if abs(pain_diff) > 3:
                direction = "upward" if pain_diff > 0 else "downward"
                insights.append(f"⚠️ **Max pain at ${max_pain:.0f}** - {abs(pain_diff):.1f}% {direction} pressure")
        
        # Find dominant expiry
        if options_summary['expiries']:
            max_prem_exp = max(
                options_summary['expiries'].items(),
                key=lambda x: x[1]['call_premium'] + x[1]['put_premium']
            )
            insights.append(f"📅 **Dominant expiry**: {max_prem_exp[1]['dte']} days out (most premium)")
        
        # Find key strike levels
        top_strikes = options_summary['top_strikes'][:5]
        if top_strikes:
            call_strikes = [s['strike'] for s in top_strikes if s['type'] == 'CALL']
            put_strikes = [s['strike'] for s in top_strikes if s['type'] == 'PUT']
            
            if call_strikes:
                insights.append(f"🎯 **Key call level**: ${call_strikes[0]:.0f} (highest call premium)")
            if put_strikes:
                insights.append(f"🎯 **Key put level**: ${put_strikes[0]:.0f} (highest put premium)")
        
        for insight in insights:
            st.markdown(insight)
    
    with col_insight2:
        st.markdown("### 💡 Quick Actions")
        
        # Generate actionable suggestions
        if options_summary['pc_premium'] < 0.7:
            st.success("**Bullish Setup**\n\n• Consider call spreads\n• Watch resistance levels\n• Monitor call walls")
        elif options_summary['pc_premium'] > 1.3:
            st.error("**Bearish Setup**\n\n• Consider put spreads\n• Watch support levels\n• Monitor put walls")
        else:
            st.info("**Neutral Setup**\n\n• Consider iron condors\n• Sell premium strategies\n• Range-bound trades")
        
        # Unusual activity call-out
        if options_summary.get('unusual_activity'):
            st.warning(f"⚡ {len(options_summary['unusual_activity'])} unusual activities detected")
    
    # Raw data expander
    with st.expander("📊 View Raw Data"):
        st.subheader("Price Data")
        st.dataframe(price_data.tail(50), use_container_width=True)
        
        st.subheader("Options Summary")
        st.json(options_summary)


if __name__ == "__main__":
    main()
