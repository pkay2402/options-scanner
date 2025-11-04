#!/usr/bin/env python3
"""
Options Flow & Unusual Activity Scanner
Detects large trades, sweeps, dark pool activity, and institutional orders in real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent; sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Configure Streamlit page
st.set_page_config(
    page_title="Options Flow Scanner",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .flow-alert {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .bullish-flow {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
        color: #155724;
    }
    .bearish-flow {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
        color: #721c24;
    }
    .neutral-flow {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
        color: #856404;
    }
    .block-trade {
        font-size: 1.2em;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .sweep-trade {
        border: 2px solid #ff6b6b;
        box-shadow: 0 0 15px rgba(255,107,107,0.4);
    }
    .unusual-volume {
        border: 2px solid #4ecdc4;
        box-shadow: 0 0 15px rgba(78,205,196,0.4);
    }
    .trade-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .trade-symbol {
        font-size: 1.5em;
        font-weight: bold;
    }
    .trade-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 5px;
    }
    .badge-call {
        background-color: #28a745;
        color: white;
    }
    .badge-put {
        background-color: #dc3545;
        color: white;
    }
    .badge-sweep {
        background-color: #ff6b6b;
        color: white;
    }
    .badge-block {
        background-color: #6c757d;
        color: white;
    }
    .badge-unusual {
        background-color: #4ecdc4;
        color: white;
    }
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .metric-item {
        flex: 1;
    }
    .live-indicator {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

def estimate_underlying_from_strikes(options_data):
    """Estimate underlying price from ATM options strikes"""
    try:
        if not options_data or 'callExpDateMap' not in options_data:
            return None
        
        exp_dates = list(options_data['callExpDateMap'].keys())
        if not exp_dates:
            return None
        
        first_exp = options_data['callExpDateMap'][exp_dates[0]]
        strikes = [float(s) for s in first_exp.keys()]
        
        if strikes:
            strike_data = []
            for strike_str, contracts in first_exp.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)
                    strike = float(strike_str)
                    activity = volume + open_interest * 0.1
                    strike_data.append((strike, activity))
            
            if strike_data:
                strike_data.sort(key=lambda x: x[1], reverse=True)
                most_active_strike = strike_data[0][0]
                
                if 50 < most_active_strike < 2000:
                    return most_active_strike
            
            strikes.sort()
            mid_index = len(strikes) // 2
            return strikes[mid_index]
        
        return None
    except:
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute for real-time feel
def get_options_data(symbol):
    """Fetch options data"""
    try:
        client = SchwabClient()
        
        quote_data = client.get_quote(symbol)
        if not quote_data or symbol not in quote_data:
            return None, None
        
        underlying_price = quote_data[symbol].get('lastPrice', 0)
        if underlying_price == 0:
            underlying_price = quote_data[symbol].get('mark', 0)
        
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not options_data:
            return None, None
        
        if 'underlying' in options_data and options_data['underlying']:
            options_underlying_price = options_data['underlying'].get('last', 0)
            if options_underlying_price and options_underlying_price > 0:
                underlying_price = options_underlying_price
        
        if underlying_price == 0 or underlying_price == 100.0:
            estimated_price = estimate_underlying_from_strikes(options_data)
            if estimated_price:
                underlying_price = estimated_price
        
        return options_data, underlying_price
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def analyze_flow(options_data, underlying_price, min_premium=10000, volume_threshold=100):
    """Analyze options flow and detect unusual activity"""
    
    if not options_data:
        return []
    
    flows = []
    
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
        
        is_call = 'call' in option_type
        exp_dates = list(options_data[option_type].keys())
        
        for exp_date in exp_dates:
            strikes_data = options_data[option_type][exp_date]
            
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                contract = contracts[0]
                
                # Extract data
                volume = contract.get('totalVolume', 0)
                open_interest = contract.get('openInterest', 0)
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                last = contract.get('last', 0)
                bid_size = contract.get('bidSize', 0)
                ask_size = contract.get('askSize', 0)
                delta = contract.get('delta', 0)
                gamma = contract.get('gamma', 0)
                vega = contract.get('vega', 0)
                implied_vol = contract.get('volatility', 0) * 100
                
                # Skip if no volume
                if volume == 0:
                    continue
                
                # Calculate metrics
                mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                premium = volume * mid_price * 100  # Premium in dollars
                
                # Skip small trades
                if premium < min_premium:
                    continue
                
                # Calculate volume/OI ratio
                vol_oi_ratio = volume / open_interest if open_interest > 0 else 0
                
                # Determine moneyness
                if is_call:
                    moneyness = "ITM" if strike < underlying_price else "OTM"
                    itm_pct = ((underlying_price - strike) / strike * 100) if strike < underlying_price else ((strike - underlying_price) / underlying_price * 100)
                else:
                    moneyness = "ITM" if strike > underlying_price else "OTM"
                    itm_pct = ((strike - underlying_price) / underlying_price * 100) if strike > underlying_price else ((underlying_price - strike) / strike * 100)
                
                # Parse expiration
                exp_date_str = exp_date.split(':')[0] if ':' in exp_date else exp_date
                try:
                    exp_dt = datetime.strptime(exp_date_str, '%Y-%m-%d')
                    days_to_exp = (exp_dt - datetime.now()).days
                except:
                    days_to_exp = 0
                
                # Detect trade types
                trade_types = []
                
                # Block Trade (large single order)
                if premium >= 100000:
                    trade_types.append("BLOCK")
                
                # Sweep (aggressive, likely multi-exchange)
                if vol_oi_ratio > 0.5 and volume > volume_threshold:
                    trade_types.append("SWEEP")
                
                # Unusual Volume
                if volume > open_interest * 2 and open_interest > 0:
                    trade_types.append("UNUSUAL")
                
                # New Position (high volume, low OI)
                if volume > 500 and open_interest < volume * 1.5:
                    trade_types.append("NEW")
                
                # Determine sentiment
                if is_call and moneyness == "OTM":
                    sentiment = "BULLISH"
                elif not is_call and moneyness == "OTM":
                    sentiment = "BEARISH"
                elif is_call and moneyness == "ITM":
                    sentiment = "BEARISH"  # Could be hedge/sell
                elif not is_call and moneyness == "ITM":
                    sentiment = "BULLISH"  # Could be hedge/sell
                else:
                    sentiment = "NEUTRAL"
                
                flows.append({
                    'strike': strike,
                    'expiry': exp_date_str,
                    'days_to_exp': days_to_exp,
                    'type': 'CALL' if is_call else 'PUT',
                    'volume': volume,
                    'open_interest': open_interest,
                    'premium': premium,
                    'price': mid_price,
                    'bid': bid,
                    'ask': ask,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'implied_vol': implied_vol,
                    'moneyness': moneyness,
                    'itm_pct': itm_pct,
                    'vol_oi_ratio': vol_oi_ratio,
                    'trade_types': trade_types,
                    'sentiment': sentiment,
                    'timestamp': datetime.now()
                })
    
    return flows

def display_flow_alert(symbol, flow, underlying_price):
    """Display a single flow alert with styling"""
    
    # Determine styling class
    if flow['sentiment'] == 'BULLISH':
        alert_class = 'bullish-flow'
    elif flow['sentiment'] == 'BEARISH':
        alert_class = 'bearish-flow'
    else:
        alert_class = 'neutral-flow'
    
    # Add special classes
    extra_classes = []
    if 'BLOCK' in flow['trade_types']:
        extra_classes.append('block-trade')
    if 'SWEEP' in flow['trade_types']:
        extra_classes.append('sweep-trade')
    if 'UNUSUAL' in flow['trade_types']:
        extra_classes.append('unusual-volume')
    
    all_classes = f"{alert_class} {' '.join(extra_classes)}"
    
    # Create badges
    badges = f'<span class="trade-badge badge-{flow["type"].lower()}">{flow["type"]}</span>'
    for trade_type in flow['trade_types']:
        badge_color = 'sweep' if trade_type == 'SWEEP' else 'block' if trade_type == 'BLOCK' else 'unusual'
        badges += f'<span class="trade-badge badge-{badge_color}">{trade_type}</span>'
    
    # Format premium
    if flow['premium'] >= 1_000_000:
        premium_str = f"${flow['premium']/1_000_000:.2f}M"
    else:
        premium_str = f"${flow['premium']/1_000:.0f}K"
    
    html = f"""
    <div class="flow-alert {all_classes}">
        <div class="trade-header">
            <div>
                <span class="trade-symbol">{symbol}</span>
                {badges}
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5em; font-weight: bold;">{premium_str}</div>
                <div style="font-size: 0.8em;">{flow['sentiment']} Flow</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <strong>Strike:</strong> ${flow['strike']:.2f} ({flow['moneyness']})
            </div>
            <div class="metric-item">
                <strong>Expiry:</strong> {flow['expiry']} ({flow['days_to_exp']} DTE)
            </div>
            <div class="metric-item">
                <strong>Price:</strong> ${flow['price']:.2f}
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <strong>Volume:</strong> {flow['volume']:,}
            </div>
            <div class="metric-item">
                <strong>OI:</strong> {flow['open_interest']:,}
            </div>
            <div class="metric-item">
                <strong>Vol/OI:</strong> {flow['vol_oi_ratio']:.2f}x
            </div>
            <div class="metric-item">
                <strong>IV:</strong> {flow['implied_vol']:.1f}%
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <strong>Delta:</strong> {flow['delta']:.3f}
            </div>
            <div class="metric-item">
                <strong>Bid/Ask:</strong> ${flow['bid']:.2f} / ${flow['ask']:.2f}
            </div>
            <div class="metric-item">
                <strong>Detected:</strong> {flow['timestamp'].strftime('%H:%M:%S')}
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def create_flow_summary_chart(flows_df):
    """Create summary charts for flow analysis"""
    
    if flows_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Premium by Type',
            'Volume Distribution',
            'Sentiment Breakdown',
            'Trade Types'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "pie"}]
        ]
    )
    
    # Premium by option type
    type_premium = flows_df.groupby('type')['premium'].sum()
    fig.add_trace(
        go.Bar(
            x=type_premium.index,
            y=type_premium.values,
            marker_color=['green', 'red'],
            name='Premium'
        ),
        row=1, col=1
    )
    
    # Volume distribution
    type_volume = flows_df.groupby('type')['volume'].sum()
    fig.add_trace(
        go.Bar(
            x=type_volume.index,
            y=type_volume.values,
            marker_color=['lightgreen', 'lightcoral'],
            name='Volume'
        ),
        row=1, col=2
    )
    
    # Sentiment breakdown
    sentiment_counts = flows_df['sentiment'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=['#28a745', '#dc3545', '#ffc107']
        ),
        row=2, col=1
    )
    
    # Trade types
    all_types = []
    for types_list in flows_df['trade_types']:
        all_types.extend(types_list)
    if all_types:
        types_series = pd.Series(all_types)
        type_counts = types_series.value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False
    )
    
    return fig

def format_number(num):
    """Format large numbers"""
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.0f}K"
    else:
        return f"${num:.0f}"

def main():
    st.title("🌊 Options Flow & Unusual Activity Scanner")
    st.markdown("Real-time detection of large trades, sweeps, and institutional orders")
    
    # Sidebar
    st.sidebar.header("Scanner Settings")
    
    # Symbol input (pre-populate from session state if available)
    default_symbols = st.session_state.get('selected_symbol', 'SPY, QQQ, AAPL, TSLA, NVDA')
    symbols_input = st.sidebar.text_input(
        "Symbols to Monitor (comma-separated)",
        value=default_symbols,
        help="Enter stock symbols to scan for unusual flow"
    )
    
    # Clear the session state after using it
    if 'selected_symbol' in st.session_state:
        del st.session_state['selected_symbol']
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Filters
    st.sidebar.subheader("Flow Filters")
    
    min_premium = st.sidebar.number_input(
        "Minimum Premium ($)",
        min_value=1000,
        max_value=5000000,
        value=70000,
        step=5000,
        help="Filter trades by minimum dollar premium"
    )
    
    min_volume = st.sidebar.number_input(
        "Minimum Volume",
        min_value=10,
        max_value=10000,
        value=1000,
        step=50,
        help="Minimum contract volume to detect"
    )
    
    flow_types = st.sidebar.multiselect(
        "Trade Types to Show",
        ["BLOCK", "SWEEP", "UNUSUAL", "NEW"],
        default=["BLOCK", "SWEEP", "UNUSUAL"],
        help="Filter by trade type"
    )
    
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment Filter",
        ["BULLISH", "BEARISH", "NEUTRAL"],
        default=["BULLISH", "BEARISH", "NEUTRAL"]
    )
    
    option_type_filter = st.sidebar.multiselect(
        "Option Type",
        ["CALL", "PUT"],
        default=["CALL", "PUT"]
    )
    
    auto_refresh = st.sidebar.checkbox("🔴 Auto-Refresh (Live)", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    
    if st.sidebar.button("🔄 Scan Now") or auto_refresh:
        st.cache_data.clear()
    
    # Main content
    if not symbols:
        st.warning("Please enter at least one symbol to monitor.")
        return
    
    # Live indicator
    if auto_refresh:
        st.markdown('<h3 class="live-indicator">🔴 LIVE MONITORING</h3>', unsafe_allow_html=True)
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_flows = []
    
    # Scan each symbol
    for idx, symbol in enumerate(symbols):
        status_text.text(f"Scanning {symbol}... ({idx+1}/{len(symbols)})")
        progress_bar.progress((idx + 1) / len(symbols))
        
        try:
            options_data, underlying_price = get_options_data(symbol)
            
            if not options_data or not underlying_price:
                continue
            
            # Analyze flow
            flows = analyze_flow(options_data, underlying_price, min_premium, min_volume)
            
            # Add symbol to each flow
            for flow in flows:
                flow['symbol'] = symbol
                flow['underlying_price'] = underlying_price
            
            all_flows.extend(flows)
            
        except Exception as e:
            st.warning(f"Error scanning {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_flows:
        st.info("No significant flow detected with current filters. Try lowering the minimum premium threshold.")
        return
    
    # Convert to DataFrame
    df_flows = pd.DataFrame(all_flows)
    
    # Apply filters
    if flow_types:
        df_flows = df_flows[df_flows['trade_types'].apply(lambda x: any(t in x for t in flow_types))]
    
    if sentiment_filter:
        df_flows = df_flows[df_flows['sentiment'].isin(sentiment_filter)]
    
    if option_type_filter:
        df_flows = df_flows[df_flows['type'].isin(option_type_filter)]
    
    if df_flows.empty:
        st.info("No flows match the current filters.")
        return
    
    # Sort by premium
    df_flows = df_flows.sort_values('premium', ascending=False)
    
    # Summary metrics
    st.header("📊 Flow Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_premium = df_flows['premium'].sum()
        st.metric("Total Premium", format_number(total_premium))
    
    with col2:
        total_volume = df_flows['volume'].sum()
        st.metric("Total Volume", f"{total_volume:,.0f}")
    
    with col3:
        num_bullish = len(df_flows[df_flows['sentiment'] == 'BULLISH'])
        num_bearish = len(df_flows[df_flows['sentiment'] == 'BEARISH'])
        st.metric("Bull/Bear Ratio", f"{num_bullish}/{num_bearish}")
    
    with col4:
        call_premium = df_flows[df_flows['type'] == 'CALL']['premium'].sum()
        put_premium = df_flows[df_flows['type'] == 'PUT']['premium'].sum()
        pc_ratio = put_premium / call_premium if call_premium > 0 else 0
        st.metric("P/C Premium Ratio", f"{pc_ratio:.2f}")
    
    with col5:
        num_blocks = len(df_flows[df_flows['trade_types'].apply(lambda x: 'BLOCK' in x)])
        st.metric("Block Trades", num_blocks)
    
    # Summary charts
    st.subheader("📈 Flow Analysis")
    flow_chart = create_flow_summary_chart(df_flows)
    st.plotly_chart(flow_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Display individual flows
    st.header(f"🔥 Detected Flows ({len(df_flows)})")
    
    # Group by symbol
    for symbol in df_flows['symbol'].unique():
        symbol_flows = df_flows[df_flows['symbol'] == symbol].head(10)  # Top 10 per symbol
        
        if symbol_flows.empty:
            continue
        
        underlying_price = symbol_flows.iloc[0]['underlying_price']
        
        with st.expander(f"💰 {symbol} - ${underlying_price:.2f} ({len(symbol_flows)} flows)", expanded=True):
            for _, flow in symbol_flows.iterrows():
                display_flow_alert(symbol, flow, underlying_price)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Total Flows: {len(df_flows)}")

if __name__ == "__main__":
    main()
