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
import yfinance as yf
import requests
import sqlite3
import logging
import hashlib
from io import StringIO
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import pytz
# --- BEGIN: CBOE/YFINANCE DATA SOURCE HELPERS ---
US_EASTERN = pytz.timezone('US/Eastern')

@st.cache_data(ttl=600)
def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        required_columns = ['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Last Price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return None
        df = df.dropna(subset=['Symbol', 'Expiration', 'Strike Price', 'Call/Put'])
        df = df[df['Volume'] >= 50].copy()
        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
        df = df.dropna(subset=['Expiration'])
        df = df[df['Expiration'].dt.date >= datetime.now().date()]
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_all_options_data() -> pd.DataFrame:
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    data_frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_data_from_url, url) for url in urls]
        for future in futures:
            df = future.result()
            if df is not None and not df.empty:
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_price(symbol: str) -> Optional[float]:
    try:
        if symbol.startswith('$') or len(symbol) > 5:
            return None
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.fast_info
            price = info.get('last_price')
            if price and price > 0:
                return float(price)
        except:
            pass
        try:
            info = ticker.info
            price = (info.get('currentPrice') or 
                    info.get('regularMarketPrice') or 
                    info.get('previousClose'))
            if price and price > 0:
                return float(price)
        except:
            pass
        return None
    except Exception:
        return None
# --- END: CBOE/YFINANCE DATA SOURCE HELPERS ---
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


# Remove SchwabClient import; use CBOE/yfinance data source

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
## REMOVED: SchwabClient and get_options_data (now using CBOE/yfinance)

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
    # support both dicts and pandas Series
    getf = lambda k, d=None: flow.get(k, d) if hasattr(flow, 'get') else (flow[k] if k in flow else d)
    sentiment = getf('sentiment', 'NEUTRAL')
    if sentiment == 'BULLISH':
        alert_class = 'bullish-flow'
    elif sentiment == 'BEARISH':
        alert_class = 'bearish-flow'
    else:
        alert_class = 'neutral-flow'
    
    # Add special classes
    extra_classes = []
    trade_types = getf('trade_types', []) or []
    if 'BLOCK' in trade_types:
        extra_classes.append('block-trade')
    if 'SWEEP' in trade_types:
        extra_classes.append('sweep-trade')
    if 'UNUSUAL' in trade_types:
        extra_classes.append('unusual-volume')
    
    all_classes = f"{alert_class} {' '.join(extra_classes)}"
    
    # Create badges
    opt_type = getf('type', 'CALL')
    badges = f'<span class="trade-badge badge-{str(opt_type).lower()}">{opt_type}</span>'
    for trade_type in trade_types:
        badge_color = 'sweep' if trade_type == 'SWEEP' else 'block' if trade_type == 'BLOCK' else 'unusual'
        badges += f'<span class="trade-badge badge-{badge_color}">{trade_type}</span>'
    
    # Format premium
    premium = getf('premium', 0) or 0
    try:
        premium_val = float(premium)
    except Exception:
        premium_val = 0
    if premium_val >= 1_000_000:
        premium_str = f"${premium_val/1_000_000:.2f}M"
    else:
        premium_str = f"${premium_val/1_000:.0f}K"
    
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
                <strong>Strike:</strong> ${getf('strike', 0):.2f} ({getf('moneyness', 'N/A')})
            </div>
            <div class="metric-item">
                <strong>Expiry:</strong> {getf('expiry', '')} ({getf('days_to_exp', 0)} DTE)
            </div>
            <div class="metric-item">
                <strong>Price:</strong> ${getf('price', 0):.2f}
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <strong>Volume:</strong> {int(getf('volume', 0)):,}
            </div>
            <div class="metric-item">
                <strong>OI:</strong> {int(getf('open_interest', 0)):,}
            </div>
            <div class="metric-item">
                <strong>Vol/OI:</strong> {float(getf('vol_oi_ratio', 0.0)):.2f}x
            </div>
            <div class="metric-item">
                <strong>IV:</strong> {float(getf('implied_vol', 0.0)):.1f}%
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <strong>Delta:</strong> {float(getf('delta', 0.0)):.3f}
            </div>
            <div class="metric-item">
                <strong>Bid/Ask:</strong> ${getf('bid', 0) or 0:.2f} / ${getf('ask', 0) or 0:.2f}
            </div>
            <div class="metric-item">
                <strong>Detected:</strong> { (getf('timestamp') if getf('timestamp') is not None else datetime.now()).strftime('%H:%M:%S') }
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

    # Scanner Settings - move to top of page
    with st.container():
        st.header("Flow Filters")
        min_premium = st.number_input(
            "Minimum Premium ($)",
            min_value=1000,
            max_value=5000000,
            value=70000,
            step=5000,
            help="Filter trades by minimum dollar premium"
        )
        min_volume = st.number_input(
            "Minimum Volume",
            min_value=10,
            max_value=10000,
            value=1000,
            step=50,
            help="Minimum contract volume to detect"
        )
        auto_refresh = st.checkbox("🔴 Auto-Refresh (Live)", value=False)
        refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
        if st.button("🔄 Scan Now") or auto_refresh:
            st.cache_data.clear()
        # Symbols input is now optional and not shown in filters
        symbols = []
        # Keep filter variables defined so downstream code is stable
        flow_types = []
        sentiment_filter = []
        option_type_filter = []
    
    # Main content
    # Symbols are optional, so do not require them
    
    # Live indicator
    if auto_refresh:
        st.markdown('<h3 class="live-indicator">🔴 LIVE MONITORING</h3>', unsafe_allow_html=True)
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_flows = []
    
    # Scan all flows (no symbol filter)
    status_text.text("Scanning all symbols...")
    progress_bar.progress(0.5)
    df = fetch_all_options_data()
    if df.empty:
        st.info("No options data available.")
        return
    # Compute premium and normalize column names so downstream code works
    # CBOE CSV uses 'Volume' and 'Last Price' columns; compute premium = volume * last_price * 100
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    if 'Last Price' in df.columns:
        df['Last Price'] = pd.to_numeric(df['Last Price'], errors='coerce').fillna(0)
    elif 'LastPrice' in df.columns:
        df['Last Price'] = pd.to_numeric(df['LastPrice'], errors='coerce').fillna(0)
    else:
        df['Last Price'] = 0

    df['Premium'] = df['Volume'] * df['Last Price'] * 100
    df['premium'] = df['Premium']
    df['volume'] = df['Volume']
    # Filter by premium and volume (use computed columns)
    df_flows = df[(df['premium'] >= min_premium) & (df['volume'] >= min_volume)].copy()
    if df_flows.empty:
        st.info("No significant flow detected with current filters. Try lowering the minimum premium threshold.")
        return
    # Normalize and enrich dataframe to match downstream expectations
    # symbol, type, strike, expiry, days_to_exp, price
    if 'Symbol' in df_flows.columns:
        df_flows['symbol'] = df_flows['Symbol'].astype(str).str.upper()
    else:
        df_flows['symbol'] = ''

    # Map call/put
    if 'Call/Put' in df_flows.columns:
        df_flows['type'] = df_flows['Call/Put'].apply(lambda x: 'CALL' if str(x).strip().upper().startswith('C') else 'PUT')
    else:
        df_flows['type'] = 'CALL'

    df_flows['strike'] = pd.to_numeric(df_flows.get('Strike Price', df_flows.get('Strike', pd.Series([0]*len(df_flows)))), errors='coerce')
    if 'Expiration' in df_flows.columns:
        df_flows['Expiration'] = pd.to_datetime(df_flows['Expiration'], errors='coerce')
        df_flows['expiry'] = df_flows['Expiration'].dt.strftime('%Y-%m-%d')
        df_flows['days_to_exp'] = (df_flows['Expiration'] - pd.Timestamp.now()).dt.days
    else:
        df_flows['expiry'] = ''
        df_flows['days_to_exp'] = 0

    df_flows['price'] = pd.to_numeric(df_flows.get('Last Price', df_flows.get('LastPrice', pd.Series([0]*len(df_flows)))), errors='coerce').fillna(0)

    # Fetch underlying prices per symbol (cached)
    unique_symbols = df_flows['symbol'].unique().tolist()
    price_map = {}
    for sym in unique_symbols:
        try:
            price_map[sym] = get_stock_price(sym) or 0
        except Exception:
            price_map[sym] = 0

    df_flows['underlying_price'] = df_flows['symbol'].map(price_map).fillna(0)

    # Defaults for missing fields used elsewhere
    df_flows['open_interest'] = 0
    df_flows['trade_types'] = df_flows.apply(lambda _: [], axis=1)
    df_flows['sentiment'] = 'NEUTRAL'
    df_flows['vol_oi_ratio'] = 0.0
    df_flows['bid'] = np.nan
    df_flows['ask'] = np.nan
    df_flows['delta'] = 0.0
    df_flows['gamma'] = 0.0
    df_flows['vega'] = 0.0
    df_flows['implied_vol'] = 0.0

    # Sort by premium descending
    df_flows = df_flows.sort_values('premium', ascending=False).reset_index(drop=True)
    progress_bar.empty()
    status_text.empty()
    
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
    
    # Summary metrics removed per user request
    
    # Display individual flows
    st.header(f"🔥 Detected Flows ({len(df_flows)})")
    
    # Show Top 15 individual plays by premium, split into Index plays and Stock plays
    st.markdown("### 🔥 Top Plays (Index vs Stocks)")
    sorted_plays = df_flows.sort_values('premium', ascending=False).reset_index(drop=True)

    # Define index-like symbols (restrict to ETF proxies only per user request)
    # Keep only SPY, QQQ, IWM, DIA as 'index plays' in the UI and ignore SPX/SPXW/NDX/RUT etc.
    index_symbols = {s.upper() for s in ['SPY', 'QQQ', 'IWM', 'DIA']}

    # Use case-insensitive matching to be safe
    index_mask = sorted_plays['symbol'].astype(str).str.upper().isin(index_symbols)
    index_plays = sorted_plays[index_mask].head(15)
    stock_plays = sorted_plays[~index_mask].head(15)

    col_idx, col_stk = st.columns(2)

    with col_idx:
        st.markdown("#### 📈 Index Plays (Top 15)")
        if index_plays.empty:
            st.info("No index plays")
        else:
            for i, row in index_plays.reset_index(drop=True).iterrows():
                sym = row.get('symbol', '')
                strike = row.get('strike', 0)
                opt_type = row.get('type', 'CALL')
                expiry = row.get('expiry', '')
                prem = float(row.get('premium', 0) or 0)
                vol = int(row.get('volume', 0) or 0)
                price = float(row.get('price', 0) or 0)
                underlying_price = float(row.get('underlying_price', 0) or 0)
                leg = f"{strike:.0f}{'C' if opt_type == 'CALL' else 'P'}"
                summary = f"{i+1}) {sym} {leg} {expiry} — ${prem:,.0f} | Vol {vol:,} | Price ${price:.2f}"
                with st.expander(summary, expanded=False):
                    display_flow_alert(sym, row, underlying_price)

    with col_stk:
        st.markdown("#### 🧾 Stock Plays (Top 15)")
        if stock_plays.empty:
            st.info("No stock plays")
        else:
            for i, row in stock_plays.reset_index(drop=True).iterrows():
                sym = row.get('symbol', '')
                strike = row.get('strike', 0)
                opt_type = row.get('type', 'CALL')
                expiry = row.get('expiry', '')
                prem = float(row.get('premium', 0) or 0)
                vol = int(row.get('volume', 0) or 0)
                price = float(row.get('price', 0) or 0)
                underlying_price = float(row.get('underlying_price', 0) or 0)
                leg = f"{strike:.0f}{'C' if opt_type == 'CALL' else 'P'}"
                summary = f"{i+1}) {sym} {leg} {expiry} — ${prem:,.0f} | Vol {vol:,} | Price ${price:.2f}"
                with st.expander(summary, expanded=False):
                    display_flow_alert(sym, row, underlying_price)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Total Flows: {len(df_flows)}")

if __name__ == "__main__":
    main()
