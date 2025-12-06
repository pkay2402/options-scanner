"""
Whale Flows Scanner
Scans top tech stocks for highest whale score options activity
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="Whale Flows Scanner",
    page_icon="🐋",
    layout="wide"
)

st.title("🐋 Whale Flows Scanner")
st.markdown("**Discover high whale activity options across top tech stocks**")

# Initialize session state for persisting results (dict keyed by expiry date)
if 'whale_flows_data' not in st.session_state:
    st.session_state.whale_flows_data = {}
# Reset if old format (DataFrame instead of dict)
elif not isinstance(st.session_state.whale_flows_data, dict):
    st.session_state.whale_flows_data = {}
if 'whale_flows_min_score' not in st.session_state:
    st.session_state.whale_flows_min_score = None
if 'whale_flows_top_n' not in st.session_state:
    st.session_state.whale_flows_top_n = None
if 'oi_flows_data' not in st.session_state:
    st.session_state.oi_flows_data = {}

# Top tech stocks
TOP_TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'AMD',
    'CRM', 'GS', 'NFLX', 'IBIT', 'COIN',
    'APP', 'PLTR', 'SNOW', 'TEAM', 'CRWD', 'SPY', 'QQQ'
]

# Value stocks watchlist
VALUE_STOCKS = [
    'AXP', 'JPM', 'C', 'WFC', 'XOM',
    'CVX', 'PG', 'JNJ', 'UNH', 'V',
    'MA', 'HD', 'WMT', 'KO', 'PEP',
    'MRK', 'ABBV', 'CAT', 'TMO', 'LLY',
    'DIA', 'IWM', 'MCD', 'NKE', 'GS', 'AMGN', 'MMM', 'BA', 'HON', 'COP'
]

def get_next_friday():
    """Get next Friday date for weekly expiry"""
    today = datetime.now().date()
    weekday = today.weekday()
    days_to_friday = (4 - weekday) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    return today + timedelta(days=days_to_friday)

def get_next_4_fridays():
    """Get next 4 weekly Friday dates"""
    fridays = []
    first_friday = get_next_friday()
    for i in range(4):
        fridays.append(first_friday + timedelta(weeks=i))
    return fridays

def send_to_discord(flows_df, expiry_date, min_whale_score):
    """Send whale flows data to Discord via webhook in chunks"""
    try:
        # Get Discord webhook URL from secrets (try multiple paths for compatibility)
        webhook_url = None
        
        # Try root level
        if "discord_webhook" in st.secrets:
            webhook_url = st.secrets["discord_webhook"]
        # Try under alerts section
        elif "alerts" in st.secrets and "discord_webhook" in st.secrets["alerts"]:
            webhook_url = st.secrets["alerts"]["discord_webhook"]
        
        if not webhook_url:
            st.error("⚠️ Discord webhook URL not configured. Please add 'discord_webhook' to secrets")
            st.info(f"Debug: Available secrets keys: {list(st.secrets.keys())}")
            return False
        
        # Format header message
        header = f"🐋 **WHALE FLOWS SCANNER RESULTS**\n"
        header += f"📅 Expiration: {expiry_date.strftime('%B %d, %Y')}\n"
        header += f"📊 Total Results: {len(flows_df)} | Min Score: {min_whale_score:,}\n"
        header += f"📈 Avg Whale Score: {flows_df['whale_score'].mean():,.0f}\n"
        header += f"🔢 Total Volume: {flows_df['volume'].sum():,.0f}\n"
        call_count = len(flows_df[flows_df['type'] == 'CALL'])
        put_count = len(flows_df[flows_df['type'] == 'PUT'])
        header += f"📊 Call/Put: {call_count}/{put_count}\n"
        header += "―" * 50 + "\n\n"
        
        # Send header
        response = requests.post(webhook_url, json={"content": header})
        if response.status_code != 204:
            st.error(f"Failed to send header to Discord: {response.status_code}")
            return False
        
        # Format data in table format using code blocks
        messages = []
        current_message = "```\n"
        current_message += f"{'Sym':<6} {'Type':<4} {'Strike':<8} {'Dist%':<7} {'Whale':>8} {'Vol':>7} {'OI':>7} {'IV%':>5}\n"
        current_message += "─" * 70 + "\n"
        
        for idx, row in flows_df.head(50).iterrows():  # Limit to top 50
            emoji = "🟢" if row['type'] == 'CALL' else "🔴"
            distance = ((row['strike'] - row['underlying_price']) / row['underlying_price'] * 100)
            
            line = f"{row['symbol']:<6} {row['type']:<4} ${row['strike']:<7.2f} {distance:>+6.1f}% {int(row['whale_score']):>8,} {int(row['volume']):>7,} {int(row['open_interest']):>7,} {row['iv']:>5.1f}\n"
            
            # Check if adding this line would exceed Discord's 2000 char limit
            # Account for closing ``` and buffer
            if len(current_message) + len(line) + 10 > 1900:
                current_message += "```"
                messages.append(current_message)
                current_message = "```\n"
                current_message += f"{'Sym':<6} {'Type':<4} {'Strike':<8} {'Dist%':<7} {'Whale':>8} {'Vol':>7} {'OI':>7} {'IV%':>5}\n"
                current_message += "─" * 70 + "\n"
            
            current_message += line
        
        # Close the final message
        if current_message != "```\n":
            current_message += "```"
            messages.append(current_message)
        
        # Send each message chunk
        for i, msg in enumerate(messages):
            response = requests.post(webhook_url, json={"content": msg})
            if response.status_code != 204:
                st.error(f"Failed to send message {i+1} to Discord: {response.status_code}")
                return False
            # Small delay between messages to avoid rate limits
            import time
            time.sleep(0.5)
        
        return True
        
    except Exception as e:
        st.error(f"Error sending to Discord: {str(e)}")
        logger.error(f"Discord webhook error: {str(e)}")
        return False

@st.cache_data(ttl=300, show_spinner=False)
def scan_stock_combined(symbol: str, expiry_date: str):
    """
    Combined scanner: fetch API data once, calculate both whale scores and OI scores
    Returns: {'whale': {...}, 'oi': [...]}
    """
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        # Get quote (SINGLE API CALL)
        quote_response = client.get_quotes([symbol])
        if not quote_response or symbol not in quote_response:
            return None
        
        underlying_price = quote_response[symbol]['quote']['lastPrice']
        underlying_volume = quote_response[symbol]['quote'].get('totalVolume', 0)
        
        # If underlying volume is 0, skip this stock (no trading activity)
        if underlying_volume == 0:
            return None
        
        # Get options chain (SINGLE API CALL)
        options_response = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=expiry_date,
            to_date=expiry_date
        )
        
        if not options_response or 'callExpDateMap' not in options_response:
            return None
        
        # Process options and calculate BOTH whale scores and OI scores
        whale_options = []
        oi_options = []
        
        # Process calls
        if 'callExpDateMap' in options_response:
            for exp_date, strikes in options_response['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Common data extraction
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        mark_price = contract.get('mark', contract.get('last', 1))
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        gamma = contract.get('gamma', 0)
                        
                        # Skip if no volume or price
                        if volume == 0 or mark_price == 0:
                            continue
                        
                        # WHALE SCORE CALCULATION (strikes within ±5%)
                        if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                            leverage = delta * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            # Calculate GEX
                            gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            whale_options.append({
                                'symbol': symbol,
                                'strike': strike,
                                'type': 'CALL',
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'gamma': gamma,
                                'gex': gex,
                                'mark': mark_price,
                                'delta': delta,
                                'iv': contract.get('volatility', 0),
                                'underlying_price': underlying_price
                            })
                        
                        # OI SCORE CALCULATION (strikes within ±10%, Vol/OI >= 3.0)
                        if abs(strike - underlying_price) / underlying_price <= 0.10:
                            vol_oi_ratio = volume / oi
                            
                            if vol_oi_ratio >= 3.0:
                                notional_value = volume * mark_price * 100
                                oi_score = vol_oi_ratio * notional_value / 1000
                                
                                oi_options.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'type': 'CALL',
                                    'volume': volume,
                                    'open_interest': oi,
                                    'vol_oi_ratio': vol_oi_ratio,
                                    'oi_score': round(oi_score, 0),
                                    'mark': mark_price,
                                    'notional': notional_value,
                                    'delta': delta,
                                    'iv': contract.get('volatility', 0),
                                    'underlying_price': underlying_price
                                })
        
        # Process puts
        if 'putExpDateMap' in options_response:
            for exp_date, strikes in options_response['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Common data extraction
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        mark_price = contract.get('mark', contract.get('last', 1))
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        gamma = contract.get('gamma', 0)
                        
                        # Skip if no volume or price
                        if volume == 0 or mark_price == 0:
                            continue
                        
                        # WHALE SCORE CALCULATION (strikes within ±5%)
                        if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                            leverage = abs(delta) * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            # Calculate GEX (negative for puts from dealer perspective)
                            gex = -gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            whale_options.append({
                                'symbol': symbol,
                                'strike': strike,
                                'type': 'PUT',
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'gamma': gamma,
                                'gex': gex,
                                'mark': mark_price,
                                'delta': delta,
                                'iv': contract.get('volatility', 0),
                                'underlying_price': underlying_price
                            })
                        
                        # OI SCORE CALCULATION (strikes within ±10%, Vol/OI >= 3.0)
                        if abs(strike - underlying_price) / underlying_price <= 0.10:
                            vol_oi_ratio = volume / oi
                            
                            if vol_oi_ratio >= 3.0:
                                notional_value = volume * mark_price * 100
                                oi_score = vol_oi_ratio * notional_value / 1000
                                
                                oi_options.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'type': 'PUT',
                                    'volume': volume,
                                    'open_interest': oi,
                                    'vol_oi_ratio': vol_oi_ratio,
                                    'oi_score': round(oi_score, 0),
                                    'mark': mark_price,
                                    'notional': notional_value,
                                    'delta': delta,
                                    'iv': contract.get('volatility', 0),
                                    'underlying_price': underlying_price
                                })
        
        # Prepare return data
        result = {'whale': None, 'oi': None}
        
        # Calculate summary stats for whale flows
        if whale_options:
            df = pd.DataFrame(whale_options)
            call_vol = df[df['type'] == 'CALL']['volume'].sum()
            put_vol = df[df['type'] == 'PUT']['volume'].sum()
            
            # Find max GEX strike
            max_gex_row = df.loc[df['gex'].abs().idxmax()]
            
            # Find call and put walls
            call_wall = df[df['type'] == 'CALL'].loc[df[df['type'] == 'CALL']['volume'].idxmax()] if len(df[df['type'] == 'CALL']) > 0 else None
            put_wall = df[df['type'] == 'PUT'].loc[df[df['type'] == 'PUT']['volume'].idxmax()] if len(df[df['type'] == 'PUT']) > 0 else None
            
            summary = {
                'call_volume': call_vol,
                'put_volume': put_vol,
                'vol_ratio': (put_vol - call_vol) / max(call_vol, put_vol, 1) * 100,
                'max_gex_strike': max_gex_row['strike'],
                'max_gex_value': max_gex_row['gex'],
                'call_wall_strike': call_wall['strike'] if call_wall is not None else None,
                'put_wall_strike': put_wall['strike'] if put_wall is not None else None
            }
            
            result['whale'] = {
                'options': whale_options,
                'summary': summary
            }
        
        # Return OI flows
        if oi_options:
            result['oi'] = oi_options
        
        return result if (result['whale'] or result['oi']) else None
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {str(e)}")
        return None

# Settings
st.markdown("## ⚙️ Scanner Settings")

# Watchlist selector
st.markdown("### 📋 Select Watchlist")
watchlist_col1, watchlist_col2 = st.columns([1, 3])

with watchlist_col1:
    watchlist_type = st.radio(
        "Watchlist Type",
        ["Tech Stocks", "Value Stocks"],
        help="Choose which stock list to scan",
        horizontal=True
    )

with watchlist_col2:
    # Display selected watchlist
    if watchlist_type == "Tech Stocks":
        selected_stocks = TOP_TECH_STOCKS
        st.info(f"**Tech Stocks ({len(selected_stocks)}):** {', '.join(selected_stocks)}")
    else:
        selected_stocks = VALUE_STOCKS
        st.info(f"**Value Stocks ({len(selected_stocks)}):** {', '.join(selected_stocks)}")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    min_whale_score = st.number_input(
        "Minimum Whale Score",
        min_value=0,
        max_value=10000,
        value=50,
        step=20,
        help="Filter options with whale score above this threshold"
    )

with col2:
    top_n = st.slider(
        "Top Results per Stock",
        min_value=1,
        max_value=10,
        value=3,
        help="Show top N whale flows per stock"
    )

with col3:
    custom_symbol = st.text_input(
        "Custom Symbol (Only)",
        placeholder="e.g., COIN, SHOP",
        help="Scan only this custom symbol (ignores default list)"
    ).upper().strip()

# Get next 4 weekly Fridays
expiry_dates = get_next_4_fridays()
st.caption(f"📅 Will scan {len(expiry_dates)} weekly expiries: {', '.join([d.strftime('%b %d') for d in expiry_dates])}")

# Scan button
scan_col1, scan_col2 = st.columns([1, 5])
with scan_col1:
    scan_button = st.button("🔍 Scan Whale Flows", type="primary", width='stretch')

if scan_button:
    st.markdown("---")
    
    # Build stock list - if custom symbol provided, scan only that
    if custom_symbol:
        stocks_to_scan = [custom_symbol]
        st.info(f"🎯 Scanning custom symbol: **{custom_symbol}**")
    else:
        # Use selected watchlist
        if watchlist_type == "Tech Stocks":
            stocks_to_scan = TOP_TECH_STOCKS.copy()
            st.info(f"📊 Scanning **{len(stocks_to_scan)} Tech Stocks**: {', '.join(stocks_to_scan[:10])}{'...' if len(stocks_to_scan) > 10 else ''}")
        else:
            stocks_to_scan = VALUE_STOCKS.copy()
            st.info(f"📊 Scanning **{len(stocks_to_scan)} Value Stocks**: {', '.join(stocks_to_scan[:10])}{'...' if len(stocks_to_scan) > 10 else ''}")
    
    st.markdown(f"### 🐋 Scanning {len(stocks_to_scan)} Stocks across {len(expiry_dates)} Expiries...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results
    st.session_state.whale_flows_data = {}
    st.session_state.oi_flows_data = {}
    
    total_operations = len(stocks_to_scan) * len(expiry_dates)
    current_operation = 0
    
    # Scan each expiry date (COMBINED - Single API call per stock!)
    for expiry_date in expiry_dates:
        all_whale_flows = []
        all_oi_flows = []
        
        for symbol in stocks_to_scan:
            current_operation += 1
            status_text.text(f"Scanning {symbol} for {expiry_date.strftime('%b %d')}... ({current_operation}/{total_operations})")
            progress_bar.progress(current_operation / total_operations)
            
            # Combined scan - fetches API data once, returns both whale and OI results
            result = scan_stock_combined(symbol, expiry_date.strftime('%Y-%m-%d'))
            
            if result:
                # Process whale flows
                if result['whale'] and result['whale']['options']:
                    whale_data = result['whale']
                    # Filter by minimum whale score and get top N
                    df = pd.DataFrame(whale_data['options'])
                    df = df[df['whale_score'] >= min_whale_score]
                    df = df.nlargest(top_n, 'whale_score')
                    
                    # Add summary data to each row
                    for _, row in df.iterrows():
                        flow_data = row.to_dict()
                        flow_data.update({
                            'call_volume': whale_data['summary']['call_volume'],
                            'put_volume': whale_data['summary']['put_volume'],
                            'vol_ratio': whale_data['summary']['vol_ratio'],
                            'max_gex_strike': whale_data['summary']['max_gex_strike'],
                            'max_gex_value': whale_data['summary']['max_gex_value'],
                            'call_wall_strike': whale_data['summary']['call_wall_strike'],
                            'put_wall_strike': whale_data['summary']['put_wall_strike']
                        })
                        all_whale_flows.append(flow_data)
                
                # Process OI flows
                if result['oi']:
                    all_oi_flows.extend(result['oi'])
        
        # Store results for this expiry
        if all_whale_flows:
            st.session_state.whale_flows_data[expiry_date.strftime('%Y-%m-%d')] = pd.DataFrame(all_whale_flows).sort_values('whale_score', ascending=False)
        
        if all_oi_flows:
            st.session_state.oi_flows_data[expiry_date.strftime('%Y-%m-%d')] = pd.DataFrame(all_oi_flows).sort_values('oi_score', ascending=False)
    
    progress_bar.empty()
    status_text.empty()
    
    # Store settings
    st.session_state.whale_flows_min_score = min_whale_score
    st.session_state.whale_flows_top_n = top_n

# Display results if they exist in session state
if st.session_state.whale_flows_data or st.session_state.oi_flows_data:
    min_whale_score = st.session_state.whale_flows_min_score
    top_n = st.session_state.whale_flows_top_n
    
    # Create main tabs for Whale Flows vs OI Flows
    main_tabs = st.tabs(["🐋 Whale Flows (VALR)", "📊 OI Flows (Fresh Positioning)"])
    
    # TAB 1: Whale Flows
    with main_tabs[0]:
        if st.session_state.whale_flows_data:
            st.markdown("### Top Whale Flows by Expiration")
            st.caption(f"Min Whale Score: {min_whale_score} | Top {top_n} per stock")
            
            # Create tabs for each expiry
            expiry_dates_with_data = sorted(st.session_state.whale_flows_data.keys())
            if expiry_dates_with_data:
                tab_labels = [datetime.strptime(d, '%Y-%m-%d').strftime('%b %d, %Y') for d in expiry_dates_with_data]
                tabs = st.tabs(tab_labels)
                
                for tab, expiry_str in zip(tabs, expiry_dates_with_data):
                    with tab:
                        flows_df = st.session_state.whale_flows_data[expiry_str]
                        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                        
                        st.markdown(f"#### {len(flows_df)} results for {expiry_date.strftime('%B %d, %Y')}")
    
                        # Display summary metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            avg_whale = flows_df['whale_score'].mean()
                            st.metric("Avg Whale Score", f"{avg_whale:,.0f}")
                        
                        with metric_col2:
                            total_vol = flows_df['volume'].sum()
                            st.metric("Total Volume", f"{total_vol:,.0f}")
                        
                        with metric_col3:
                            call_count = len(flows_df[flows_df['type'] == 'CALL'])
                            put_count = len(flows_df[flows_df['type'] == 'PUT'])
                            st.metric("Call/Put Split", f"{call_count}/{put_count}")
                        
                        with metric_col4:
                            unique_stocks = flows_df['symbol'].nunique()
                            st.metric("Stocks with Activity", unique_stocks)
                        
                        st.markdown("---")
                        
                        # Create display DataFrame
                        display_df = flows_df.copy()
                        display_df['Distance'] = ((display_df['strike'] - display_df['underlying_price']) / display_df['underlying_price'] * 100).round(2)
                        display_df['Distance'] = display_df['Distance'].apply(lambda x: f"{x:+.2f}%")
                        
                        # Format columns
                        display_cols = {
                            'symbol': 'Stock',
                            'type': 'Type',
                            'strike': 'Strike',
                            'Distance': 'Distance',
                            'whale_score': 'Whale Score',
                            'volume': 'Volume',
                            'open_interest': 'OI',
                            'vol_ratio': 'Vol Ratio',
                            'max_gex_strike': 'Max GEX Strike',
                            'max_gex_value': 'Max GEX',
                            'call_wall_strike': 'Call Wall',
                            'put_wall_strike': 'Put Wall',
                            'mark': 'Premium',
                            'iv': 'IV%'
                        }
                        
                        result_df = display_df[list(display_cols.keys())].copy()
                        result_df.columns = list(display_cols.values())
                        
                        # Format numeric columns
                        result_df['Whale Score'] = result_df['Whale Score'].apply(lambda x: f"{int(x):,}")
                        result_df['Volume'] = result_df['Volume'].apply(lambda x: f"{int(x):,}")
                        result_df['OI'] = result_df['OI'].apply(lambda x: f"{int(x):,}")
                        result_df['Vol Ratio'] = result_df['Vol Ratio'].apply(lambda x: f"{x:+.1f}%")
                        result_df['Max GEX Strike'] = result_df['Max GEX Strike'].apply(lambda x: f"${x:.2f}")
                        result_df['Max GEX'] = result_df['Max GEX'].apply(lambda x: f"${x/1e6:.2f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K")
                        result_df['Call Wall'] = result_df['Call Wall'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                        result_df['Put Wall'] = result_df['Put Wall'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                        result_df['Strike'] = result_df['Strike'].apply(lambda x: f"${x:.2f}")
                        result_df['Premium'] = result_df['Premium'].apply(lambda x: f"${x:.2f}")
                        result_df['IV%'] = result_df['IV%'].apply(lambda x: f"{x:.1f}%")
                        
                        # Style the dataframe
                        def color_type(val):
                            if val == 'CALL':
                                return 'background-color: #22c55e; color: white; font-weight: bold'
                            elif val == 'PUT':
                                return 'background-color: #ef4444; color: white; font-weight: bold'
                            return ''
                        
                        styled_df = result_df.style.map(color_type, subset=['Type'])
                        
                        st.dataframe(
                            styled_df,
                            width='stretch',
                            height=600
                        )
                        
                        # Download and Discord buttons
                        btn_col1, btn_col2 = st.columns([1, 1])
                        
                        with btn_col1:
                            csv = flows_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"whale_flows_{expiry_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                width='stretch',
                                key=f"download_{expiry_str}"
                            )
                        
                        with btn_col2:
                            if st.button("📤 Send to Discord", width='stretch', type="secondary", key=f"discord_{expiry_str}"):
                                with st.spinner("Sending to Discord..."):
                                    if send_to_discord(flows_df, expiry_date, min_whale_score):
                                        st.success("✅ Successfully sent to Discord!")
                                    else:
                                        st.error("❌ Failed to send to Discord")
        else:
            st.info("No whale flows data available. Click 'Scan Whale Flows' to begin.")
    
    # TAB 2: OI Flows (Fresh Positioning)
    with main_tabs[1]:
        if st.session_state.oi_flows_data:
            st.markdown("### 🆕 Fresh Institutional Positioning by Expiration")
            st.caption("📊 Volume ≥ 3.0x Open Interest = Massive new positions being opened")
            
            # Create tabs for each expiry
            expiry_dates_with_oi = sorted(st.session_state.oi_flows_data.keys())
            if expiry_dates_with_oi:
                oi_tab_labels = [datetime.strptime(d, '%Y-%m-%d').strftime('%b %d, %Y') for d in expiry_dates_with_oi]
                oi_tabs = st.tabs(oi_tab_labels)
                
                for oi_tab, oi_expiry_str in zip(oi_tabs, expiry_dates_with_oi):
                    with oi_tab:
                        oi_df = st.session_state.oi_flows_data[oi_expiry_str]
                        oi_expiry_date = datetime.strptime(oi_expiry_str, '%Y-%m-%d').date()
                        
                        st.markdown(f"#### {len(oi_df)} options with fresh positioning for {oi_expiry_date.strftime('%B %d, %Y')}")
                        
                        # Display summary metrics
                        oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)
                        
                        with oi_col1:
                            avg_ratio = oi_df['vol_oi_ratio'].mean()
                            st.metric("Avg Vol/OI Ratio", f"{avg_ratio:.2f}x")
                        
                        with oi_col2:
                            total_notional = oi_df['notional'].sum()
                            st.metric("Total Notional", f"${total_notional/1e6:.1f}M")
                        
                        with oi_col3:
                            call_count_oi = len(oi_df[oi_df['type'] == 'CALL'])
                            put_count_oi = len(oi_df[oi_df['type'] == 'PUT'])
                            st.metric("Call/Put Split", f"{call_count_oi}/{put_count_oi}")
                        
                        with oi_col4:
                            unique_stocks_oi = oi_df['symbol'].nunique()
                            st.metric("Stocks with Fresh OI", unique_stocks_oi)
                        
                        st.markdown("---")
                        
                        # Create display DataFrame for OI flows
                        oi_display_df = oi_df.copy()
                        oi_display_df['Distance'] = ((oi_display_df['strike'] - oi_display_df['underlying_price']) / oi_display_df['underlying_price'] * 100).round(2)
                        oi_display_df['Distance'] = oi_display_df['Distance'].apply(lambda x: f"{x:+.2f}%")
                        
                        # Format columns for OI display
                        oi_display_cols = {
                            'symbol': 'Stock',
                            'type': 'Type',
                            'strike': 'Strike',
                            'Distance': 'Distance',
                            'oi_score': 'OI Score',
                            'vol_oi_ratio': 'Vol/OI',
                            'volume': 'Volume',
                            'open_interest': 'OI',
                            'notional': 'Notional $',
                            'mark': 'Premium',
                            'delta': 'Delta',
                            'iv': 'IV%'
                        }
                        
                        oi_result_df = oi_display_df[list(oi_display_cols.keys())].copy()
                        oi_result_df.columns = list(oi_display_cols.values())
                        
                        # Format numeric columns
                        oi_result_df['OI Score'] = oi_result_df['OI Score'].apply(lambda x: f"{int(x):,}")
                        oi_result_df['Vol/OI'] = oi_result_df['Vol/OI'].apply(lambda x: f"{x:.2f}x")
                        oi_result_df['Volume'] = oi_result_df['Volume'].apply(lambda x: f"{int(x):,}")
                        oi_result_df['OI'] = oi_result_df['OI'].apply(lambda x: f"{int(x):,}")
                        oi_result_df['Notional $'] = oi_result_df['Notional $'].apply(lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
                        oi_result_df['Strike'] = oi_result_df['Strike'].apply(lambda x: f"${x:.2f}")
                        oi_result_df['Premium'] = oi_result_df['Premium'].apply(lambda x: f"${x:.2f}")
                        oi_result_df['Delta'] = oi_result_df['Delta'].apply(lambda x: f"{x:.3f}")
                        oi_result_df['IV%'] = oi_result_df['IV%'].apply(lambda x: f"{x:.1f}%")
                        
                        # Style the dataframe
                        def color_type_oi(val):
                            if val == 'CALL':
                                return 'background-color: #22c55e; color: white; font-weight: bold'
                            elif val == 'PUT':
                                return 'background-color: #ef4444; color: white; font-weight: bold'
                            return ''
                        
                        styled_oi_df = oi_result_df.style.map(color_type_oi, subset=['Type'])
                        
                        st.dataframe(
                            styled_oi_df,
                            width='stretch',
                            height=600
                        )
                        
                        # Download button for OI flows
                        oi_csv = oi_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download OI Flows (CSV)",
                            data=oi_csv,
                            file_name=f"oi_flows_{oi_expiry_date.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            width='stretch',
                            key=f"download_oi_{oi_expiry_str}"
                        )
        else:
            st.info("No OI flows data available. Click 'Scan Whale Flows' to begin.")
    
    # Check if no results at all
    if not st.session_state.whale_flows_data and not st.session_state.oi_flows_data:
        st.warning("⚠️ No flows found matching your criteria. Try lowering the minimum whale score.")

if not scan_button and not st.session_state.whale_flows_data and not st.session_state.oi_flows_data:
    # Show comprehensive trader's guide when not scanning
    with st.expander("📖 Complete Trader's Guide - Whale Flows Scanner", expanded=False):
        st.markdown("""
        ## **Whale Flows Scanner - Trader's Guide**
        
        ### **What This Tool Does**
        
        The Whale Flows Scanner identifies **institutional-grade options activity** across 20+ tech stocks by analyzing:
        1. **Whale Flows (VALR)** - Sophisticated institutional positioning using leverage and volatility
        2. **Fresh OI Flows** - Brand new positions being opened TODAY (Vol/OI ≥ 3.0x)
        
        This tool scans **4 weekly Friday expiries** simultaneously, giving you a complete view of near-term institutional positioning.
        
        ---
        
        ## **Key Concepts**
        
        ### **Whale Score (VALR Formula)**
        ```
        Whale Score = (Leverage Ratio × IV) × (Vol/OI) × (Option $ / Stock $) × 1000
        ```
        
        **Components:**
        - **Leverage Ratio**: Delta × Stock Price / Option Price (how much bang for buck)
        - **Implied Volatility (IV)**: Volatility adjusted leverage effect
        - **Vol/OI**: Fresh activity vs existing positions
        - **Dollar Volume Ratio**: Option notional vs underlying volume
        
        **What It Means:**
        - High whale score = Sophisticated institutional positioning
        - Combines price leverage, volatility exposure, and activity levels
        - Filters ATM options (±5% from current price) for maximum impact
        
        ### **OI Score (Fresh Positioning)**
        ```
        OI Score = (Vol/OI Ratio) × Notional Value / 1000
        ```
        
        **Logic:**
        - **Vol ≥ 3.0x OI** = Massive new positions opening
        - If OI = 1,000 and Vol = 3,000 → 3x more contracts traded TODAY than exist total
        - Catches institutions aggressively building positions
        - Filters strikes within ±10% of current price
        
        ---
        
        ## **How to Use - Step by Step**
        
        ### **1. Configure Scanner Settings**
        
        **Minimum Whale Score (0-10,000):**
        - **Default: 50** - Balanced filter
        - **Low (0-30)**: See all activity, noisy
        - **Medium (50-150)**: Quality institutional flows
        - **High (200+)**: Only extreme whale activity
        - *Recommendation: Start at 50, adjust based on results*
        
        **Top Results per Stock (1-10):**
        - Shows top N highest-scoring flows per symbol
        - **3 results** = Focus on strongest signals only
        - **5-7 results** = Balanced view
        - **10 results** = Full picture of activity
        - *Recommendation: 3-5 for actionable insights*
        
        **Custom Symbol:**
        - Override default tech stocks list
        - Scan only your specific ticker
        - Useful for earnings, events, or focused analysis
        - Leave blank to scan all 20 default tech stocks
        
        ### **2. Click "Scan Whale Flows"**
        
        **What Happens:**
        - Scans 20-22 stocks across 4 weekly Friday expiries
        - Single API call per stock = efficient & fast
        - Calculates both Whale Scores AND OI Scores simultaneously
        - Filters and ranks results
        - Stores data in tabs by expiration date
        
        **Scan Coverage:**
        - **Tech Stocks (22)**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO, ORCL, AMD, CRM, GS, NFLX, IBIT, COIN, APP, PLTR, SNOW, TEAM, CRWD, SPY, QQQ
        - **Value Stocks (22)**: AXP, JPM, C, WFC, XOM, CVX, PG, JNJ, UNH, V, MA, HD, WMT, KO, PEP, MRK, ABBV, CAT, TMO, LLY, DIA, IWM
        - **Expiries**: Next 4 weekly Fridays
        - **Time**: ~1-2 minutes for full scan
        
        ### **3. Analyze Results by Tab**
        
        ---
        
        ## **Tab 1: 🐋 Whale Flows (VALR)**
        
        ### **What It Shows**
        Institutional positioning using the sophisticated VALR formula - catches smart money making leveraged, volatility-adjusted plays.
        
        ### **Key Metrics Explained**
        
        | Metric | What It Means | Trading Use |
        |--------|---------------|-------------|
        | **Whale Score** | Combined institutional activity score | Higher = stronger institutional interest |
        | **Distance** | % from current price | Directional target or hedge distance |
        | **Volume** | Contracts traded today | Activity level, liquidity |
        | **OI** | Open Interest (total contracts) | Existing positioning size |
        | **Vol Ratio** | (Put Vol - Call Vol) / Max | **Positive = Bearish, Negative = Bullish** |
        | **Max GEX Strike** | Highest gamma exposure strike | Dealer hedging magnet, price attractor |
        | **Max GEX Value** | Dollar gamma at strike | Strength of hedging pressure |
        | **Call Wall** | Strike with max call volume | **Upside resistance level** |
        | **Put Wall** | Strike with max put volume | **Downside support level** |
        | **Premium** | Option price | Entry cost per contract |
        | **IV%** | Implied Volatility | Volatility pricing, elevated = expensive |
        
        ### **Summary Metrics (Top of Tab)**
        
        **Avg Whale Score:**
        - Average activity level across all results
        - >100 = Very active institutional session
        - <50 = Quieter day
        
        **Total Volume:**
        - Sum of all option contracts traded
        - Compare across expiries to find where institutions are focused
        
        **Call/Put Split:**
        - Number of call flows vs put flows
        - **More Calls** = Bullish bias
        - **More Puts** = Bearish bias or protection buying
        - **Balanced** = Neutral or complex strategies
        
        **Stocks with Activity:**
        - How many symbols triggered flows
        - Lower number = focused institutional attention
        - Higher number = broad market positioning
        
        ### **Trading Applications**
        
        **1. Identify Directional Bias**
        ```
        Example:
        NVDA - Multiple high whale score CALLS
        Vol Ratio: -35% (call heavy)
        Call Wall: $500
        → Institutions positioning for upside to $500
        ```
        
        **2. Find Price Targets**
        - Look at strikes with high whale scores
        - Distance % tells you institutional target
        - Max GEX Strike = magnetic price level
        
        **3. Spot Hedging Activity**
        ```
        High PUT whale scores + negative distance
        Vol Ratio: +45% (put heavy)
        → Institutions hedging downside risk
        → Often precedes volatility or protection buying
        ```
        
        **4. Confirm Levels**
        - Call Wall = Strong resistance (dealers sell into rallies)
        - Put Wall = Strong support (dealers buy dips)
        - Max GEX = Hedging magnet (price tends to pin here)
        
        **5. Follow Smart Money**
        - Highest whale scores = most sophisticated positioning
        - Look for patterns: multiple strikes same direction
        - Match with your technical analysis for confluence
        
        ### **Complete Trading Workflow**
        
        **Morning Scan (Pre-Market):**
        1. Run scan with default settings (min score 50, top 3)
        2. Check "Avg Whale Score" - is it elevated?
        3. Look at Call/Put Split - directional bias?
        4. Note Max GEX Strikes - potential pin levels
        5. Identify Call/Put Walls - key support/resistance
        
        **Intraday Monitoring:**
        1. Re-scan every 1-2 hours to catch fresh flows
        2. Compare with morning scan - new patterns?
        3. Watch price action at Walls and GEX strikes
        4. Volume spike at specific strike = confirmation
        
        **Position Entry:**
        ```
        Example Setup:
        Stock: AAPL at $185
        High Whale Score: $190 CALL (Distance +2.7%)
        Vol Ratio: -25% (bullish)
        Call Wall: $190
        Max GEX: $187.50
        
        Trade: Buy $187.50 calls
        Logic: Below call wall, at GEX magnet, institutions bullish
        Target: $190 (call wall)
        Stop: Below $185 (if breaks down)
        ```
        
        ---
        
        ## **Tab 2: 📊 OI Flows (Fresh Positioning)**
        
        ### **What It Shows**
        **BRAND NEW institutional positions opening TODAY** - catches institutions aggressively building positions in real-time.
        
        ### **Core Logic**
        ```
        Volume ≥ 3.0x Open Interest = Fresh Positioning
        
        Example:
        Open Interest: 1,000 contracts (total existing)
        Today's Volume: 3,000+ contracts (3x+ OI)
        → Institutions opening MASSIVE new positions
        ```
        
        ### **Key Metrics Explained**
        
        | Metric | What It Means | Trading Use |
        |--------|---------------|-------------|
        | **OI Score** | Vol/OI × Notional value | Ranks fresh positioning by size & conviction |
        | **Vol/OI Ratio** | Volume / Open Interest | **≥3.0 = 300%+ new positions** |
        | **Volume** | Contracts traded today | Raw activity level |
        | **OI** | Existing open interest | Baseline to compare against |
        | **Notional $** | Volume × Premium × 100 | Total dollar value deployed |
        | **Distance** | % from current price | Target or hedge distance |
        | **Premium** | Option price | Cost to follow the flow |
        | **Delta** | Price sensitivity | Directional exposure |
        | **IV%** | Implied Volatility | How expensive the option is |
        
        ### **Summary Metrics**
        
        **Avg Vol/OI Ratio:**
        - Average freshness across results
        - **3.0-5.0x** = Strong new positioning
        - **5.0-10x** = Very aggressive positioning
        - **>10x** = Extreme institutional conviction
        
        **Total Notional:**
        - Dollar value of all new positions
        - $50M+ = Significant institutional deployment
        - $100M+ = Massive capital allocation
        
        **Call/Put Split:**
        - Direction of fresh positioning
        - More calls = institutions betting on upside
        - More puts = institutions hedging or betting downside
        
        ### **Why Vol/OI ≥ 3.0x Matters**
        
        **Normal Market:**
        - Vol/OI typically 0.5-1.5x (low turnover)
        - Existing positions rolling or adjusting
        
        **Vol/OI ≥ 3.0x:**
        - 300%+ MORE contracts traded than exist
        - Institutions opening brand new positions
        - Strong directional conviction
        - Often precedes significant moves
        
        **Example:**
        ```
        TSLA $250 CALL
        Open Interest: 2,000
        Today's Volume: 8,000
        Vol/OI: 4.0x (400% fresh positioning!)
        Notional: $4.8M
        
        → Institutions aggressively bullish on TSLA $250
        → 8,000 contracts = $250M notional exposure
        → This wasn't here yesterday - brand new conviction
        ```
        
        ### **Trading Applications**
        
        **1. Follow Fresh Institutional Money**
        ```
        High OI Score + High Vol/OI = Follow this trade
        
        If institutions deploy $5M+ in fresh calls
        → They expect upside move
        → Consider buying same or nearby strikes
        ```
        
        **2. Spot Early Positioning**
        - Fresh OI often appears BEFORE the move
        - Institutions position ahead of catalysts
        - Use this as early warning system
        
        **3. Gauge Conviction Level**
        ```
        Vol/OI Ratio Interpretation:
        3.0-4.0x = Strong conviction
        4.0-6.0x = Very strong conviction
        6.0-10x = Extreme conviction
        >10x = Potentially parabolic setup (rare)
        ```
        
        **4. Identify Event Positioning**
        ```
        Multiple fresh OI flows same direction
        Same expiry date
        → Institutions positioning for event
        → Check earnings, Fed, data releases
        ```
        
        **5. Confirm Directional Bias**
        - Fresh calls = Bullish positioning
        - Fresh puts = Bearish or hedging
        - Mixed = Complex strategies or market-neutral
        
        ### **Complete Trading Workflow**
        
        **Daily Routine:**
        1. Run scan at market open
        2. Check "Avg Vol/OI" - is it elevated (>4.0)?
        3. Look for Vol/OI >5.0x - extreme conviction
        4. Note highest Notional values - biggest deployments
        5. Match fresh OI with Whale Flows for confirmation
        
        **Position Selection:**
        ```
        Example Setup:
        Stock: AMD at $145
        Fresh OI: $150 CALL
        Vol/OI: 6.2x (extreme!)
        Notional: $3.2M
        Distance: +3.4%
        Delta: 0.35
        
        Trade Decision:
        ✓ Extreme Vol/OI (6.2x)
        ✓ Large notional ($3.2M)
        ✓ Reasonable distance (+3.4%)
        ✓ Good delta (0.35)
        
        Action: Buy $147.50 or $150 calls
        Logic: Institutions betting big on AMD $150
        Target: $150-152
        Stop: Below $143
        Time: Match their expiry or go slightly longer
        ```
        
        **Risk Management:**
        - Don't blindly follow every flow
        - Verify with Vol Ratio and GEX from Whale tab
        - Consider using spreads to reduce cost
        - Size appropriately - institutions can be wrong
        
        ---
        
        ## **Comparing Both Tabs**
        
        ### **Whale Flows vs OI Flows**
        
        | Aspect | Whale Flows (VALR) | OI Flows (Fresh) |
        |--------|-------------------|------------------|
        | **Focus** | Sophisticated positioning | Brand new positions |
        | **Formula** | Complex (leverage + volatility) | Simple (Vol/OI ratio) |
        | **Strike Filter** | ±5% ATM | ±10% ATM |
        | **Best For** | Institutional strategy | Following fresh money |
        | **Timeframe** | Overall positioning | Intraday conviction |
        | **Signal** | Smart money placement | New capital deployment |
        
        ### **How to Use Together**
        
        **Maximum Conviction Setup:**
        ```
        1. High Whale Score in Whale Flows tab
        2. Same strike appears in OI Flows tab with Vol/OI >4.0
        3. Whale metrics support direction (Vol Ratio, GEX)
        
        → STRONGEST possible signal
        → Institutional positioning + fresh capital
        → Trade with high confidence
        ```
        
        **Divergence Alert:**
        ```
        Whale Flows: Bullish (call heavy)
        OI Flows: Bearish (fresh puts)
        
        → Institutions hedging
        → Or conflicting views
        → Proceed with caution
        ```
        
        **Confirmation Pattern:**
        ```
        Morning: Whale Flows show $190 call positioning
        Afternoon: Fresh OI flows at $190 calls (Vol/OI 5.0x)
        
        → Institutions doubling down
        → High conviction maintained
        → Strong bullish signal
        ```
        
        ---
        
        ## **Expiry Date Analysis**
        
        ### **How to Read Multiple Expiries**
        
        **This Week's Expiry:**
        - **0-3 DTE (Days to Expiry)**
        - Immediate positioning
        - Gamma risk highest
        - Usually for event plays or momentum
        - *Use for: Day trades, scalps, event trading*
        
        **Next Week's Expiry:**
        - **4-10 DTE**
        - Short-term directional views
        - Balance of theta decay and price movement
        - Most common for swing trades
        - *Use for: Weekly swings, earnings runs*
        
        **2-3 Weeks Out:**
        - **11-21 DTE**
        - Longer-term institutional positioning
        - Lower theta decay pressure
        - Room for thesis to play out
        - *Use for: Swing trades, trend following*
        
        **4 Weeks Out:**
        - **22-30 DTE**
        - Strategic positioning
        - Often pre-earnings or event positioning
        - Longer runway for movement
        - *Use for: Position trades, pre-event setups*
        
        ### **Cross-Expiry Analysis**
        
        **Same Strike Across Multiple Expiries:**
        ```
        This Week: AAPL $190 calls - High whale score
        Next Week: AAPL $190 calls - High whale score
        2 Weeks: AAPL $190 calls - High whale score
        
        → $190 is institutional TARGET
        → Multiple time horizons betting on same level
        → Very strong conviction
        ```
        
        **Different Strikes Per Expiry:**
        ```
        This Week: $185 calls (ATM)
        Next Week: $190 calls (+2.7%)
        2 Weeks: $195 calls (+5.4%)
        
        → Institutions expecting gradual uptrend
        → Staggered targets by time
        → Calendar spread opportunity
        ```
        
        **Put Protection Pattern:**
        ```
        This Week: Few puts
        2-4 Weeks: Heavy put positioning
        
        → Institutions hedging medium-term risk
        → Not concerned short-term
        → Potential event or volatility ahead
        ```
        
        ---
        
        ## **Advanced Trading Strategies**
        
        ### **Strategy 1: Follow the Whale**
        ```
        Setup:
        - Whale Score >200
        - Vol/OI >4.0x
        - Notional >$2M
        - Call/Put split confirms direction
        
        Execution:
        - Buy same strike as institutions
        - Or buy 1 strike closer to ATM for better delta
        - Match or extend expiry slightly
        - Size: 1-3% of portfolio per position
        
        Exit:
        - Target: 25-50% gain
        - Stop: -20% to -30% loss
        - Trail stops as it moves in your favor
        ```
        
        ### **Strategy 2: GEX Magnet Play**
        ```
        Setup:
        - Max GEX Strike identified
        - Multiple whale flows at/near GEX strike
        - Price within 2-3% of GEX strike
        
        Execution:
        - Sell premium (credit spreads) around GEX
        - Price likely to pin at GEX into expiry
        - Iron condor with GEX at center
        
        Exit:
        - Target: 50% profit
        - Stop: If price breaks GEX level decisively
        - Manage 2-3 days before expiry
        ```
        
        ### **Strategy 3: Fresh OI Momentum**
        ```
        Setup:
        - Vol/OI >6.0x (extreme)
        - Multiple stocks showing same pattern
        - Sector-wide fresh positioning
        
        Execution:
        - Enter immediately (fresh OI = early signal)
        - Use tighter stops (momentum can reverse)
        - Scale in if Vol/OI increases further
        
        Exit:
        - Target: 30-100% gain (momentum can be explosive)
        - Stop: -15% (tight, as signal is time-sensitive)
        - Exit if Vol/OI drops below 3.0x on next scan
        ```
        
        ### **Strategy 4: Wall Fade**
        ```
        Setup:
        - Clear Call Wall identified
        - Price approaching Call Wall
        - High gamma at wall
        
        Execution:
        - Sell calls at/above wall (covered or spreads)
        - Or buy puts if expecting rejection
        - Gamma pin likely at wall
        
        Exit:
        - Target: Wall holds, premium decays
        - Stop: If price breaks above wall with volume
        ```
        
        ### **Strategy 5: Divergence Hedge**
        ```
        Setup:
        - Long stock position
        - Fresh OI in puts appearing
        - Vol/OI >4.0x on puts
        
        Execution:
        - Buy same puts as institutions
        - Hedge your long position
        - Institutions know something
        
        Exit:
        - Keep hedge while fresh put OI continues
        - Remove if market stabilizes
        - Adjust strikes as needed
        ```
        
        ---
        
        ## **Key Indicators Summary**
        
        ### **Bullish Signals**
        ✅ Multiple CALL flows with high whale scores  
        ✅ Vol Ratio negative (call volume > put volume)  
        ✅ Fresh OI in calls with Vol/OI >4.0x  
        ✅ Call Wall above current price  
        ✅ Max GEX above current price  
        ✅ Call/Put split favors calls (>2:1)  
        ✅ Increasing notional dollar deployment in calls  
        ✅ Same strikes across multiple expiries (calls)  
        
        ### **Bearish Signals**
        ⚠️ Multiple PUT flows with high whale scores  
        ⚠️ Vol Ratio positive (put volume > call volume)  
        ⚠️ Fresh OI in puts with Vol/OI >4.0x  
        ⚠️ Put Wall below current price  
        ⚠️ Max GEX below current price  
        ⚠️ Call/Put split favors puts (>2:1)  
        ⚠️ Increasing notional dollar deployment in puts  
        ⚠️ Same strikes across multiple expiries (puts)  
        
        ### **Neutral/Range Signals**
        ⚪ Balanced Call/Put split  
        ⚪ Vol Ratio near zero  
        ⚪ Max GEX at current price  
        ⚪ Call Wall above + Put Wall below (range)  
        ⚪ Low average whale scores  
        ⚪ Mixed fresh OI direction  
        
        ### **High Conviction Signals**
        🔥 Whale Score >200  
        🔥 Vol/OI >6.0x  
        🔥 Notional >$5M  
        🔥 Multiple stocks same pattern  
        🔥 Same strike in both tabs  
        🔥 Vol Ratio >|40%|  
        
        ---
        
        ## **Common Patterns to Recognize**
        
        ### **Pre-Earnings Straddle**
        ```
        Pattern:
        - Fresh OI in both calls AND puts
        - Same strike (ATM)
        - Same expiry (just after earnings)
        - High IV%
        
        Interpretation:
        → Institutions expect big move (either direction)
        → Volatility play, not directional
        → Earnings expected to be volatile
        ```
        
        ### **Protective Put Sweep**
        ```
        Pattern:
        - Sudden fresh OI in OTM puts
        - Vol/OI >5.0x
        - Vol Ratio positive
        - Stock near highs
        
        Interpretation:
        → Institutions hedging longs
        → Expecting short-term pullback
        → Or protecting profits
        → Consider trimming longs or hedging
        ```
        
        ### **Bull Call Spread**
        ```
        Pattern:
        - High whale score at lower call strike
        - Simultaneous fresh OI at higher call strike
        - Same expiry
        
        Interpretation:
        → Institutions buying call spread
        → Bullish but capping upside (selling higher strike)
        → Expect move to higher strike
        ```
        
        ### **Gamma Squeeze Setup**
        ```
        Pattern:
        - Massive fresh OI in ATM/ITM calls
        - Vol/OI >8.0x (extreme)
        - Max GEX below current price
        - Low put activity
        
        Interpretation:
        → Dealers short gamma
        → Must buy stock as price rises
        → Potential for explosive upside
        → Classic gamma squeeze setup
        ```
        
        ### **Distribution Setup**
        ```
        Pattern:
        - High call OI but low fresh OI
        - Fresh put OI increasing (Vol/OI >4.0)
        - Vol Ratio turning positive
        - Stock at resistance
        
        Interpretation:
        → Institutions distributing (selling)
        → Hedging with fresh puts
        → Potential top forming
        → Consider bearish positioning
        ```
        
        ---
        
        ## **Practical Examples**
        
        ### **Example 1: Following Whale Calls**
        ```
        Scan Results - Friday Expiry:
        
        NVDA - $520 CALL
        Whale Score: 285
        Vol/OI: 5.2x
        Volume: 8,200
        OI: 1,580
        Notional: $6.4M
        Distance: +2.1%
        Vol Ratio: -32% (bullish)
        Max GEX: $515
        Call Wall: $520
        
        Analysis:
        ✓ High whale score (>200)
        ✓ Extreme fresh positioning (5.2x)
        ✓ Large notional ($6.4M)
        ✓ Bullish Vol Ratio
        ✓ Strike = Call Wall = institutional target
        
        Trade:
        Entry: Buy NVDA $517.50 or $520 calls (same expiry)
        Rationale: Institutions betting $6.4M on $520
        Target: $520-525
        Stop: Below $510 (below Max GEX)
        Risk: 1.5% of portfolio
        
        Result: Monitor for continuation or reversal
        ```
        
        ### **Example 2: GEX Pin Trade**
        ```
        Scan Results - Next Week Expiry:
        
        TSLA - $245 Strike
        Max GEX: $245 ($120M gamma)
        Call Wall: $250
        Put Wall: $240
        Current Price: $247
        Vol Ratio: +5% (neutral)
        
        Analysis:
        ✓ Massive GEX at $245
        ✓ Call Wall $5 above
        ✓ Put Wall $5 below
        ✓ Price between walls
        → Classic pin setup
        
        Trade:
        Entry: Sell $250/$255 call spread + $240/$235 put spread
        Structure: Iron Condor centered on $245
        Rationale: Price will pin at Max GEX ($245)
        Target: Collect 50% of premium
        Stop: If price breaks $250 or $240
        Days to Manage: 5-7 DTE optimal
        ```
        
        ### **Example 3: Fresh OI Momentum**
        ```
        Morning Scan - This Week Expiry:
        
        AMD - $150 CALL
        Vol/OI: 8.5x (EXTREME!)
        Volume: 12,300
        OI: 1,447
        Notional: $4.9M
        Distance: +3.2%
        Current Price: $145.35
        
        11am Scan (2 hours later):
        AMD - $150 CALL
        Vol/OI: 11.2x (INCREASING!)
        Volume: 16,800 (+4,500)
        Notional: $6.8M (+$1.9M)
        
        Analysis:
        🔥 Vol/OI >8.0x initially, now >11x
        🔥 Institutions ADDING to position
        🔥 Fresh $1.9M deployed in 2 hours
        🔥 Extreme conviction building
        
        Trade:
        Entry: Buy AMD $147.50 calls immediately
        Rationale: Momentum building, follow the money
        Target: $150+ (their target)
        Stop: -15% (tight for momentum)
        Position Size: 2% (high conviction)
        
        Management:
        - Re-scan every hour
        - If Vol/OI drops <5.0, take profits
        - Trail stop as it moves up
        ```
        
        ### **Example 4: Earnings Protection**
        ```
        Scan Results - 2 Weeks Out:
        
        AAPL (Earnings in 10 days)
        
        Whale Flows:
        $185 CALL - Whale Score: 165
        $175 PUT - Whale Score: 185 (HIGHER!)
        
        Fresh OI:
        $175 PUT - Vol/OI: 6.8x, Notional: $8.2M
        $180 PUT - Vol/OI: 5.3x, Notional: $6.1M
        
        Vol Ratio: +42% (very put heavy)
        Current Price: $182
        
        Analysis:
        ⚠️ Heavy put buying
        ⚠️ Puts have higher whale scores than calls
        ⚠️ Massive fresh OI in puts
        ⚠️ Institutions protecting downside
        
        Trade (If Long AAPL):
        Entry: Buy $180 or $175 puts (match institutions)
        Rationale: Smart money hedging aggressively
        Size: 25-50% of long position
        Duration: Through earnings
        
        Trade (If Neutral):
        Entry: Sell $185/$190 call spread
        Rationale: Institutions not betting on upside
        Target: Collect premium as stock stays below $185
        ```
        
        ---
        
        ## **Using Discord Integration**
        
        ### **Send Results to Discord**
        After scanning, use "Send to Discord" button to:
        1. Share findings with trading group
        2. Create permanent record of flows
        3. Track patterns over time
        4. Collaborate on trade ideas
        
        **Discord Message Includes:**
        - Expiration date
        - Total results found
        - Avg whale score
        - Total volume
        - Call/Put split
        - Top 50 flows formatted as table
        
        ---
        
        ## **Pro Tips & Best Practices**
        
        ### **Scanning Schedule**
        - **9:45 AM ET**: First scan (15 min after open)
        - **11:00 AM ET**: Mid-morning check
        - **1:00 PM ET**: Post-lunch scan
        - **3:00 PM ET**: Final hour check
        - **Ad-hoc**: After major news/volatility
        
        ### **Filtering Strategy**
        - Start with min score 50
        - If >200 results, raise to 100
        - If <20 results, lower to 30
        - Focus on quality over quantity
        
        ### **Position Sizing**
        - Single flow: 1-1.5% of portfolio
        - Confirmed (both tabs): 2-2.5% of portfolio
        - Multiple confluences: Up to 3% of portfolio
        - Never exceed 5% on one idea
        
        ### **Risk Management**
        - Set stops BEFORE entry
        - Trail stops aggressively on winners
        - Cut losers quickly (institutions can be wrong)
        - Take partial profits at 25-30%
        - Don't fight sustained adverse flows
        
        ### **What to Avoid**
        ❌ Following every single flow  
        ❌ Ignoring your own analysis  
        ❌ Over-leveraging on one signal  
        ❌ Chasing after big moves already happened  
        ❌ Trading against clear flow patterns  
        ❌ Holding through expiry without plan  
        
        ### **Confirmation Checklist**
        Before taking a trade, verify:
        - [ ] High whale score OR high Vol/OI (preferably both)
        - [ ] Vol Ratio supports direction
        - [ ] Notional value significant (>$1M)
        - [ ] Strike makes sense (distance reasonable)
        - [ ] Max GEX/Walls confirm thesis
        - [ ] Pattern matches historical winners
        - [ ] Technical analysis aligns
        - [ ] Risk/reward favorable (2:1 minimum)
        
        ---
        
        ## **Troubleshooting**
        
        **No Results Found:**
        - Lower minimum whale score
        - Check if markets are open
        - Verify API connection
        - Try custom symbol (may be issue with defaults)
        
        **Too Many Results:**
        - Raise minimum whale score
        - Reduce top N results
        - Focus on specific expiry dates
        
        **Conflicting Signals:**
        - Check Vol Ratio for bias
        - Compare multiple expiries
        - Look at notional values (bigger = more important)
        - Consider market may be neutral/ranging
        
        **Discord Webhook Not Working:**
        - Check secrets configuration
        - Verify webhook URL is valid
        - Ensure proper permissions
        
        ---
        
        ## **Bottom Line**
        
        This scanner gives you **institutional-grade flow analysis** that reveals:
        - Where smart money is positioning
        - How much capital they're deploying
        - Their conviction levels (Vol/OI ratios)
        - Price targets and risk levels
        
        **Master these flows → Trade with institutions → Dramatically improve edge**
        
        The combination of Whale Flows (sophisticated analysis) and Fresh OI (new positioning) gives you a complete picture of institutional options activity. Use both tabs together for maximum insight.
        """)

