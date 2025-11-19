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

# Top 20 tech stocks
TOP_TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'AMD',
    'CRM', 'GS', 'NFLX', 'IBIT', 'COIN',
    'APP', 'PLTR', 'SNOW', 'TEAM', 'CRWD','SPY','QQQ'
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
        st.info(f"🎯 Scanning custom symbol: {custom_symbol}")
    else:
        stocks_to_scan = TOP_TECH_STOCKS.copy()
        st.info(f"📊 Scanning {len(stocks_to_scan)} default tech stocks")
    
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
    # Show explanation when not scanning
    with st.expander("📖 How to Use Whale Flows Scanner", expanded=False):
        st.markdown("""
        ### 🐋 Whale Flows Tab (VALR-Based)
        
        Identifies options with the highest **whale scores** using the VALR formula.
        The whale score combines leverage, volatility, and volume activity to find institutional positioning.
        
        **Key Metrics:**
        - **Whale Score**: Combined measure of leverage, volatility, and volume activity
        - **Vol Ratio**: Put volume - Call volume (positive = bearish, negative = bullish)
        - **Max GEX Strike**: Strike with highest gamma exposure (dealer hedging point)
        - **Call/Put Walls**: Strikes with maximum call/put volume (resistance/support)
        
        ---
        
        ### 🆕 OI Flows Tab (Fresh Positioning)
        
        Identifies **NEW institutional positions** being opened TODAY based on Volume/OI ratio.
        
        **Logic:**
        - If Volume ≥ 3.0x Open Interest → Massive fresh positioning
        - High Vol/OI ratio = institutions aggressively opening new positions
        - Sorted by OI Score (Vol/OI × Notional Value)
        
        **Key Metrics:**
        - **OI Score**: Combines Vol/OI ratio with notional dollar size
        - **Vol/OI Ratio**: How much volume relative to existing OI (3.0+ = 300%+ new positions)
        - **Notional**: Total dollar value of contracts traded (Volume × Premium × 100)
        
        ---
        
        ### 🎯 How to Use:
        
        1. **Set Minimum Score**: Filter for whale activity above threshold (Whale Flows tab)
        2. **Choose Top N**: Number of top results per stock
        3. **Click Scan**: Analyzes all 20 tech stocks across 4 weekly expiries
        4. **Compare Tabs**: Whale Flows for overall activity, OI Flows for fresh positioning
        
        ### 💡 Trading Insights:
        
        - **High Whale Score** = Institutional interest, potential price magnet
        - **High Vol/OI (>3.0)** = Massive fresh positions, extremely strong conviction
        - **Calls > Puts** = Bullish positioning
        - **Strike near Max GEX** = Strong support/resistance zone
        - **High Notional** = Big money at work
        """)

