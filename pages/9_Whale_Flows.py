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

# Initialize session state for persisting results
if 'whale_flows_data' not in st.session_state:
    st.session_state.whale_flows_data = None
if 'whale_flows_expiry' not in st.session_state:
    st.session_state.whale_flows_expiry = None
if 'whale_flows_min_score' not in st.session_state:
    st.session_state.whale_flows_min_score = None

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

def send_to_discord(flows_df, expiry_date, min_whale_score):
    """Send whale flows data to Discord via webhook in chunks"""
    try:
        # Get Discord webhook URL from secrets only (for security)
        webhook_url = st.secrets.get("discord_webhook")
        
        if not webhook_url:
            st.error("⚠️ Discord webhook URL not configured. Please add 'discord_webhook' to .streamlit/secrets.toml")
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
def scan_stock_whale_flows(symbol: str, expiry_date: str):
    """
    Scan a single stock for whale flows
    Returns top whale score options with comprehensive data
    """
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        # Get quote
        quote_response = client.get_quotes([symbol])
        if not quote_response or symbol not in quote_response:
            return None
        
        underlying_price = quote_response[symbol]['quote']['lastPrice']
        
        # Get options chain
        options_response = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=expiry_date,
            to_date=expiry_date
        )
        
        if not options_response or 'callExpDateMap' not in options_response:
            return None
        
        # Process options and calculate whale scores
        whale_options = []
        
        # Process calls
        if 'callExpDateMap' in options_response:
            for exp_date, strikes in options_response['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Filter: Only consider strikes within 5% of underlying price
                        if abs(strike - underlying_price) / underlying_price > 0.05:
                            continue
                        
                        # Calculate whale score
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        mark_price = contract.get('mark', contract.get('last', 1))
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        gamma = contract.get('gamma', 0)
                        
                        if mark_price > 0 and delta != 0 and volume > 0:
                            leverage = delta * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * 100000
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
        
        # Process puts
        if 'putExpDateMap' in options_response:
            for exp_date, strikes in options_response['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Filter: Only consider strikes within 5% of underlying price
                        if abs(strike - underlying_price) / underlying_price > 0.05:
                            continue
                        
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        mark_price = contract.get('mark', contract.get('last', 1))
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        gamma = contract.get('gamma', 0)
                        
                        if mark_price > 0 and delta != 0 and volume > 0:
                            leverage = abs(delta) * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * 100000
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
        
        # Calculate summary stats for the stock
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
            
            return {
                'options': whale_options,
                'summary': summary
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {str(e)}")
        return None

# Settings
st.markdown("## ⚙️ Scanner Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    expiry_date = st.date_input(
        "Expiration Date",
        value=get_next_friday(),
        help="Weekly expiration date to scan"
    )

with col2:
    min_whale_score = st.number_input(
        "Minimum Whale Score",
        min_value=0,
        max_value=10000,
        value=500,
        step=100,
        help="Filter options with whale score above this threshold"
    )

with col3:
    top_n = st.slider(
        "Top Results per Stock",
        min_value=1,
        max_value=10,
        value=3,
        help="Show top N whale flows per stock"
    )

with col4:
    custom_symbol = st.text_input(
        "Custom Symbol (Only)",
        placeholder="e.g., COIN, SHOP",
        help="Scan only this custom symbol (ignores default list)"
    ).upper().strip()

# Scan button
scan_col1, scan_col2 = st.columns([1, 5])
with scan_col1:
    scan_button = st.button("🔍 Scan Whale Flows", type="primary", use_container_width=True)

if scan_button:
    st.markdown("---")
    
    # Build stock list - if custom symbol provided, scan only that
    if custom_symbol:
        stocks_to_scan = [custom_symbol]
        st.info(f"🎯 Scanning custom symbol: {custom_symbol}")
    else:
        stocks_to_scan = TOP_TECH_STOCKS.copy()
        st.info(f"📊 Scanning {len(stocks_to_scan)} default tech stocks")
    
    st.markdown(f"### 🐋 Scanning {len(stocks_to_scan)} Stocks...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_whale_flows = []
    
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.text(f"Scanning {symbol}... ({idx+1}/{len(stocks_to_scan)})")
        progress_bar.progress((idx + 1) / len(stocks_to_scan))
        
        result = scan_stock_whale_flows(symbol, expiry_date.strftime('%Y-%m-%d'))
        
        if result and result['options']:
            # Filter by minimum whale score and get top N
            df = pd.DataFrame(result['options'])
            df = df[df['whale_score'] >= min_whale_score]
            df = df.nlargest(top_n, 'whale_score')
            
            # Add summary data to each row
            for _, row in df.iterrows():
                flow_data = row.to_dict()
                flow_data.update({
                    'call_volume': result['summary']['call_volume'],
                    'put_volume': result['summary']['put_volume'],
                    'vol_ratio': result['summary']['vol_ratio'],
                    'max_gex_strike': result['summary']['max_gex_strike'],
                    'max_gex_value': result['summary']['max_gex_value'],
                    'call_wall_strike': result['summary']['call_wall_strike'],
                    'put_wall_strike': result['summary']['put_wall_strike']
                })
                all_whale_flows.append(flow_data)
    
    progress_bar.empty()
    status_text.empty()
    
    # Store results in session state
    if all_whale_flows:
        st.session_state.whale_flows_data = pd.DataFrame(all_whale_flows).sort_values('whale_score', ascending=False)
        st.session_state.whale_flows_expiry = expiry_date
        st.session_state.whale_flows_min_score = min_whale_score

# Display results if they exist in session state
if st.session_state.whale_flows_data is not None:
    flows_df = st.session_state.whale_flows_data
    expiry_date = st.session_state.whale_flows_expiry
    min_whale_score = st.session_state.whale_flows_min_score
    
    st.markdown(f"### 📊 Top Whale Flows ({len(flows_df)} results)")
    st.caption(f"Expiration: {expiry_date.strftime('%B %d, %Y')} | Min Whale Score: {min_whale_score}")
    
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
    
    styled_df = result_df.style.applymap(color_type, subset=['Type'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
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
            use_container_width=True
        )
    
    with btn_col2:
        if st.button("📤 Send to Discord", use_container_width=True, type="secondary"):
            with st.spinner("Sending to Discord..."):
                if send_to_discord(flows_df, expiry_date, min_whale_score):
                    st.success("✅ Successfully sent to Discord!")
                else:
                    st.error("❌ Failed to send to Discord")

elif scan_button:
    # Only show this if scan was just performed and no results found
    st.warning("⚠️ No whale flows found matching your criteria. Try lowering the minimum whale score.")

if not scan_button and st.session_state.whale_flows_data is None:
    # Show explanation when not scanning
    with st.expander("📖 How to Use Whale Flows Scanner", expanded=False):
        st.markdown("""
        ### 🐋 What is Whale Flows?
        
        This scanner identifies options with the highest **whale scores** across top tech stocks. 
        The whale score (VALR-based) indicates where large institutional players are positioning.
        
        ### 📊 Key Metrics Explained:
        
        - **Whale Score**: Combined measure of leverage, volatility, and volume activity
        - **Vol Ratio**: Put volume - Call volume (positive = bearish, negative = bullish)
        - **Max GEX Strike**: Strike with highest gamma exposure (dealer hedging point)
        - **Call/Put Walls**: Strikes with maximum call/put volume (resistance/support)
        
        ### 🎯 How to Use:
        
        1. **Select Expiration**: Choose weekly expiry date (default: next Friday)
        2. **Set Minimum Score**: Filter for whale activity above threshold
        3. **Choose Top N**: Number of top results per stock
        4. **Click Scan**: Analyze all 20 tech stocks
        
        ### 💡 Trading Insights:
        
        - **High Whale Score** = Institutional interest, potential price magnet
        - **Calls > Puts** = Bullish positioning
        - **Strike near Max GEX** = Strong support/resistance zone
        - **High IV%** = Expected volatility, premium opportunities
        """)
