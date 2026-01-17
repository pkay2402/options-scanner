"""
Watchlist Table - Custom Stock Monitoring
Real-time options analysis for your custom watchlist
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.utils.cached_client import get_client

st.set_page_config(
    page_title="Watchlist Table",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .stock-input-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'custom_watchlist' not in st.session_state:
    st.session_state.custom_watchlist = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN']
if 'auto_refresh_watchlist' not in st.session_state:
    st.session_state.auto_refresh_watchlist = False

def get_next_friday():
    """Get next Friday for weekly options expiry"""
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.cache_data(ttl=60, show_spinner=False)
def get_market_snapshot(symbol: str, expiry_date: str):
    """Fetches complete market data snapshot for a symbol"""
    client = get_client()
    
    if not client:
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
        
        return {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options,
            'fetched_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

def calculate_comprehensive_analysis(options_data, underlying_price):
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
        
        return {
            'call_wall': call_wall,
            'put_wall': put_wall,
            'max_gex': max_gex,
            'flip_level': flip_level,
            'pc_ratio': total_put_vol / total_call_vol if total_call_vol > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return None

@st.fragment(run_every="120s")
def live_watchlist_table(watchlist, expiry_date):
    """Auto-refreshing table showing multiple symbols - updates every 120 seconds"""
    exp_date_str = expiry_date.strftime('%Y-%m-%d')
    
    st.caption(f"🔄 Auto-updates every 120s • Last: {datetime.now().strftime('%H:%M:%S')}")
    
    table_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(watchlist):
        status_text.text(f"Loading {symbol}... ({idx + 1}/{len(watchlist)})")
        progress_bar.progress((idx + 1) / len(watchlist))
        
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
    
    progress_bar.empty()
    status_text.empty()
    
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
            height=min(len(table_data) * 35 + 38, 600)
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Unable to load watchlist data")

# ===== HEADER =====
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0;">📊 Custom Watchlist Table</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">Real-time options analysis for your custom stock list</p>
</div>
""", unsafe_allow_html=True)

# ===== CONTROLS =====
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### ⚙️ Settings")
    
    # Expiry date selector
    today = datetime.now().date()
    weekday = today.weekday()
    if weekday == 5:
        default_expiry = today + timedelta(days=2)
    elif weekday == 6:
        default_expiry = today + timedelta(days=1)
    else:
        default_expiry = today
    
    next_friday = get_next_friday()
    
    expiry_options = {
        "Today (0DTE)": default_expiry,
        "Next Friday (Weekly)": next_friday,
        "Custom Date": None
    }
    
    expiry_choice = st.selectbox("Expiry Date", list(expiry_options.keys()))
    
    if expiry_choice == "Custom Date":
        selected_expiry = st.date_input("Select Date", value=default_expiry, min_value=today)
    else:
        selected_expiry = expiry_options[expiry_choice]
    
    st.info(f"📅 Using expiry: {selected_expiry.strftime('%b %d, %Y')}")

with col2:
    st.markdown("### 🔄 Refresh")
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Fragment-based auto-refresh (non-blocking)
    @st.fragment(run_every="120s")
    def auto_refresh_fragment():
        st.cache_data.clear()
    
    if st.checkbox("Auto-refresh (2min)", value=st.session_state.get('auto_refresh_watchlist', False), key="auto_refresh_checkbox"):
        auto_refresh_fragment()

with col3:
    st.markdown("### 📊 Stats")
    st.metric("Total Symbols", len(st.session_state.custom_watchlist))
    st.metric("Expiry", selected_expiry.strftime('%b %d'))

st.markdown("---")

# ===== WATCHLIST MANAGEMENT =====
st.markdown("### 📝 Manage Your Watchlist")

col_input, col_buttons = st.columns([3, 1])

with col_input:
    new_symbols = st.text_input(
        "Add Symbols (comma-separated)",
        placeholder="e.g., MSFT, GOOGL, AMZN",
        help="Enter stock symbols separated by commas. Press Enter or click 'Add' to add them to your watchlist."
    )

with col_buttons:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("➕ Add", type="primary", use_container_width=True):
        if new_symbols:
            symbols = [s.strip().upper() for s in new_symbols.split(',') if s.strip()]
            for symbol in symbols:
                if symbol not in st.session_state.custom_watchlist:
                    st.session_state.custom_watchlist.append(symbol)
            st.success(f"Added {len(symbols)} symbol(s)")
            st.rerun()

# Display current watchlist with remove buttons
st.markdown("#### Current Watchlist:")
cols_per_row = 6
watchlist_copy = st.session_state.custom_watchlist.copy()

for i in range(0, len(watchlist_copy), cols_per_row):
    cols = st.columns(cols_per_row)
    for idx, symbol in enumerate(watchlist_copy[i:i+cols_per_row]):
        with cols[idx]:
            if st.button(f"❌ {symbol}", key=f"remove_{symbol}", use_container_width=True):
                st.session_state.custom_watchlist.remove(symbol)
                st.rerun()

col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)

with col_preset1:
    if st.button("📊 Load: Indices", use_container_width=True):
        st.session_state.custom_watchlist = ['SPY', 'QQQ', 'IWM', 'DIA', '$SPX']
        st.rerun()

with col_preset2:
    if st.button("💻 Load: Tech Giants", use_container_width=True):
        st.session_state.custom_watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX']
        st.rerun()

with col_preset3:
    if st.button("🔥 Load: Meme Stocks", use_container_width=True):
        st.session_state.custom_watchlist = ['GME', 'AMC', 'BBBY', 'PLTR', 'COIN', 'HOOD', 'RIVN']
        st.rerun()

with col_preset4:
    if st.button("🚀 Load: High IV", use_container_width=True):
        st.session_state.custom_watchlist = ['TSLA', 'NVDA', 'AMD', 'PLTR', 'CRWD', 'OKLO', 'COIN']
        st.rerun()

st.markdown("---")

# ===== WATCHLIST TABLE =====
if len(st.session_state.custom_watchlist) > 0:
    st.markdown(f"### 📊 Live Watchlist - {selected_expiry.strftime('%b %d, %Y')}")
    
    if st.session_state.auto_refresh_watchlist:
        live_watchlist_table(st.session_state.custom_watchlist, selected_expiry)
    else:
        # Non-auto-refreshing version
        exp_date_str = selected_expiry.strftime('%Y-%m-%d')
        
        table_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(st.session_state.custom_watchlist):
            status_text.text(f"Loading {symbol}... ({idx + 1}/{len(st.session_state.custom_watchlist)})")
            progress_bar.progress((idx + 1) / len(st.session_state.custom_watchlist))
            
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
                            '% Change': daily_change_pct,
                            'Flip Level': f"${ana['flip_level']:.2f}" if ana['flip_level'] else "-",
                            'Call Wall': f"${ana['call_wall']['strike']:.2f}" if ana['call_wall'] is not None else "-",
                            'Put Wall': f"${ana['put_wall']['strike']:.2f}" if ana['put_wall'] is not None else "-",
                            'Max GEX': f"${ana['max_gex']['strike']:.2f}" if ana['max_gex'] is not None else "-",
                            'P/C': f"{ana['pc_ratio']:.2f}"
                        })
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if table_data:
            df = pd.DataFrame(table_data)
            df = df.sort_values('% Change', ascending=False)
            df['% Change'] = df['% Change'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(len(table_data) * 35 + 38, 600)
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Unable to load watchlist data")
else:
    st.info("👆 Add symbols to your watchlist to get started!")
