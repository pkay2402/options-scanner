"""
Option Volume Walls & Levels (NetSPY Indicator)
Visualizes key support/resistance levels based on option volume analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import time
import logging
from streamlit_autorefresh import st_autorefresh

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

# ===== CACHED MARKET DATA FETCHER =====
# This function caches raw market data for 60 seconds
# Multiple users watching the same symbol share the cached data
# This reduces API calls from N users to 1 call per symbol per minute

@st.cache_data(ttl=60, show_spinner=False)
def get_market_snapshot(symbol: str, expiry_date: str):
    """
    Fetches complete market data snapshot for a symbol
    
    Cache Strategy:
    - Cached for 60 seconds based on (symbol, expiry_date)
    - All users watching same symbol+expiry share this cache
    - Example: 10 users watching SPY 2025-11-08 = 1 API call/min (not 10)
    
    Returns:
        dict: {
            'symbol': str,
            'underlying_price': float,
            'quote': dict,
            'options_chain': dict,
            'price_history': dict,
            'fetched_at': datetime,
            'cache_key': str
        }
    """
    client = SchwabClient()
    
    # Authenticate
    if not client.authenticate():
        st.error("Failed to authenticate with Schwab API")
        return None
    
    try:
        # Special handling for SPX (S&P 500 Index)
        # For Schwab API: Use $SPX for quotes, but SPX (no $) for options chain
        query_symbol_quote = symbol
        query_symbol_options = symbol
        
        if symbol == 'SPX':
            query_symbol_quote = '$SPX'  # Quotes need $ prefix
            query_symbol_options = 'SPX'  # Options chain does NOT need $ prefix
        
        # Get quote (use $SPX for index symbols)
        quote = client.get_quote(query_symbol_quote)
        if not quote:
            st.error(f"Failed to get quote for {symbol}")
            return None
        
        # Extract price - use the quote symbol format (with $ if applicable)
        underlying_price = quote.get(query_symbol_quote, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            st.error(f"Could not extract price for {symbol}. Response: {quote}")
            return None
        
        # Get options chain - use symbol WITHOUT $ prefix (SPX not $SPX)
        # For SPX: Limit strikes to prevent timeout (SPX has 1000+ strikes)
        chain_params = {
            'symbol': query_symbol_options,
            'contract_type': 'ALL',
            'from_date': expiry_date,
            'to_date': expiry_date
        }
        
        # For index symbols, limit strikes to ±50 around current price
        if symbol in ['SPX', 'DJX', 'NDX', 'RUT']:
            chain_params['strike_count'] = 50  # Get 50 strikes above and below current price
        
        # Log for debugging
        logger.info(f"Fetching options chain with params: {chain_params}")
        
        options = client.get_options_chain(**chain_params)
        
        if not options or 'callExpDateMap' not in options:
            st.warning(f"No options data available for {symbol} on {expiry_date}. Symbol used: {query_symbol_options}")
            return None
        
        # Get intraday price history (48 hours to ensure we capture 2 full trading days)
        # This accounts for weekends, holidays, and after-hours gaps
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(hours=48)).timestamp() * 1000)
        
        price_history = client.get_price_history(
            symbol=query_symbol_quote,  # Use quote symbol format (with $ if index)
            frequency_type='minute',
            frequency=5,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        # Return snapshot with the actual symbols used
        return {
            'symbol': symbol,  # Original symbol for display
            'query_symbol_quote': query_symbol_quote,  # Symbol used for quotes/price history
            'query_symbol_options': query_symbol_options,  # Symbol used for options chain
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options,
            'price_history': price_history,
            'fetched_at': datetime.now(),
            'cache_key': f"{symbol}_{expiry_date}"
        }
        
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_multi_expiry_snapshot(symbol: str, from_date: str, to_date: str):
    """
    Fetches options data for multiple expiration dates
    Used for heatmaps that need to compare multiple expiries
    
    Cache Strategy:
    - Cached for 60 seconds based on (symbol, from_date, to_date)
    - Separate cache from single-expiry snapshot
    
    Returns:
        dict: {
            'symbol': str,
            'options_chain': dict (with multiple expiries),
            'fetched_at': datetime,
            'cache_key': str
        }
    """
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        # Special handling for SPX - options chain uses SPX (no $ prefix)
        query_symbol_options = symbol
        if symbol == 'SPX':
            query_symbol_options = 'SPX'  # Options API does NOT use $ prefix
        
        # Get options chain for date range - use symbol without $ prefix
        # For index symbols, limit strikes to prevent timeout
        chain_params = {
            'symbol': query_symbol_options,
            'contract_type': 'ALL',
            'from_date': from_date,
            'to_date': to_date
        }
        
        # For index symbols, limit strikes to ±50 around current price
        if symbol in ['SPX', 'DJX', 'NDX', 'RUT']:
            chain_params['strike_count'] = 50  # Get 50 strikes above and below current price
        
        # Log for debugging
        logger.info(f"Fetching multi-expiry options chain with params: {chain_params}")
        
        options = client.get_options_chain(**chain_params)
        
        if not options or 'callExpDateMap' not in options:
            st.warning(f"No multi-expiry options data for {symbol}. Symbol used: {query_symbol_options}")
            return None
        
        return {
            'symbol': symbol,  # Original symbol for display
            'query_symbol_options': query_symbol_options,  # Symbol used for options
            'options_chain': options,
            'fetched_at': datetime.now(),
            'cache_key': f"{symbol}_{from_date}_{to_date}"
        }
        
    except Exception as e:
        st.error(f"Error fetching multi-expiry data: {str(e)}")
        return None

st.set_page_config(
    page_title="Option Volume Walls",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Option Volume Walls & Key Levels")
st.markdown("**Identify support/resistance levels based on massive option volume concentrations**")

# ===== AUTO-REFRESH MECHANISM =====
# Initialize session state for auto-refresh control
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False

# Only auto-refresh if user has enabled it AND has run analysis at least once
if st.session_state.get('auto_refresh_enabled', False) and st.session_state.get('run_analysis', False):
    # Auto-refresh every 60 seconds (60000 milliseconds)
    refresh_count = st_autorefresh(interval=60000, key="data_refresh")
    
    # Display refresh indicator
    if refresh_count > 0:
        st.info(f"🔄 Auto-refreshed {refresh_count} time(s). Data updates every 60 seconds.")

# Settings at the top
st.markdown("## ⚙️ Settings")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    symbol = st.text_input("Symbol", value="SPY").upper()

# Show SPX notice if user enters SPX
if symbol in ['SPX', 'DJX', 'NDX', 'RUT']:
    st.info(f"📝 **Index Options Note:** {symbol} has 1000+ strikes. API call limited to 50 strikes above/below current price to prevent timeout. Data filtered to 4 expiries max. Consider using SPY/QQQ/IWM for more liquid ETF options.")

with col2:
    expiry_date = st.date_input(
        "Expiration Date",
        value=datetime.now() + timedelta(days=7),
        help="Select the options expiration date to analyze"
    )

with col3:
    strike_spacing = st.number_input(
        "Strike Spacing",
        min_value=0.5,
        max_value=10.0,
        value=5.0 if symbol in ['SPY', 'QQQ'] else 5.0,
        step=0.5,
        help="Distance between strikes (e.g., 5 for SPY)"
    )

with col4:
    num_strikes = st.slider(
        "Number of Strikes (each side)",
        min_value=10,
        max_value=30,
        value=20,
        help="How many strikes above/below current price to analyze"
    )

col5, col6, col7, col8 = st.columns([2, 2, 2, 1])

with col5:
    multi_expiry = st.checkbox(
        "📅 Compare Multiple Expirations",
        value=True,
        help="Analyze walls across multiple expiration dates to find stacked levels"
    )

with col6:
    show_heatmap = st.checkbox(
        "🔥 Show Gamma Heatmap",
        value=True,
        help="Display NetGEX heatmap showing gamma exposure across strikes and expirations"
    )

with col7:
    auto_refresh = st.checkbox(
        "🔄 Auto-Refresh (1 min)",
        value=st.session_state.get('auto_refresh_enabled', False),
        help="Automatically refresh data every 60 seconds using cached data (efficient for multiple users)"
    )
    # Update session state
    st.session_state.auto_refresh_enabled = auto_refresh

with col8:
    analyze_button = st.button("🔍 Calculate Levels", type="primary", use_container_width=True)

# Add cache clear button for debugging (especially useful for SPX)
if symbol in ['SPX', 'DJX', 'NDX', 'RUT']:
    if st.button("🗑️ Clear Cache (Force Refresh)", help="Clear cached data and fetch fresh data from API"):
        st.cache_data.clear()
        st.success("✅ Cache cleared! Click 'Calculate Levels' to fetch fresh data.")
        st.rerun()

def calculate_option_walls(options_data, underlying_price, strike_spacing, num_strikes):
    """
    Calculate key levels based on option volume
    Returns: call wall, put wall, net call wall, net put wall, flip level
    """
    try:
        # Round underlying price to nearest 10
        base_strike = np.floor(underlying_price / 10) * 10
        
        # Generate strike range
        strikes_above = [base_strike + strike_spacing * i for i in range(num_strikes + 1)]
        strikes_below = [base_strike - strike_spacing * i for i in range(1, num_strikes + 1)]
        all_strikes = sorted(strikes_below + strikes_above)
        
        # Extract volumes, OI, and greeks from ALL strikes in options data
        # Don't filter by all_strikes yet - collect everything first
        call_volumes = {}
        put_volumes = {}
        call_oi = {}
        put_oi = {}
        call_gamma = {}
        put_gamma = {}
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        strike = float(strike_str)
                        contract = contracts[0]
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        call_volumes[strike] = call_volumes.get(strike, 0) + volume
                        call_oi[strike] = call_oi.get(strike, 0) + oi
                        call_gamma[strike] = call_gamma.get(strike, 0) + gamma
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        strike = float(strike_str)
                        contract = contracts[0]
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        put_volumes[strike] = put_volumes.get(strike, 0) + volume
                        put_oi[strike] = put_oi.get(strike, 0) + oi
                        put_gamma[strike] = put_gamma.get(strike, 0) + gamma
        
        # Update all_strikes to include ALL strikes that have data
        all_strikes_with_data = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
        all_strikes = all_strikes_with_data  # Use actual strikes from data instead of generated list
        
        # Calculate net volumes (Put - Call, positive = bearish, negative = bullish)
        net_volumes = {}
        for strike in all_strikes:
            call_vol = call_volumes.get(strike, 0)
            put_vol = put_volumes.get(strike, 0)
            net_volumes[strike] = put_vol - call_vol
        
        # Calculate Gamma Exposure (GEX) for each strike
        # GEX = Gamma * Open Interest * 100 * Spot^2 * 0.01
        # Dealer is short gamma, so: Call GEX is positive, Put GEX is negative
        gex_by_strike = {}
        for strike in all_strikes:
            call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01
            put_gex = put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01 * -1
            gex_by_strike[strike] = call_gex + put_gex
        
        # Find walls (max volumes)
        call_wall_strike = max(call_volumes.items(), key=lambda x: x[1])[0] if call_volumes else None
        call_wall_volume = call_volumes.get(call_wall_strike, 0) if call_wall_strike else 0
        call_wall_oi = call_oi.get(call_wall_strike, 0) if call_wall_strike else 0
        call_wall_gex = gex_by_strike.get(call_wall_strike, 0) if call_wall_strike else 0
        
        put_wall_strike = max(put_volumes.items(), key=lambda x: x[1])[0] if put_volumes else None
        put_wall_volume = put_volumes.get(put_wall_strike, 0) if put_wall_strike else 0
        put_wall_oi = put_oi.get(put_wall_strike, 0) if put_wall_strike else 0
        put_wall_gex = gex_by_strike.get(put_wall_strike, 0) if put_wall_strike else 0
        
        # Find net walls (max absolute net volumes)
        bullish_strikes = {k: abs(v) for k, v in net_volumes.items() if v < 0}  # negative = call dominant
        bearish_strikes = {k: abs(v) for k, v in net_volumes.items() if v > 0}  # positive = put dominant
        
        net_call_wall_strike = max(bullish_strikes.items(), key=lambda x: x[1])[0] if bullish_strikes else None
        net_call_wall_volume = net_volumes.get(net_call_wall_strike, 0) if net_call_wall_strike else 0
        
        net_put_wall_strike = max(bearish_strikes.items(), key=lambda x: x[1])[0] if bearish_strikes else None
        net_put_wall_volume = net_volumes.get(net_put_wall_strike, 0) if net_put_wall_strike else 0
        
        # Find flip level (where net volume changes sign near current price)
        strikes_near_price = sorted([s for s in all_strikes if abs(s - underlying_price) < strike_spacing * 5])
        flip_strike = None
        for i in range(len(strikes_near_price) - 1):
            s1, s2 = strikes_near_price[i], strikes_near_price[i + 1]
            net_vol_s1 = net_volumes.get(s1, 0)
            net_vol_s2 = net_volumes.get(s2, 0)
            if net_vol_s1 * net_vol_s2 < 0:  # Sign change
                # Pick the strike with smallest absolute net volume (closest to neutral)
                flip_strike = s1 if abs(net_vol_s1) < abs(net_vol_s2) else s2
                break
        
        # Calculate totals
        total_call_vol = sum(call_volumes.values())
        total_put_vol = sum(put_volumes.values())
        total_net_vol = total_put_vol - total_call_vol
        
        return {
            'all_strikes': all_strikes,
            'call_volumes': call_volumes,
            'put_volumes': put_volumes,
            'net_volumes': net_volumes,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'gex_by_strike': gex_by_strike,
            'call_wall': {
                'strike': call_wall_strike, 
                'volume': call_wall_volume,
                'oi': call_wall_oi,
                'gex': call_wall_gex
            },
            'put_wall': {
                'strike': put_wall_strike, 
                'volume': put_wall_volume,
                'oi': put_wall_oi,
                'gex': put_wall_gex
            },
            'net_call_wall': {'strike': net_call_wall_strike, 'volume': net_call_wall_volume},
            'net_put_wall': {'strike': net_put_wall_strike, 'volume': net_put_wall_volume},
            'flip_level': flip_strike,
            'totals': {
                'call_vol': total_call_vol,
                'put_vol': total_put_vol,
                'net_vol': total_net_vol,
                'total_gex': sum(gex_by_strike.values())
            }
        }
        
    except Exception as e:
        st.error(f"Error calculating walls: {str(e)}")
        return None

def calculate_most_valuable_strike(options_data, underlying_price):
    """
    Calculate the most valuable strike based on GEX, proximity, and activity
    Returns the strike that is most likely to influence price movement
    """
    try:
        strikes_analysis = []
        
        # Extract data from options
        if 'callExpDateMap' not in options_data or 'putExpDateMap' not in options_data:
            return None
        
        # Collect all strikes with their metrics
        all_strikes = set()
        call_data = {}
        put_data = {}
        
        # Process calls
        for exp_date, strikes in options_data['callExpDateMap'].items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    strike = float(strike_str)
                    all_strikes.add(strike)
                    
                    if strike not in call_data:
                        call_data[strike] = {
                            'gamma': contract.get('gamma', 0),
                            'oi': contract.get('openInterest', 0),
                            'volume': contract.get('totalVolume', 0)
                        }
        
        # Process puts
        for exp_date, strikes in options_data['putExpDateMap'].items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    strike = float(strike_str)
                    all_strikes.add(strike)
                    
                    if strike not in put_data:
                        put_data[strike] = {
                            'gamma': contract.get('gamma', 0),
                            'oi': contract.get('openInterest', 0),
                            'volume': contract.get('totalVolume', 0)
                        }
        
        # Calculate value score for each strike
        for strike in all_strikes:
            call_info = call_data.get(strike, {'gamma': 0, 'oi': 0, 'volume': 0})
            put_info = put_data.get(strike, {'gamma': 0, 'oi': 0, 'volume': 0})
            
            # 1. Net GEX (absolute value - we care about magnitude)
            call_gex = call_info['gamma'] * call_info['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = put_info['gamma'] * put_info['oi'] * 100 * underlying_price * underlying_price * 0.01 * -1
            net_gex = call_gex + put_gex
            
            # 2. Proximity to current price (exponential decay - closer = more valuable)
            distance_pct = abs(strike - underlying_price) / underlying_price
            proximity_weight = np.exp(-distance_pct * 20)  # Heavy decay for distant strikes
            
            # 3. Total activity (volume + OI)
            total_volume = call_info['volume'] + put_info['volume']
            total_oi = call_info['oi'] + put_info['oi']
            
            # 4. Volume-to-OI ratio (measures fresh activity)
            vol_oi_ratio = total_volume / total_oi if total_oi > 0 else 0
            
            # 5. Composite value score
            # High weight on proximity and GEX magnitude
            value_score = (
                abs(net_gex) * 0.001 * 0.4 +        # 40% weight on GEX magnitude (scaled)
                proximity_weight * 1000 * 0.35 +    # 35% weight on proximity
                total_volume * 0.15 +               # 15% weight on volume
                vol_oi_ratio * 100 * 0.10           # 10% weight on activity ratio
            )
            
            strikes_analysis.append({
                'strike': strike,
                'net_gex': net_gex,
                'proximity_weight': proximity_weight,
                'total_volume': total_volume,
                'total_oi': total_oi,
                'vol_oi_ratio': vol_oi_ratio,
                'value_score': value_score,
                'distance_pct': distance_pct * 100
            })
        
        if not strikes_analysis:
            return None
        
        # Sort by value score and get top strike
        strikes_df = pd.DataFrame(strikes_analysis).sort_values('value_score', ascending=False)
        most_valuable = strikes_df.iloc[0]
        
        return {
            'strike': most_valuable['strike'],
            'value_score': most_valuable['value_score'],
            'net_gex': most_valuable['net_gex'],
            'distance_pct': most_valuable['distance_pct'],
            'total_volume': most_valuable['total_volume'],
            'total_oi': most_valuable['total_oi']
        }
        
    except Exception as e:
        logger.error(f"Error calculating most valuable strike: {str(e)}")
        return None

def calculate_expected_move(options_data, underlying_price, days_to_expiry):
    """
    Calculate expected move based on GEX and implied volatility
    Returns: dict with expected_move_up, expected_move_down, and reasoning
    """
    try:
        # Calculate total gamma exposure
        total_call_gex = 0
        total_put_gex = 0
        atm_iv = 0
        atm_strike = None
        min_distance = float('inf')
        
        # Find ATM strike and sum GEX
        for exp_date, strikes in options_data.get('callExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    strike = float(strike_str)
                    gamma = contract.get('gamma', 0)
                    oi = contract.get('openInterest', 0)
                    iv = contract.get('volatility', 0)
                    
                    # Calculate GEX
                    gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                    total_call_gex += gex
                    
                    # Track ATM IV
                    distance = abs(strike - underlying_price)
                    if distance < min_distance:
                        min_distance = distance
                        atm_strike = strike
                        atm_iv = iv
        
        for exp_date, strikes in options_data.get('putExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    strike = float(strike_str)
                    gamma = contract.get('gamma', 0)
                    oi = contract.get('openInterest', 0)
                    
                    # Calculate GEX (negative for puts)
                    gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                    total_put_gex += gex
        
        # Net GEX (positive = resistance to movement, negative = acceleration)
        net_gex = total_call_gex - total_put_gex
        
        # Calculate expected move using ATM IV
        # Expected Move = Stock Price × IV × √(Days to Expiry / 365)
        if atm_iv > 0 and days_to_expiry > 0:
            time_factor = np.sqrt(days_to_expiry / 365)
            base_expected_move = underlying_price * atm_iv * time_factor
            
            # Adjust based on GEX positioning
            # High positive GEX = compress expected move (dealers dampen volatility)
            # High negative GEX = expand expected move (dealers amplify volatility)
            gex_adjustment = 1.0
            if abs(net_gex) > 0:
                # Normalize GEX impact (arbitrary scaling for visualization)
                gex_factor = net_gex / (underlying_price * underlying_price * 10000)
                gex_adjustment = 1.0 - (gex_factor * 0.2)  # ±20% max adjustment
                gex_adjustment = max(0.7, min(1.3, gex_adjustment))  # Clamp between 0.7x and 1.3x
            
            adjusted_move = base_expected_move * gex_adjustment
            
            return {
                'expected_move_up': underlying_price + adjusted_move,
                'expected_move_down': underlying_price - adjusted_move,
                'expected_move_pct': (adjusted_move / underlying_price) * 100,
                'atm_iv': atm_iv * 100,  # Convert to percentage
                'net_gex': net_gex,
                'gex_adjustment': gex_adjustment,
                'days_to_expiry': days_to_expiry,
                'regime': 'Dampening' if net_gex > 0 else 'Accelerating'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error calculating expected move: {str(e)}")
        return None

def analyze_volume_flow(options_data, price_history):
    """
    Analyze call vs put volume flow over time
    Returns: dict with timestamps and volume data for sparklines
    """
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        # Get price data
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True).dt.tz_convert('America/New_York')
        
        # Filter to today's market hours only
        today = pd.Timestamp.now(tz='America/New_York').date()
        df = df[
            (df['datetime'].dt.date == today) &
            (
                ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
                ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
            )
        ].copy()
        
        if df.empty:
            return None
        
        # Aggregate option volumes
        total_call_volume = 0
        total_put_volume = 0
        total_call_premium = 0
        total_put_premium = 0
        
        # Sum up all call volumes and premiums
        for exp_date, strikes in options_data.get('callExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0)
                    mark = contract.get('mark', 0)
                    total_call_volume += volume
                    total_call_premium += volume * mark * 100  # Premium in dollars
        
        # Sum up all put volumes and premiums
        for exp_date, strikes in options_data.get('putExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0)
                    mark = contract.get('mark', 0)
                    total_put_volume += volume
                    total_put_premium += volume * mark * 100  # Premium in dollars
        
        # Calculate ratios
        total_volume = total_call_volume + total_put_volume
        call_volume_pct = (total_call_volume / total_volume * 100) if total_volume > 0 else 50
        put_volume_pct = (total_put_volume / total_volume * 100) if total_volume > 0 else 50
        
        total_premium = total_call_premium + total_put_premium
        call_premium_pct = (total_call_premium / total_premium * 100) if total_premium > 0 else 50
        put_premium_pct = (total_put_premium / total_premium * 100) if total_premium > 0 else 50
        
        # Put/Call ratio
        pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        return {
            'call_volume': total_call_volume,
            'put_volume': total_put_volume,
            'call_volume_pct': call_volume_pct,
            'put_volume_pct': put_volume_pct,
            'call_premium': total_call_premium,
            'put_premium': total_put_premium,
            'call_premium_pct': call_premium_pct,
            'put_premium_pct': put_premium_pct,
            'pc_ratio': pc_ratio,
            'price_data': df[['datetime', 'close']].tail(60),  # Last 60 candles for sparkline
            'sentiment': 'Bullish' if call_volume_pct > 55 else 'Bearish' if put_volume_pct > 55 else 'Neutral'
        }
        
    except Exception as e:
        logger.error(f"Error analyzing volume flow: {str(e)}")
        return None

def create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol, most_valuable_strike=None, expected_move=None):
    """Create intraday chart with key levels overlaid"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        df['date'] = df['datetime'].dt.date
        
        # Filter to market hours only (9:30 AM - 4:00 PM)
        df = df[
            (
                ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
                ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
            )
        ].copy()
        
        if df.empty:
            return None
        
        # Get the last 2 unique trading days from the data
        unique_dates = sorted(df['date'].unique(), reverse=True)
        
        if len(unique_dates) == 0:
            return None
        elif len(unique_dates) == 1:
            # Only one day of data available - show just that day
            target_dates = [unique_dates[0]]
        else:
            # Show the last 2 trading days
            target_dates = unique_dates[:2]
        
        # Filter to only the last 2 trading days
        df = df[df['date'].isin(target_dates)].copy()
        
        if df.empty:
            return None
        
        # Log what dates we're showing for debugging
        logger.info(f"Chart displaying {len(target_dates)} day(s): {target_dates}")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create a continuous index to remove gaps
        df['chart_index'] = range(len(df))
        
        # Find where the most recent trading day starts (for separate VWAP calculation)
        most_recent_date = df['date'].max()
        today_start_idx = df[df['date'] == most_recent_date].index.min() if any(df['date'] == most_recent_date) else len(df)
        
        fig = go.Figure()
        
        # Calculate VWAP from first day's open (continuous through both days)
        df['vwap_from_yesterday'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Add candlesticks FIRST so they're behind the lines
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Yesterday's VWAP - thicker and cyan color for visibility
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['vwap_from_yesterday'],
            mode='lines',
            name='VWAP (From Day 1 Open)',
            line=dict(color='#00bcd4', width=3),
            hovertemplate='<b>VWAP from Day 1 Open</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Calculate most recent day's VWAP (from that day's open only)
        if today_start_idx < len(df):
            df_today = df.iloc[today_start_idx:].copy()
            df_today['vwap_today'] = (df_today['volume'] * (df_today['high'] + df_today['low'] + df_today['close']) / 3).cumsum() / df_today['volume'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=df_today['datetime'],
                y=df_today['vwap_today'],
                mode='lines',
                name='VWAP (Latest Day Open)',
                line=dict(color='#9c27b0', width=2.5),
                hovertemplate='<b>Latest Day VWAP</b>: $%{y:.2f}<extra></extra>'
            ))
        
        # Calculate 21 EMA on all data
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ema21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='#ff9800', width=2),
            hovertemplate='<b>21 EMA</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add level lines - clean, prominent styling for better visibility
        if levels['call_wall']['strike']:
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[levels['call_wall']['strike'], levels['call_wall']['strike']],
                mode='lines',
                name=f"📈 ${levels['call_wall']['strike']:.2f}",
                line=dict(color='#22c55e', width=3, dash='dot'),
                opacity=0.9,
                hovertemplate=f'<b>Call Wall (Resistance)</b><br>${levels["call_wall"]["strike"]:.2f}<extra></extra>',
                showlegend=True
            ))
        
        if levels['put_wall']['strike']:
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[levels['put_wall']['strike'], levels['put_wall']['strike']],
                mode='lines',
                name=f"📉 ${levels['put_wall']['strike']:.2f}",
                line=dict(color='#ef4444', width=3, dash='dot'),
                opacity=0.9,
                hovertemplate=f'<b>Put Wall (Support)</b><br>${levels["put_wall"]["strike"]:.2f}<extra></extra>',
                showlegend=True
            ))
        
        if levels['flip_level']:
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[levels['flip_level'], levels['flip_level']],
                mode='lines',
                name=f"🔄 Flip ${levels['flip_level']:.2f}",
                line=dict(color='#a855f7', width=3.5, dash='solid'),
                opacity=0.85,
                hovertemplate=f'<b>Flip Level (Sentiment Pivot)</b><br>${levels["flip_level"]:.2f}<extra></extra>',
                showlegend=True
            ))
        
        # Add Most Valuable Strike line if provided
        if most_valuable_strike:
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[most_valuable_strike, most_valuable_strike],
                mode='lines',
                name=f"🎯 MVS ${most_valuable_strike:.2f}",
                line=dict(color='#9c27b0', width=4, dash='dashdot'),
                opacity=0.95,
                hovertemplate=f'<b>Most Valuable Strike</b><br>${most_valuable_strike:.2f}<br><i>Highest impact level</i><extra></extra>',
                showlegend=True
            ))
        
        # Add Expected Move zones if provided
        if expected_move:
            em_up = expected_move['expected_move_up']
            em_down = expected_move['expected_move_down']
            
            # Add shaded zone for expected move
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1], df['datetime'].iloc[-1], df['datetime'].iloc[0]],
                y=[em_up, em_up, em_down, em_down],
                fill='toself',
                fillcolor='rgba(255, 193, 7, 0.1)',
                line=dict(width=0),
                name=f"Expected Move ±{expected_move['expected_move_pct']:.1f}%",
                hoverinfo='skip',
                showlegend=True
            ))
            
            # Add boundary lines
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[em_up, em_up],
                mode='lines',
                name=f"EM Upper ${em_up:.2f}",
                line=dict(color='#fbbf24', width=2, dash='dot'),
                opacity=0.7,
                hovertemplate=f'<b>Expected Move Upper</b><br>${em_up:.2f}<extra></extra>',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[df['datetime'].iloc[0], df['datetime'].iloc[-1]],
                y=[em_down, em_down],
                mode='lines',
                name=f"EM Lower ${em_down:.2f}",
                line=dict(color='#fbbf24', width=2, dash='dot'),
                opacity=0.7,
                hovertemplate=f'<b>Expected Move Lower</b><br>${em_down:.2f}<extra></extra>',
                showlegend=False
            ))
        
        # Add annotations on the right side for key levels
        annotations = []
        
        if levels['call_wall']['strike']:
            annotations.append(dict(
                x=1.01,  # Position to the right of the plot
                y=levels['call_wall']['strike'],
                xref='paper',
                yref='y',
                text=f"📈 Call Wall<br>${levels['call_wall']['strike']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#22c55e',
                ax=40,
                ay=0,
                xanchor='left',
                font=dict(size=10, color='#ffffff', family='Arial, sans-serif', weight='bold'),
                bgcolor='rgba(34, 197, 94, 0.95)',
                bordercolor='#22c55e',
                borderwidth=2,
                borderpad=6
            ))
        
        if levels['put_wall']['strike']:
            annotations.append(dict(
                x=1.01,
                y=levels['put_wall']['strike'],
                xref='paper',
                yref='y',
                text=f"📉 Put Wall<br>${levels['put_wall']['strike']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#ef4444',
                ax=40,
                ay=0,
                xanchor='left',
                font=dict(size=10, color='#ffffff', family='Arial, sans-serif', weight='bold'),
                bgcolor='rgba(239, 68, 68, 0.95)',
                bordercolor='#ef4444',
                borderwidth=2,
                borderpad=6
            ))
        
        if levels['flip_level']:
            annotations.append(dict(
                x=1.01,
                y=levels['flip_level'],
                xref='paper',
                yref='y',
                text=f"🔄 ${levels['flip_level']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#a855f7',
                ax=40,
                ay=0,
                xanchor='left',
                font=dict(size=11, color='#ffffff', family='Arial, sans-serif', weight='bold'),
                bgcolor='rgba(168, 85, 247, 0.95)',
                bordercolor='#a855f7',
                borderwidth=2,
                borderpad=6
            ))
        
        # Add Most Valuable Strike annotation
        if most_valuable_strike:
            annotations.append(dict(
                x=1.01,
                y=most_valuable_strike,
                xref='paper',
                yref='y',
                text=f"🎯 MVS<br>${most_valuable_strike:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2.5,
                arrowcolor='#9c27b0',
                ax=40,
                ay=0,
                xanchor='left',
                font=dict(size=11, color='#ffffff', family='Arial, sans-serif', weight='bold'),
                bgcolor='rgba(156, 39, 176, 0.95)',
                bordercolor='#9c27b0',
                borderwidth=2.5,
                borderpad=6
            ))
        
        # Add current price annotation - larger and more prominent
        annotations.append(dict(
            x=1.01,
            y=underlying_price,
            xref='paper',
            yref='y',
            text=f"💰 ${underlying_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor='#ffd700',
            ax=40,
            ay=0,
            xanchor='left',
            font=dict(size=11, color='#000000', family='Arial, sans-serif', weight='bold'),
            bgcolor='rgba(255, 215, 0, 0.95)',
            bordercolor='#ffd700',
            borderwidth=3,
            borderpad=6
        ))
        
        # Add space on right side of chart for better visibility
        # Extend x-axis by 10% to create breathing room
        time_range = df['datetime'].max() - df['datetime'].min()
        x_axis_end = df['datetime'].max() + time_range * 0.1
        
        # Calculate intelligent Y-axis range with better padding
        # Include all key levels in the range calculation
        all_prices = [df['high'].max(), df['low'].min(), underlying_price]
        if levels['call_wall']['strike']:
            all_prices.append(levels['call_wall']['strike'])
        if levels['put_wall']['strike']:
            all_prices.append(levels['put_wall']['strike'])
        if levels['flip_level']:
            all_prices.append(levels['flip_level'])
        if most_valuable_strike:
            all_prices.append(most_valuable_strike)
        if expected_move:
            all_prices.append(expected_move['expected_move_up'])
            all_prices.append(expected_move['expected_move_down'])
        
        price_range = max(all_prices) - min(all_prices)
        y_padding = price_range * 0.15  # 15% padding on each side for breathing room
        y_min = min(all_prices) - y_padding
        y_max = max(all_prices) + y_padding
        
        fig.update_layout(
            title=dict(
                text=f"{symbol} - Intraday + Walls",
                font=dict(size=18, color='#1f2937'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Time (ET)",
            yaxis_title="Price ($)",
            height=550,
            template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                type='date',
                tickformat='%I:%M %p\n%b %d',  # Show time and date
                dtick=3600000,  # Tick every hour (in milliseconds)
                tickangle=0,
                range=[df['datetime'].min(), x_axis_end],  # Extend x-axis by 10%
                rangebreaks=[
                    dict(bounds=[16, 9.5], pattern="hour"),  # Hide hours between 4 PM and 9:30 AM
                ],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis=dict(
                range=[y_min, y_max],  # Set fixed range with padding
                tickformat='$.2f',
                gridcolor='rgba(0,0,0,0.08)',
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10),
                itemwidth=30
            ),
            annotations=annotations,  # Add the annotations
            margin=dict(t=120, r=150, l=80, b=80),  # Better margins all around
            plot_bgcolor='rgba(250, 250, 250, 0.5)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_macd_chart(price_history, symbol):
    """Create MACD indicator chart"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        df['date'] = df['datetime'].dt.date
        
        # Filter to market hours only (9:30 AM - 4:00 PM)
        df = df[
            (
                ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
                ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
            )
        ].copy()
        
        if df.empty:
            return None
        
        # Get the last 2 unique trading days from the data
        unique_dates = sorted(df['date'].unique(), reverse=True)
        
        if len(unique_dates) == 0:
            return None
        elif len(unique_dates) == 1:
            target_dates = [unique_dates[0]]
        else:
            target_dates = unique_dates[:2]
        
        # Filter to only the last 2 trading days
        df = df[df['date'].isin(target_dates)].copy()
        
        if df.empty:
            return None
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        fig = go.Figure()
        
        # Add histogram
        colors = ['#22c55e' if val >= 0 else '#ef4444' for val in df['histogram']]
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['histogram'],
            name='Histogram',
            marker=dict(color=colors),
            hovertemplate='<b>Histogram</b>: %{y:.4f}<extra></extra>'
        ))
        
        # Add MACD line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='#2196f3', width=2),
            hovertemplate='<b>MACD</b>: %{y:.4f}<extra></extra>'
        ))
        
        # Add Signal line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff9800', width=2),
            hovertemplate='<b>Signal</b>: %{y:.4f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        time_range = df['datetime'].max() - df['datetime'].min()
        x_axis_end = df['datetime'].max() + time_range * 0.1
        
        fig.update_layout(
            title=f"{symbol} - MACD (12, 26, 9)",
            xaxis_title="Time (ET)",
            yaxis_title="MACD",
            height=250,
            template='plotly_white',
            hovermode='x unified',
            xaxis=dict(
                type='date',
                tickformat='%I:%M %p\n%b %d',
                dtick=3600000,
                tickangle=0,
                range=[df['datetime'].min(), x_axis_end],
                rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.08)',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(t=80, r=20, l=80, b=60),
            plot_bgcolor='rgba(250, 250, 250, 0.5)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating MACD chart: {str(e)}")
        return None

def get_gex_by_strike(options_data, underlying_price, expiry_date):
    """Extract GEX values by strike for a specific expiry to display alongside intraday chart"""
    try:
        gamma_data = []
        
        # Convert expiry_date to string format if it's a datetime object
        if hasattr(expiry_date, 'strftime'):
            target_expiry = expiry_date.strftime('%Y-%m-%d')
        else:
            target_expiry = str(expiry_date)
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                if expiry == target_expiry:
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            contract = contracts[0]
                            strike = float(strike_str)
                            gamma = contract.get('gamma', 0)
                            oi = contract.get('openInterest', 0)
                            
                            # Calculate signed notional gamma (positive for calls)
                            signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            gamma_data.append({
                                'strike': strike,
                                'signed_notional_gamma': signed_gamma,
                                'type': 'call'
                            })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                if expiry == target_expiry:
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            contract = contracts[0]
                            strike = float(strike_str)
                            gamma = contract.get('gamma', 0)
                            oi = contract.get('openInterest', 0)
                            
                            # Calculate signed notional gamma (negative for puts)
                            signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01 * -1
                            
                            gamma_data.append({
                                'strike': strike,
                                'signed_notional_gamma': signed_gamma,
                                'type': 'put'
                            })
        
        if not gamma_data:
            logger.warning("No gamma data found in options chain")
            return None
        
        # Create dataframe and aggregate by strike
        df_gamma = pd.DataFrame(gamma_data)
        strike_gex = df_gamma.groupby('strike')['signed_notional_gamma'].sum().reset_index()
        strike_gex.columns = ['strike', 'net_gex']
        
        # Filter to reasonable range around current price (±10%)
        min_strike = underlying_price * 0.90
        max_strike = underlying_price * 1.10
        strike_gex = strike_gex[(strike_gex['strike'] >= min_strike) & (strike_gex['strike'] <= max_strike)]
        
        if len(strike_gex) == 0:
            logger.warning(f"No strikes found in range ${min_strike:.2f} - ${max_strike:.2f}")
            return None
        
        return strike_gex.sort_values('strike')
        
    except Exception as e:
        logger.error(f"Error calculating GEX by strike: {str(e)}")
        return None

def create_gamma_heatmap(options_data, underlying_price, num_expiries=6):
    """Create a NetGEX heatmap showing gamma exposure across strikes and expirations"""
    
    try:
        # Parse options data to create gamma dataframe
        gamma_data = []
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                # Extract just the date from the format "2025-11-08:7"
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        oi = contract.get('openInterest', 0)
                        
                        # Calculate signed notional gamma (positive for calls from dealer perspective)
                        signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'strike': strike,
                            'expiry': expiry,
                            'signed_notional_gamma': signed_gamma,
                            'type': 'call'
                        })
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        oi = contract.get('openInterest', 0)
                        
                        # Calculate signed notional gamma (negative for puts from dealer perspective)
                        signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01 * -1
                        
                        gamma_data.append({
                            'strike': strike,
                            'expiry': expiry,
                            'signed_notional_gamma': signed_gamma,
                            'type': 'put'
                        })
        
        if not gamma_data:
            return None
        
        df_gamma = pd.DataFrame(gamma_data)
        
        # Get unique expiries and strikes
        expiries = sorted(df_gamma['expiry'].unique())[:min(num_expiries, 4)]  # Max 4 expiries for better spacing
        all_strikes = sorted(df_gamma['strike'].unique())
        
        # Filter strikes to a tighter range for better readability (±5%)
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        
        filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # Limit to 12 strikes max for readability
        if not filtered_strikes or len(filtered_strikes) > 12:
            sorted_by_distance = sorted(all_strikes, key=lambda x: abs(x - underlying_price))
            filtered_strikes = sorted(sorted_by_distance[:12])
        
        # Create the data matrix for the heat map
        heat_data = []
        
        for strike in filtered_strikes:
            row = []
            for expiry in expiries:
                # Get net GEX for this strike/expiry combination
                mask = (df_gamma['strike'] == strike) & (df_gamma['expiry'] == expiry)
                strike_exp_data = df_gamma[mask]
                
                if not strike_exp_data.empty:
                    # Calculate Net GEX = sum of all signed gamma for this strike/expiry
                    net_gex = strike_exp_data['signed_notional_gamma'].sum()
                    row.append(net_gex)
                else:
                    row.append(0)
            
            heat_data.append(row)
        
        # Create labels
        strike_labels = [f"${s:.0f}" for s in filtered_strikes]
        expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        # Create custom colorscale for better visibility
        # Negative (red) = dealer short gamma = acceleration
        # Positive (blue) = dealer long gamma = resistance
        custom_colorscale = [
            [0.0, '#d32f2f'],   # Dark red (very negative GEX)
            [0.25, '#ef5350'],  # Red
            [0.4, '#ffcdd2'],   # Light red
            [0.5, '#ffffff'],   # White (zero GEX)
            [0.6, '#bbdefb'],   # Light blue
            [0.75, '#42a5f5'],  # Blue
            [1.0, '#1565c0']    # Dark blue (very positive GEX)
        ]
        
        # Calculate max absolute value for better text contrast
        max_abs_value = max(abs(val) for row in heat_data for val in row) if heat_data else 1
        
        # Create text annotations with smart color based on intensity
        text_annotations = []
        text_colors = []
        for row in heat_data:
            text_row = []
            color_row = []
            for val in row:
                # Format the value
                if abs(val) >= 1e6:
                    text_row.append(f"${val/1e6:.1f}M")
                elif abs(val) >= 1e3:
                    text_row.append(f"${val/1e3:.0f}K")
                else:
                    text_row.append("")
                
                # Choose text color based on background intensity
                if abs(val) > max_abs_value * 0.4:
                    color_row.append('white')
                else:
                    color_row.append('black')
            
            text_annotations.append(text_row)
            text_colors.append(color_row)
        
        # Create the heat map with larger cells
        fig = go.Figure(data=go.Heatmap(
            z=heat_data,
            x=expiry_labels,
            y=strike_labels,
            colorscale=custom_colorscale,
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Net GEX ($)",
                tickformat='$,.0s',
                len=0.7,
                thickness=20
            ),
            hovertemplate='<b>Strike: %{y}</b><br>Expiry: %{x}<br>Net GEX: $%{z:,.0f}<extra></extra>',
            text=text_annotations,
            texttemplate='%{text}',
            textfont=dict(size=13, family='Arial Black')
        ))
        
        # Find closest strike to current price for yellow line
        closest_strike = min(filtered_strikes, key=lambda x: abs(x - underlying_price))
        
        # Find the index of the closest strike in the filtered list
        try:
            current_price_idx = filtered_strikes.index(closest_strike)
            
            # Add current price line at the correct position
            fig.add_hline(
                y=current_price_idx,
                line=dict(color="yellow", width=3, dash="dash"),
                annotation_text=f"  ${underlying_price:.2f}",
                annotation_position="right",
                annotation=dict(font_size=11, font_color="yellow", bgcolor="rgba(0,0,0,0.7)")
            )
        except (ValueError, IndexError):
            pass  # Skip if strike not found
        
        fig.update_layout(
            title=dict(
                text=f"Net Gamma Exposure (GEX) Heatmap - Current: ${underlying_price:.2f}",
                font=dict(size=18, color='black', family='Arial Black')
            ),
            xaxis=dict(
                title=dict(
                    text="Expiration Date",
                    font=dict(size=14)
                ),
                tickfont=dict(size=13)
            ),
            yaxis=dict(
                title=dict(
                    text="Strike Price",
                    font=dict(size=14)
                ),
                tickfont=dict(size=13),
                dtick=1  # Show every strike for clarity
            ),
            height=650,  # Adjusted for fewer strikes
            template='plotly_white',
            font=dict(size=13),
            margin=dict(l=100, r=100, t=100, b=80)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating gamma heatmap: {str(e)}")
        return None

def create_net_premium_heatmap(options_data, underlying_price, num_expiries=6):
    """Create a Net Premium heatmap showing call premium - put premium across strikes and expirations"""
    
    try:
        # Create a dictionary to aggregate premiums by strike and expiry
        premium_matrix = {}  # {(strike, expiry): {'call': X, 'put': Y}}
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        mark_price = contract.get('mark', 0)
                        
                        # Calculate notional premium (volume * mark * 100 shares per contract)
                        notional_premium = volume * mark_price * 100
                        
                        key = (strike, expiry)
                        if key not in premium_matrix:
                            premium_matrix[key] = {'call': 0, 'put': 0}
                        premium_matrix[key]['call'] += notional_premium
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        mark_price = contract.get('mark', 0)
                        
                        # Calculate notional premium
                        notional_premium = volume * mark_price * 100
                        
                        key = (strike, expiry)
                        if key not in premium_matrix:
                            premium_matrix[key] = {'call': 0, 'put': 0}
                        premium_matrix[key]['put'] += notional_premium
        
        if not premium_matrix:
            st.warning("No premium data available for the selected expiry")
            return None
        
        # Get unique expiries and strikes
        all_strikes = sorted(set(k[0] for k in premium_matrix.keys()))
        all_expiries = sorted(set(k[1] for k in premium_matrix.keys()))
        
        if not all_strikes or not all_expiries:
            st.warning("Insufficient strike or expiry data")
            return None
        
        # Limit to nearest 3-4 expiries
        expiries = all_expiries[:min(4, len(all_expiries))]
        
        # Filter strikes to ±5% range (tighter than before)
        min_strike = underlying_price * 0.95
        max_strike = underlying_price * 1.05
        
        # Get all strikes in range
        strikes_in_range = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # Calculate total activity per strike across all expiries
        strike_activity = {}
        for strike in strikes_in_range:
            total = 0
            for expiry in expiries:
                key = (strike, expiry)
                if key in premium_matrix:
                    total += abs(premium_matrix[key]['call']) + abs(premium_matrix[key]['put'])
            if total > 0:
                strike_activity[strike] = total
        
        # Sort by activity and take top 12, then sort by strike price
        if strike_activity:
            top_strikes = sorted(strike_activity.items(), key=lambda x: x[1], reverse=True)[:12]
            filtered_strikes = sorted([s[0] for s in top_strikes])
        elif strikes_in_range:
            # Fallback: get 12 closest strikes to current price
            filtered_strikes = sorted(strikes_in_range, key=lambda x: abs(x - underlying_price))[:12]
        else:
            # Last resort: get any 12 strikes closest to current price
            filtered_strikes = sorted(all_strikes, key=lambda x: abs(x - underlying_price))[:12]
        
        if not filtered_strikes:
            return None
        
        # Create the data matrix for the heat map
        heat_data = []
        
        for strike in filtered_strikes:
            row = []
            for expiry in expiries:
                # Get premium for this strike/expiry combination
                key = (strike, expiry)
                if key in premium_matrix:
                    # Calculate Net Premium = Call Premium - Put Premium
                    # Positive = call-heavy (bullish bias)
                    # Negative = put-heavy (bearish bias)
                    net_premium = premium_matrix[key]['call'] - premium_matrix[key]['put']
                    row.append(net_premium)
                else:
                    row.append(0)
            
            heat_data.append(row)
        
        # Create labels - match GEX format exactly
        strike_labels = [f"${s:.0f}" for s in filtered_strikes]
        expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        # Custom colorscale - match GEX colors exactly
        custom_colorscale = [
            [0.0, '#d32f2f'],   # Dark red (very bearish)
            [0.25, '#ef5350'],  # Red
            [0.4, '#ffcdd2'],   # Light red
            [0.5, '#ffffff'],   # White (neutral)
            [0.6, '#c8e6c9'],   # Light green
            [0.75, '#66bb6a'],  # Green
            [1.0, '#2e7d32']    # Dark green (very bullish)
        ]
        
        # Calculate max absolute value for better text contrast
        max_abs_value = max(abs(val) for row in heat_data for val in row) if heat_data else 1
        
        # Create text annotations matching GEX style
        text_annotations = []
        
        for row in heat_data:
            row_text = []
            for val in row:
                # Format based on magnitude
                if abs(val) >= 1e6:
                    formatted = f"${val/1e6:.1f}M"
                elif abs(val) >= 1e3:
                    formatted = f"${val/1e3:.0f}K"
                elif abs(val) > 0:
                    formatted = f"${val:.0f}"
                else:
                    formatted = ""
                
                row_text.append(formatted)
            
            text_annotations.append(row_text)
        
        # Create heatmap - exactly matching GEX heatmap style
        fig = go.Figure(data=go.Heatmap(
            z=heat_data,
            x=expiry_labels,
            y=strike_labels,
            colorscale=custom_colorscale,
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Net Premium ($)",
                tickformat='$,.0s',
                len=0.7,
                thickness=20
            ),
            hovertemplate='<b>Strike: %{y}</b><br>Expiry: %{x}<br>Net Premium: $%{z:,.0f}<extra></extra>',
            text=text_annotations,
            texttemplate='%{text}',
            textfont=dict(size=13, family='Arial Black')
        ))
        
        # Find closest strike to current price for yellow line
        closest_strike = min(filtered_strikes, key=lambda x: abs(x - underlying_price))
        
        # Find the index of the closest strike in the filtered list
        try:
            current_price_idx = filtered_strikes.index(closest_strike)
            
            # Add current price line at the correct position
            fig.add_hline(
                y=current_price_idx,
                line=dict(color="yellow", width=3, dash="dash"),
                annotation_text=f"  ${underlying_price:.2f}",
                annotation_position="right",
                annotation=dict(font_size=11, font_color="yellow", bgcolor="rgba(0,0,0,0.7)")
            )
        except (ValueError, IndexError):
            pass  # Skip if strike not found
        
        # Layout matching GEX heatmap exactly
        fig.update_layout(
            title=dict(
                text=f"Net Premium Flow (Call - Put) - Current: ${underlying_price:.2f}",
                font=dict(size=18, color='black', family='Arial Black')
            ),
            xaxis=dict(
                title=dict(
                    text="Expiration Date",
                    font=dict(size=14)
                ),
                tickfont=dict(size=13)
            ),
            yaxis=dict(
                title=dict(
                    text="Strike Price",
                    font=dict(size=14)
                ),
                tickfont=dict(size=13),
                dtick=1  # Show every strike for clarity
            ),
            height=650,  # Match GEX height
            template='plotly_white',
            font=dict(size=13),
            margin=dict(l=100, r=100, t=100, b=80)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating net premium heatmap: {str(e)}")
        return None

def create_interval_map(price_history, options_data, underlying_price, symbol):
    """Create an interval map showing price movement with gamma exposure bubbles"""
    try:
        if 'candles' not in price_history or not price_history['candles']:
            return None
        
        # Process price data
        df = pd.DataFrame(price_history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        
        # Filter to today's market hours only
        today = pd.Timestamp.now(tz='America/New_York').date()
        df['date'] = df['datetime'].dt.date
        
        df = df[
            (df['date'] == today) &
            (
                ((df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute >= 30)) |
                ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour < 16))
            )
        ].copy()
        
        if df.empty:
            return None
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate GEX for each strike using VOLUME (not OI)
        # Volume reflects intraday activity, OI is static
        gamma_by_strike = {}
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        volume = contract.get('totalVolume', 0)
                        
                        # Use VOLUME instead of OI for intraday gamma exposure
                        # Positive GEX for calls (dealer long gamma = resistance)
                        gex = gamma * volume * 100 * underlying_price * underlying_price * 0.01
                        gamma_by_strike[strike] = gamma_by_strike.get(strike, 0) + gex
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        volume = contract.get('totalVolume', 0)
                        
                        # Use VOLUME instead of OI for intraday gamma exposure
                        # Negative GEX for puts (dealer short gamma = acceleration)
                        gex = gamma * volume * 100 * underlying_price * underlying_price * 0.01 * -1
                        gamma_by_strike[strike] = gamma_by_strike.get(strike, 0) + gex
        
        # Filter strikes near current price (±5%)
        min_strike = underlying_price * 0.93
        max_strike = underlying_price * 1.07
        filtered_strikes = {k: v for k, v in gamma_by_strike.items() if min_strike <= k <= max_strike}
        
        # Sort and get top strikes by absolute GEX
        sorted_strikes = sorted(filtered_strikes.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        
        # Create figure
        fig = go.Figure()
        
        # Add gamma exposure bubbles FIRST (so they appear behind the price line)
        # Create time grid for bubbles (every 15 minutes for clarity)
        time_points = pd.date_range(
            start=df['datetime'].min(),
            end=df['datetime'].max(),
            freq='15min'
        )
        
        # Calculate price direction and distance for dynamic coloring
        # Get price at each time point for comparison
        price_at_time = {}
        for tp in time_points:
            # Find closest price data point
            closest_idx = (df['datetime'] - tp).abs().idxmin()
            if pd.notna(closest_idx):
                price_at_time[tp] = df.loc[closest_idx, 'close']
        
        # For each strike, add discrete bubbles with dynamic coloring
        for strike, gex in sorted_strikes:
            if abs(gex) < 5e5:  # Skip very small GEX
                continue
            
            # Create color array for each time point based on price distance
            colors = []
            sizes = []
            
            for tp in time_points:
                current_price = price_at_time.get(tp, underlying_price)
                distance_pct = abs((strike - current_price) / current_price)
                
                # Calculate brightness based on distance
                # Closer = brighter (higher alpha), further = dimmer (lower alpha)
                if distance_pct < 0.005:  # Within 0.5%
                    alpha = 0.8  # Very bright
                elif distance_pct < 0.01:  # Within 1%
                    alpha = 0.6  # Bright
                elif distance_pct < 0.02:  # Within 2%
                    alpha = 0.4  # Medium
                else:
                    alpha = 0.25  # Dim
                
                # Determine base color
                if gex > 0:
                    # Green for positive (resistance)
                    colors.append(f'rgba(34, 197, 94, {alpha})')
                else:
                    # Red for negative (acceleration)
                    colors.append(f'rgba(239, 68, 68, {alpha})')
                
                # Size also slightly increases when price is near
                base_size = min(max(abs(gex) / 1e6, 4), 15)
                if distance_pct < 0.01:
                    sizes.append(base_size * 1.3)  # 30% larger when price is very close
                else:
                    sizes.append(base_size)
            
            # Add scatter points with varying colors
            fig.add_trace(go.Scatter(
                x=time_points,
                y=[strike] * len(time_points),
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol='circle',
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                ),
                showlegend=False,
                hovertemplate=f'<b>Strike ${strike:.2f}</b><br>GEX: ${gex/1e6:.1f}M<extra></extra>'
            ))
        
        # Add price line ON TOP
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#00bcd4', width=3),
            hovertemplate='<b>Price: $%{y:.2f}</b><br>%{x|%I:%M %p}<extra></extra>'
        ))
        
        # Add current price indicator
        fig.add_hline(
            y=underlying_price,
            line=dict(color='#ffd700', width=2, dash='dash'),
            annotation=dict(
                text=f'Underlying (${underlying_price:.2f})',
                font=dict(size=11, color='#333333'),
                bgcolor='#ffd700',
                bordercolor='#333333',
                borderwidth=1,
                borderpad=4,
                xanchor='left',
                x=0.01
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f'Interval Map (GEX) - {symbol}',
                font=dict(size=16, color='#333333')
            ),
            xaxis_title='Time (ET)',
            yaxis_title='Strike Price ($)',
            height=600,
            template='plotly_white',
            hovermode='closest',
            showlegend=False,
            plot_bgcolor='rgba(20, 30, 48, 1)',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.15)',
                tickformat='%I:%M %p',
                color='white'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.15)',
                color='white'
            ),
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating interval map: {str(e)}")
        return None

def generate_tradeable_alerts(levels, underlying_price, symbol, price_data=None):
    """Generate immediate tradeable alerts based on current price action"""
    alerts = []
    
    try:
        # Calculate distances to key levels
        put_wall = levels['put_wall']['strike']
        call_wall = levels['call_wall']['strike']
        flip_level = levels['flip_level']
        net_call_wall = levels['net_call_wall']['strike']
        net_put_wall = levels['net_put_wall']['strike']
        
        # Alert 1: Near Put Wall (Support Test)
        if put_wall:
            dist_to_put = ((underlying_price - put_wall) / put_wall) * 100
            if 0 <= dist_to_put <= 1:  # Within 1% above put wall
                alerts.append({
                    'priority': '🔴 HIGH',
                    'type': 'Support Test',
                    'message': f'Price at ${underlying_price:.2f} testing Put Wall support at ${put_wall:.2f}',
                    'action': f'WATCH: Bounce opportunity if holds above ${put_wall:.2f}. Break below = sell signal',
                    'level': put_wall
                })
            elif -0.5 <= dist_to_put < 0:  # Just broke below
                alerts.append({
                    'priority': '🔴 HIGH',
                    'type': 'Support Break',
                    'message': f'BREAKDOWN: Price ${underlying_price:.2f} broke below Put Wall ${put_wall:.2f}',
                    'action': f'BEARISH: Consider puts or exit longs. Next support at ${put_wall * 0.98:.2f}',
                    'level': put_wall
                })
        
        # Alert 2: Near Call Wall (Resistance Test)
        if call_wall:
            dist_to_call = ((call_wall - underlying_price) / underlying_price) * 100
            if 0 <= dist_to_call <= 1:  # Within 1% below call wall
                alerts.append({
                    'priority': '🔴 HIGH',
                    'type': 'Resistance Test',
                    'message': f'Price at ${underlying_price:.2f} approaching Call Wall resistance at ${call_wall:.2f}',
                    'action': f'WATCH: Rejection likely. Break above ${call_wall:.2f} = bullish breakout',
                    'level': call_wall
                })
            elif -0.5 <= dist_to_call < 0:  # Just broke above
                alerts.append({
                    'priority': '🔴 HIGH',
                    'type': 'Resistance Break',
                    'message': f'BREAKOUT: Price ${underlying_price:.2f} broke above Call Wall ${call_wall:.2f}',
                    'action': f'BULLISH: Consider calls. Next resistance at ${call_wall * 1.02:.2f}',
                    'level': call_wall
                })
        
        # Alert 3: Flip Level Cross
        if flip_level:
            dist_to_flip = abs((underlying_price - flip_level) / flip_level) * 100
            if dist_to_flip <= 0.5:  # Within 0.5% of flip level
                sentiment = "BULLISH" if underlying_price > flip_level else "BEARISH"
                alerts.append({
                    'priority': '🟡 MEDIUM',
                    'type': 'Flip Level',
                    'message': f'Price at ${underlying_price:.2f} near Flip Level ${flip_level:.2f}',
                    'action': f'{sentiment} bias. Watch for volatility expansion if crossed',
                    'level': flip_level
                })
        
        # Alert 4: Between Walls (Pin Risk)
        if call_wall and put_wall:
            range_pct = ((call_wall - put_wall) / put_wall) * 100
            if put_wall < underlying_price < call_wall and range_pct < 5:  # Tight range
                mid_point = (call_wall + put_wall) / 2
                alerts.append({
                    'priority': '🟢 LOW',
                    'type': 'Range Bound',
                    'message': f'Price pinned between ${put_wall:.2f} and ${call_wall:.2f} ({range_pct:.1f}% range)',
                    'action': f'RANGE TRADE: Sell near ${call_wall:.2f}, buy near ${put_wall:.2f}. Or wait for breakout',
                    'level': mid_point
                })
        
        # Alert 5: Strong GEX Levels
        if net_call_wall and net_put_wall:
            if net_call_wall < underlying_price < call_wall:
                alerts.append({
                    'priority': '🟡 MEDIUM',
                    'type': 'Gamma Resistance',
                    'message': f'Price in gamma resistance zone (${net_call_wall:.2f} - ${call_wall:.2f})',
                    'action': f'CAUTION: Dealers hedging will suppress upside. Need strong momentum to break higher',
                    'level': net_call_wall
                })
            elif put_wall < underlying_price < net_put_wall:
                alerts.append({
                    'priority': '🟡 MEDIUM',
                    'type': 'Gamma Support',
                    'message': f'Price in gamma support zone (${put_wall:.2f} - ${net_put_wall:.2f})',
                    'action': f'SUPPORT: Dealers hedging will limit downside. Dips may be bought',
                    'level': net_put_wall
                })
        
        return alerts
        
    except Exception as e:
        st.error(f"Error generating alerts: {str(e)}")
        return []

def generate_trade_setups(levels, underlying_price, symbol):
    """Generate actionable trade setups based on key levels"""
    setups = []
    
    try:
        # Setup 1: Put Wall Bounce (Support)
        if levels['put_wall']['strike']:
            put_wall = levels['put_wall']['strike']
            distance_to_put_wall = ((underlying_price - put_wall) / underlying_price) * 100
            
            if -2 <= distance_to_put_wall <= 2:  # Within 2% of put wall
                setups.append({
                    'type': '🎯 Bounce Trade at Put Wall',
                    'bias': 'BULLISH',
                    'entry': put_wall,
                    'stop': put_wall * 0.98,  # 2% below
                    'target1': levels['flip_level'] if levels['flip_level'] else underlying_price * 1.02,
                    'target2': levels['call_wall']['strike'] if levels['call_wall']['strike'] else underlying_price * 1.05,
                    'risk_reward': 2.5,
                    'confidence': 'High' if levels['put_wall']['oi'] > 5000 else 'Medium',
                    'reasoning': f"Price near Put Wall support (${put_wall:.2f}) with {levels['put_wall']['volume']:,} volume and {levels['put_wall']['oi']:,} OI"
                })
        
        # Setup 2: Call Wall Breakout (Resistance break)
        if levels['call_wall']['strike']:
            call_wall = levels['call_wall']['strike']
            distance_to_call_wall = ((call_wall - underlying_price) / underlying_price) * 100
            
            if 0 <= distance_to_call_wall <= 3:  # Price below and approaching call wall
                setups.append({
                    'type': '🚀 Breakout Trade Above Call Wall',
                    'bias': 'BULLISH',
                    'entry': call_wall * 1.002,  # 0.2% above wall
                    'stop': call_wall * 0.995,  # Back below wall
                    'target1': call_wall * 1.015,
                    'target2': call_wall * 1.03,
                    'risk_reward': 2.0,
                    'confidence': 'Medium',
                    'reasoning': f"Breakout above Call Wall resistance (${call_wall:.2f}) with {levels['call_wall']['volume']:,} volume acts as magnet"
                })
        
        # Setup 3: Flip Level Cross (Sentiment Change)
        if levels['flip_level']:
            flip = levels['flip_level']
            distance_to_flip = ((flip - underlying_price) / underlying_price) * 100
            
            if abs(distance_to_flip) <= 1.5:  # Within 1.5% of flip
                bias = 'BULLISH' if underlying_price > flip else 'BEARISH'
                setups.append({
                    'type': '🔄 Flip Level Momentum Trade',
                    'bias': bias,
                    'entry': flip,
                    'stop': flip * 0.99 if bias == 'BULLISH' else flip * 1.01,
                    'target1': levels['call_wall']['strike'] if bias == 'BULLISH' and levels['call_wall']['strike'] else flip * 1.02,
                    'target2': levels['net_call_wall']['strike'] if bias == 'BULLISH' and levels['net_call_wall']['strike'] else flip * 1.04,
                    'risk_reward': 2.0,
                    'confidence': 'High',
                    'reasoning': f"Flip level (${flip:.2f}) marks sentiment transition - momentum likely continues"
                })
        
        # Setup 4: Range Trade (Between Walls)
        if levels['put_wall']['strike'] and levels['call_wall']['strike']:
            put_wall = levels['put_wall']['strike']
            call_wall = levels['call_wall']['strike']
            range_pct = ((call_wall - put_wall) / put_wall) * 100
            
            if 3 <= range_pct <= 10:  # Reasonable range
                position_in_range = (underlying_price - put_wall) / (call_wall - put_wall)
                
                if 0.4 <= position_in_range <= 0.6:  # In middle of range
                    setups.append({
                        'type': '📊 Range Trade (Pin Risk)',
                        'bias': 'NEUTRAL',
                        'entry': underlying_price,
                        'stop': None,
                        'target1': put_wall,
                        'target2': call_wall,
                        'risk_reward': 1.5,
                        'confidence': 'Medium',
                        'reasoning': f"Price in middle of range ${put_wall:.2f}-${call_wall:.2f}. Fade extremes, take profit at walls"
                    })
        
    except Exception as e:
        st.error(f"Error generating trade setups: {str(e)}")
    
    return setups

def create_volume_profile_chart(levels, underlying_price, symbol):
    """Create horizontal volume profile showing net volumes by strike"""
    try:
        all_strikes = levels['all_strikes']
        
        # For major ETFs, show tight ±$10 range dynamically centered on current price
        # For other symbols, show wider range
        if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            range_buffer = 10  # ±$10 for ETFs
        elif underlying_price < 100:
            range_buffer = underlying_price * 0.15  # ±15% for low-priced stocks
        elif underlying_price < 500:
            range_buffer = underlying_price * 0.10  # ±10% for mid-priced stocks
        else:
            range_buffer = underlying_price * 0.08  # ±8% for high-priced stocks
        
        min_strike = underlying_price - range_buffer
        max_strike = underlying_price + range_buffer
        
        # Filter strikes to the dynamic range
        strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        if not strikes:
            # Fallback to closest strikes if range is too tight
            strikes = sorted(all_strikes, key=lambda x: abs(x - underlying_price))[:20]
        
        strikes = sorted(strikes)
        
        net_vols = [levels['net_volumes'].get(s, 0) for s in strikes]
        call_vols = [levels['call_volumes'].get(s, 0) for s in strikes]
        put_vols = [levels['put_volumes'].get(s, 0) for s in strikes]
        
        fig = go.Figure()
        
        # Net volume bars (horizontal) with better visibility
        colors = ['#ef4444' if v > 0 else '#22c55e' for v in net_vols]  # Brighter colors
        fig.add_trace(go.Bar(
            y=strikes,
            x=net_vols,
            orientation='h',
            name='Net Volume (Put - Call)',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=0.5)
            ),
            text=[f"{abs(v):,.0f}" if abs(v) > 1000 else "" for v in net_vols],
            textposition='outside',
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>Strike: $%{y:.2f}</b><br>Net Volume: %{x:,.0f}<br>(Positive = Put-heavy, Negative = Call-heavy)<extra></extra>'
        ))
        
        # Mark key levels
        annotations = []
        
        if levels['net_call_wall']['strike']:
            annotations.append(dict(
                y=levels['net_call_wall']['strike'],
                x=levels['net_call_wall']['volume'],
                text="💚 Net Call Wall",
                showarrow=True,
                arrowhead=2,
                arrowcolor="darkgreen"
            ))
        
        if levels['net_put_wall']['strike']:
            annotations.append(dict(
                y=levels['net_put_wall']['strike'],
                x=levels['net_put_wall']['volume'],
                text="❤️ Net Put Wall",
                showarrow=True,
                arrowhead=2,
                arrowcolor="darkred"
            ))
        
        if levels['flip_level']:
            fig.add_hline(
                y=levels['flip_level'],
                line_dash="dash",
                line_color="purple",
                annotation_text="🔄 Flip",
                annotation_position="top left"
            )
        
        # Determine appropriate tick interval based on symbol and price
        # For ETFs showing tight ±$10 range, use $1 ticks to show every dollar
        if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            tick_interval = 1  # Show every $1 for major ETFs
        elif underlying_price < 100:
            tick_interval = 2  # $2 ticks for low-priced stocks
        elif underlying_price < 500:
            tick_interval = 5  # $5 ticks for mid-priced stocks
        else:
            tick_interval = 10  # $10 ticks for high-priced stocks
        
        # Generate tick values within the filtered strike range
        min_strike_val = min(strikes)
        max_strike_val = max(strikes)
        tick_start = int(min_strike_val / tick_interval) * tick_interval
        tick_values = list(range(tick_start, int(max_strike_val) + tick_interval, tick_interval))
        
        fig.update_layout(
            title=dict(
                text="Net Option Volume Profile by Strike",
                font=dict(size=16, color='black')
            ),
            xaxis_title="Net Volume (Put - Call)",
            yaxis_title="Strike Price ($)",
            height=900,  # Increased from 700 for better visibility
            template='plotly_white',
            annotations=annotations,
            showlegend=False,
            xaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=tick_values,
                ticktext=[f"${x:.0f}" for x in tick_values],
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                tickfont=dict(size=11)
            ),
            font=dict(size=12),
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating volume profile: {str(e)}")
        return None

# Main analysis
# Use session state to track if we should run analysis
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if analyze_button:
    st.session_state.run_analysis = True

if st.session_state.run_analysis:
    with st.spinner(f"🔄 Analyzing option volumes for {symbol}..."):
        try:
            # ===== FETCH CACHED MARKET SNAPSHOT =====
            # This uses @st.cache_data with 60-second TTL
            # Multiple users watching same symbol share this cached data
            exp_date_str = expiry_date.strftime('%Y-%m-%d')
            
            snapshot = get_market_snapshot(symbol, exp_date_str)
            
            if not snapshot:
                st.error("Failed to fetch market data")
                st.stop()
            
            # Extract data from snapshot
            underlying_price = snapshot['underlying_price']
            options = snapshot['options_chain']
            price_history = snapshot['price_history']
            
            # Display cache status with timestamp
            cache_age = (datetime.now() - snapshot['fetched_at']).total_seconds()
            cache_status_color = "🟢" if cache_age < 30 else "🟡" if cache_age < 60 else "🔴"
            
            # Calculate Most Valuable Strike
            mvs = calculate_most_valuable_strike(options, underlying_price)
            
            col_price, col_mvs, col_cache = st.columns([2, 2, 1])
            with col_price:
                st.info(f"💰 Current Price: **${underlying_price:.2f}**")
            with col_mvs:
                if mvs:
                    # Display Most Valuable Strike in bold purple
                    mvs_display = f"🎯 Most Valuable Strike: **<span style='color: #9c27b0; font-weight: 900; font-size: 1.1em;'>${mvs['strike']:.2f}</span>**"
                    distance_direction = "above" if mvs['strike'] > underlying_price else "below"
                    st.markdown(mvs_display, unsafe_allow_html=True)
                    st.caption(f"📍 {abs(mvs['distance_pct']):.1f}% {distance_direction} | Vol: {int(mvs['total_volume']):,}")
                else:
                    st.info("🎯 Most Valuable Strike: Calculating...")
            with col_cache:
                st.caption(f"{cache_status_color} Data age: {cache_age:.0f}s | Cached: {snapshot['fetched_at'].strftime('%I:%M:%S %p')}")
            
            # ===== NEW: EXPECTED MOVE & VOLUME FLOW ANALYSIS =====
            st.markdown("---")
            
            # Calculate days to expiry
            today = datetime.now().date()
            days_to_expiry = (expiry_date - today).days
            
            # Calculate expected move and volume flow
            expected_move = calculate_expected_move(options, underlying_price, days_to_expiry)
            volume_flow = analyze_volume_flow(options, price_history)
            
            # Display in 2-column layout
            col_expected, col_flow = st.columns(2)
            
            with col_expected:
                st.markdown("### 📈 Expected Move (GEX-Adjusted)")
                
                if expected_move:
                    # Main expected move display
                    em_up = expected_move['expected_move_up']
                    em_down = expected_move['expected_move_down']
                    em_pct = expected_move['expected_move_pct']
                    
                    # Color based on regime
                    regime_color = "#4caf50" if expected_move['regime'] == 'Dampening' else "#f44336"
                    regime_icon = "🛡️" if expected_move['regime'] == 'Dampening' else "⚡"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {regime_color}22, {regime_color}11);
                        border-left: 4px solid {regime_color};
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                            {regime_icon} <strong>{expected_move['regime']} Regime</strong> • {days_to_expiry} DTE
                        </div>
                        <div style="font-size: 1.3em; font-weight: 700; color: {regime_color}; margin: 10px 0;">
                            ±${em_pct:.2f}%
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.95em; margin-top: 10px;">
                            <div>
                                <span style="color: #22c55e;">▲ Upside:</span> 
                                <strong>${em_up:.2f}</strong>
                            </div>
                            <div>
                                <span style="color: #ef4444;">▼ Downside:</span> 
                                <strong>${em_down:.2f}</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    col_iv, col_gex = st.columns(2)
                    with col_iv:
                        st.metric("ATM IV", f"{expected_move['atm_iv']:.1f}%", 
                                 help="At-the-money implied volatility")
                    with col_gex:
                        gex_display = f"{expected_move['gex_adjustment']:.2f}x"
                        st.metric("GEX Impact", gex_display,
                                 delta=f"{(expected_move['gex_adjustment']-1)*100:+.0f}%",
                                 help="How GEX positioning affects expected move")
                else:
                    st.info("📊 Calculating expected move...")
            
            with col_flow:
                st.markdown("### 🌊 Volume Flow Analysis")
                
                if volume_flow:
                    # Volume sentiment indicator
                    sentiment = volume_flow['sentiment']
                    sentiment_color = "#22c55e" if sentiment == 'Bullish' else "#ef4444" if sentiment == 'Bearish' else "#fbbf24"
                    sentiment_icon = "🐂" if sentiment == 'Bullish' else "🐻" if sentiment == 'Bearish' else "⚖️"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {sentiment_color}22, {sentiment_color}11);
                        border-left: 4px solid {sentiment_color};
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                            Today's Flow Sentiment
                        </div>
                        <div style="font-size: 1.3em; font-weight: 700; color: {sentiment_color}; margin: 10px 0;">
                            {sentiment_icon} {sentiment}
                        </div>
                        <div style="font-size: 0.85em; color: #666;">
                            P/C Ratio: <strong>{volume_flow['pc_ratio']:.2f}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Volume breakdown with progress bars
                    st.markdown("**Volume Distribution:**")
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin-bottom: 3px;">
                            <span>📞 Calls</span>
                            <span><strong>{volume_flow['call_volume']:,}</strong> ({volume_flow['call_volume_pct']:.1f}%)</span>
                        </div>
                        <div style="background: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #22c55e, #16a34a); width: {volume_flow['call_volume_pct']:.1f}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin-bottom: 3px;">
                            <span>📉 Puts</span>
                            <span><strong>{volume_flow['put_volume']:,}</strong> ({volume_flow['put_volume_pct']:.1f}%)</span>
                        </div>
                        <div style="background: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #ef4444, #dc2626); width: {volume_flow['put_volume_pct']:.1f}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Premium flow
                    st.markdown("**Premium Flow:**")
                    col_c, col_p = st.columns(2)
                    with col_c:
                        st.metric("Call $", f"${volume_flow['call_premium']/1e6:.1f}M",
                                 delta=f"{volume_flow['call_premium_pct']:.0f}%")
                    with col_p:
                        st.metric("Put $", f"${volume_flow['put_premium']/1e6:.1f}M",
                                 delta=f"{volume_flow['put_premium_pct']:.0f}%")
                else:
                    st.info("📊 Analyzing volume flow...")
            
            # ===== APPLY USER-SPECIFIC FILTERS =====
            # Calculate levels using user's custom filters
            # This runs on every refresh but uses cached raw data
            levels = calculate_option_walls(
                options, 
                underlying_price, 
                strike_spacing,  # User's filter
                num_strikes      # User's filter
            )
            
            if not levels:
                st.error("Failed to calculate levels")
                st.stop()
            
            # ===== TRADER DASHBOARD - 4 CORNER LAYOUT =====
            st.markdown("## 🎯 Trading Command Center")
            
            # Quick Bias Indicator Banner
            net_vol_preview = levels['totals']['net_vol']
            if net_vol_preview > 10000:
                bias_color = "#f44336"  # Red for strong bearish
                bias_text = "🐻 STRONG BEARISH BIAS"
                bias_emoji = "📉"
            elif net_vol_preview > 0:
                bias_color = "#ff9800"  # Orange for mild bearish
                bias_text = "🐻 MILD BEARISH BIAS"
                bias_emoji = "📊"
            elif net_vol_preview < -10000:
                bias_color = "#4caf50"  # Green for strong bullish
                bias_text = "🐂 STRONG BULLISH BIAS"
                bias_emoji = "📈"
            else:
                bias_color = "#2196f3"  # Blue for mild bullish
                bias_text = "🐂 MILD BULLISH BIAS"
                bias_emoji = "📊"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, {bias_color} 0%, {bias_color}cc 100%);
                color: white;
                padding: 15px 25px;
                border-radius: 8px;
                text-align: center;
                font-size: 20px;
                font-weight: 800;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                letter-spacing: 2px;
            ">
                {bias_emoji} MARKET BIAS: {bias_text} {bias_emoji}
            </div>
            """, unsafe_allow_html=True)
            
            # CSS for the command center boxes
            dashboard_style = """
            <style>
            .corner-box {
                border: 3px solid;
                border-radius: 12px;
                padding: 20px;
                color: white;
                height: 150px;
                box-shadow: 0 6px 12px rgba(0,0,0,0.2);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.2s;
            }
            .corner-box:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            }
            .corner-box-bearish {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border-color: #f5576c;
            }
            .corner-box-bullish {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                border-color: #00f2fe;
            }
            .corner-box-resistance {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                border-color: #fa709a;
            }
            .corner-box-support {
                background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
                border-color: #30cfd0;
            }
            .corner-box-flip {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                border-color: #a8edea;
                color: #333 !important;
            }
            .corner-title {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
                opacity: 0.95;
                text-transform: uppercase;
            }
            .corner-value {
                font-size: 36px;
                font-weight: 900;
                margin: 5px 0;
                text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
                line-height: 1;
            }
            .corner-subtitle {
                font-size: 11px;
                opacity: 0.85;
                margin-top: 3px;
                font-weight: 500;
            }
            .corner-delta {
                font-size: 14px;
                font-weight: 700;
                margin-top: 5px;
                opacity: 0.95;
            }
            </style>
            """
            st.markdown(dashboard_style, unsafe_allow_html=True)
            
            # Top row - 4 corner boxes
            corner_col1, corner_col2, corner_col3, corner_col4 = st.columns(4)
            
            with corner_col1:
                # Current Price & Sentiment
                net_vol = levels['totals']['net_vol']
                sentiment = "🐻 BEARISH" if net_vol > 0 else "🐂 BULLISH"
                sentiment_pct = abs(net_vol) / max(levels['totals']['call_vol'], levels['totals']['put_vol'], 1) * 100
                box_class = 'corner-box-bearish' if net_vol > 0 else 'corner-box-bullish'
                
                st.markdown(f"""
                <div class="corner-box {box_class}">
                    <div class="corner-title">💰 LIVE PRICE</div>
                    <div class="corner-value">${underlying_price:.2f}</div>
                    <div class="corner-subtitle">{sentiment}</div>
                    <div class="corner-delta">Flow Bias: {sentiment_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with corner_col2:
                # Resistance (Call Wall)
                if levels['call_wall']['strike']:
                    distance_pct = ((levels['call_wall']['strike'] - underlying_price) / underlying_price) * 100
                    resistance_strength = min(levels['call_wall']['gex']/1e6 / 5, 1.0) * 100
                    
                    st.markdown(f"""
                    <div class="corner-box corner-box-resistance">
                        <div class="corner-title">🔴 RESISTANCE</div>
                        <div class="corner-value">${levels['call_wall']['strike']:.2f}</div>
                        <div class="corner-subtitle">Call Wall</div>
                        <div class="corner-delta">{abs(distance_pct):.2f}% away • {resistance_strength:.0f}% str</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="corner-box corner-box-resistance">
                        <div class="corner-title">🔴 RESISTANCE</div>
                        <div class="corner-value">-</div>
                        <div class="corner-subtitle">No clear wall detected</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with corner_col3:
                # Support (Put Wall)
                if levels['put_wall']['strike']:
                    distance_pct = ((levels['put_wall']['strike'] - underlying_price) / underlying_price) * 100
                    support_strength = min(levels['put_wall']['gex']/1e6 / 5, 1.0) * 100
                    
                    st.markdown(f"""
                    <div class="corner-box corner-box-support">
                        <div class="corner-title">🟢 SUPPORT</div>
                        <div class="corner-value">${levels['put_wall']['strike']:.2f}</div>
                        <div class="corner-subtitle">Put Wall</div>
                        <div class="corner-delta">{abs(distance_pct):.2f}% away • {support_strength:.0f}% str</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="corner-box corner-box-support">
                        <div class="corner-title">🟢 SUPPORT</div>
                        <div class="corner-value">-</div>
                        <div class="corner-subtitle">No clear wall detected</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with corner_col4:
                # Flip Level
                if levels['flip_level']:
                    flip_distance = ((levels['flip_level'] - underlying_price) / underlying_price) * 100
                    flip_status = "ABOVE ⬆️" if underlying_price > levels['flip_level'] else "BELOW ⬇️"
                    
                    st.markdown(f"""
                    <div class="corner-box corner-box-flip">
                        <div class="corner-title">🔄 FLIP LEVEL</div>
                        <div class="corner-value">${levels['flip_level']:.2f}</div>
                        <div class="corner-subtitle">Sentiment Pivot</div>
                        <div class="corner-delta">{flip_status} ({abs(flip_distance):.2f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="corner-box corner-box-flip">
                        <div class="corner-title">🔄 FLIP LEVEL</div>
                        <div class="corner-value">-</div>
                        <div class="corner-subtitle">No clear flip detected</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ===== TRADEABLE ALERTS - COMPACT VIEW =====
            alerts = generate_tradeable_alerts(levels, underlying_price, symbol)
            
            if alerts:
                # Sort by priority (HIGH first)
                priority_order = {'🔴 HIGH': 0, '🟡 MEDIUM': 1, '🟢 LOW': 2}
                alerts_sorted = sorted(alerts, key=lambda x: priority_order.get(x['priority'], 3))
                
                with st.expander("🚨 Live Trade Alerts", expanded=False):
                    # Show only top 3 alerts for speed
                    for alert in alerts_sorted[:3]:
                        if '🔴' in alert['priority']:
                            st.error(f"**{alert['type']}**: {alert['message']} → {alert['action']}", icon="🔴")
                        elif '🟡' in alert['priority']:
                            st.warning(f"**{alert['type']}**: {alert['message']} → {alert['action']}", icon="🟡")
                        else:
                            st.info(f"**{alert['type']}**: {alert['message']} → {alert['action']}", icon="🟢")
            
            st.markdown("---")
            
            # Create full-width intraday chart
            st.markdown("### 📊 Intraday + Walls")
            
            # Show date range info
            if 'candles' in price_history and price_history['candles']:
                df_temp = pd.DataFrame(price_history['candles'])
                df_temp['datetime'] = pd.to_datetime(df_temp['datetime'], unit='ms', utc=True).dt.tz_convert('America/New_York')
                df_temp['date'] = df_temp['datetime'].dt.date
                # Filter to market hours
                df_temp = df_temp[
                    ((df_temp['datetime'].dt.hour == 9) & (df_temp['datetime'].dt.minute >= 30)) |
                    ((df_temp['datetime'].dt.hour >= 10) & (df_temp['datetime'].dt.hour < 16))
                ]
                unique_dates = sorted(df_temp['date'].unique(), reverse=True)
                if len(unique_dates) >= 2:
                    date_info = f"Showing last 2 trading days: {unique_dates[1].strftime('%b %d')} & {unique_dates[0].strftime('%b %d, %Y')}"
                elif len(unique_dates) == 1:
                    date_info = f"Showing {unique_dates[0].strftime('%b %d, %Y')}"
                else:
                    date_info = "Price action with VWAP, key support/resistance levels"
                st.caption(f"📅 **{date_info}**")
            else:
                st.caption("**Price action with VWAP, key support/resistance levels**")
            
            # Get GEX data for the selected expiry
            strike_gex = get_gex_by_strike(options, underlying_price, expiry_date)
            
            # Create the main intraday chart with Most Valuable Strike and Expected Move
            mvs_strike = mvs['strike'] if mvs else None
            chart = create_intraday_chart_with_levels(price_history, levels, underlying_price, symbol, mvs_strike, expected_move)
            
            # Controls row - GEX sidebar checkbox and Refresh button side by side
            col_gex, col_refresh = st.columns([3, 1])
            with col_gex:
                show_gex_sidebar = st.checkbox("📊 Show Net GEX Sidebar", value=True, help="Display gamma exposure levels next to the chart")
            with col_refresh:
                if st.button("🔄 Refresh", key="refresh_charts_btn", type="secondary", use_container_width=True):
                    with st.spinner("🔄 Refreshing data..."):
                        # Clear cache and force fresh data fetch
                        st.cache_data.clear()
                        st.success("✅ Cache cleared! Data will refresh automatically.")
                        st.rerun()
            
            # Display chart with or without GEX sidebar
            if not chart:
                st.error("❌ Unable to create price chart. Please check if market data is available.")
            elif show_gex_sidebar and strike_gex is not None and len(strike_gex) > 0:
                # Create figure with subplots - GEX bar on left, main chart on right
                from plotly.subplots import make_subplots
                
                # Extract the main chart data
                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.15, 0.85],  # GEX takes 15%, chart takes 85%
                    horizontal_spacing=0.02,
                    shared_yaxes=True,
                    subplot_titles=("Net GEX ($)", f"{symbol} - Intraday + Walls")
                )
                
                # Format GEX values for display
                max_gex = strike_gex['net_gex'].abs().max()
                formatted_gex = []
                for val in strike_gex['net_gex']:
                    if abs(val) >= 1e6:
                        formatted_gex.append(f"${val/1e6:.1f}M")
                    elif abs(val) >= 1e3:
                        formatted_gex.append(f"${val/1e3:.0f}K")
                    else:
                        formatted_gex.append(f"${val:.0f}")
                
                # Add GEX bar chart on the left
                colors = ['#22c55e' if x > 0 else '#ef4444' for x in strike_gex['net_gex']]
                fig.add_trace(
                    go.Bar(
                        y=strike_gex['strike'],
                        x=strike_gex['net_gex'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.8),
                        text=formatted_gex,
                        textposition='outside',
                        textfont=dict(size=9, color='black'),
                        hovertemplate='<b>Strike: $%{y:.2f}</b><br>Net GEX: $%{x:,.0f}<extra></extra>',
                        showlegend=False,
                        name='Net GEX',
                        width=0.8
                    ),
                    row=1, col=1
                )
                
                # Add all traces from the original chart to the second subplot
                for trace in chart.data:
                    fig.add_trace(trace, row=1, col=2)
                
                # Extract x-axis range from the original chart (it has the correct filtered time range)
                original_xaxis_range = chart.layout.xaxis.range if chart.layout.xaxis.range else None
                
                # Calculate Y-axis range (same as in original chart)
                all_prices = [underlying_price]
                if levels['call_wall']['strike']:
                    all_prices.append(levels['call_wall']['strike'])
                if levels['put_wall']['strike']:
                    all_prices.append(levels['put_wall']['strike'])
                if levels['flip_level']:
                    all_prices.append(levels['flip_level'])
                
                # Also include price history range from filtered data
                # Use the actual chart traces to get the price range, not raw candles
                for trace in chart.data:
                    if hasattr(trace, 'high') and trace.high is not None:
                        all_prices.extend([h for h in trace.high if h is not None])
                    if hasattr(trace, 'low') and trace.low is not None:
                        all_prices.extend([l for l in trace.low if l is not None])
                    if hasattr(trace, 'y') and trace.y is not None:
                        all_prices.extend([y for y in trace.y if y is not None and not pd.isna(y)])
                
                price_range = max(all_prices) - min(all_prices)
                y_padding = price_range * 0.15
                y_min = min(all_prices) - y_padding
                y_max = max(all_prices) + y_padding
                
                # Update x-axes
                fig.update_xaxes(
                    title_text="",
                    row=1, col=1,
                    tickformat='$,.0s',
                    side='top',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)'
                )
                fig.update_xaxes(
                    title_text="Time (ET)",
                    row=1, col=2,
                    type='date',
                    tickformat='%I:%M %p\n%b %d',
                    dtick=3600000,
                    tickangle=0,
                    range=original_xaxis_range,  # Use the range from the original chart
                    rangebreaks=[dict(bounds=[16, 9.5], pattern="hour")],
                    gridcolor='rgba(0,0,0,0.05)'
                )
                
                # Update y-axes
                fig.update_yaxes(
                    title_text="Strike Price ($)",
                    row=1, col=1,
                    range=[y_min, y_max],
                    tickformat='$.0f',
                    gridcolor='rgba(0,0,0,0.08)',
                    zeroline=False,
                    showticklabels=True
                )
                fig.update_yaxes(
                    title_text="",
                    row=1, col=2,
                    range=[y_min, y_max],
                    tickformat='$.2f',
                    gridcolor='rgba(0,0,0,0.08)',
                    zeroline=False,
                    showticklabels=False  # Hide duplicate y-axis labels
                )
                
                fig.update_layout(
                    height=600,
                    template='plotly_white',
                    hovermode='closest',
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    xaxis2_rangeslider_visible=False,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,
                        xanchor="center",
                        x=0.55,
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1,
                        font=dict(size=10)
                    ),
                    margin=dict(t=100, r=150, l=100, b=80),
                    plot_bgcolor='rgba(250, 250, 250, 0.5)',
                    annotations=list(chart.layout.annotations) if chart.layout.annotations else []
                )
                
                st.plotly_chart(fig, use_container_width=True, key="combined_chart")
            
            elif show_gex_sidebar and (strike_gex is None or len(strike_gex) == 0):
                # GEX sidebar requested but no GEX data available
                st.warning("⚠️ GEX data not available for this expiration. Showing chart without GEX sidebar.")
                st.plotly_chart(chart, use_container_width=True, key="intraday_chart")
            
            else:
                # Show just the intraday chart (GEX sidebar disabled by user)
                st.plotly_chart(chart, use_container_width=True, key="intraday_chart")
            
            # Add MACD indicator below the main chart
            st.markdown("---")
            st.markdown("#### 📈 MACD Indicator")
            macd_chart = create_macd_chart(price_history, symbol)
            if macd_chart:
                st.plotly_chart(macd_chart, use_container_width=True, key="macd_chart")
            else:
                st.info("MACD chart not available")
            
            # Show GEX Heatmap below if enabled (collapsed by default)
            if show_heatmap:
                st.markdown("---")
                with st.expander("🔥 GEX Heatmap - Dealer gamma positioning across strikes and expirations", expanded=False):
                    heatmap = create_gamma_heatmap(options, underlying_price, num_expiries=6)
                    if heatmap:
                        heatmap.update_layout(height=400)
                        st.plotly_chart(heatmap, use_container_width=True, key="gex_heatmap")
                    else:
                        st.info("Gamma heatmap not available - insufficient options data")
            
            # Bottom row: Volume Profile + Net Premium Flow
            st.markdown("---")
            st.markdown("### 📊 Additional Analysis")
            chart_row2_col1, chart_row2_col2 = st.columns(2)
            
            with chart_row2_col1:
                st.markdown("#### 📏 Volume Profile")
                st.caption(f"Net volume by strike • Showing ±$10 range around ${underlying_price:.2f}" if symbol in ['SPY', 'QQQ', 'IWM', 'DIA'] else "Net volume by strike (Put - Call)")
                profile_chart = create_volume_profile_chart(levels, underlying_price, symbol)
                if profile_chart:
                    # Don't override the height set in the function (900px for better visibility)
                    st.plotly_chart(profile_chart, use_container_width=True, key="volume_profile")
            
            with chart_row2_col2:
                st.markdown("#### 💰 Net Premium Flow")
                st.caption("Call premium minus put premium across strikes and expirations")
                
                # Use cached multi-expiry snapshot for date range
                # Fetch options for next 30 days to get multiple expiries
                from_date = expiry_date.strftime('%Y-%m-%d')
                to_date = (expiry_date + timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Get multi-expiry snapshot (cached separately from single-expiry)
                multi_snapshot = get_multi_expiry_snapshot(symbol, from_date, to_date)
                
                if multi_snapshot and multi_snapshot['options_chain']:
                    multi_expiry_options = multi_snapshot['options_chain']
                    premium_map = create_net_premium_heatmap(multi_expiry_options, underlying_price, num_expiries=4)
                    if premium_map:
                        st.plotly_chart(premium_map, use_container_width=True, key="premium_heatmap")
                    else:
                        st.info("Net premium heatmap not available - insufficient options data")
                else:
                    st.info("Could not fetch multi-expiry options data for heatmap")
            
            # Expandable detailed explanations below the charts
            st.markdown("---")
            with st.expander("📖 Chart Interpretation Guide", expanded=False):
                st.markdown("""
                ### 📈 Intraday + Walls
                - **Green/Red Candlesticks**: Price movement (green=up, red=down)
                - **Cyan Line**: VWAP from yesterday's open
                - **Purple Line**: VWAP from today's open
                - **Orange Line**: 21 EMA
                - **Horizontal Lines**: Call wall (resistance), Put wall (support), Flip level (pivot)
                
                ### � Net Premium Flow
                - **Green cells**: Call premium > Put premium = Bullish positioning
                - **Red cells**: Put premium > Call premium = Bearish positioning
                - **White/light cells**: Balanced call/put premiums = Neutral
                - **Yellow line**: Current price level
                - **Darker colors**: Larger premium imbalances (stronger directional bias)
                
                ### 📏 Volume Profile
                - **Red Bars** (right): Put-heavy = bearish pressure
                - **Green Bars** (left): Call-heavy = bullish pressure
                
                ### 🔥 GEX Heatmap
                - **Blue**: Dealers long gamma → Resistance/Support
                - **Red**: Dealers short gamma → Acceleration Zone
                - **White**: Neutral positioning
                
                ### 🎯 Trading Implications
                1. **Near Put Wall + Blue GEX above** → Support likely holds, resistance forms
                2. **Near Call Wall + Blue GEX below** → Resistance likely holds
                3. **Green Premium Flow + Price near strike** → Bullish bias, call buying
                4. **Red Premium Flow + Price near strike** → Bearish bias, put buying
                5. **Flip Level Cross** → Sentiment shift, momentum trade opportunity
                6. **Stacked Blue GEX zones** → Strong pin risk at expiration
                """)
            
            # Trade Setups - HIDDEN FOR NOW
            # st.markdown("---")
            # st.markdown("## 🎯 Trade Setup Recommendations")
            # 
            # trade_setups = generate_trade_setups(levels, underlying_price, symbol)
            # 
            # if trade_setups:
            #     for idx, setup in enumerate(trade_setups):
            #         with st.expander(f"{setup['type']} - {setup['bias']} (Confidence: {setup['confidence']})", expanded=True):
            #             st.markdown(f"**Reasoning:** {setup['reasoning']}")
            #             
            #             setup_col1, setup_col2, setup_col3, setup_col4 = st.columns(4)
            #             
            #             with setup_col1:
            #                 st.metric("Entry", f"${setup['entry']:.2f}")
            #             
            #             with setup_col2:
            #                 if setup['stop']:
            #                     risk = abs(setup['entry'] - setup['stop'])
            #                     st.metric("Stop Loss", f"${setup['stop']:.2f}", delta=f"-${risk:.2f}")
            #                 else:
            #                     st.metric("Stop Loss", "Use time-based")
            #             
            #             with setup_col3:
            #                 if setup['target1']:
            #                     reward1 = abs(setup['target1'] - setup['entry'])
            #                     st.metric("Target 1", f"${setup['target1']:.2f}", delta=f"+${reward1:.2f}")
            #             
            #             with setup_col4:
            #                 if setup['target2']:
            #                     reward2 = abs(setup['target2'] - setup['entry'])
            #                     st.metric("Target 2", f"${setup['target2']:.2f}", delta=f"+${reward2:.2f}")
            #             
            #             # Risk/Reward
            #             if setup['stop'] and setup['target1']:
            #                 risk = abs(setup['entry'] - setup['stop'])
            #                 reward = abs(setup['target1'] - setup['entry'])
            #                 rr_ratio = reward / risk if risk > 0 else 0
            #                 
            #                 st.progress(min(rr_ratio / 3, 1.0))
            #                 st.caption(f"Risk/Reward: 1:{rr_ratio:.1f}")
            # else:
            #     st.info("No clear trade setups at current price levels. Wait for price to approach key walls.")
            
            # Multi-Expiry Comparison
            if multi_expiry:
                st.markdown("---")
                st.markdown("## 📅 Multi-Expiry Wall Comparison")
                
                # Get next 3 weekly expirations
                expiry_dates = []
                current_date = datetime.now()
                for weeks_ahead in range(1, 4):
                    next_date = current_date + timedelta(weeks=weeks_ahead)
                    # Find next Friday
                    days_until_friday = (4 - next_date.weekday()) % 7
                    friday = next_date + timedelta(days=days_until_friday)
                    expiry_dates.append(friday.strftime('%Y-%m-%d'))
                
                multi_expiry_data = []
                
                for exp_date in expiry_dates:
                    try:
                        # Use cached snapshot for each expiry date
                        exp_snapshot = get_market_snapshot(symbol, exp_date)
                        
                        if exp_snapshot and exp_snapshot['options_chain']:
                            exp_options = exp_snapshot['options_chain']
                            exp_levels = calculate_option_walls(exp_options, underlying_price, strike_spacing, num_strikes)
                            if exp_levels:
                                multi_expiry_data.append({
                                    'expiry': exp_date,
                                    'call_wall': exp_levels['call_wall']['strike'],
                                    'put_wall': exp_levels['put_wall']['strike'],
                                    'flip_level': exp_levels['flip_level']
                                })
                    except:
                        continue
                
                if multi_expiry_data:
                    df_multi = pd.DataFrame(multi_expiry_data)
                    
                    # Display as table
                    st.dataframe(
                        df_multi.style.format({
                            'call_wall': '${:.2f}',
                            'put_wall': '${:.2f}',
                            'flip_level': '${:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Identify stacked walls (same strikes across expirations)
                    st.markdown("### 🔥 Stacked Walls (High Confidence Levels)")
                    
                    # Check for call walls within 1% of each other
                    call_walls = [d['call_wall'] for d in multi_expiry_data if d['call_wall']]
                    put_walls = [d['put_wall'] for d in multi_expiry_data if d['put_wall']]
                    
                    stacked_calls = []
                    stacked_puts = []
                    
                    for i, cw1 in enumerate(call_walls):
                        matches = sum(1 for cw2 in call_walls if abs(cw1 - cw2) / cw1 < 0.01)
                        if matches >= 2 and cw1 not in stacked_calls:
                            stacked_calls.append(cw1)
                    
                    for i, pw1 in enumerate(put_walls):
                        matches = sum(1 for pw2 in put_walls if abs(pw1 - pw2) / pw1 < 0.01)
                        if matches >= 2 and pw1 not in stacked_puts:
                            stacked_puts.append(pw1)
                    
                    if stacked_calls or stacked_puts:
                        col_stack1, col_stack2 = st.columns(2)
                        
                        with col_stack1:
                            if stacked_calls:
                                st.success(f"📈 **Stacked Call Walls:** {', '.join([f'${x:.2f}' for x in stacked_calls])}")
                                st.caption("Strong resistance - multiple expirations aligned")
                        
                        with col_stack2:
                            if stacked_puts:
                                st.success(f"📉 **Stacked Put Walls:** {', '.join([f'${x:.2f}' for x in stacked_puts])}")
                                st.caption("Strong support - multiple expirations aligned")
                    else:
                        st.info("No stacked walls found - levels vary across expirations")
            
            # Interpretation
            st.markdown("---")
            
            with st.expander("💡 How to Read This", expanded=False):
                st.markdown("""
                ### 🧱 Volume Walls (Simple)
                - **Call Wall** 📈: Strike with highest call volume = **Resistance** (price ceiling)
                - **Put Wall** 📉: Strike with highest put volume = **Support** (price floor)
                
                ### 💎 Net Walls (Advanced)
                - **Net Call Wall** 💚: Strike with highest *net call dominance* = **Strong upside magnet**
                - **Net Put Wall** ❤️: Strike with highest *net put dominance* = **Strong downside magnet**
                
                ### 🔄 Flip Level (Critical)
                - Where net volume flips from **bearish → bullish** (or vice versa)
                - Above flip = bullish territory | Below flip = bearish territory
                - Breaking flip level often triggers momentum
                
                ### 📊 Trading Strategy
                1. **Breakout Play**: Price breaking through Call Wall with volume = bullish breakout
                2. **Bounce Play**: Price bouncing off Put Wall = support holding
                3. **Flip Trade**: Crossing flip level = sentiment change, momentum shift
                4. **Pin Risk**: Price gravitates toward max pain (highest volume strikes)
                
                ### ⚠️ Important Notes
                - These levels are **dynamic** - recalculate as volume changes
                - Most effective on **expiration day** when gamma is highest
                - Use with **price action** confirmation, not in isolation
                """)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())

else:
    with st.expander("🧱 What Are Option Volume Walls?", expanded=False):
        st.markdown("""
        Option volume walls are **key price levels** where massive option activity creates support or resistance.
        
        **Think of it like this:**
        - Market makers are **short options** to customers
        - They must **hedge** by buying/selling the underlying
        - **High volume strikes** = lots of hedging activity
        - This creates **price magnets** or **barriers**
        
        ### 📊 The Five Key Levels
        
        1. **Call Wall** 📈 - Highest call volume strike (typical resistance)
        2. **Put Wall** 📉 - Highest put volume strike (typical support)
        3. **Net Call Wall** 💚 - Where calls dominate most (strong bullish level)
        4. **Net Put Wall** ❤️ - Where puts dominate most (strong bearish level)
        5. **Flip Level** 🔄 - Where sentiment flips between bullish/bearish
        
        ### 🎯 Why This Matters
        
        - **Pin Risk**: Price often pins to high volume strikes at expiration
        - **Breakout Targets**: Breaking call wall = next leg up
        - **Support Levels**: Put walls act as price floors
        - **Sentiment Gauge**: Net volumes show true directional bias
        
        **Configure settings and click 'Calculate Levels' to start analyzing!**
        """)
