#!/usr/bin/env python3
"""
Streamlit Max Gamma Strike Scanner
Scans multiple stocks and displays top 3 highest gamma strikes with expiration dates
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Import yfinance at module level
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

@st.cache_resource
def get_schwab_client():
    """Create a singleton Schwab API client for the session"""
    return SchwabClient()

# Configure Streamlit page
st.set_page_config(
    page_title="Max Gamma Strike Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile-friendly styling
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .stDataFrame {
            font-size: 12px !important;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
    }
    
    /* Compact headers */
    .stock-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Compact gamma level cards */
    .gamma-level-card {
        background-color: #f8f9fa;
        border-left: 4px solid;
        padding: 10px;
        margin: 8px 0;
        border-radius: 6px;
        font-size: 0.9em;
    }
    
    .gamma-level-card.resistance {
        border-left-color: #28a745;
        background-color: #e8f5e9;
    }
    
    .gamma-level-card.support {
        border-left-color: #dc3545;
        background-color: #ffebee;
    }
    
    /* Compact strike cards */
    .strike-card {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .strike-card.call {
        border-left: 4px solid #28a745;
    }
    
    .strike-card.put {
        border-left: 4px solid #dc3545;
    }
    
    /* Better table styling */
    .dataframe {
        font-size: 0.85em;
    }
    
    /* Compact metric displays */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 5px 0;
    }
    
    .metric-label {
        font-weight: 600;
        color: #495057;
    }
    
    .metric-value {
        font-weight: bold;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

def estimate_underlying_from_strikes(options_data):
    """Estimate underlying price from ATM options strikes"""
    try:
        if not options_data or 'callExpDateMap' not in options_data:
            return None
        
        # Get the first expiry with data
        exp_dates = list(options_data['callExpDateMap'].keys())
        if not exp_dates:
            return None
        
        first_exp = options_data['callExpDateMap'][exp_dates[0]]
        strikes = [float(s) for s in first_exp.keys()]
        
        if strikes:
            # Look for highest volume/open interest strikes (likely ATM)
            strike_data = []
            for strike_str, contracts in first_exp.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0) or 0
                    open_interest = contract.get('openInterest', 0) or 0
                    strike = float(strike_str)
                    activity = volume + open_interest * 0.1  # Weight OI less than volume
                    strike_data.append((strike, activity))
            
            if strike_data:
                # Sort by activity and get the most active strike
                strike_data.sort(key=lambda x: x[1], reverse=True)
                most_active_strike = strike_data[0][0]
                
                # If the most active strike seems reasonable, use it
                if 1 < most_active_strike < 10000:  # Reasonable stock price range
                    return most_active_strike
            
            # Fallback to middle strike
            strikes.sort()
            mid_index = len(strikes) // 2
            return strikes[mid_index]
        
        return None
    except Exception:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_options_data(symbol, num_expiries=5):
    """
    Fetch options data for multiple expiries.
    Returns: (options_data, underlying_price, error_message)
    """
    try:
        client = get_schwab_client()
        
        # Get quote data
        quote_data = client.get_quote(symbol)
        
        if not quote_data:
            return None, None, "Failed to get quote data. Check API connection."
        
        # Extract underlying price from quote
        underlying_price = None
        if symbol in quote_data:
            price_fields = ['lastPrice', 'mark', 'bidPrice', 'askPrice', 'closePrice']
            for field in price_fields:
                price = quote_data[symbol].get(field, 0)
                if price and price > 0:
                    underlying_price = price
                    break
        else:
            return None, None, f"Symbol {symbol} not found in quote response"
        
        # Get options chain
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not options_data:
            return None, None, "No options data available for this symbol."
        
        # Check if options data has the expected structure
        if 'callExpDateMap' not in options_data and 'putExpDateMap' not in options_data:
            return None, None, "Invalid options data structure."
        
        # Try to get underlying price from options data if not already set or invalid
        yf_info_msg = None
        if not underlying_price or underlying_price == 0 or underlying_price == 100.0:
            # Try options data fields
            if 'underlyingPrice' in options_data and options_data['underlyingPrice']:
                underlying_price = options_data['underlyingPrice']
            elif 'underlying' in options_data and options_data['underlying']:
                price_fields = ['last', 'mark', 'lastPrice', 'close']
                for field in price_fields:
                    price = options_data['underlying'].get(field, 0)
                    if price and price > 0:
                        underlying_price = price
                        break
            
            # If still no valid price, try yfinance
            if not underlying_price or underlying_price == 0 or underlying_price == 100.0:
                if YFINANCE_AVAILABLE:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
                        for field in price_fields:
                            price = info.get(field)
                            if price and price > 0:
                                underlying_price = price
                                yf_info_msg = f"Using yfinance for {symbol} price: ${underlying_price:.2f}"
                                break
                    except Exception:
                        pass
                
                # Last resort - estimate from strike prices
                if not underlying_price or underlying_price == 0:
                    estimated_price = estimate_underlying_from_strikes(options_data)
                    if estimated_price:
                        underlying_price = estimated_price
                        yf_info_msg = f"Estimated {symbol} price from strikes: ${underlying_price:.2f}"
        
        # Final validation
        if not underlying_price or underlying_price <= 0:
            return None, None, f"Could not determine valid price for {symbol}"
        
        return options_data, underlying_price, yf_info_msg
        
    except Exception as e:
        error_msg = f"Error fetching options data: {str(e)}"
        # Try yfinance as fallback
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                underlying_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                if underlying_price and underlying_price > 0:
                    client = get_schwab_client()
                    options_data = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
                    if options_data:
                        return options_data, underlying_price, f"API error - using yfinance for {symbol} price: ${underlying_price:.2f}"
            except Exception:
                pass
        return None, None, error_msg

def calculate_gamma_strikes(options_data, underlying_price, num_expiries=5, debug_mode=False):
    """
    Calculate gamma for all strikes and identify top gamma strikes.
    
    Simplified approach: Use absolute gamma values and distinguish calls/puts by type,
    not by gamma sign (which may be unreliable from API).
    """
    
    debug_info = {'total_contracts': 0, 'contracts_with_gamma': 0, 'sample_contract': None}
    
    if not options_data:
        return pd.DataFrame(), debug_info
    
    results = []
    
    # Process both calls and puts
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
            
        exp_dates = list(options_data[option_type].keys())[:num_expiries]
        is_call = 'call' in option_type.lower()
        
        for exp_date in exp_dates:
            strikes_data = options_data[option_type][exp_date]
            
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                
                try:
                    strike = float(strike_str)
                    contract = contracts[0]
                    
                    # Debug: Store first contract for inspection
                    debug_info['total_contracts'] += 1
                    if debug_info['sample_contract'] is None:
                        debug_info['sample_contract'] = contract
                    
                    # Extract data with safe defaults
                    gamma = abs(contract.get('gamma', 0) or 0)  # Use absolute value
                    delta = contract.get('delta', 0) or 0
                    vega = contract.get('vega', 0) or 0
                    volume = contract.get('totalVolume', 0) or 0
                    open_interest = contract.get('openInterest', 0) or 0
                    bid = contract.get('bid', 0) or 0
                    ask = contract.get('ask', 0) or 0
                    last = contract.get('last', 0) or 0
                    implied_volatility = (contract.get('volatility', 0) or 0) * 100
                    
                    # Skip if no meaningful data
                    if gamma == 0 and open_interest == 0:
                        continue
                    
                    # Count valid gamma contracts for debug
                    if gamma > 0:
                        debug_info['contracts_with_gamma'] += 1
                    
                    # Estimate gamma if not provided and we have vega
                    if gamma == 0 and vega > 0 and implied_volatility > 0 and underlying_price > 0:
                        gamma = abs(vega / (underlying_price * implied_volatility / 100))
                    
                    # Calculate gamma exposure metrics
                    # Use absolute values for ranking, track option type separately
                    notional_gamma = gamma * open_interest * 100 * underlying_price if underlying_price > 0 else gamma * open_interest * 100
                    gamma_exposure_shares = gamma * open_interest * 100
                    
                    # Calculate moneyness
                    moneyness = (strike / underlying_price - 1) * 100 if underlying_price > 0 else 0
                    
                    # Parse expiration date
                    exp_date_str = exp_date.split(':')[0] if ':' in exp_date else exp_date
                    
                    # Calculate days to expiration
                    try:
                        exp_datetime = datetime.strptime(exp_date_str, '%Y-%m-%d')
                        days_to_exp = max(0, (exp_datetime - datetime.now()).days)
                    except Exception:
                        days_to_exp = 0
                    
                    results.append({
                        'strike': strike,
                        'expiry': exp_date_str,
                        'days_to_exp': days_to_exp,
                        'option_type': 'Call' if is_call else 'Put',
                        'gamma': gamma,
                        'delta': delta,
                        'vega': vega,
                        'volume': volume,
                        'open_interest': open_interest,
                        'bid': bid,
                        'ask': ask,
                        'last': last,
                        'notional_gamma': notional_gamma,
                        'gamma_exposure_shares': gamma_exposure_shares,
                        'moneyness': moneyness,
                        'implied_volatility': implied_volatility
                    })
                    
                except Exception as e:
                    if debug_mode:
                        debug_info.setdefault('errors', []).append(str(e))
                    continue
    
    if not results:
        return pd.DataFrame(), debug_info
    
    df = pd.DataFrame(results)
    
    # Sort by notional gamma (absolute value)
    df_sorted = df.sort_values('notional_gamma', ascending=False)
    
    return df_sorted, debug_info

def format_large_number(num):
    """Format large numbers for display"""
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num/1e3:.1f}K"
    else:
        return f"${num:.0f}"

def create_gamma_table(df_gamma, underlying_price, num_expiries=5):
    """Create a clean table showing gamma exposure across strikes and expiries"""
    
    if df_gamma.empty:
        return pd.DataFrame()
    
    # Get unique expiries (limit to num_expiries)
    expiries = sorted(df_gamma['expiry'].unique())[:num_expiries]
    
    # Filter dataframe to only include selected expiries
    df_filtered = df_gamma[df_gamma['expiry'].isin(expiries)].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Create signed notional gamma for display (calls positive, puts negative)
    df_filtered['display_gamma'] = df_filtered.apply(
        lambda row: row['notional_gamma'] if row['option_type'] == 'Call' else -row['notional_gamma'],
        axis=1
    )
    
    # Use pandas pivot_table for efficient aggregation
    df_pivot = df_filtered.pivot_table(
        index='strike',
        columns='expiry',
        values='display_gamma',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Rename index column
    df_pivot.rename(columns={'strike': 'Strike'}, inplace=True)
    
    # Mark current price row (within 1% of underlying)
    df_pivot['is_current'] = df_pivot['Strike'].apply(
        lambda x: abs(x - underlying_price) < (underlying_price * 0.01)
    )
    
    return df_pivot

def display_gamma_strike_card(row, rank, underlying_price):
    """Display a compact, mobile-friendly card for a gamma strike"""
    
    card_class = "call" if row['option_type'] == 'Call' else "put"
    
    # Determine if ITM or OTM
    if row['option_type'] == 'Call':
        status = "ITM" if row['strike'] < underlying_price else "OTM"
        status_color = "#28a745" if status == "ITM" else "#6c757d"
    else:
        status = "ITM" if row['strike'] > underlying_price else "OTM"
        status_color = "#dc3545" if status == "ITM" else "#6c757d"
    
    gamma_color = "#28a745" if row['option_type'] == 'Call' else "#dc3545"
    
    html = f"""
    <div class="strike-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <strong style="font-size: 1.1em;">#{rank} ${row['strike']:.2f} {row['option_type']}</strong>
                <span style="color: {status_color}; font-size: 0.85em; margin-left: 8px;">{status}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2em; font-weight: bold; color: {gamma_color};">
                    {format_large_number(row['notional_gamma'])}
                </div>
            </div>
        </div>
        <div style="font-size: 0.85em; color: #6c757d; margin-bottom: 8px;">
            {row['expiry']} • {row['days_to_exp']} days • {row['moneyness']:+.1f}% Money
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.85em;">
            <div><span style="color: #6c757d;">OI:</span> <strong>{row['open_interest']:,.0f}</strong></div>
            <div><span style="color: #6c757d;">Vol:</span> <strong>{row['volume']:,.0f}</strong></div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def main():
    # Compact header
    st.title("🎯 Advanced Max Gamma Scanner")
    st.caption("Find the highest gamma strikes driving dealer hedging activity")
    
    # Settings at top of page (4 columns)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Symbols (comma-separated)", 
            value="AMZN",
            help="Enter stock symbols separated by commas"
        )
    
    with col2:
        num_expiries = st.selectbox(
            "Expiries to Scan", 
            [3, 4, 5, 6, 7, 8, 10], 
            index=2
        )
    
    with col3:
        top_n = st.selectbox(
            "Top N Strikes",
            [3, 5, 10],
            index=0
        )
    
    with col4:
        if st.button("🔄 Scan Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Filters in expander
    with st.expander("🔍 Filters & Settings", expanded=False):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            option_type_filter = st.selectbox(
                "Option Type",
                ["All", "Calls Only", "Puts Only"]
            )
        
        with filter_col2:
            min_open_interest = st.number_input(
                "Min Open Interest",
                min_value=0,
                max_value=10000,
                value=100,
                step=50
            )
        
        with filter_col3:
            moneyness_range = st.slider(
                "Moneyness Range (% from spot)",
                -50, 50, (-20, 20),
                help="Filter strikes by distance from current price"
            )
        
        with filter_col4:
            debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show detailed error messages")
    
    # Main content
    if not symbols:
        st.warning("Please enter at least one symbol to scan.")
        return
    
    # Test API connection
    with st.spinner("Testing API connection..."):
        try:
            test_client = get_schwab_client()
            test_quote = test_client.get_quote("SPY")
            if not test_quote:
                st.error("❌ API connection failed. Please check your authentication.")
                st.info("Run `python scripts/auth_setup.py` to authenticate.")
                return
            else:
                st.success("✅ API connection successful!")
        except Exception as e:
            st.error(f"❌ API connection failed: {str(e)}")
            st.info("Run `python scripts/auth_setup.py` to authenticate.")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            return
    
    st.info(f"Scanning {len(symbols)} symbol(s): {', '.join(symbols)}")
    
    # Dictionary to store results
    all_results = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch and analyze data for each symbol
    for idx, symbol in enumerate(symbols):
        status_text.text(f"Fetching data for {symbol}...")
        progress_bar.progress((idx + 1) / len(symbols))
        
        try:
            options_data, underlying_price, info_msg = get_options_data(symbol, num_expiries)
            
            # Display any info messages
            if info_msg:
                st.info(info_msg)
            
            if options_data is None or underlying_price is None:
                all_results[symbol] = {
                    'underlying_price': None,
                    'top_strikes': pd.DataFrame(),
                    'error': True
                }
                continue
            
            # Calculate gamma strikes
            df_gamma, debug_info = calculate_gamma_strikes(options_data, underlying_price, num_expiries, debug_mode)
            
            # Show debug info if enabled
            if debug_mode and debug_info['sample_contract']:
                with st.expander(f"🐛 Debug Info for {symbol}"):
                    st.write(f"**Total contracts processed:** {debug_info['total_contracts']}")
                    st.write(f"**Contracts with gamma:** {debug_info['contracts_with_gamma']}")
                    st.write(f"**Sample contract keys:** {list(debug_info['sample_contract'].keys())}")
                    st.write(f"**Sample gamma:** {debug_info['sample_contract'].get('gamma', 'N/A')}")
                    st.write(f"**Sample delta:** {debug_info['sample_contract'].get('delta', 'N/A')}")
            
            if df_gamma.empty:
                st.warning(f"⚠️ No gamma data available for {symbol}")
                all_results[symbol] = {
                    'underlying_price': underlying_price,
                    'top_strikes': pd.DataFrame(),
                    'error': True
                }
                continue
            
            # Apply filters
            if option_type_filter == "Calls Only":
                df_gamma = df_gamma[df_gamma['option_type'] == 'Call']
            elif option_type_filter == "Puts Only":
                df_gamma = df_gamma[df_gamma['option_type'] == 'Put']
            
            df_gamma = df_gamma[df_gamma['open_interest'] >= min_open_interest]
            df_gamma = df_gamma[
                (df_gamma['moneyness'] >= moneyness_range[0]) & 
                (df_gamma['moneyness'] <= moneyness_range[1])
            ]
            
            all_results[symbol] = {
                'underlying_price': underlying_price,
                'top_strikes': df_gamma.head(top_n),
                'all_gamma': df_gamma,
                'error': False
            }
            
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            all_results[symbol] = {
                'underlying_price': None,
                'top_strikes': pd.DataFrame(),
                'error': True
            }
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if not any(not result['error'] for result in all_results.values()):
        st.error("Failed to fetch data for all symbols. Please check your symbols and try again.")
        return
    
    # Compact summary section
    if len(symbols) == 1:
        result = all_results[symbols[0]]
        if not result['error'] and not result['top_strikes'].empty:
            underlying_price = result['underlying_price']
            top_gamma = result['top_strikes'].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${underlying_price:.2f}")
            with col2:
                st.metric("Max Gamma Strike", f"${top_gamma['strike']:.2f} {top_gamma['option_type']}")
            with col3:
                st.metric("Gamma Exposure", format_large_number(top_gamma['notional_gamma']))
    else:
        st.subheader("📊 Summary")
        cols = st.columns(min(len(symbols), 4))
        for idx, symbol in enumerate(symbols):
            with cols[idx % len(cols)]:
                result = all_results[symbol]
                if not result['error'] and not result['top_strikes'].empty:
                    underlying_price = result['underlying_price']
                    top_gamma = result['top_strikes'].iloc[0]
                    
                    st.metric(symbol, f"${underlying_price:.2f}")
                    st.caption(f"Max: ${top_gamma['strike']:.2f} {top_gamma['option_type']}")
                    st.caption(f"{format_large_number(top_gamma['notional_gamma'])}")
                else:
                    st.metric(symbol, "N/A")
    
    # Detailed results for each symbol
    st.markdown("---")
    st.header("🔍 Detailed Analysis")
    
    for symbol in symbols:
        result = all_results[symbol]
        
        if result['error']:
            st.warning(f"No data available for {symbol}")
            continue
        
        underlying_price = result['underlying_price']
        all_gamma = result.get('all_gamma', pd.DataFrame())
        
        if all_gamma.empty:
            st.warning(f"No gamma data available for {symbol}")
            continue
        
        # Symbol header
        st.markdown(f'<div class="stock-header">{symbol} - Current Price: ${underlying_price:.2f}</div>', unsafe_allow_html=True)
        
        # Key gamma levels
        st.subheader("🎯 Key Gamma Levels to Watch")
        
        calls_data = all_gamma[all_gamma['option_type'] == 'Call']
        puts_data = all_gamma[all_gamma['option_type'] == 'Put']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 💵 Max $ Call")
            if not calls_data.empty:
                row = calls_data.iloc[0]
                st.metric(f"${row['strike']:.2f}", format_large_number(row['notional_gamma']))
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        with col2:
            st.markdown("### 🎯 Highest Γ Call")
            if not calls_data.empty:
                row = calls_data.nlargest(1, 'gamma').iloc[0]
                st.metric(f"${row['strike']:.2f}", f"Γ: {row['gamma']:.4f}")
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        with col3:
            st.markdown("### 💵 Max $ Put")
            if not puts_data.empty:
                row = puts_data.iloc[0]
                st.metric(f"${row['strike']:.2f}", format_large_number(row['notional_gamma']))
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        with col4:
            st.markdown("### 🎯 Highest Γ Put")
            if not puts_data.empty:
                row = puts_data.nlargest(1, 'gamma').iloc[0]
                st.metric(f"${row['strike']:.2f}", f"Γ: {row['gamma']:.4f}")
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        st.markdown("---")
        
        # Detailed top strikes by category
        st.subheader("📊 Top Strikes by Category")
        
        max_positive = calls_data.head(20) if not calls_data.empty else pd.DataFrame()
        max_negative = puts_data.head(20) if not puts_data.empty else pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Call Gamma (Resistance)")
            if not max_positive.empty:
                for idx, (_, row) in enumerate(max_positive.iterrows(), 1):
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    card_html = f"""
                    <div class="gamma-level-card resistance">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <strong style="font-size: 1.1em;">#{idx} ${row['strike']:.2f}</strong>
                            <span style="color: #28a745; font-weight: bold;">{distance:+.1f}%</span>
                        </div>
                        <div style="font-size: 0.9em; color: #495057;">
                            <strong>💵 {format_large_number(row['notional_gamma'])}</strong> • <strong>Γ: {row['gamma']:.4f}</strong>
                        </div>
                        <div style="font-size: 0.85em; color: #6c757d; margin-top: 3px;">
                            {row['expiry']} ({row['days_to_exp']}d) • OI: {row['open_interest']:,.0f} | Vol: {row['volume']:,.0f}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No significant call gamma")
        
        with col2:
            st.markdown("### 📉 Put Gamma (Support)")
            if not max_negative.empty:
                for idx, (_, row) in enumerate(max_negative.iterrows(), 1):
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    card_html = f"""
                    <div class="gamma-level-card support">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <strong style="font-size: 1.1em;">#{idx} ${row['strike']:.2f}</strong>
                            <span style="color: #dc3545; font-weight: bold;">{distance:+.1f}%</span>
                        </div>
                        <div style="font-size: 0.9em; color: #495057;">
                            <strong>💵 {format_large_number(row['notional_gamma'])}</strong> • <strong>Γ: {row['gamma']:.4f}</strong>
                        </div>
                        <div style="font-size: 0.85em; color: #6c757d; margin-top: 3px;">
                            {row['expiry']} ({row['days_to_exp']}d) • OI: {row['open_interest']:,.0f} | Vol: {row['volume']:,.0f}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No significant put gamma")
        
        st.markdown("---")
        
        # Create gamma table
        gamma_table = create_gamma_table(all_gamma, underlying_price, num_expiries=min(num_expiries, 6))
        
        if not gamma_table.empty:
            with st.expander("📊 Gamma Exposure by Strike and Expiry (Click to Expand)", expanded=False):
                # Prepare display dataframe
                display_cols = [col for col in gamma_table.columns if col not in ['is_current']]
                display_df = gamma_table[display_cols].copy()
                
                # Format strike column
                display_df['Strike'] = gamma_table['Strike'].apply(lambda x: f"${x:.2f}")
                
                # Format gamma columns with color coding
                def style_gamma(val):
                    if pd.isna(val) or val == 0:
                        return ''
                    abs_val = abs(val)
                    if abs_val > 1e9:
                        color = '#006400' if val > 0 else '#8B0000'
                        return f'background-color: {color}; color: white; font-weight: bold'
                    elif abs_val > 5e8:
                        color = '#228B22' if val > 0 else '#DC143C'
                        return f'background-color: {color}; color: white'
                    elif abs_val > 1e8:
                        color = '#90EE90' if val > 0 else '#FF6347'
                        return f'background-color: {color}'
                    else:
                        color = '#e8f5e9' if val > 0 else '#ffebee'
                        return f'background-color: {color}'
                
                # Apply styling
                styled_df = gamma_table.style
                
                # Highlight current price row
                def highlight_current(row):
                    if row.get('is_current', False):
                        return ['background-color: yellow; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_df = styled_df.apply(highlight_current, axis=1)
                
                # Apply color to gamma columns
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        styled_df = styled_df.applymap(style_gamma, subset=[col])
                
                # Format display values
                format_dict = {}
                for col in gamma_table.columns:
                    if col == 'Strike':
                        format_dict[col] = '${:.2f}'
                    elif col != 'is_current':
                        format_dict[col] = lambda x: format_large_number(x) if x != 0 else ''
                
                st.dataframe(
                    styled_df.format(format_dict).hide(axis='index').hide(columns=['is_current']),
                    use_container_width=True,
                    height=500
                )
        
        # Show top strikes summary
        with st.expander(f"📊 Top {top_n} Gamma Strikes (Click to Expand)", expanded=True):
            top_strikes = result['top_strikes']
            if not top_strikes.empty:
                for rank, (_, row) in enumerate(top_strikes.iterrows(), 1):
                    display_gamma_strike_card(row, rank, underlying_price)
            else:
                st.info("No strikes match the current filters")
        
        # Add expandable detailed table
        with st.expander(f"📋 Full Data Table"):
            top_strikes = result['top_strikes']
            if not top_strikes.empty:
                display_df = top_strikes[[
                    'strike', 'expiry', 'days_to_exp', 'option_type', 
                    'notional_gamma', 'open_interest', 'volume', 'moneyness'
                ]].copy()
                
                display_df.columns = [
                    'Strike', 'Expiry', 'DTE', 'Type', 
                    'Notional Gamma', 'OI', 'Volume', 'Moneyness %'
                ]
                
                # Format columns
                display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
                display_df['Notional Gamma'] = display_df['Notional Gamma'].apply(format_large_number)
                display_df['OI'] = display_df['OI'].apply(lambda x: f"{x:,.0f}")
                display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
                display_df['Moneyness %'] = display_df['Moneyness %'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No data to display")
        
        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Expiries Scanned: {num_expiries}")

if __name__ == "__main__":
    main()