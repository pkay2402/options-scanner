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
project_root = Path(__file__).parent.parent; sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

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
                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)
                    strike = float(strike_str)
                    activity = volume + open_interest * 0.1  # Weight OI less than volume
                    strike_data.append((strike, activity))
            
            if strike_data:
                # Sort by activity and get the most active strike
                strike_data.sort(key=lambda x: x[1], reverse=True)
                most_active_strike = strike_data[0][0]
                
                # If the most active strike seems reasonable, use it
                if 50 < most_active_strike < 2000:  # Reasonable stock price range
                    return most_active_strike
            
            # Fallback to middle strike
            strikes.sort()
            mid_index = len(strikes) // 2
            return strikes[mid_index]
        
        return None
    except:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_options_data(symbol, num_expiries=5):
    """Fetch options data for multiple expiries"""
    try:
        client = SchwabClient()
        
        # Get quote data
        quote_data = client.get_quote(symbol)
        
        if not quote_data:
            st.error("Failed to get quote data. Check API connection.")
            return None, None
        
        # Extract underlying price from quote
        underlying_price = None
        if symbol in quote_data:
            underlying_price = quote_data[symbol].get('lastPrice', 0)
            if underlying_price == 0:
                # Try other price fields
                underlying_price = quote_data[symbol].get('mark', 0)
                if underlying_price == 0:
                    underlying_price = quote_data[symbol].get('bidPrice', 0)
                if underlying_price == 0:
                    underlying_price = quote_data[symbol].get('askPrice', 0)
        else:
            st.error(f"Symbol {symbol} not found in quote response")
            return None, None
        
        # Get options chain
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not options_data:
            st.error("No options data available for this symbol.")
            return None, None
        
        # Check if options data has the expected structure
        if 'callExpDateMap' not in options_data and 'putExpDateMap' not in options_data:
            st.error("Invalid options data structure.")
            return None, None
        
        # Try to get underlying price from options data if available and valid
        if 'underlyingPrice' in options_data and options_data['underlyingPrice']:
            underlying_price = options_data['underlyingPrice']
        elif 'underlying' in options_data and options_data['underlying']:
            options_underlying_price = (
                options_data['underlying'].get('last') or
                options_data['underlying'].get('mark') or
                options_data['underlying'].get('lastPrice') or
                options_data['underlying'].get('close') or 0
            )
            if options_underlying_price and options_underlying_price > 0:
                underlying_price = options_underlying_price
        
        # If price still not valid, try yfinance
        if not underlying_price or underlying_price == 0 or underlying_price == 100.0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                underlying_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                if underlying_price:
                    st.info(f"Using yfinance for {symbol} price: ${underlying_price:.2f}")
            except:
                # Last resort - estimate from strike prices
                estimated_price = estimate_underlying_from_strikes(options_data)
                if estimated_price:
                    underlying_price = estimated_price
        
        return options_data, underlying_price
        
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        # Try yfinance as fallback
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            underlying_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if underlying_price:
                st.warning(f"API error - using yfinance for {symbol} price: ${underlying_price:.2f}")
                # Still try to get options from Schwab
                client = SchwabClient()
                options_data = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
                if options_data:
                    return options_data, underlying_price
        except:
            pass
        return None, None

def calculate_gamma_strikes(options_data, underlying_price, num_expiries=5, debug_mode=False):
    """Calculate gamma for all strikes and identify top gamma strikes"""
    
    debug_info = {'total_contracts': 0, 'contracts_with_gamma': 0, 'sample_contract': None}
    
    if not options_data:
        return pd.DataFrame()
    
    results = []
    
    # Process both calls and puts
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
            
        exp_dates = list(options_data[option_type].keys())[:num_expiries]
        
        for exp_date in exp_dates:
            strikes_data = options_data[option_type][exp_date]
            
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                    
                strike = float(strike_str)
                contract = contracts[0]  # Take first contract
                
                # Debug: Store first contract for inspection
                debug_info['total_contracts'] += 1
                if debug_info['sample_contract'] is None:
                    debug_info['sample_contract'] = contract
                
                # Extract data - simple approach like Index Positioning
                gamma = contract.get('gamma', 0)
                delta = contract.get('delta', 0)
                vega = contract.get('vega', 0)
                volume = contract.get('totalVolume', 0)
                open_interest = contract.get('openInterest', 0)
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                last = contract.get('last', 0)
                volatility = contract.get('volatility', 0)
                implied_volatility = contract.get('volatility', 0) * 100  # Convert to percentage
                
                # Count valid gamma contracts for debug
                if gamma > 0:
                    debug_info['contracts_with_gamma'] += 1
                
                # Estimate gamma if not provided (same as Index Positioning)
                if gamma == 0 and vega > 0 and implied_volatility > 0:
                    gamma = vega / (underlying_price * implied_volatility / 100)
                
                # Calculate different gamma metrics
                # Apply dealer GEX sign convention:
                # - Calls: negative gamma (dealers short calls need to buy on rallies)
                # - Puts: positive gamma (dealers short puts need to sell on dips)
                is_call = 'call' in option_type.lower()
                dealer_gamma_sign = -1 if is_call else 1
                
                # 1. Notional gamma exposure (dollar value) - from dealer's perspective
                notional_gamma = dealer_gamma_sign * gamma * open_interest * 100 * underlying_price if underlying_price > 0 else dealer_gamma_sign * gamma * open_interest * 100
                
                # 2. Gamma exposure (shares that need to be hedged)
                gamma_exposure_shares = dealer_gamma_sign * gamma * open_interest * 100
                
                # 3. Absolute gamma (for ranking by gamma concentration)
                abs_gamma_oi = abs(gamma) * open_interest
                
                # Calculate moneyness
                moneyness = (strike / underlying_price - 1) * 100 if underlying_price > 0 else 0
                
                # Parse expiration date
                exp_date_str = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                # Calculate days to expiration
                try:
                    exp_datetime = datetime.strptime(exp_date_str, '%Y-%m-%d')
                    days_to_exp = (exp_datetime - datetime.now()).days
                except:
                    days_to_exp = 0
                
                results.append({
                    'strike': strike,
                    'expiry': exp_date_str,
                    'days_to_exp': days_to_exp,
                    'option_type': 'Call' if 'call' in option_type else 'Put',
                    'gamma': gamma,
                    'delta': delta,
                    'vega': vega,
                    'volume': volume,
                    'open_interest': open_interest,
                    'bid': bid,
                    'ask': ask,
                    'last': last,
                    'notional_gamma': abs(notional_gamma),  # Dollar value for ranking
                    'signed_notional_gamma': notional_gamma,  # Keep signed value for display
                    'gamma_exposure_shares': gamma_exposure_shares,  # Shares to hedge
                    'abs_gamma_oi': abs_gamma_oi,  # Pure gamma concentration
                    'moneyness': moneyness,
                    'implied_volatility': implied_volatility
                })
    
    if not results:
        return pd.DataFrame(), debug_info
    
    df = pd.DataFrame(results)
    
    # Sort by absolute notional gamma and get top strikes
    df_sorted = df.sort_values('notional_gamma', ascending=False)
    
    return df_sorted, debug_info

def format_large_number(num):
    """Format large numbers for display"""
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.1f}K"
    else:
        return f"${num:.0f}"

def create_gamma_table(df_gamma, underlying_price, num_expiries=5):
    """Create a clean table showing gamma exposure across strikes and expiries"""
    
    if df_gamma.empty:
        return pd.DataFrame()
    
    # Get unique expiries (limit to num_expiries)
    expiries = sorted(df_gamma['expiry'].unique())[:num_expiries]
    
    # Create pivot table
    pivot_data = []
    
    # Get all strikes
    strikes = sorted(df_gamma['strike'].unique())
    
    for strike in strikes:
        row = {'Strike': strike}
        for expiry in expiries:
            mask = (df_gamma['strike'] == strike) & (df_gamma['expiry'] == expiry)
            matches = df_gamma[mask]
            
            if not matches.empty:
                gamma_sum = matches['signed_notional_gamma'].sum()
                row[expiry] = gamma_sum
            else:
                row[expiry] = 0
        
        pivot_data.append(row)
    
    df_pivot = pd.DataFrame(pivot_data)
    
    # Mark current price row
    if not df_pivot.empty:
        df_pivot['is_current'] = df_pivot['Strike'].apply(
            lambda x: abs(x - underlying_price) < (underlying_price * 0.01)
        )
    
    return df_pivot

def create_gamma_comparison_chart(all_results):
    """Create a comparison chart showing top gamma strikes across all symbols"""
    
    if not all_results:
        return go.Figure()
    
    fig = go.Figure()
    
    symbols = list(all_results.keys())
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
    
    for idx, symbol in enumerate(symbols):
        df = all_results[symbol]['top_strikes']
        if df.empty:
            continue
        
        # Take top 3
        top_3 = df.head(10)
        
        # Create hover text
        hover_text = [
            f"Symbol: {symbol}<br>Strike: ${row['strike']:.2f}<br>Gamma: {format_large_number(row['signed_notional_gamma'])}<br>Expiry: {row['expiry']}<br>Type: {row['option_type']}"
            for _, row in top_3.iterrows()
        ]
        
        fig.add_trace(go.Bar(
            name=symbol,
            x=[f"${row['strike']:.2f} ({row['option_type'][0]})" for _, row in top_3.iterrows()],
            y=top_3['notional_gamma'].values,
            marker_color=colors[idx % len(colors)],
            hovertext=hover_text,
            hoverinfo='text',
            text=[format_large_number(val) for val in top_3['signed_notional_gamma'].values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Top 3 Gamma Strikes Comparison",
        xaxis_title="Strike (Type)",
        yaxis_title="Absolute Notional Gamma",
        height=500,
        barmode='group',
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
    
    gamma_color = "#28a745" if row['signed_notional_gamma'] > 0 else "#dc3545"
    
    html = f"""
    <div class="strike-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <strong style="font-size: 1.1em;">#{rank} ${row['strike']:.2f} {row['option_type']}</strong>
                <span style="color: {status_color}; font-size: 0.85em; margin-left: 8px;">{status}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2em; font-weight: bold; color: {gamma_color};">
                    {format_large_number(row['signed_notional_gamma'])}
                </div>
            </div>
        </div>
        <div style="font-size: 0.85em; color: #6c757d; margin-bottom: 8px;">
            {row['expiry']} • {row['days_to_exp']} days • {row['moneyness']:+.1f}% Money
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.85em;">
            <div><span style="color: #6c757d;">Γ:</span> <strong>{row['gamma']:.4f}</strong></div>
            <div><span style="color: #6c757d;">Δ:</span> <strong>{row['delta']:.3f}</strong></div>
            <div><span style="color: #6c757d;">IV:</span> <strong>{row['implied_volatility']:.1f}%</strong></div>
            <div><span style="color: #6c757d;">OI:</span> <strong>{row['open_interest']:,.0f}</strong></div>
            <div><span style="color: #6c757d;">Vol:</span> <strong>{row['volume']:,.0f}</strong></div>
            <div><span style="color: #6c757d;">Vega:</span> <strong>{row['vega']:.3f}</strong></div>
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
            index=5
        )
    
    with col3:
        top_n = st.selectbox(
            "Top N Strikes",
            [3, 5, 10],
            index=2
        )
    
    with col4:
        if st.button("🔄 Scan Now", type="primary", use_container_width=True):
            st.cache_data.clear()
    
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
            test_client = SchwabClient()
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
            options_data, underlying_price = get_options_data(symbol, num_expiries)
            
            if options_data is None or underlying_price is None:
                if not debug_mode:
                    st.warning(f"⚠️ Could not fetch data for {symbol}")
                all_results[symbol] = {
                    'underlying_price': None,
                    'top_strikes': pd.DataFrame(),
                    'error': True
                }
                continue
            
            # Calculate gamma strikes
            df_gamma, debug_info = calculate_gamma_strikes(options_data, underlying_price, num_expiries, debug_mode)
            
            # Show debug info if enabled
            if debug_mode:
                st.info(f"""
                **Debug Info for {symbol}:**
                - Total contracts processed: {debug_info['total_contracts']}
                - Contracts with gamma: {debug_info['contracts_with_gamma']}
                - Sample contract keys: {list(debug_info['sample_contract'].keys()) if debug_info['sample_contract'] else 'None'}
                - Sample gamma value: {debug_info['sample_contract'].get('gamma', 'N/A') if debug_info['sample_contract'] else 'N/A'}
                - Sample delta value: {debug_info['sample_contract'].get('delta', 'N/A') if debug_info['sample_contract'] else 'N/A'}
                """)
            
            if df_gamma.empty:
                st.warning(f"⚠️ No gamma data available for {symbol}")
                all_results[symbol] = {
                    'underlying_price': underlying_price,
                    'top_strikes': pd.DataFrame(),
                    'error': True
                }
                continue
            
            # Sort by notional gamma (default)
            df_gamma = df_gamma.sort_values('notional_gamma', ascending=False)
            
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
                'all_gamma': df_gamma,  # Store all gamma data for heatmap
                'error': False
            }
            
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
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
        # Single symbol - show inline
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
                st.metric("Gamma Exposure", format_large_number(top_gamma['signed_notional_gamma']))
    else:
        # Multiple symbols - show compact cards
        st.subheader("📊 Summary")
        cols = st.columns(len(symbols))
        for idx, symbol in enumerate(symbols):
            with cols[idx]:
                result = all_results[symbol]
                if not result['error'] and not result['top_strikes'].empty:
                    underlying_price = result['underlying_price']
                    top_gamma = result['top_strikes'].iloc[0]
                    
                    st.metric(symbol, f"${underlying_price:.2f}")
                    st.caption(f"Max: ${top_gamma['strike']:.2f} {top_gamma['option_type']}")
                    st.caption(f"{format_large_number(top_gamma['signed_notional_gamma'])}")
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
        
        # Add summary of max gamma areas
        st.subheader("🎯 Key Gamma Levels to Watch")
        
        # Find max strikes by BOTH metrics
        # Filter by option type (not by gamma sign, since API returns -999 for all)
        calls_data = all_gamma[all_gamma['option_type'] == 'Call']
        puts_data = all_gamma[all_gamma['option_type'] == 'Put']
        
        # 1. By dollar value (notional gamma)
        max_positive_dollar = calls_data.nlargest(1, 'notional_gamma') if not calls_data.empty else pd.DataFrame()
        max_negative_dollar = puts_data.nlargest(1, 'notional_gamma') if not puts_data.empty else pd.DataFrame()
        
        # 2. By concentration (gamma × OI)
        max_positive_conc = calls_data.nlargest(1, 'abs_gamma_oi') if not calls_data.empty else pd.DataFrame()
        max_negative_conc = puts_data.nlargest(1, 'abs_gamma_oi') if not puts_data.empty else pd.DataFrame()
        
        # Display both metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 💵 Max $ Call")
            if not max_positive_dollar.empty:
                row = max_positive_dollar.iloc[0]
                st.metric(f"${row['strike']:.2f}", format_large_number(row['signed_notional_gamma']))
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        with col2:
            st.markdown("### 🎯 Max Γ Call")
            if not max_positive_conc.empty:
                row = max_positive_conc.iloc[0]
                st.metric(f"${row['strike']:.2f}", f"Γ×OI: {row['abs_gamma_oi']:,.0f}")
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
            else:
                st.info("None")
        
        with col3:
            st.markdown("### 💵 Max $ Put")
            if not max_negative_dollar.empty:
                row = max_negative_dollar.iloc[0]
                st.metric(f"${row['strike']:.2f}", format_large_number(row['signed_notional_gamma']))
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
                if debug_mode:
                    st.caption(f"Debug: OI={row['open_interest']:,.0f}, Gamma={row['gamma']:.4f}")
            else:
                st.info("None")
        
        with col4:
            st.markdown("### 🎯 Max Γ Put")
            if not max_negative_conc.empty:
                row = max_negative_conc.iloc[0]
                st.metric(f"${row['strike']:.2f}", f"Γ×OI: {row['abs_gamma_oi']:,.0f}")
                st.caption(f"{row['expiry']} ({row['days_to_exp']}d)")
                if debug_mode:
                    st.caption(f"Debug: ${format_large_number(row['signed_notional_gamma'])}")
            else:
                st.info("None")
        
        st.markdown("---")
        
        # Show max gamma by expiry (to understand the breakdown)
        if debug_mode:
            st.subheader("🔍 Debug: Max Gamma by Expiry")
            expiries = sorted(all_gamma['expiry'].unique())
            for expiry in expiries[:3]:  # Show first 3 expiries
                exp_data = all_gamma[all_gamma['expiry'] == expiry]
                # Show top 5 puts for this expiry
                top_puts = exp_data[exp_data['signed_notional_gamma'] < 0].nlargest(5, 'notional_gamma')
                if not top_puts.empty:
                    st.write(f"**{expiry}** - Top 5 Puts:")
                    for idx, (_, row) in enumerate(top_puts.iterrows(), 1):
                        st.write(f"  {idx}. ${row['strike']:.2f} = {format_large_number(row['signed_notional_gamma'])} (OI: {row['open_interest']:,.0f}, Γ: {row['gamma']:.4f})")
        
        # Detailed top 20 for each
        st.subheader("📊 Top 20 Strikes by Category")
        
        # Filter by option type instead of gamma sign
        max_positive = calls_data.nlargest(20, 'notional_gamma') if not calls_data.empty else pd.DataFrame()
        max_negative = puts_data.nlargest(20, 'notional_gamma') if not puts_data.empty else pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Resistance Zones")
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
                            <strong>💵 {format_large_number(row['signed_notional_gamma'])}</strong> • <strong>🎯 Γ×OI: {row['abs_gamma_oi']:,.0f}</strong>
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
            st.markdown("### 📉 Support Zones")
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
                            <strong>💵 {format_large_number(abs(row['signed_notional_gamma']))}</strong> • <strong>🎯 Γ×OI: {row['abs_gamma_oi']:,.0f}</strong>
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
            # Style the dataframe
            def highlight_current_price(row):
                if row.get('is_current', False):
                    return ['background-color: yellow; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            def color_gamma(val):
                if pd.isna(val) or val == 0:
                    return 'background-color: white'
                elif val > 1e9:
                    return 'background-color: #006400; color: white; font-weight: bold'
                elif val > 5e8:
                    return 'background-color: #228B22; color: white'
                elif val > 1e8:
                    return 'background-color: #90EE90'
                elif val > 0:
                    return 'background-color: #e8f5e9'
                elif val < -1e9:
                    return 'background-color: #8B0000; color: white; font-weight: bold'
                elif val < -5e8:
                    return 'background-color: #DC143C; color: white'
                elif val < -1e8:
                    return 'background-color: #FF6347'
                else:
                    return 'background-color: #ffebee'
            
            # Prepare display dataframe
            display_cols = [col for col in gamma_table.columns if col not in ['is_current']]
            display_df = gamma_table[display_cols].copy()
            
            # Format strike column
            display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
            
            # Format gamma columns
            for col in display_df.columns:
                if col != 'Strike':
                    display_df[col] = gamma_table[col].apply(
                        lambda x: format_large_number(x) if x != 0 else ''
                    )
            
            with st.expander("📊 Gamma Exposure by Strike and Expiry (Click to Expand)", expanded=False):
                # Apply styling
                styled_df = gamma_table.style.apply(highlight_current_price, axis=1)
                
                # Apply color to gamma columns
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        styled_df = styled_df.applymap(color_gamma, subset=[col])
                
                # Format values
                format_dict = {'Strike': '${:.2f}'}
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        format_dict[col] = lambda x: format_large_number(x) if x != 0 else ''
                
                st.dataframe(
                    styled_df.format(format_dict).hide(axis='index'),
                    use_container_width=True,
                    height=500
                )
        else:
            st.warning("No gamma data to display")
        
        # Show top strikes summary in expander
        with st.expander(f"📊 Top {top_n} Gamma Strikes (Click to Expand)", expanded=True):
            top_strikes = result['top_strikes']
            for rank, (_, row) in enumerate(top_strikes.iterrows(), 1):
                display_gamma_strike_card(row, rank, underlying_price)
        
        # Add expandable detailed table
        with st.expander(f"📋 Full Data Table"):
            # Create a formatted dataframe
            display_df = top_strikes[[
                'strike', 'expiry', 'days_to_exp', 'option_type', 
                'signed_notional_gamma', 'gamma', 'delta', 'vega',
                'open_interest', 'volume', 'implied_volatility', 'moneyness'
            ]].copy()
            
            display_df.columns = [
                'Strike', 'Expiry', 'DTE', 'Type', 
                'Notional Gamma', 'Gamma', 'Delta', 'Vega',
                'OI', 'Volume', 'IV %', 'Moneyness %'
            ]
            
            # Format columns
            display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
            display_df['Notional Gamma'] = display_df['Notional Gamma'].apply(format_large_number)
            # Display all valid greeks - gamma/vega can legitimately be 0.00 for far OTM
            display_df['Gamma'] = display_df['Gamma'].apply(lambda x: f"{x:.4f}")
            display_df['Delta'] = display_df['Delta'].apply(lambda x: f"{x:.3f}" if abs(x) <= 1 else "N/A")
            display_df['Vega'] = display_df['Vega'].apply(lambda x: f"{x:.3f}")
            display_df['OI'] = display_df['OI'].apply(lambda x: f"{x:,.0f}")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            display_df['IV %'] = display_df['IV %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            display_df['Moneyness %'] = display_df['Moneyness %'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Expiries Scanned: {num_expiries}")

if __name__ == "__main__":
    main()
