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

# Add the project root to Python path (works in both local and cloud)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

def calculate_gamma_strikes(options_data, underlying_price, num_expiries=5):
    """Calculate gamma for all strikes and identify top gamma strikes"""
    
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
                
                # Extract data
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
                
                # If gamma is not provided, try to estimate it
                if gamma == 0 and delta != 0 and volatility > 0 and underlying_price > 0:
                    if vega > 0:
                        gamma = vega / (underlying_price * volatility)
                    else:
                        time_to_exp = 30
                        if abs(strike - underlying_price) / underlying_price < 0.05:
                            gamma = 1 / (underlying_price * volatility * (time_to_exp ** 0.5))
                
                # Calculate different gamma metrics
                # 1. Notional gamma exposure (dollar value)
                notional_gamma = gamma * open_interest * 100 * underlying_price if underlying_price > 0 else gamma * open_interest * 100
                
                # 2. Gamma exposure (shares that need to be hedged)
                gamma_exposure_shares = gamma * open_interest * 100
                
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
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Sort by absolute notional gamma and get top strikes
    df_sorted = df.sort_values('notional_gamma', ascending=False)
    
    return df_sorted

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
    # Retail-focused header
    st.title("🎯 Options Strike Finder")
    st.caption("💡 Discover which strikes and expiries have the most market-moving potential")
    
    # Sidebar controls
    st.sidebar.header("Scanner Settings")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed error messages")
    
    # Symbol input - allow multiple symbols
    symbols_input = st.sidebar.text_input(
        "Symbols (comma-separated)", 
        value="AMZN",
        help="Enter stock symbols separated by commas"
    )
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Number of expiries to scan
    num_expiries = st.sidebar.selectbox(
        "Number of Expiries to Scan", 
        [3, 4, 5, 6, 7, 8, 10], 
        index=5
    )
    
    # Number of top strikes to show per symbol
    top_n = st.sidebar.selectbox(
        "Top N Strikes per Symbol",
        [3, 5, 10],
        index=2
    )
    
    # Filter options
    st.sidebar.subheader("Filters")
    
    option_type_filter = st.sidebar.selectbox(
        "Option Type",
        ["All", "Calls Only", "Puts Only"]
    )
    
    min_open_interest = st.sidebar.number_input(
        "Min Open Interest",
        min_value=0,
        max_value=10000,
        value=100,
        step=50
    )
    
    moneyness_range = st.sidebar.slider(
        "Moneyness Range (% from spot)",
        -50, 50, (-20, 20),
        help="Filter strikes by distance from current price"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Scan Now"):
        st.cache_data.clear()
    
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
            df_gamma = calculate_gamma_strikes(options_data, underlying_price, num_expiries)
            
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
                'options_data': options_data,  # Store for flow analysis
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
        # Single symbol - show inline with EMAs
        result = all_results[symbols[0]]
        if not result['error'] and not result['top_strikes'].empty:
            underlying_price = result['underlying_price']
            top_gamma = result['top_strikes'].iloc[0]
            
            # Calculate EMAs using yfinance
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbols[0])
                hist = ticker.history(period="1y")
                
                if not hist.empty:
                    ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
                    ema_21 = hist['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
                    ema_50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                    ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                else:
                    ema_8 = ema_21 = ema_50 = ema_200 = None
            except:
                ema_8 = ema_21 = ema_50 = ema_200 = None
            
            # Display metrics with EMAs
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${underlying_price:.2f}")
                if ema_8 and ema_21 and ema_50 and ema_200:
                    st.caption(f"📊 EMA-8: ${ema_8:.2f}")
                    st.caption(f"📊 EMA-21: ${ema_21:.2f}")
            with col2:
                st.metric("Max Gamma Strike", f"${top_gamma['strike']:.2f} {top_gamma['option_type']}")
                if ema_50:
                    st.caption(f"📊 EMA-50: ${ema_50:.2f}")
                if ema_200:
                    st.caption(f"📊 EMA-200: ${ema_200:.2f}")
            with col3:
                st.metric("Gamma Exposure", format_large_number(top_gamma['signed_notional_gamma']))
                # Show trend based on EMA positioning
                if ema_8 and ema_21 and ema_50 and ema_200:
                    if underlying_price > ema_8 > ema_21 > ema_50:
                        st.caption("📈 Strong uptrend")
                    elif underlying_price < ema_8 < ema_21 < ema_50:
                        st.caption("📉 Strong downtrend")
                    elif underlying_price > ema_50 > ema_200:
                        st.caption("🟢 Bullish trend")
                    elif underlying_price < ema_50 < ema_200:
                        st.caption("🔴 Bearish trend")
                    else:
                        st.caption("📊 Mixed/Ranging")
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
        
        # Symbol header with clear current price
        st.markdown(f'<div class="stock-header">{symbol} - Current Price: ${underlying_price:.2f}</div>', unsafe_allow_html=True)
        
        # Gamma Heatmap - MOVED TO TOP
        gamma_table = create_gamma_table(all_gamma, underlying_price, num_expiries=min(num_expiries, 6))
        
        if not gamma_table.empty:
            with st.expander("📊 Gamma Heatmap - All Strikes & Expiries (Click to Expand)", expanded=False):
                st.caption("💡 Darker colors = Stronger gamma levels. Yellow row = Current price.")
                
                # CLEANER color scheme - Blue for calls, Green for puts
                def color_gamma_clean(val):
                    if pd.isna(val) or val == 0:
                        return 'background-color: white'
                    # Positive (Calls) - BLUE shades
                    elif val > 5e10:  # >$50B
                        return 'background-color: #0D47A1; color: white; font-weight: bold'
                    elif val > 2e10:  # >$20B
                        return 'background-color: #1976D2; color: white'
                    elif val > 1e10:  # >$10B
                        return 'background-color: #42A5F5; color: black'
                    elif val > 0:
                        return 'background-color: #E3F2FD'
                    # Negative (Puts) - GREEN shades
                    elif val < -5e10:  # <-$50B
                        return 'background-color: #1B5E20; color: white; font-weight: bold'
                    elif val < -2e10:  # <-$20B
                        return 'background-color: #388E3C; color: white'
                    elif val < -1e10:  # <-$10B
                        return 'background-color: #66BB6A; color: black'
                    else:
                        return 'background-color: #E8F5E9'
                
                def highlight_current_price(row):
                    if row.get('is_current', False):
                        return ['background-color: #FFEB3B; font-weight: bold; border: 2px solid black'] * len(row)
                    return [''] * len(row)
                
                # Apply styling
                styled_df = gamma_table.style.apply(highlight_current_price, axis=1)
                
                # Apply color to gamma columns
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        styled_df = styled_df.applymap(color_gamma_clean, subset=[col])
                
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
        
        # Latest Options Flow section
        with st.expander("🌊 Latest Options Flow Activity (Click to Expand)", expanded=False):
            st.caption("💡 Recent large trades and unusual options activity")
            
            # Analyze flow from the options data
            try:
                flows = []
                options_data = result.get('options_data')
                
                if options_data:
                    for option_type in ['callExpDateMap', 'putExpDateMap']:
                        if option_type not in options_data:
                            continue
                        
                        is_call = 'call' in option_type
                        exp_dates = list(options_data[option_type].keys())[:3]  # Top 3 expiries
                        
                        for exp_date in exp_dates:
                            strikes_data = options_data[option_type][exp_date]
                            
                            for strike_str, contracts in strikes_data.items():
                                if not contracts:
                                    continue
                                
                                strike = float(strike_str)
                                contract = contracts[0]
                                
                                volume = contract.get('totalVolume', 0)
                                open_interest = contract.get('openInterest', 0)
                                bid = contract.get('bid', 0)
                                ask = contract.get('ask', 0)
                                last = contract.get('last', 0)
                                
                                if volume == 0:
                                    continue
                                
                                # Calculate premium
                                mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                                premium = volume * mid_price * 100
                                
                                # Only show trades > $50k
                                if premium >= 50000:
                                    exp_date_str = exp_date.split(':')[0] if ':' in exp_date else exp_date
                                    try:
                                        exp_dt = datetime.strptime(exp_date_str, '%Y-%m-%d')
                                        days_to_exp = (exp_dt - datetime.now()).days
                                    except:
                                        days_to_exp = 0
                                    
                                    flows.append({
                                        'type': 'Call' if is_call else 'Put',
                                        'strike': strike,
                                        'expiry': exp_date_str,
                                        'days': days_to_exp,
                                        'volume': volume,
                                        'oi': open_interest,
                                        'premium': premium,
                                        'price': mid_price
                                    })
                    
                    # Sort by premium and show top 10
                    flows.sort(key=lambda x: x['premium'], reverse=True)
                    flows = flows[:10]
                    
                    if flows:
                        for idx, flow in enumerate(flows, 1):
                            sentiment = "🟢 Bullish" if flow['type'] == 'Call' else "🔴 Bearish"
                            distance = ((flow['strike'] - underlying_price) / underlying_price) * 100
                            
                            flow_html = f"""
                            <div style="background: {'#e8f5e9' if flow['type'] == 'Call' else '#ffebee'}; 
                                        padding: 12px; margin: 8px 0; border-radius: 8px; 
                                        border-left: 4px solid {'#28a745' if flow['type'] == 'Call' else '#dc3545'};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="font-size: 1.1em;">#{idx} ${flow['strike']:.2f} {flow['type']}</strong>
                                        <span style="margin-left: 10px;">{sentiment}</span>
                                    </div>
                                    <div style="text-align: right;">
                                        <strong style="font-size: 1.15em;">{format_large_number(flow['premium'])}</strong>
                                    </div>
                                </div>
                                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                    📅 {flow['expiry']} ({flow['days']}d) • 
                                    📊 Vol: {flow['volume']:,} • OI: {flow['oi']:,} • 
                                    📍 {distance:+.1f}% from price
                                </div>
                            </div>
                            """
                            st.markdown(flow_html, unsafe_allow_html=True)
                    else:
                        st.info("No significant flow activity detected (minimum $50k premium)")
                else:
                    st.warning("Flow data not available")
            except Exception as e:
                st.error(f"Error analyzing flow: {str(e)}")
        
        # Quick explanation for retail traders
        with st.expander("💡 How to Use This Scanner", expanded=False):
            st.markdown("""
            ### What This Shows You:
            - **High Gamma Strikes** = Where dealers need to hedge heavily
            - **More hedging** = More price magnetism to these strikes
            - **Strike Distance** = How far price needs to move
            
            ### How to Trade It:
            1. **Look at "Upside/Downside Targets"** - These are the strongest levels
            2. **Check the expiry dates** - Closer dates = more immediate impact
            3. **Compare distance** - Strikes 2-5% away often have best risk/reward
            
            ### What the Labels Mean:
            - 🎯 **AT THE MONEY**: Price is very close - expect high volatility
            - ✅ **NEAR MONEY**: Good balance of probability and profit potential
            - 🚀 **OUT OF MONEY**: Cheaper but riskier - needs bigger move
            - 💰 **IN THE MONEY**: Already profitable if you bought earlier
            """)
        
        # Trading recommendation section
        st.subheader("💰 Main Trading Targets")
        
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
        
        # Key trading levels - CLEAR and SIMPLE
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Strongest CALL Strike")
            if not max_positive_dollar.empty:
                row = max_positive_dollar.iloc[0]
                distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                st.metric(
                    label=f"${row['strike']:.2f} Strike",
                    value=f"{distance:+.1f}% away",
                    delta=f"{row['days_to_exp']} days"
                )
                st.caption(f"📅 Expiry: {row['expiry']}")
                st.caption(f"💪 Strength: {format_large_number(abs(row['signed_notional_gamma']))}")
                if distance > 0:
                    st.success(f"📈 Need {abs(distance):.1f}% UPWARD move")
                    st.caption("✅ OUT-OF-MONEY - buy if bullish")
                else:
                    st.info(f"📍 {abs(distance):.1f}% IN-THE-MONEY")
                    st.caption("💰 Already profitable")
            else:
                st.info("No significant call activity")
        
        with col2:
            st.markdown("### 🎯 Strongest PUT Strike")
            if not max_negative_dollar.empty:
                row = max_negative_dollar.iloc[0]
                distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                st.metric(
                    label=f"${row['strike']:.2f} Strike",
                    value=f"{distance:+.1f}% away",
                    delta=f"{row['days_to_exp']} days"
                )
                st.caption(f"📅 Expiry: {row['expiry']}")
                st.caption(f"💪 Strength: {format_large_number(abs(row['signed_notional_gamma']))}")
                if distance < 0:
                    st.error(f"📉 Need {abs(distance):.1f}% DOWNWARD move")
                    st.caption("✅ OUT-OF-MONEY - buy if bearish")
                else:
                    st.info(f"📍 {abs(distance):.1f}% IN-THE-MONEY")
                    st.caption("💰 Already profitable")
                if debug_mode:
                    st.caption(f"Debug: OI={row['open_interest']:,.0f}, Gamma={row['gamma']:.4f}")
            else:
                st.info("No significant put activity")
        
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
        
        # Top strikes by expiry - ACTIONABLE trading plan
        st.markdown("---")
        st.subheader("📅 Top Strikes to Trade by Expiry Date")
        
        if not all_gamma.empty:
            # Get next 3 expiries
            expiries = sorted(all_gamma['expiry'].unique())[:3]
            
            exp_cols = st.columns(3)
            for idx, expiry in enumerate(expiries):
                exp_data = all_gamma[all_gamma['expiry'] == expiry]
                days = exp_data['days_to_exp'].iloc[0]
                
                # Get top call and put for this expiry
                exp_calls = exp_data[exp_data['option_type'] == 'Call']
                exp_puts = exp_data[exp_data['option_type'] == 'Put']
                
                top_call = exp_calls.nlargest(1, 'notional_gamma')
                top_put = exp_puts.nlargest(1, 'notional_gamma')
                
                with exp_cols[idx]:
                    st.markdown(f"### {expiry}")
                    st.caption(f"⏰ {days} days away")
                    
                    # Show top call
                    if not top_call.empty:
                        call_row = top_call.iloc[0]
                        call_distance = ((call_row['strike'] - underlying_price) / underlying_price) * 100
                        st.markdown(f"**🟢 CALL: ${call_row['strike']:.2f}**")
                        st.caption(f"📈 {abs(call_distance):.1f}% {'up' if call_distance > 0 else 'ITM'}")
                        st.caption(f"💪 {format_large_number(abs(call_row['signed_notional_gamma']))}")
                        st.caption(f"OI: {call_row['open_interest']:,.0f}")
                    else:
                        st.caption("No calls")
                    
                    st.markdown("---")
                    
                    # Show top put
                    if not top_put.empty:
                        put_row = top_put.iloc[0]
                        put_distance = ((put_row['strike'] - underlying_price) / underlying_price) * 100
                        st.markdown(f"**🔴 PUT: ${put_row['strike']:.2f}**")
                        st.caption(f"📉 {abs(put_distance):.1f}% {'down' if put_distance < 0 else 'ITM'}")
                        st.caption(f"💪 {format_large_number(abs(put_row['signed_notional_gamma']))}")
                        st.caption(f"OI: {put_row['open_interest']:,.0f}")
                    else:
                        st.caption("No puts")
        
        st.markdown("---")
        
        # Detailed strike list
        st.subheader("🎯 All Key Strikes (Top 20)")
        
        # Filter by option type instead of gamma sign
        max_positive = calls_data.nlargest(20, 'notional_gamma') if not calls_data.empty else pd.DataFrame()
        max_negative = puts_data.nlargest(20, 'notional_gamma') if not puts_data.empty else pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### � CALL Strikes (Bullish Targets)")
            if not max_positive.empty:
                for idx, (_, row) in enumerate(max_positive.iterrows(), 1):
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    
                    # Simple recommendation
                    if abs(distance) < 2:
                        recommendation = "🎯 AT THE MONEY - High activity expected"
                    elif 0 < distance < 5:
                        recommendation = "✅ NEAR MONEY - Good risk/reward"
                    elif distance > 5:
                        recommendation = "🚀 OUT OF MONEY - Cheaper, riskier"
                    else:
                        recommendation = "💰 IN THE MONEY - Already profitable"
                    
                    card_html = f"""
                    <div class="gamma-level-card resistance">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <strong style="font-size: 1.15em;">#{idx} ${row['strike']:.2f} Call</strong>
                            <span style="color: #28a745; font-weight: bold; font-size: 1.1em;">{distance:+.1f}%</span>
                        </div>
                        <div style="font-size: 0.95em; color: #2c3e50; margin-bottom: 5px; font-weight: 500;">
                            {recommendation}
                        </div>
                        <div style="font-size: 0.88em; color: #495057;">
                            📅 {row['expiry']} ({row['days_to_exp']}d) • � {format_large_number(abs(row['signed_notional_gamma']))}
                        </div>
                        <div style="font-size: 0.82em; color: #6c757d; margin-top: 3px;">
                            OI: {row['open_interest']:,.0f} | Vol: {row['volume']:,.0f}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("💡 No significant call activity - consider puts instead")
        
        with col2:
            st.markdown("### � PUT Strikes (Bearish Targets)")
            if not max_negative.empty:
                for idx, (_, row) in enumerate(max_negative.iterrows(), 1):
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    
                    # Simple recommendation
                    if abs(distance) < 2:
                        recommendation = "🎯 AT THE MONEY - High activity expected"
                    elif -5 < distance < 0:
                        recommendation = "✅ NEAR MONEY - Good risk/reward"
                    elif distance < -5:
                        recommendation = "🚀 OUT OF MONEY - Cheaper, riskier"
                    else:
                        recommendation = "💰 IN THE MONEY - Already profitable"
                    
                    card_html = f"""
                    <div class="gamma-level-card support">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <strong style="font-size: 1.15em;">#{idx} ${row['strike']:.2f} Put</strong>
                            <span style="color: #dc3545; font-weight: bold; font-size: 1.1em;">{distance:+.1f}%</span>
                        </div>
                        <div style="font-size: 0.95em; color: #2c3e50; margin-bottom: 5px; font-weight: 500;">
                            {recommendation}
                        </div>
                        <div style="font-size: 0.88em; color: #495057;">
                            📅 {row['expiry']} ({row['days_to_exp']}d) • � {format_large_number(abs(row['signed_notional_gamma']))}
                        </div>
                        <div style="font-size: 0.82em; color: #6c757d; margin-top: 3px;">
                            OI: {row['open_interest']:,.0f} | Vol: {row['volume']:,.0f}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("💡 No significant put activity - consider calls instead")
        
        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Expiries Scanned: {num_expiries}")

if __name__ == "__main__":
    main()
