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

# Add the project root to Python path
sys.path.append('.')

from src.api.schwab_client import SchwabClient

# Configure Streamlit page
st.set_page_config(
    page_title="Max Gamma Strike Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .high-gamma {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stock-header {
        background-color: #343a40;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 1.5em;
        font-weight: bold;
    }
    .gamma-strike {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .call-option {
        border-left: 5px solid #28a745;
    }
    .put-option {
        border-left: 5px solid #dc3545;
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
        if 'underlying' in options_data and options_data['underlying']:
            options_underlying_price = options_data['underlying'].get('last', 0)
            if options_underlying_price and options_underlying_price > 0:
                underlying_price = options_underlying_price
        
        # Final check - if still zero, try to estimate from strike prices
        if underlying_price == 0 or underlying_price == 100.0:
            estimated_price = estimate_underlying_from_strikes(options_data)
            if estimated_price:
                underlying_price = estimated_price
        
        return options_data, underlying_price
        
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
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
                
                # OFFICIAL PROFESSIONAL NET GEX FORMULA
                # Net GEX_K = Γ × 100 × OI × S² × 0.01
                if underlying_price > 0:
                    notional_gamma = gamma * 100 * open_interest * underlying_price * underlying_price * 0.01
                else:
                    notional_gamma = gamma * 100 * open_interest * 100  # Fallback
                
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
                    'notional_gamma': abs(notional_gamma),  # Use absolute value for ranking
                    'signed_notional_gamma': notional_gamma,  # Keep signed value for display
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
        top_3 = df.head(20)
        
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
    """Display a formatted card for a gamma strike"""
    
    option_class = "call-option" if row['option_type'] == 'Call' else "put-option"
    
    # Determine if ITM or OTM
    if row['option_type'] == 'Call':
        status = "ITM" if row['strike'] < underlying_price else "OTM"
    else:
        status = "ITM" if row['strike'] > underlying_price else "OTM"
    
    # Color code the gamma value
    gamma_color = "#28a745" if row['signed_notional_gamma'] > 0 else "#dc3545"
    
    html = f"""
    <div class="gamma-strike {option_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h3 style="margin: 0; font-size: 1.2em;">#{rank} - ${row['strike']:.2f} {row['option_type']}</h3>
                <p style="margin: 5px 0; font-size: 0.9em;">Expires: {row['expiry']} ({row['days_to_exp']} days) | {status}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5em; font-weight: bold; color: {gamma_color};">
                    {format_large_number(row['signed_notional_gamma'])}
                </div>
                <div style="font-size: 0.8em;">Notional Gamma</div>
            </div>
        </div>
        <div style="display: flex; gap: 20px; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
            <div><strong>Gamma:</strong> {row['gamma']:.4f}</div>
            <div><strong>Delta:</strong> {row['delta']:.3f}</div>
            <div><strong>OI:</strong> {row['open_interest']:,}</div>
            <div><strong>Volume:</strong> {row['volume']:,}</div>
            <div><strong>IV:</strong> {row['implied_volatility']:.1f}%</div>
            <div><strong>Moneyness:</strong> {row['moneyness']:+.1f}%</div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def main():
    # Header
    st.title("🎯 Max Gamma Strike Scanner")
    st.markdown("Scan multiple stocks to find the highest gamma strikes across expiration dates")
    
    # Sidebar controls
    st.sidebar.header("Scanner Settings")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed error messages")
    
    # Symbol input - allow multiple symbols
    symbols_input = st.sidebar.text_input(
        "Symbols (comma-separated)", 
        value="AAPL, NVDA, TSLA",
        help="Enter stock symbols separated by commas"
    )
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Number of expiries to scan
    num_expiries = st.sidebar.selectbox(
        "Number of Expiries to Scan", 
        [3, 4, 5, 6, 7, 8, 10], 
        index=2
    )
    
    # Number of top strikes to show per symbol
    top_n = st.sidebar.selectbox(
        "Top N Strikes per Symbol",
        [3, 5, 10],
        index=0
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
    
    # Summary section
    st.header("📊 Summary")
    
    cols = st.columns(len(symbols))
    for idx, symbol in enumerate(symbols):
        with cols[idx]:
            result = all_results[symbol]
            if not result['error'] and not result['top_strikes'].empty:
                underlying_price = result['underlying_price']
                top_gamma = result['top_strikes'].iloc[0]
                
                st.metric(
                    symbol,
                    f"${underlying_price:.2f}",
                    help="Current Price"
                )
                st.markdown(f"**Max Gamma Strike:**")
                st.markdown(f"${top_gamma['strike']:.2f} {top_gamma['option_type']}")
                st.markdown(f"{format_large_number(top_gamma['signed_notional_gamma'])}")
            else:
                st.metric(symbol, "N/A", help="Data unavailable")
    
    # Comparison chart
    st.header("📈 Comparison Chart")
    comparison_chart = create_gamma_comparison_chart(all_results)
    st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Detailed results for each symbol
    st.header("🔍 Detailed Results")
    
    for symbol in symbols:
        result = all_results[symbol]
        
        if result['error'] or result['top_strikes'].empty:
            st.warning(f"No data available for {symbol}")
            continue
        
        underlying_price = result['underlying_price']
        top_strikes = result['top_strikes']
        
        # Symbol header
        st.markdown(f'<div class="stock-header">{symbol} - Current Price: ${underlying_price:.2f}</div>', unsafe_allow_html=True)
        
        # Display top strikes
        for rank, (_, row) in enumerate(top_strikes.iterrows(), 1):
            display_gamma_strike_card(row, rank, underlying_price)
        
        # Add expandable detailed table
        with st.expander(f"📋 Show Full Data Table for {symbol}"):
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
            display_df['Gamma'] = display_df['Gamma'].apply(lambda x: f"{x:.4f}")
            display_df['Delta'] = display_df['Delta'].apply(lambda x: f"{x:.3f}")
            display_df['Vega'] = display_df['Vega'].apply(lambda x: f"{x:.3f}")
            display_df['OI'] = display_df['OI'].apply(lambda x: f"{x:,.0f}")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            display_df['IV %'] = display_df['IV %'].apply(lambda x: f"{x:.1f}%")
            display_df['Moneyness %'] = display_df['Moneyness %'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Expiries Scanned: {num_expiries}")

if __name__ == "__main__":
    main()
