#!/usr/bin/env python3
"""
Stock Option Finder
Discover which strikes and expiries have the most market-moving potential
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
from src.utils.dark_pool import get_7day_dark_pool_sentiment, format_dark_pool_display

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Option Finder",
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
                # Gamma Exposure (GEX) Convention:
                # - Gamma from API is buyer's gamma (always positive for long options)
                # - Dealers are SHORT options to customers
                # - For dealer gamma exposure: CALLS get negative sign, PUTS get positive sign
                # - Net GEX = -Call_Gamma + Put_Gamma
                # 
                # Why? As price rises:
                # - Dealers short calls need to BUY more stock (negative gamma = destabilizing)
                # - Dealers short puts need to SELL stock (positive gamma = stabilizing)
                
                # Apply sign convention based on option type
                is_call = 'call' in option_type.lower()
                dealer_gamma_sign = -1 if is_call else 1
                
                # 1. OFFICIAL PROFESSIONAL NET GEX FORMULA
                # Source: Standard industry formula for Net Gamma Exposure
                # Net GEX_K = Σ(Γ_c × 100 × OI_c × S² × 0.01) + Σ(-Γ_p × 100 × OI_p × S² × 0.01)
                # 
                # Key components:
                # - Γ: Gamma (per share, from Black-Scholes)
                # - 100: Contract multiplier (shares per contract)
                # - OI: Open Interest (number of contracts)
                # - S² × 0.01: Dollar conversion for 1% move in underlying
                # - Sign: +1 for calls (stabilizing), -1 for puts (destabilizing from dealer perspective)
                
                if underlying_price > 0:
                    # Official formula: Γ × 100 × OI × S² × 0.01
                    dollar_gex = gamma * 100 * open_interest * underlying_price * underlying_price * 0.01
                    
                    # Professional sign convention for Net GEX
                    if is_call:
                        signed_gex = dollar_gex  # Calls: positive contribution
                    else:
                        signed_gex = -dollar_gex  # Puts: negative contribution
                else:
                    # Fallback for edge cases
                    dollar_gex = gamma * 100 * open_interest * 100  # Simplified
                    signed_gex = dollar_gex if is_call else -dollar_gex
                
                notional_gamma = abs(dollar_gex)  # For ranking purposes
                
                # 2. Gamma exposure (shares that need to be hedged) - traditional dealer perspective
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
                    'notional_gamma': notional_gamma,  # Raw GEX value for ranking
                    'signed_notional_gamma': signed_gex,  # Signed GEX for Net GEX calculation
                    'gamma_exposure_shares': gamma_exposure_shares,  # Traditional dealer shares to hedge
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

def create_professional_netgex_heatmap(df_gamma, underlying_price, num_expiries=6):
    """Create a professional-style NetGEX heat map similar to institutional tools"""
    
    if df_gamma.empty:
        return None
    
    try:
        # Get unique expiries and strikes
        expiries = sorted(df_gamma['expiry'].unique())[:num_expiries]
        all_strikes = sorted(df_gamma['strike'].unique())
        
        # Filter strikes to a reasonable range around current price
        min_strike = underlying_price * 0.85  # 15% below
        max_strike = underlying_price * 1.15  # 15% above
        
        filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
        
        # If no strikes in range, take the closest ones
        if not filtered_strikes:
            sorted_by_distance = sorted(all_strikes, key=lambda x: abs(x - underlying_price))
            filtered_strikes = sorted(sorted_by_distance[:20])
        
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
        expiry_labels = [exp.split('-')[1] + '-' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
        
        # Create the heat map
        fig = go.Figure(data=go.Heatmap(
            z=heat_data,
            x=expiry_labels,
            y=strike_labels,
            colorscale='RdYlGn',  # Red-Yellow-Green colorscale
            zmid=0,
            showscale=True,
            hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>Net GEX: $%{z:,.0f}<extra></extra>'
        ))
        
        # Find current price position for yellow line
        current_price_y = None
        for i, strike in enumerate(filtered_strikes):
            if abs(strike - underlying_price) <= (underlying_price * 0.025):  # Within 2.5%
                current_price_y = i
                break
        
        # Add current price line
        if current_price_y is not None:
            fig.add_hline(
                y=current_price_y,
                line=dict(color="yellow", width=3),
                annotation_text=f"Current: ${underlying_price:.2f}",
                annotation_position="right"
            )
        
        fig.update_layout(
            title=f"Net GEX Heat Map - ${underlying_price:.2f}",
            xaxis_title="Expiration Date",
            yaxis_title="Strike Price",
            height=500,
            font=dict(size=10)
        )
        
        return fig
        
    except Exception as e:
        return None

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
    st.title("🎯 Stock Option Finder")
    st.caption("💡 Discover which strikes and expiries have the most market-moving potential")
    
    # Move settings to SIDEBAR for cleaner main view
    with st.sidebar:
        st.subheader("⚙️ Scanner Settings")
        
        # Debug mode
        debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show detailed error messages")
        
        # Number of expiries to scan
        num_expiries = st.selectbox(
            "Expiries to Scan", 
            [3, 4, 5, 6, 7, 8, 10], 
            index=2  # Default to 5
        )
        
        # Number of top strikes to show per symbol
        top_n = st.selectbox(
            "Top Strikes",
            [3, 5, 10, 20],
            index=1  # Default to 5
        )
        
        st.markdown("---")
        st.subheader("� Filters")
        
        option_type_filter = st.selectbox(
            "Option Type",
            ["All", "Calls Only", "Puts Only"]
        )
        
        min_open_interest = st.number_input(
            "Min Open Interest",
            min_value=0,
            max_value=10000,
            value=100,
            step=50
        )
        
        moneyness_range = st.slider(
            "Moneyness Range (%)",
            -50, 50, (-20, 20),
            help="Filter strikes by distance from current price"
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Scan Now", use_container_width=True):
            st.cache_data.clear()
    
    # Symbol input - compact, single line
    col1, col2 = st.columns([3, 1])
    with col1:
        default_symbol = st.session_state.get('selected_symbol', 'AMZN')
        symbols_input = st.text_input(
            "Symbols (comma-separated)", 
            value=default_symbol,
            label_visibility="collapsed",
            placeholder="Enter symbols (e.g., AAPL, TSLA, AMZN)"
        )
    with col2:
        st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
        st.caption("Enter symbols ↖️")
    
    # Clear the session state after using it
    if 'selected_symbol' in st.session_state:
        del st.session_state['selected_symbol']
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Main content
    if not symbols:
        st.info("👆 Enter symbols above to start scanning")
        return
    
    # Test API connection - SILENT unless error
    try:
        test_client = SchwabClient()
        test_quote = test_client.get_quote("SPY")
        if not test_quote:
            st.error("❌ API connection failed. Run `python scripts/auth_setup.py` to authenticate.")
            return
    except Exception as e:
        st.error(f"❌ API connection failed: {str(e)}")
        if debug_mode:
            import traceback
            st.code(traceback.format_exc())
        return
    
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
                        st.caption("� Bullish trend")
                    elif underlying_price < ema_50 < ema_200:
                        st.caption("� Bearish trend")
                    else:
                        st.caption("📊 Mixed/Ranging")
            
            # Add Dark Pool Sentiment (7-day)
            try:
                dark_pool_data = get_7day_dark_pool_sentiment(symbols[0])
                display_text, color, icon = format_dark_pool_display(dark_pool_data)
                
                ratio = dark_pool_data['ratio']
                days = dark_pool_data['days_available']
                bought = dark_pool_data['total_bought']
                sold = dark_pool_data['total_sold']
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, {color}20 0%, {color}10 100%);
                    border-left: 4px solid {color};
                    padding: 12px 20px;
                    border-radius: 8px;
                    margin-top: 15px;
                    margin-bottom: 15px;
                ">
                    <div style="font-size: 16px; font-weight: bold; color: {color}; margin-bottom: 5px;">
                        {display_text}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        Bought: {bought:,} | Sold: {sold:,} | Data: {days} days
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                pass  # Silently fail if dark pool data unavailable
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
    
    # ========== DASHBOARD LAYOUT - NO SCROLLING ==========
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
        st.markdown(f'<div class="stock-header">{symbol} - ${underlying_price:.2f}</div>', unsafe_allow_html=True)
        
        # Filter by option type
        calls_data = all_gamma[all_gamma['option_type'] == 'Call']
        puts_data = all_gamma[all_gamma['option_type'] == 'Put']
        
        # Get top strikes
        max_positive_dollar = calls_data.nlargest(1, 'notional_gamma') if not calls_data.empty else pd.DataFrame()
        max_negative_dollar = puts_data.nlargest(1, 'notional_gamma') if not puts_data.empty else pd.DataFrame()
        
        # ========== DASHBOARD TABS ==========
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Top Strikes", "🌊 Flow", "🔥 Heatmap", "� Advanced"])
        
        with tab1:
            # ========== OVERVIEW DASHBOARD (3 COLUMNS) ==========
            col1, col2, col3 = st.columns(3)
            
            # Left: Top CALL
            with col1:
                st.markdown("### 🚀 Top CALL Strike")
                if not max_positive_dollar.empty:
                    row = max_positive_dollar.iloc[0]
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    st.metric(
                        label=f"${row['strike']:.2f}",
                        value=f"{distance:+.1f}% away",
                        delta=f"{row['days_to_exp']}d"
                    )
                    st.caption(f"💪 {format_large_number(abs(row['signed_notional_gamma']))}")
                    st.caption(f"📅 {row['expiry']}")
                    st.caption(f"OI: {row['open_interest']:,.0f}")
                else:
                    st.info("No calls")
            
            # Middle: Top PUT
            with col2:
                st.markdown("### 🎯 Top PUT Strike")
                if not max_negative_dollar.empty:
                    row = max_negative_dollar.iloc[0]
                    distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                    st.metric(
                        label=f"${row['strike']:.2f}",
                        value=f"{distance:+.1f}% away",
                        delta=f"{row['days_to_exp']}d"
                    )
                    st.caption(f"💪 {format_large_number(abs(row['signed_notional_gamma']))}")
                    st.caption(f"📅 {row['expiry']}")
                    st.caption(f"OI: {row['open_interest']:,.0f}")
                else:
                    st.info("No puts")
            
            # Right: Quick Stats
            with col3:
                st.markdown("### 📈 Quick Stats")
                total_call_oi = calls_data['open_interest'].sum() if not calls_data.empty else 0
                total_put_oi = puts_data['open_interest'].sum() if not puts_data.empty else 0
                put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                
                st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
                st.caption(f"Total Call OI: {total_call_oi:,.0f}")
                st.caption(f"Total Put OI: {total_put_oi:,.0f}")
                
                # Sentiment indicator
                if put_call_ratio > 1.2:
                    st.caption("🔴 Bearish Bias")
                elif put_call_ratio < 0.8:
                    st.caption("🟢 Bullish Bias")
                else:
                    st.caption("⚪ Neutral")
        
        with tab2:
            # ========== TOP STRIKES DASHBOARD (2 COLUMNS) ==========
            st.markdown("### 📊 Top 5 Gamma Strikes by Type")
            
            top_5_calls = calls_data.head(5) if not calls_data.empty else pd.DataFrame()
            top_5_puts = puts_data.head(5) if not puts_data.empty else pd.DataFrame()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Calls")
                if not top_5_calls.empty:
                    for idx, (_, row) in enumerate(top_5_calls.iterrows(), 1):
                        distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                        
                        # Create cleaner card-style display
                        strike_html = f"""
                        <div style="background: #f8f9fa; padding: 8px 12px; margin: 6px 0; border-radius: 6px; border-left: 3px solid #28a745;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.05em; color: #212529;">{idx}. ${row['strike']:.2f}</strong>
                                    <span style="color: #28a745; margin-left: 8px; font-weight: 600;">{distance:+.1f}%</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-weight: 600; color: #495057;">{format_large_number(abs(row['signed_notional_gamma']))}</div>
                                    <div style="font-size: 0.85em; color: #6c757d;">{row['days_to_exp']} days</div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(strike_html, unsafe_allow_html=True)
                else:
                    st.info("No call strikes available")
            
            with col2:
                st.markdown("### 🔴 Puts")
                if not top_5_puts.empty:
                    for idx, (_, row) in enumerate(top_5_puts.iterrows(), 1):
                        distance = ((row['strike'] - underlying_price) / underlying_price) * 100
                        
                        # Create cleaner card-style display
                        strike_html = f"""
                        <div style="background: #f8f9fa; padding: 8px 12px; margin: 6px 0; border-radius: 6px; border-left: 3px solid #dc3545;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.05em; color: #212529;">{idx}. ${row['strike']:.2f}</strong>
                                    <span style="color: #dc3545; margin-left: 8px; font-weight: 600;">{distance:+.1f}%</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-weight: 600; color: #495057;">{format_large_number(abs(row['signed_notional_gamma']))}</div>
                                    <div style="font-size: 0.85em; color: #6c757d;">{row['days_to_exp']} days</div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(strike_html, unsafe_allow_html=True)
                else:
                    st.info("No put strikes available")
        
        # ========== OPTIONS FLOW DASHBOARD ==========
        with tab3:
            st.markdown("### 🌊 Latest Options Flow Activity")
            st.caption("💡 Recent large trades and unusual options activity (powered by Flow Scanner)")

            # Import flow scanner logic
            try:
                from flow_scanner import analyze_flow
            except ImportError:
                st.error("Could not import flow scanner. Please check your installation.")
                return

            options_data = result.get('options_data')
            underlying_price = result.get('underlying_price')
            if not options_data or not underlying_price:
                st.warning("Flow data not available for this symbol.")
                return

            # Use flow scanner to get flows
            try:
                flows = analyze_flow(options_data, underlying_price, min_premium=50000, volume_threshold=100)
            except Exception as e:
                st.error(f"Error analyzing flow: {str(e)}")
                flows = []

            if not flows:
                st.info("No significant flow activity detected (minimum $50k premium)")
            else:
                # Sort by premium and show top 7
                flows = sorted(flows, key=lambda x: x['premium'], reverse=True)[:7]
                for idx, flow in enumerate(flows, 1):
                    sentiment = "🟢 Bullish" if flow['type'] == 'CALL' else "🔴 Bearish"
                    distance = ((flow['strike'] - underlying_price) / underlying_price) * 100
                    flow_html = f"""
                    <div style="background: {'#e8f5e9' if flow['type'] == 'CALL' else '#ffebee'}; 
                                padding: 10px; margin: 6px 0; border-radius: 6px; 
                                border-left: 4px solid {'#28a745' if flow['type'] == 'CALL' else '#dc3545'};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.05em;">#{idx} ${flow['strike']:.2f} {flow['type']}</strong>
                                <span style="margin-left: 8px; font-size: 0.9em;">{sentiment}</span>
                            </div>
                            <div style="text-align: right;">
                                <strong style="font-size: 1.1em;">{format_large_number(flow['premium'])}</strong>
                            </div>
                        </div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 4px;">
                            📅 {flow['expiry']} ({flow['days_to_exp']}d) • 
                            📊 Vol: {flow['volume']:,} • OI: {flow['open_interest']:,} • 
                            📍 {distance:+.1f}% from price
                        </div>
                    </div>
                    """
                    st.markdown(flow_html, unsafe_allow_html=True)
        
        # ========== TAB 4: GAMMA HEATMAP ==========
        with tab4:
            st.markdown("### � Gamma Heatmap")
            st.caption("📊 Full gamma exposure matrix across strikes and expiries")
            
            # ========== GAMMA HEATMAP ==========
            gamma_table = create_gamma_table(all_gamma, underlying_price, num_expiries=min(num_expiries, 6))
            
            if not gamma_table.empty:
                def color_gamma_clean(val):
                    if pd.isna(val) or val == 0:
                        return 'background-color: white'
                    elif val > 5e10:
                        return 'background-color: #0D47A1; color: white; font-weight: bold'
                    elif val > 2e10:
                        return 'background-color: #1976D2; color: white'
                    elif val > 1e10:
                        return 'background-color: #42A5F5; color: black'
                    elif val > 0:
                        return 'background-color: #E3F2FD'
                    elif val < -5e10:
                        return 'background-color: #1B5E20; color: white; font-weight: bold'
                    elif val < -2e10:
                        return 'background-color: #388E3C; color: white'
                    elif val < -1e10:
                        return 'background-color: #66BB6A; color: black'
                    else:
                        return 'background-color: #E8F5E9'
                
                def highlight_current_price(row):
                    if row.get('is_current', False):
                        return ['background-color: #FFEB3B; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_df = gamma_table.style.apply(highlight_current_price, axis=1)
                
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        styled_df = styled_df.map(color_gamma_clean, subset=[col])
                
                format_dict = {'Strike': '${:.2f}'}
                for col in gamma_table.columns:
                    if col not in ['Strike', 'is_current']:
                        format_dict[col] = lambda x: format_large_number(x) if x != 0 else ''
                
                st.dataframe(
                    styled_df.format(format_dict).hide(axis='index'),
                    use_container_width=True,
                    height=600
                )
                
                st.info("💡 **Blue** = Call exposure | **Green** = Put exposure | **Yellow row** = Current price")
            else:
                st.warning("No heatmap data available")
        
        # ========== TAB 5: ADVANCED ANALYSIS ==========
        with tab5:
            st.markdown("### 🔬 Advanced Analysis")
            st.caption("📊 Detailed debug information and strike breakdowns")
            
            # ========== DEBUG INFO (COLLAPSED BY DEFAULT) ==========
            if debug_mode:
                st.subheader("🔍 Debug: Max Gamma by Expiry")
                expiries = sorted(all_gamma['expiry'].unique())
                for expiry in expiries[:3]:  # Show first 3 expiries
                    exp_data = all_gamma[all_gamma['expiry'] == expiry]
                    
                    st.write(f"**{expiry}** - Net GEX Analysis:")
                    
                    # Show top strikes by Net GEX (calls - puts combined)
                    strikes_net_gex = {}
                    for strike in exp_data['strike'].unique():
                        strike_data = exp_data[exp_data['strike'] == strike]
                        calls = strike_data[strike_data['option_type'] == 'Call']
                        puts = strike_data[strike_data['option_type'] == 'Put']
                        
                        call_gex = calls['signed_notional_gamma'].sum() if not calls.empty else 0
                        put_gex = puts['signed_notional_gamma'].sum() if not puts.empty else 0
                        net_gex = call_gex + put_gex  # Since puts are already negative
                        
                        if abs(net_gex) > 1e6:  # Only show significant values
                            strikes_net_gex[strike] = {
                                'net_gex': net_gex,
                                'call_gex': call_gex,
                                'put_gex': put_gex,
                                'call_oi': calls['open_interest'].sum() if not calls.empty else 0,
                                'put_oi': puts['open_interest'].sum() if not puts.empty else 0
                            }
                    
                    # Sort by absolute Net GEX and show top 5
                    sorted_strikes = sorted(strikes_net_gex.items(), 
                                          key=lambda x: abs(x[1]['net_gex']), reverse=True)[:5]
                    
                    for strike, data in sorted_strikes:
                        st.write(f"${strike:.2f}: **Net GEX = {data['net_gex']/1e6:.1f}M** "
                                f"(Calls: {data['call_gex']/1e6:.1f}M, Puts: {data['put_gex']/1e6:.1f}M)")
                        st.write(f"    OI - Calls: {data['call_oi']:,}, Puts: {data['put_oi']:,}")
                    
                    st.write("---")
                
                # Aggregated Net GEX across all expiries (likely what professional tool shows)
                st.subheader("🔍 Debug: Aggregated Net GEX (All Expiries)")
                st.write("**This likely matches the professional heat map methodology**")
                
                aggregated_strikes = {}
                for strike in all_gamma['strike'].unique():
                    strike_data = all_gamma[all_gamma['strike'] == strike]
                    calls = strike_data[strike_data['option_type'] == 'Call']
                    puts = strike_data[strike_data['option_type'] == 'Put']
                    
                    call_gex_total = calls['signed_notional_gamma'].sum() if not calls.empty else 0
                    put_gex_total = puts['signed_notional_gamma'].sum() if not puts.empty else 0
                    net_gex_total = call_gex_total + put_gex_total  # Puts already negative
                    
                    if abs(net_gex_total) > 5e6:  # Only show values > $5M
                        aggregated_strikes[strike] = {
                            'net_gex': net_gex_total,
                            'call_gex': call_gex_total,
                            'put_gex': put_gex_total,
                            'num_expiries': len(strike_data['expiry'].unique())
                        }
                
                # Sort by absolute Net GEX
                sorted_agg = sorted(aggregated_strikes.items(), 
                                  key=lambda x: abs(x[1]['net_gex']), reverse=True)[:10]
                
                st.write("**Top 10 Strikes by Aggregated Net GEX (All Expiries):**")
                for strike, data in sorted_agg:
                    color = "🟢" if data['net_gex'] > 0 else "🔴"
                    st.write(f"{color} ${strike:.2f}: **{data['net_gex']/1e6:.1f}M** "
                            f"(across {data['num_expiries']} expiries)")
                    st.write(f"    Calls: {data['call_gex']/1e6:.1f}M, Puts: {data['put_gex']/1e6:.1f}M")
                
                st.write("💡 **Compare these aggregated values with the professional heat map!**")
                
                # NetGEX Calculation Analysis
                st.subheader("🔍 Debug: NetGEX Calculation Analysis")
                st.info("""
                **Professional Dollar Gamma Formula (Industry Standard):**
                
                **Dollar Gamma:** Dollar_Gamma = Γ × S² × Position × Contract_Multiplier
                **Net Gamma:** Net_Gamma = Γ × Position × Contract_Multiplier
            
                
                Where:
                - Γ = Option's Gamma per contract
                - S² = Spot Price squared
                - Position = Signed position (negative for short, positive for long)
                - Contract_Multiplier = 100 (standard options)
                
                **For Market Makers (Dealer Perspective):**
                - Dealers are typically SHORT options to customers
                - Position = -Open_Interest (negative because dealers are short)
                - Call Dollar Gamma = negative (destabilizing for dealers)
                - Put Dollar Gamma = negative (but stabilizing effect when flipped)
                
                **This matches professional trading systems and risk management**
                """)
                
                # Specific debugging for PLTR $210 strike Nov 7
                if symbol == 'PLTR':
                    st.write("**🎯 PLTR $210 Strike Nov 7 Debug Analysis:**")
                    target_strike = 210.0
                    target_expiry = "2025-11-07"
                    
                    pltr_debug = all_gamma[
                        (all_gamma['strike'] == target_strike) & 
                        (all_gamma['expiry'] == target_expiry)
                    ]
                    
                    if not pltr_debug.empty:
                        st.write(f"Found {len(pltr_debug)} option(s) for ${target_strike} {target_expiry}")
                        
                        for idx, row in pltr_debug.iterrows():
                            st.write(f"**{row['option_type']} Option:**")
                            st.write(f"- Gamma: {row['gamma']:.6f}")
                            st.write(f"- Open Interest: {row['open_interest']:,}")
                            st.write(f"- Underlying Price: ${underlying_price:.2f}")
                            
                            # Manual calculation - OFFICIAL PROFESSIONAL FORMULA
                            # Net GEX = Γ × 100 × OI × S² × 0.01
                            oi = row['open_interest']
                            dollar_gex_calc = row['gamma'] * 100 * oi * underlying_price * underlying_price * 0.01
                            
                            st.write(f"**Official Professional Formula:**")
                            st.write(f"Net GEX = Γ × 100 × OI × S² × 0.01")
                            st.write(f"Gamma (Γ): {row['gamma']:.6f}")
                            st.write(f"Contract Multiplier: 100")
                            st.write(f"Open Interest (OI): {oi:,}")
                            st.write(f"Spot Price (S): ${underlying_price:.2f}")
                            st.write(f"S² × 0.01: {underlying_price:.2f}² × 0.01 = {underlying_price * underlying_price * 0.01:,.0f}")
                            st.write(f"**Calculation**: {row['gamma']:.6f} × 100 × {oi:,} × {underlying_price * underlying_price * 0.01:,.0f}")
                            st.write(f"**Result**: {dollar_gex_calc:,.0f}")
                            st.write(f"**Formatted**: {format_large_number(dollar_gex_calc)}")
                            st.write(f"**Our tool shows**: {format_large_number(row['signed_notional_gamma'])}")
                            st.write(f"**Professional shows**: 13.43M")
                            st.write(f"**Difference ratio**: {abs(dollar_gex_calc)/13.43e6:.1f}x")
                            st.write("---")
                    else:
                        st.write(f"❌ No data found for ${target_strike} {target_expiry}")
                        st.write("Available strikes:", sorted(all_gamma['strike'].unique()))
                        st.write("Available expiries:", sorted(all_gamma['expiry'].unique()))
                
                # Show calculation breakdown for a sample strike
                if not all_gamma.empty:
                    sample_row = all_gamma.iloc[0]
                    st.write("**Sample Calculation Breakdown:**")
                    st.write(f"Strike: ${sample_row['strike']:.2f} {sample_row['option_type']}")
                    st.write(f"Gamma: {sample_row['gamma']:.4f}")
                    st.write(f"Open Interest: {sample_row['open_interest']:,}")
                    st.write(f"Underlying Price: ${underlying_price:.2f}")
                    
                    # Show professional calculation
                    dealer_position = -sample_row['open_interest']  # Dealers are short
                    dollar_gamma = sample_row['gamma'] * underlying_price * underlying_price * dealer_position * 100
                    net_gamma = sample_row['gamma'] * dealer_position * 100
                    
                    st.write(f"**Professional Formula**:")
                    st.write(f"Dealer Position: {dealer_position:,} (short to customers)")
                    st.write(f"Net Gamma: {sample_row['gamma']:.4f} × {dealer_position:,} × 100 = {net_gamma:.0f}")
                    st.write(f"Dollar Gamma: {sample_row['gamma']:.4f} × ${underlying_price:.2f}² × {dealer_position:,} × 100")
                    st.write(f"**Result**: {format_large_number(dollar_gamma)}")
                    st.write(f"**Our Tool Shows**: {format_large_number(sample_row['signed_notional_gamma'])}")
            
            # EXPANDER: Top strikes by expiry (collapsed by default)
            with st.expander("📅 View Top Strikes by Expiry Date", expanded=False):
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
            
            # EXPANDER: Detailed strike list (collapsed)
            with st.expander("🎯 View All Key Strikes (Top 10 Each)", expanded=False):
                # Filter by option type and limit to top 10
                max_positive = calls_data.nlargest(10, 'notional_gamma') if not calls_data.empty else pd.DataFrame()
                max_negative = puts_data.nlargest(10, 'notional_gamma') if not puts_data.empty else pd.DataFrame()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🟢 CALL Strikes")
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
                                    📅 {row['expiry']} ({row['days_to_exp']}d) • 💪 {format_large_number(abs(row['signed_notional_gamma']))}
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
                    st.markdown("### 🔴 PUT Strikes")
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
                                    📅 {row['expiry']} ({row['days_to_exp']}d) • 💪 {format_large_number(abs(row['signed_notional_gamma']))}
                                </div>
                                <div style="font-size: 0.82em; color: #6c757d; margin-top: 3px;">
                                    OI: {row['open_interest']:,.0f} | Vol: {row['volume']:,.0f}
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        st.info("💡 No significant put activity - consider calls instead")
            
            # ========== HOW TO READ THIS (MOVED TO END) ==========
            with st.expander("💡 How to Read This", expanded=False):
                st.markdown("""
                **High Gamma Strikes** = Where dealers hedge heavily = Price magnets  
                **Distance %** = How far price needs to move  
                **Days** = Time until expiry (closer = stronger effect)  
                **OI** = Open Interest (higher = more contracts = stronger level)
                """)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbols: {', '.join(symbols)} | Expiries Scanned: {num_expiries}")

if __name__ == "__main__":
    main()
