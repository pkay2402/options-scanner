#!/usr/bin/env python3
"""
SPY/SPX/QQQ Positioning Dashboard
Analyzes options positioning, key levels, and dealer hedging across next 3 expiries
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
    page_title="Index Positioning Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .support-level {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .resistance-level {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .current-price {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .gamma-wall {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .expiry-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    .level-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .call-heavy {
        border-left: 4px solid #28a745;
    }
    .put-heavy {
        border-left: 4px solid #dc3545;
    }
    .balanced {
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def estimate_underlying_from_strikes(options_data):
    """Estimate underlying price from ATM options strikes"""
    try:
        if not options_data or 'callExpDateMap' not in options_data:
            return None
        
        exp_dates = list(options_data['callExpDateMap'].keys())
        if not exp_dates:
            return None
        
        first_exp = options_data['callExpDateMap'][exp_dates[0]]
        strikes = [float(s) for s in first_exp.keys()]
        
        if strikes:
            strike_data = []
            for strike_str, contracts in first_exp.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)
                    strike = float(strike_str)
                    activity = volume + open_interest * 0.1
                    strike_data.append((strike, activity))
            
            if strike_data:
                strike_data.sort(key=lambda x: x[1], reverse=True)
                most_active_strike = strike_data[0][0]
                
                if 50 < most_active_strike < 2000:
                    return most_active_strike
            
            strikes.sort()
            mid_index = len(strikes) // 2
            return strikes[mid_index]
        
        return None
    except:
        return None

@st.cache_data(ttl=300)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_options_data(symbol):
    """Fetch options data"""
    try:
        client = SchwabClient()
        
        # Get quote data
        quote_data = client.get_quote(symbol)
        if not quote_data:
            st.error(f"Failed to get quote data for {symbol}")
            return None, None
        
        # Extract underlying price
        underlying_price = None
        if symbol in quote_data:
            underlying_price = quote_data[symbol].get('lastPrice', 0)
            if underlying_price == 0:
                underlying_price = quote_data[symbol].get('mark', 0)
                if underlying_price == 0:
                    underlying_price = quote_data[symbol].get('bidPrice', 0)
                if underlying_price == 0:
                    underlying_price = quote_data[symbol].get('askPrice', 0)
        else:
            st.error(f"Symbol {symbol} not found in quote response")
            return None, None
        
        # Get options chain (reduced strike count for faster loading)
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50  # Reduced from 100 for better performance
        )
        
        if not options_data:
            st.error(f"No options data available for {symbol}")
            return None, None
        
        # Check structure
        if 'callExpDateMap' not in options_data and 'putExpDateMap' not in options_data:
            st.error(f"Invalid options data structure for {symbol}")
            return None, None
        
        # Try to get underlying price from options data if available
        if 'underlying' in options_data and options_data['underlying']:
            options_underlying_price = options_data['underlying'].get('last', 0)
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
                    st.info(f"✓ Using live market price for {symbol}: ${underlying_price:.2f}")
            except:
                # Last resort - estimate from strike prices
                estimated_price = estimate_underlying_from_strikes(options_data)
                if estimated_price:
                    underlying_price = estimated_price
        
        if not underlying_price or underlying_price == 0:
            st.error(f"Could not determine valid price for {symbol}")
            return None, None
        
        return options_data, underlying_price
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        # Try yfinance as fallback
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            underlying_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if underlying_price:
                st.warning(f"⚠️ API error - using live market price for {symbol}: ${underlying_price:.2f}")
                # Still try to get options from Schwab
                client = SchwabClient()
                options_data = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
                if options_data:
                    return options_data, underlying_price
        except:
            pass
        
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
        return None, None

def analyze_expiry_positioning(options_data, underlying_price, expiry):
    """Analyze positioning for a specific expiry"""
    
    results = {
        'calls': [],
        'puts': []
    }
    
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data or expiry not in options_data[option_type]:
            continue
        
        strikes_data = options_data[option_type][expiry]
        option_side = 'calls' if 'call' in option_type else 'puts'
        
        for strike_str, contracts in strikes_data.items():
            if not contracts:
                continue
            
            strike = float(strike_str)
            contract = contracts[0]
            
            gamma = contract.get('gamma', 0)
            delta = contract.get('delta', 0)
            vega = contract.get('vega', 0)
            theta = contract.get('theta', 0)
            volume = contract.get('totalVolume', 0)
            open_interest = contract.get('openInterest', 0)
            bid = contract.get('bid', 0)
            ask = contract.get('ask', 0)
            implied_vol = contract.get('volatility', 0) * 100
            
            # Estimate gamma if not provided
            if gamma == 0 and vega > 0 and implied_vol > 0:
                gamma = vega / (underlying_price * implied_vol / 100)
            
            # Calculate notional exposures
            notional_gamma = gamma * open_interest * 100 * underlying_price
            notional_delta = delta * open_interest * 100 * underlying_price
            notional_vega = vega * open_interest * 100
            
            results[option_side].append({
                'strike': strike,
                'gamma': gamma,
                'delta': delta,
                'vega': vega,
                'theta': theta,
                'volume': volume,
                'open_interest': open_interest,
                'notional_gamma': notional_gamma,
                'notional_delta': notional_delta,
                'notional_vega': notional_vega,
                'implied_vol': implied_vol,
                'distance_pct': ((strike - underlying_price) / underlying_price) * 100
            })
    
    return results

def calculate_key_levels(df_calls, df_puts, underlying_price):
    """Calculate key support/resistance levels based on gamma and open interest"""
    
    # Combine calls and puts
    df_all = pd.concat([df_calls.assign(type='call'), df_puts.assign(type='put')])
    
    # Group by strike
    strike_summary = df_all.groupby('strike').agg({
        'notional_gamma': 'sum',
        'notional_delta': 'sum',
        'open_interest': 'sum',
        'volume': 'sum'
    }).reset_index()
    
    strike_summary['abs_gamma'] = strike_summary['notional_gamma'].abs()
    strike_summary['distance'] = abs(strike_summary['strike'] - underlying_price)
    
    # Find gamma walls (highest gamma concentrations)
    gamma_walls = strike_summary.nlargest(5, 'abs_gamma')
    
    # Find support (below current price with high put OI)
    below_price = df_puts[df_puts['strike'] < underlying_price].copy()
    if not below_price.empty:
        support_levels = below_price.nlargest(3, 'open_interest')
    else:
        support_levels = pd.DataFrame()
    
    # Find resistance (above current price with high call OI)
    above_price = df_calls[df_calls['strike'] > underlying_price].copy()
    if not above_price.empty:
        resistance_levels = above_price.nlargest(3, 'open_interest')
    else:
        resistance_levels = pd.DataFrame()
    
    return {
        'gamma_walls': gamma_walls,
        'support': support_levels,
        'resistance': resistance_levels,
        'strike_summary': strike_summary
    }

def calculate_dealer_positioning(df_calls, df_puts):
    """Calculate dealer positioning metrics"""
    
    total_call_gamma = df_calls['notional_gamma'].sum()
    total_put_gamma = df_puts['notional_gamma'].sum()
    net_gamma = total_call_gamma + total_put_gamma
    
    total_call_delta = df_calls['notional_delta'].sum()
    total_put_delta = df_puts['notional_delta'].sum()
    net_delta = total_call_delta + total_put_delta
    
    total_call_vega = df_calls['notional_vega'].sum()
    total_put_vega = df_puts['notional_vega'].sum()
    net_vega = total_call_vega + total_put_vega
    
    # Put/Call ratio by OI
    total_call_oi = df_calls['open_interest'].sum()
    total_put_oi = df_puts['open_interest'].sum()
    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    # Determine positioning
    if net_gamma > 0:
        gamma_position = "Long Gamma (Dealers Short)"
    else:
        gamma_position = "Short Gamma (Dealers Long)"
    
    if pc_ratio > 1.0:
        sentiment = "Bearish (Put Heavy)"
    elif pc_ratio < 0.7:
        sentiment = "Bullish (Call Heavy)"
    else:
        sentiment = "Neutral"
    
    return {
        'net_gamma': net_gamma,
        'net_delta': net_delta,
        'net_vega': net_vega,
        'total_call_gamma': total_call_gamma,
        'total_put_gamma': total_put_gamma,
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi,
        'pc_ratio': pc_ratio,
        'gamma_position': gamma_position,
        'sentiment': sentiment
    }

def create_gamma_profile(strike_summary, underlying_price, title="Gamma Profile"):
    """Create gamma profile chart"""
    
    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in strike_summary['notional_gamma']]
    
    fig.add_trace(go.Bar(
        x=strike_summary['strike'],
        y=strike_summary['notional_gamma'],
        marker_color=colors,
        name='Net Gamma',
        hovertemplate='Strike: $%{x}<br>Gamma: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="blue",
        line_width=3,
        annotation_text=f"Spot: ${underlying_price:.2f}"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Strike Price",
        yaxis_title="Net Gamma Exposure",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_oi_distribution(df_calls, df_puts, underlying_price):
    """Create open interest distribution chart"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Open Interest', 'Put Open Interest'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Calls
    fig.add_trace(
        go.Bar(
            x=df_calls['strike'],
            y=df_calls['open_interest'],
            marker_color='green',
            name='Calls',
            hovertemplate='Strike: $%{x}<br>OI: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Puts
    fig.add_trace(
        go.Bar(
            x=df_puts['strike'],
            y=df_puts['open_interest'],
            marker_color='red',
            name='Puts',
            hovertemplate='Strike: $%{x}<br>OI: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add current price lines
    for col in [1, 2]:
        fig.add_vline(
            x=underlying_price,
            line_dash="dash",
            line_color="blue",
            line_width=2,
            row=1, col=col
        )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def format_number(num):
    """Format large numbers"""
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.0f}K"
    else:
        return f"${num:.0f}"

def main():
    st.title("📈 SPY/SPX/QQQ Positioning Dashboard")
    st.markdown("Real-time options positioning analysis and key level identification")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    symbol = st.sidebar.selectbox(
        "Select Index",
        ["SPY", "IWM", "QQQ"],
        index=0
    )
    
    num_expiries = st.sidebar.selectbox(
        "Number of Expiries",
        [1, 2, 3, 4, 5],
        index=2
    )
    
    strike_range_pct = st.sidebar.slider(
        "Strike Range (% from spot)",
        1, 20, 10,
        help="Filter strikes within this % range from current price"
    )
    
    show_detailed_tables = st.sidebar.checkbox("Show Detailed Tables", value=False)
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Fetch data (with timeout handling)
    with st.spinner(f"Fetching options data for {symbol}..."):
        try:
            options_data, underlying_price = get_options_data(symbol)
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.info("This might be due to API rate limits or timeout. Try refreshing in a moment.")
            return
    
    if not options_data or not underlying_price:
        st.error(f"Failed to fetch data for {symbol}")
        return
    
    # Current price banner
    st.markdown(f'<div class="big-metric current-price">{symbol}: ${underlying_price:.2f}</div>', 
                unsafe_allow_html=True)
    
    # Get available expiries
    exp_dates = list(options_data.get('callExpDateMap', {}).keys())[:num_expiries]
    
    if not exp_dates:
        st.error("No expiration dates available")
        return
    
    st.markdown("---")
    
    # Analyze each expiry
    expiry_data = {}
    all_positioning = []
    
    for expiry in exp_dates:
        expiry_date = expiry.split(':')[0]
        
        # Parse expiry date
        try:
            exp_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            days_to_exp = (exp_dt - datetime.now()).days
            expiry_label = f"{expiry_date} ({days_to_exp} DTE)"
        except:
            expiry_label = expiry_date
            days_to_exp = 0
        
        # Analyze positioning
        positioning = analyze_expiry_positioning(options_data, underlying_price, expiry)
        
        df_calls = pd.DataFrame(positioning['calls'])
        df_puts = pd.DataFrame(positioning['puts'])
        
        if df_calls.empty and df_puts.empty:
            continue
        
        # Filter by strike range
        price_range = (
            underlying_price * (1 - strike_range_pct/100),
            underlying_price * (1 + strike_range_pct/100)
        )
        
        if not df_calls.empty:
            df_calls = df_calls[
                (df_calls['strike'] >= price_range[0]) & 
                (df_calls['strike'] <= price_range[1])
            ]
        
        if not df_puts.empty:
            df_puts = df_puts[
                (df_puts['strike'] >= price_range[0]) & 
                (df_puts['strike'] <= price_range[1])
            ]
        
        # Calculate metrics
        key_levels = calculate_key_levels(df_calls, df_puts, underlying_price)
        dealer_pos = calculate_dealer_positioning(df_calls, df_puts)
        
        expiry_data[expiry_label] = {
            'df_calls': df_calls,
            'df_puts': df_puts,
            'key_levels': key_levels,
            'dealer_pos': dealer_pos,
            'days_to_exp': days_to_exp
        }
        
        all_positioning.append(dealer_pos)
    
    # Overall summary
    st.header("📊 Overall Positioning Summary")
    
    cols = st.columns(4)
    
    total_net_gamma = sum([p['net_gamma'] for p in all_positioning])
    total_pc_ratio = np.mean([p['pc_ratio'] for p in all_positioning])
    
    with cols[0]:
        st.metric(
            "Net Gamma Exposure",
            format_number(total_net_gamma),
            help="Positive = Long Gamma (Dealers Short)"
        )
    
    with cols[1]:
        st.metric(
            "Put/Call Ratio",
            f"{total_pc_ratio:.2f}",
            help="Based on Open Interest. >1 = Bearish, <0.7 = Bullish"
        )
    
    with cols[2]:
        gamma_bias = "Long Gamma" if total_net_gamma > 0 else "Short Gamma"
        st.metric(
            "Market Bias",
            gamma_bias
        )
    
    with cols[3]:
        if total_pc_ratio > 1.0:
            sentiment = "🔴 Bearish"
        elif total_pc_ratio < 0.7:
            sentiment = "🟢 Bullish"
        else:
            sentiment = "🟡 Neutral"
        st.metric("Sentiment", sentiment)
    
    st.markdown("---")
    
    # Display each expiry
    for expiry_label, data in expiry_data.items():
        st.header(f"📅 {expiry_label}")
        
        df_calls = data['df_calls']
        df_puts = data['df_puts']
        key_levels = data['key_levels']
        dealer_pos = data['dealer_pos']
        
        # Expiry metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Net Gamma",
                format_number(dealer_pos['net_gamma']),
                delta=dealer_pos['gamma_position']
            )
        
        with col2:
            st.metric(
                "P/C Ratio",
                f"{dealer_pos['pc_ratio']:.2f}",
                delta=dealer_pos['sentiment']
            )
        
        with col3:
            st.metric(
                "Total Call OI",
                f"{dealer_pos['total_call_oi']:,.0f}"
            )
        
        with col4:
            st.metric(
                "Total Put OI",
                f"{dealer_pos['total_put_oi']:,.0f}"
            )
        
        # Key levels
        st.subheader("🎯 Key Levels to Watch")
        
        level_cols = st.columns(3)
        
        # Gamma Walls
        with level_cols[0]:
            st.markdown("### 🔥 Gamma Walls")
            for idx, row in key_levels['gamma_walls'].head(3).iterrows():
                gamma_val = row['notional_gamma']
                gamma_color = "green" if gamma_val > 0 else "red"
                st.markdown(f"""
                <div class="level-card">
                    <strong style="font-size: 1.2em;">${row['strike']:.2f}</strong><br>
                    <span style="color: {gamma_color}; font-weight: bold;">
                        {format_number(abs(gamma_val))}
                    </span><br>
                    <small>OI: {row['open_interest']:,.0f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Support Levels
        with level_cols[1]:
            st.markdown("### 🟢 Support Levels")
            if not key_levels['support'].empty:
                for idx, row in key_levels['support'].iterrows():
                    st.markdown(f"""
                    <div class="level-card put-heavy">
                        <strong style="font-size: 1.2em;">${row['strike']:.2f}</strong><br>
                        <span>OI: {row['open_interest']:,.0f}</span><br>
                        <small>Gamma: {format_number(abs(row['notional_gamma']))}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strong support levels identified")
        
        # Resistance Levels
        with level_cols[2]:
            st.markdown("### 🔴 Resistance Levels")
            if not key_levels['resistance'].empty:
                for idx, row in key_levels['resistance'].iterrows():
                    st.markdown(f"""
                    <div class="level-card call-heavy">
                        <strong style="font-size: 1.2em;">${row['strike']:.2f}</strong><br>
                        <span>OI: {row['open_interest']:,.0f}</span><br>
                        <small>Gamma: {format_number(abs(row['notional_gamma']))}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strong resistance levels identified")
        
        # Charts
        st.subheader("📈 Gamma & Open Interest Profile")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            gamma_chart = create_gamma_profile(
                key_levels['strike_summary'], 
                underlying_price,
                f"Gamma Profile - {expiry_label}"
            )
            st.plotly_chart(gamma_chart, use_container_width=True)
        
        with chart_col2:
            oi_chart = create_oi_distribution(df_calls, df_puts, underlying_price)
            st.plotly_chart(oi_chart, use_container_width=True)
        
        # Detailed tables (optional)
        if show_detailed_tables:
            with st.expander(f"📋 Detailed Strike Data - {expiry_label}"):
                tab1, tab2 = st.tabs(["Calls", "Puts"])
                
                with tab1:
                    if not df_calls.empty:
                        display_calls = df_calls.sort_values('strike').copy()
                        display_calls = display_calls[[
                            'strike', 'open_interest', 'volume', 'notional_gamma',
                            'delta', 'gamma', 'vega', 'implied_vol', 'distance_pct'
                        ]]
                        display_calls.columns = [
                            'Strike', 'OI', 'Volume', 'Notional Gamma',
                            'Delta', 'Gamma', 'Vega', 'IV %', 'Distance %'
                        ]
                        st.dataframe(display_calls, use_container_width=True)
                
                with tab2:
                    if not df_puts.empty:
                        display_puts = df_puts.sort_values('strike').copy()
                        display_puts = display_puts[[
                            'strike', 'open_interest', 'volume', 'notional_gamma',
                            'delta', 'gamma', 'vega', 'implied_vol', 'distance_pct'
                        ]]
                        display_puts.columns = [
                            'Strike', 'OI', 'Volume', 'Notional Gamma',
                            'Delta', 'Gamma', 'Vega', 'IV %', 'Distance %'
                        ]
                        st.dataframe(display_puts, use_container_width=True)
        
        st.markdown("---")
    
    # Footer
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {symbol} @ ${underlying_price:.2f}")

if __name__ == "__main__":
    main()
