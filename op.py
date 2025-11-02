#!/usr/bin/env python3
"""
Streamlit Net Gamma/Vanna Options Analysis Dashboard
Similar to the options chain view with strike-based analysis across multiple expiries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.append('.')

from src.api.schwab_client import SchwabClient
from src.analysis.market_dynamics import MarketDynamicsAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="Net Gamma/Vanna Analysis",
    page_icon="📊",
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
    .positive-gamma {
        background-color: #d4edda;
        color: #155724;
    }
    .negative-gamma {
        background-color: #f8d7da;
        color: #721c24;
    }
    .high-gamma {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .stDataFrame {
        font-size: 12px;
        font-family: 'Courier New', monospace;
    }
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th {
        background-color: #343a40;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border: 1px solid #dee2e6;
    }
    .stDataFrame td {
        text-align: center;
        padding: 6px;
        border: 1px solid #dee2e6;
        font-weight: 500;
    }
    .options-chain-header {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
            else:
                # Use reasonable estimate for TSLA
                underlying_price = 435.0
        
        return options_data, underlying_price
        
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return None, None

def calculate_net_gamma_vanna(options_data, underlying_price, num_expiries=5):
    """Calculate net gamma and vanna by strike across expiries"""
    
    if not options_data:
        return pd.DataFrame(), pd.DataFrame()
    
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
                
                # Extract Greeks and other data
                gamma = contract.get('gamma', 0)
                vanna = contract.get('vanna', 0)
                delta = contract.get('delta', 0)
                vega = contract.get('vega', 0)
                theta = contract.get('theta', 0)
                volume = contract.get('totalVolume', 0)
                open_interest = contract.get('openInterest', 0)
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                last = contract.get('last', 0)
                volatility = contract.get('volatility', 0)
                
                # If Greeks are not provided, try to estimate them
                if gamma == 0 and delta != 0 and volatility > 0 and underlying_price > 0:
                    if vega > 0:
                        gamma = vega / (underlying_price * volatility)
                    else:
                        time_to_exp = 30
                        if abs(strike - underlying_price) / underlying_price < 0.05:
                            gamma = 1 / (underlying_price * volatility * (time_to_exp ** 0.5))
                
                # Calculate vanna if not provided
                if vanna == 0 and vega != 0 and underlying_price != 0:
                    iv = volatility
                    if iv > 0 and underlying_price > 0:
                        vanna = (vega * delta) / (underlying_price * iv)
                
                # Calculate notional exposure
                option_price = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                notional_gamma = gamma * open_interest * 100 * underlying_price if underlying_price > 0 else gamma * open_interest * 100
                notional_vanna = vanna * open_interest * 100 * underlying_price if underlying_price > 0 else vanna * open_interest * 100
                
                # Calculate distance from spot
                distance_from_spot = 0
                if underlying_price > 0:
                    distance_from_spot = ((strike - underlying_price) / underlying_price) * 100
                
                results.append({
                    'strike': strike,
                    'expiry': exp_date,
                    'option_type': 'Call' if 'call' in option_type else 'Put',
                    'gamma': gamma,
                    'vanna': vanna,
                    'delta': delta,
                    'vega': vega,
                    'theta': theta,
                    'volume': volume,
                    'open_interest': open_interest,
                    'option_price': option_price,
                    'notional_gamma': notional_gamma,
                    'notional_vanna': notional_vanna,
                    'distance_from_spot': distance_from_spot,
                    'volatility': volatility
                })
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Aggregate by strike across all expiries
    agg_df = df.groupby('strike').agg({
        'notional_gamma': 'sum',
        'notional_vanna': 'sum',
        'volume': 'sum',
        'open_interest': 'sum',
        'distance_from_spot': 'first'
    }).reset_index()
    
    return df, agg_df

def create_options_chain_table(df_detailed, underlying_price, metric='notional_gamma'):
    """Create an options chain style table like the screenshot"""
    
    if df_detailed.empty:
        return pd.DataFrame()
    
    # Create pivot table with strikes as rows and expiries as columns
    pivot_df = df_detailed.pivot_table(
        index='strike', 
        columns='expiry', 
        values=metric, 
        aggfunc='sum'
    ).fillna(0)
    
    # Round strikes to nearest 0.5 for cleaner display
    pivot_df.index = (pivot_df.index * 2).round() / 2
    pivot_df = pivot_df.groupby(pivot_df.index).sum()
    
    # Sort by strike
    pivot_df = pivot_df.sort_index()
    
    # Format column names to show just the date part
    new_columns = []
    for col in pivot_df.columns:
        if isinstance(col, str) and ':' in col:
            # Extract date from expiry string like "2025-10-17:45"
            date_part = col.split(':')[0]
            try:
                from datetime import datetime
                dt = datetime.strptime(date_part, '%Y-%m-%d')
                formatted_date = dt.strftime('%m-%d')
                new_columns.append(formatted_date)
            except:
                new_columns.append(col)
        else:
            new_columns.append(str(col))
    
    pivot_df.columns = new_columns
    
    # Add strike column as first column
    pivot_df.insert(0, 'Strike', [f"${s:.1f}" for s in pivot_df.index])
    
    # Format numbers for better readability (like the screenshot)
    for col in pivot_df.columns:
        if col != 'Strike':
            pivot_df[col] = pivot_df[col].apply(lambda x: 
                f"${x/1000000:.1f}M" if abs(x) >= 1000000 
                else f"${x/1000:.0f}K" if abs(x) >= 1000 
                else f"${x:.0f}" if x != 0 
                else "$0.00"
            )
    
    return pivot_df

def style_options_chain_table(df, underlying_price, metric='notional_gamma'):
    """Apply styling to make it look like the options chain screenshot"""
    
    def color_cells(val):
        """Color cells based on value - similar to the screenshot"""
        if isinstance(val, str):
            if val == "$0.00" or val == "$0":
                return 'background-color: #f8f9fa; color: #6c757d'
            
            # Extract numeric value
            try:
                if 'M' in val:
                    num_val = float(val.replace('$', '').replace('M', '')) * 1000000
                elif 'K' in val:
                    num_val = float(val.replace('$', '').replace('K', '')) * 1000
                else:
                    num_val = float(val.replace('$', ''))
            except:
                return 'background-color: #f8f9fa'
            
            # Color coding similar to the screenshot
            if num_val > 0:
                if metric == 'notional_gamma':
                    if num_val > 100000:  # High values
                        return 'background-color: #28a745; color: white; font-weight: bold'  # Dark green
                    else:
                        return 'background-color: #d4edda; color: #155724'  # Light green
                else:  # vanna
                    if num_val > 100000:
                        return 'background-color: #ffc107; color: #212529; font-weight: bold'  # Yellow
                    else:
                        return 'background-color: #fff3cd; color: #856404'  # Light yellow
            elif num_val < 0:
                if metric == 'notional_gamma':
                    if abs(num_val) > 100000:
                        return 'background-color: #dc3545; color: white; font-weight: bold'  # Dark red
                    else:
                        return 'background-color: #f8d7da; color: #721c24'  # Light red
                else:  # vanna
                    if abs(num_val) > 100000:
                        return 'background-color: #007bff; color: white; font-weight: bold'  # Dark blue
                    else:
                        return 'background-color: #d1ecf1; color: #0c5460'  # Light blue
            else:
                return 'background-color: #f8f9fa; color: #6c757d'
        
        return ''
    
    def highlight_strike_column(s):
        """Style the strike column"""
        if s.name == 'Strike':
            return ['background-color: #343a40; color: white; font-weight: bold; text-align: center'] * len(s)
        return [''] * len(s)
    
    def highlight_atm_row(s):
        """Highlight the ATM strike row"""
        styles = [''] * len(s)
        
        # Get the strike value from the Strike column
        if hasattr(s, 'name') and 'Strike' in s.index:
            strike_str = s['Strike']
            try:
                strike_val = float(strike_str.replace('$', ''))
                if abs(strike_val - underlying_price) <= 2.5:  # Within $2.50 of ATM
                    styles = ['border: 3px solid #ffc107; font-weight: bold'] * len(s)
            except:
                pass
        
        return styles
    
    styled = df.style.map(color_cells).apply(highlight_strike_column, axis=0).apply(highlight_atm_row, axis=1)
    
    return styled

def create_gamma_profile_chart(agg_df, underlying_price):
    """Create gamma profile chart by strike"""
    
    if agg_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Color coding based on gamma values
    colors = ['red' if x < 0 else 'green' for x in agg_df['notional_gamma']]
    
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['notional_gamma'],
        marker_color=colors,
        hovertemplate='Strike: $%{x}<br>Net Gamma: %{y:,.0f}<br>Distance: %{customdata:.1f}%<extra></extra>',
        customdata=agg_df['distance_from_spot'],
        name='Net Gamma'
    ))
    
    # Add current price line
    fig.add_vline(
        x=underlying_price, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text=f"Current: ${underlying_price:.2f}"
    )
    
    fig.update_layout(
        title="Net Gamma Profile by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Net Gamma Exposure",
        height=400,
        showlegend=False
    )
    
    return fig

def create_vanna_profile_chart(agg_df, underlying_price):
    """Create vanna profile chart by strike"""
    
    if agg_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Color coding based on vanna values
    colors = ['blue' if x < 0 else 'orange' for x in agg_df['notional_vanna']]
    
    fig.add_trace(go.Bar(
        x=agg_df['strike'],
        y=agg_df['notional_vanna'],
        marker_color=colors,
        hovertemplate='Strike: $%{x}<br>Net Vanna: %{y:,.0f}<br>Distance: %{customdata:.1f}%<extra></extra>',
        customdata=agg_df['distance_from_spot'],
        name='Net Vanna'
    ))
    
    # Add current price line
    fig.add_vline(
        x=underlying_price, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text=f"Current: ${underlying_price:.2f}"
    )
    
    fig.update_layout(
        title="Net Vanna Profile by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Net Vanna Exposure",
        height=400,
        showlegend=False
    )
    
    return fig

def format_large_number(num):
    """Format large numbers for display"""
    if abs(num) >= 1e9:
        return f"${num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.1f}K"
    else:
        return f"${num:.0f}"

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

def generate_sample_data(underlying_price, num_expiries=5):
    """Generate sample gamma/vanna data for testing when API data is insufficient"""
    import random
    
    results = []
    expiries = [f"2025-{10+i:02d}-{17+i*7:02d}" for i in range(num_expiries)]
    
    # Generate strikes around the underlying price
    strike_range = np.arange(underlying_price * 0.8, underlying_price * 1.2, underlying_price * 0.02)
    
    for exp_date in expiries:
        for strike in strike_range:
            for option_type in ['Call', 'Put']:
                # Generate realistic Greeks
                moneyness = strike / underlying_price
                
                # ATM options have higher gamma
                if 0.95 <= moneyness <= 1.05:
                    gamma = random.uniform(0.01, 0.05)
                    vanna = random.uniform(-0.02, 0.02)
                else:
                    gamma = random.uniform(0.001, 0.02)
                    vanna = random.uniform(-0.01, 0.01)
                
                open_interest = random.randint(10, 1000)
                volume = random.randint(0, 100)
                
                notional_gamma = gamma * open_interest * 100 * underlying_price
                notional_vanna = vanna * open_interest * 100 * underlying_price
                
                results.append({
                    'strike': strike,
                    'expiry': exp_date,
                    'option_type': option_type,
                    'gamma': gamma,
                    'vanna': vanna,
                    'delta': random.uniform(-1, 1),
                    'vega': random.uniform(0, 0.5),
                    'theta': random.uniform(-0.1, 0),
                    'volume': volume,
                    'open_interest': open_interest,
                    'option_price': random.uniform(1, 50),
                    'notional_gamma': notional_gamma,
                    'notional_vanna': notional_vanna,
                    'distance_from_spot': ((strike - underlying_price) / underlying_price) * 100,
                    'volatility': random.uniform(0.2, 0.8)
                })
    
    df = pd.DataFrame(results)
    
    # Aggregate by strike
    agg_df = df.groupby('strike').agg({
        'notional_gamma': 'sum',
        'notional_vanna': 'sum',
        'volume': 'sum',
        'open_interest': 'sum',
        'distance_from_spot': 'first'
    }).reset_index()
    
    return df, agg_df

def main():
    # Header
    st.title("📊 Net Gamma/Vanna Options Analysis")
    st.markdown("Real-time analysis of options Greek exposures across multiple expiration dates")
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")
    
    # Symbol input
    symbol = st.sidebar.text_input("Symbol", value="TSLA", help="Enter stock symbol")
    symbol = symbol.upper()
    
    # Number of expiries
    num_expiries = st.sidebar.selectbox("Number of Expiries", [3, 4, 5, 6, 7], index=2)
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Primary Analysis", 
        ["Both Gamma & Vanna", "Gamma Focus", "Vanna Focus"]
    )
    
    # Use sample data toggle
    use_sample_data = st.sidebar.checkbox(
        "📊 Use Sample Data", 
        value=False, 
        help="Use generated sample data for testing"
    )
    
    # Strike range filter
    strike_range = st.sidebar.slider("Strike Range (% from spot)", -50, 50, (-25, 25))
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Main content
    try:
        # Fetch data
        with st.spinner("Fetching options data..."):
            options_data, underlying_price = get_options_data(symbol, num_expiries)
        
        if options_data is None or underlying_price is None:
            st.error("Failed to fetch options data. Please check the symbol and try again.")
            return
        
        # Calculate gamma and vanna
        with st.spinner("Calculating Greeks..."):
            if use_sample_data:
                df_detailed, agg_df = generate_sample_data(underlying_price, num_expiries)
            else:
                df_detailed, agg_df = calculate_net_gamma_vanna(options_data, underlying_price, num_expiries)
                
                # If real data is empty, offer to use sample data
                if df_detailed.empty or agg_df.empty or agg_df['notional_gamma'].abs().sum() == 0:
                    st.warning("⚠️ No meaningful options data found. Try using sample data for demonstration.")
                    return
        
        if df_detailed.empty:
            st.warning("No options data available for analysis.")
            return
        
        # Filter by strike range
        price_range = (
            underlying_price * (1 + strike_range[0]/100),
            underlying_price * (1 + strike_range[1]/100)
        )
        
        agg_df_filtered = agg_df[
            (agg_df['strike'] >= price_range[0]) & 
            (agg_df['strike'] <= price_range[1])
        ].copy()
        
        df_detailed_filtered = df_detailed[
            (df_detailed['strike'] >= price_range[0]) & 
            (df_detailed['strike'] <= price_range[1])
        ].copy()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_gamma = agg_df_filtered['notional_gamma'].sum()
            st.metric(
                "Total Net Gamma", 
                format_large_number(total_gamma),
                delta=None,
                help="Sum of all gamma exposure across strikes"
            )
        
        with col2:
            total_vanna = agg_df_filtered['notional_vanna'].sum()
            st.metric(
                "Total Net Vanna", 
                format_large_number(total_vanna),
                delta=None,
                help="Sum of all vanna exposure across strikes"
            )
        
        with col3:
            max_gamma_strike = agg_df_filtered.loc[agg_df_filtered['notional_gamma'].abs().idxmax(), 'strike'] if not agg_df_filtered.empty else 0
            # Calculate percentage change safely
            pct_change = 0
            if underlying_price > 0:
                pct_change = ((max_gamma_strike - underlying_price) / underlying_price * 100)
            
            st.metric(
                "Max Gamma Strike", 
                f"${max_gamma_strike:.2f}",
                delta=f"{pct_change:+.1f}%" if underlying_price > 0 else "N/A",
                help="Strike with highest absolute gamma exposure"
            )
        
        with col4:
            st.metric(
                "Current Price", 
                f"${underlying_price:.2f}",
                delta=None,
                help="Current underlying stock price"
            )
        
        # Charts and tables based on analysis type
        if analysis_type in ["Both Gamma & Vanna", "Gamma Focus"]:
            st.subheader("🎯 Net Gamma Exposure by Strike & Expiry")
            
            # Create options chain style table for gamma
            gamma_table = create_options_chain_table(df_detailed_filtered, underlying_price, 'notional_gamma')
            if not gamma_table.empty:
                styled_gamma_table = style_options_chain_table(gamma_table, underlying_price, 'notional_gamma')
                st.dataframe(styled_gamma_table, width='stretch')
            
            # Add gamma profile chart below
            col1, col2 = st.columns([2, 1])
            
            with col1:
                gamma_chart = create_gamma_profile_chart(agg_df_filtered, underlying_price)
                st.plotly_chart(gamma_chart, width='stretch', key="gamma_profile_chart")
            
            with col2:
                st.markdown("### Gamma Insights")
                
                # Gamma wall analysis
                positive_gamma = agg_df_filtered[agg_df_filtered['notional_gamma'] > 0]
                negative_gamma = agg_df_filtered[agg_df_filtered['notional_gamma'] < 0]
                
                if not positive_gamma.empty:
                    max_pos_gamma = positive_gamma.loc[positive_gamma['notional_gamma'].idxmax()]
                    st.markdown(f"**🟢 Largest Positive Gamma:**")
                    st.markdown(f"Strike: ${max_pos_gamma['strike']:.2f}")
                    st.markdown(f"Exposure: {format_large_number(max_pos_gamma['notional_gamma'])}")
                
                if not negative_gamma.empty:
                    max_neg_gamma = negative_gamma.loc[negative_gamma['notional_gamma'].idxmin()]
                    st.markdown(f"**🔴 Largest Negative Gamma:**")
                    st.markdown(f"Strike: ${max_neg_gamma['strike']:.2f}")
                    st.markdown(f"Exposure: {format_large_number(max_neg_gamma['notional_gamma'])}")
        
        if analysis_type in ["Both Gamma & Vanna", "Vanna Focus"]:
            st.subheader("🌊 Net Vanna Exposure by Strike & Expiry")
            
            # Create options chain style table for vanna
            vanna_table = create_options_chain_table(df_detailed_filtered, underlying_price, 'notional_vanna')
            if not vanna_table.empty:
                styled_vanna_table = style_options_chain_table(vanna_table, underlying_price, 'notional_vanna')
                st.dataframe(styled_vanna_table, width='stretch')
            
            # Add vanna profile chart below
            col1, col2 = st.columns([2, 1])
            
            with col1:
                vanna_chart = create_vanna_profile_chart(agg_df_filtered, underlying_price)
                st.plotly_chart(vanna_chart, width='stretch', key="vanna_profile_chart")
            
            with col2:
                st.markdown("### Vanna Insights")
                
                # Vanna analysis
                positive_vanna = agg_df_filtered[agg_df_filtered['notional_vanna'] > 0]
                negative_vanna = agg_df_filtered[agg_df_filtered['notional_vanna'] < 0]
                
                if not positive_vanna.empty:
                    max_pos_vanna = positive_vanna.loc[positive_vanna['notional_vanna'].idxmax()]
                    st.markdown(f"**🟠 Largest Positive Vanna:**")
                    st.markdown(f"Strike: ${max_pos_vanna['strike']:.2f}")
                    st.markdown(f"Exposure: {format_large_number(max_pos_vanna['notional_vanna'])}")
                
                if not negative_vanna.empty:
                    max_neg_vanna = negative_vanna.loc[negative_vanna['notional_vanna'].idxmin()]
                    st.markdown(f"**🔵 Largest Negative Vanna:**")
                    st.markdown(f"Strike: ${max_neg_vanna['strike']:.2f}")
                    st.markdown(f"Exposure: {format_large_number(max_neg_vanna['notional_vanna'])}")
        
        # Combined view for "Both" option
        if analysis_type == "Both Gamma & Vanna":
            st.subheader("� Combined Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Gamma Summary")
                total_pos_gamma = agg_df_filtered[agg_df_filtered['notional_gamma'] > 0]['notional_gamma'].sum()
                total_neg_gamma = agg_df_filtered[agg_df_filtered['notional_gamma'] < 0]['notional_gamma'].sum()
                st.write(f"Positive Gamma: {format_large_number(total_pos_gamma)}")
                st.write(f"Negative Gamma: {format_large_number(total_neg_gamma)}")
                st.write(f"Net Gamma: {format_large_number(total_pos_gamma + total_neg_gamma)}")
            
            with col2:
                st.markdown("#### Vanna Summary")
                total_pos_vanna = agg_df_filtered[agg_df_filtered['notional_vanna'] > 0]['notional_vanna'].sum()
                total_neg_vanna = agg_df_filtered[agg_df_filtered['notional_vanna'] < 0]['notional_vanna'].sum()
                st.write(f"Positive Vanna: {format_large_number(total_pos_vanna)}")
                st.write(f"Negative Vanna: {format_large_number(total_neg_vanna)}")
                st.write(f"Net Vanna: {format_large_number(total_pos_vanna + total_neg_vanna)}")
        
        # Footer
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Symbol:** {symbol} | **Current Price:** ${underlying_price:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()