"""
GEX Analysis - Comprehensive Options Chain Analysis with Gamma Exposure
Similar to NSE options chain view with full greeks and GEX calculations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cached_client import get_client
from src.utils.config import get_settings

st.set_page_config(page_title="GEX Analysis", page_icon="📊", layout="wide")

def get_next_fridays(n=8):
    """Get next N Fridays"""
    fridays = []
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    
    for i in range(n):
        friday = today + timedelta(days=days_ahead + (i * 7))
        fridays.append(friday)
    
    return fridays

def calculate_gex(row, spot_price, contracts_per_point=100):
    """Calculate Gamma Exposure (GEX) for an option"""
    # GEX = Gamma * Open Interest * Contract Multiplier * Spot Price
    # For calls: positive GEX, for puts: negative GEX
    gamma = row.get('gamma', 0) or 0
    oi = row.get('openInterest', 0) or 0
    
    if gamma == 0 or oi == 0:
        return 0
    
    # GEX in $ millions
    gex = gamma * oi * contracts_per_point * spot_price / 1_000_000
    
    # Calls are positive, puts are negative (dealer's perspective)
    if row.get('putCall') == 'PUT':
        gex = -gex
    
    return gex

@st.cache_data(ttl=300)
def fetch_options_chain(symbol, expiry_date):
    """Fetch full options chain for symbol and expiry"""
    try:
        client = get_client()
        if not client:
            st.error("⚠️ Schwab API connection failed")
            return None, None, None
        
        # Format date
        date_str = expiry_date.strftime('%Y-%m-%d')
        
        # Fetch options chain
        chain_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=date_str,
            to_date=date_str,
            include_quotes=True
        )
        
        if not chain_data:
            return None, None
        
        # Get underlying price
        underlying_price = chain_data.get('underlyingPrice', 0)
        
        # Parse call and put maps
        call_map = chain_data.get('callExpDateMap', {})
        put_map = chain_data.get('putExpDateMap', {})
        
        calls = []
        puts = []
        
        # Process calls
        for exp_date, strikes in call_map.items():
            for strike, options_list in strikes.items():
                for option in options_list:
                    option['strike'] = float(strike)
                    calls.append(option)
        
        # Process puts
        for exp_date, strikes in put_map.items():
            for strike, options_list in strikes.items():
                for option in options_list:
                    option['strike'] = float(strike)
                    puts.append(option)
        
        return pd.DataFrame(calls), pd.DataFrame(puts), underlying_price
    
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None, None, None

def create_chain_table(calls_df, puts_df, spot_price):
    """Create combined options chain table like NSE format"""
    if calls_df.empty or puts_df.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    call_cols = ['openInterest', 'totalVolume', 'delta', 'gamma', 'theta', 'vega', 
                 'volatility', 'bid', 'ask', 'last', 'strike']
    put_cols = ['strike', 'bid', 'ask', 'last', 'volatility', 'delta', 'gamma', 
                'theta', 'vega', 'totalVolume', 'openInterest']
    
    # Prepare call side
    call_data = calls_df[call_cols].copy()
    call_data.columns = ['Call_OI', 'Call_Vol', 'Call_Delta', 'Call_Gamma', 'Call_Theta', 
                         'Call_Vega', 'Call_IV', 'Call_Bid', 'Call_Ask', 'Call_LTP', 'Strike']
    
    # Prepare put side
    put_data = puts_df[put_cols].copy()
    put_data.columns = ['Strike', 'Put_Bid', 'Put_Ask', 'Put_LTP', 'Put_IV', 'Put_Delta', 
                        'Put_Gamma', 'Put_Theta', 'Put_Vega', 'Put_Vol', 'Put_OI']
    
    # Merge on strike
    merged = pd.merge(call_data, put_data, on='Strike', how='outer').sort_values('Strike')
    
    # Calculate GEX for each side
    merged['Call_GEX'] = merged.apply(
        lambda x: calculate_gex({'gamma': x.get('Call_Gamma'), 'openInterest': x.get('Call_OI'), 'putCall': 'CALL'}, spot_price),
        axis=1
    )
    merged['Put_GEX'] = merged.apply(
        lambda x: calculate_gex({'gamma': x.get('Put_Gamma'), 'openInterest': x.get('Put_OI'), 'putCall': 'PUT'}, spot_price),
        axis=1
    )
    
    # Calculate PCR (Put-Call Ratio)
    merged['PCR_OI'] = merged['Put_OI'] / merged['Call_OI']
    merged['PCR_Vol'] = merged['Put_Vol'] / merged['Call_Vol']
    
    # Mark ATM strike
    merged['ATM'] = abs(merged['Strike'] - spot_price)
    atm_idx = merged['ATM'].idxmin()
    
    # Fill NaN with 0
    merged = merged.fillna(0)
    
    return merged, atm_idx

def generate_market_summary(merged_df, spot_price, symbol, total_call_gex, total_put_gex, net_gex, pcr):
    """Generate AI-powered market analysis summary"""
    
    # Find max GEX strikes
    max_call_gex_strike = merged_df.loc[merged_df['Call_GEX'].idxmax(), 'Strike'] if not merged_df['Call_GEX'].empty else 0
    max_call_gex_value = merged_df['Call_GEX'].max()
    
    max_put_gex_strike = merged_df.loc[merged_df['Put_GEX'].abs().idxmax(), 'Strike'] if not merged_df['Put_GEX'].empty else 0
    max_put_gex_value = merged_df['Put_GEX'].min()  # Most negative
    
    # Find max OI strikes
    max_call_oi_strike = merged_df.loc[merged_df['Call_OI'].idxmax(), 'Strike'] if not merged_df['Call_OI'].empty else 0
    max_put_oi_strike = merged_df.loc[merged_df['Put_OI'].idxmax(), 'Strike'] if not merged_df['Put_OI'].empty else 0
    
    # Average IV
    avg_call_iv = merged_df['Call_IV'].mean() * 100
    avg_put_iv = merged_df['Put_IV'].mean() * 100
    
    # Generate summary
    summary = f"""
### 📊 Market Analysis Summary for {symbol}

**Current Positioning:**
- **Spot Price:** ${spot_price:,.2f}
- **Net GEX:** ${net_gex:,.2f}M {'(Positive - Stabilizing Market)' if net_gex > 0 else '(Negative - Volatile Market)'}
- **Put/Call Ratio:** {pcr:.2f} {'(Bearish sentiment)' if pcr > 1.0 else '(Bullish sentiment)'}

**Market Maker Positioning:**
"""
    
    if net_gex > 0:
        summary += f"""
- **Positive GEX Environment:** Market makers are net long gamma. They will hedge by:
  - **Selling into rallies** (providing resistance)
  - **Buying into dips** (providing support)
  - This typically **suppresses volatility** and keeps price range-bound
"""
    else:
        summary += f"""
- **Negative GEX Environment:** Market makers are net short gamma. They will hedge by:
  - **Buying into rallies** (accelerating moves up)
  - **Selling into dips** (accelerating moves down)
  - This typically **amplifies volatility** and creates explosive moves
"""
    
    summary += f"""
**Key Levels to Watch:**

**Call Side (Resistance Levels):**
- **Max Call GEX Strike:** ${max_call_gex_strike:,.2f} (${max_call_gex_value:,.2f}M GEX)
- **Max Call OI:** ${max_call_oi_strike:,.2f} - Heavy call positioning suggests resistance
- Dealers likely selling calls here, creating a ceiling

**Put Side (Support Levels):**
- **Max Put GEX Strike:** ${max_put_gex_strike:,.2f} (${max_put_gex_value:,.2f}M GEX)
- **Max Put OI:** ${max_put_oi_strike:,.2f} - Heavy put positioning suggests support
- Dealers likely selling puts here, creating a floor

**Volatility Environment:**
- **Avg Call IV:** {avg_call_iv:.1f}%
- **Avg Put IV:** {avg_put_iv:.1f}%
- **IV Skew:** {abs(avg_put_iv - avg_call_iv):.1f}% {'(Put skew - fear premium)' if avg_put_iv > avg_call_iv else '(Call skew - bullish positioning)'}

**Trading Implications:**
"""
    
    # Distance from key strikes
    distance_to_max_call = ((max_call_gex_strike - spot_price) / spot_price) * 100
    distance_to_max_put = ((spot_price - max_put_gex_strike) / spot_price) * 100
    
    if abs(distance_to_max_call) < 2:
        summary += f"\n- ⚠️ **Near Max Call GEX ({distance_to_max_call:+.1f}%):** Strong resistance overhead"
    elif distance_to_max_call > 0:
        summary += f"\n- 📈 **Room to Max Call GEX ({distance_to_max_call:+.1f}%):** Upside potential to ${max_call_gex_strike:,.2f}"
    
    if abs(distance_to_max_put) < 2:
        summary += f"\n- ⚠️ **Near Max Put GEX ({distance_to_max_put:+.1f}% below):** Strong support nearby"
    elif distance_to_max_put > 0:
        summary += f"\n- 📉 **Room to Max Put GEX ({distance_to_max_put:+.1f}% below):** Downside to ${max_put_gex_strike:,.2f}"
    
    if pcr > 1.3:
        summary += "\n- 🐻 **High PCR:** Elevated put buying suggests defensive positioning or bearish bets"
    elif pcr < 0.7:
        summary += "\n- 🐂 **Low PCR:** Heavy call buying suggests bullish positioning"
    
    if net_gex > 0 and abs(distance_to_max_call) < 5 and abs(distance_to_max_put) < 5:
        summary += f"\n- 🎯 **Range-Bound Setup:** Positive GEX with price between key strikes (${max_put_gex_strike:,.0f}-${max_call_gex_strike:,.0f})"
    
    if net_gex < 0:
        summary += "\n- ⚡ **High Volatility Risk:** Negative GEX environment can lead to accelerated moves in either direction"
    
    summary += """

**Disclaimer:** This analysis is based on current options positioning and assumes standard market maker hedging behavior. 
Market conditions can change rapidly. Use this as one input among many for trading decisions.
"""
    
    return summary

def create_visualizations(merged_df, spot_price):
    """Create all visualization charts"""
    
    # Filter around ATM for better visualization (±20% strikes)
    price_range = spot_price * 0.20
    viz_df = merged_df[
        (merged_df['Strike'] >= spot_price - price_range) & 
        (merged_df['Strike'] <= spot_price + price_range)
    ].copy()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Open Interest', 'OI Change', 'Volume',
            'Put/Call Theta', 'Put/Call Vega', 'GEX',
            'Put Gex vs Strike', 'Call Gex vs Strike', 'Total GEX Distribution'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    strikes_str = viz_df['Strike'].astype(str)
    
    # Row 1, Col 1: Open Interest
    fig.add_trace(
        go.Bar(name='Put OI', x=strikes_str, y=viz_df['Put_OI'], 
               marker_color='red', showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Call OI', x=strikes_str, y=viz_df['Call_OI'], 
               marker_color='blue', showlegend=True),
        row=1, col=1
    )
    
    # Row 1, Col 2: OI Change (using volume as proxy since we don't have OI change)
    fig.add_trace(
        go.Bar(name='Put Vol', x=strikes_str, y=viz_df['Put_Vol'], 
               marker_color='red', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name='Call Vol', x=strikes_str, y=viz_df['Call_Vol'], 
               marker_color='blue', showlegend=False),
        row=1, col=2
    )
    
    # Row 1, Col 3: Volume
    fig.add_trace(
        go.Bar(name='Put Volume', x=strikes_str, y=viz_df['Put_Vol'], 
               marker_color='red', showlegend=False),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(name='Call Volume', x=strikes_str, y=viz_df['Call_Vol'], 
               marker_color='blue', showlegend=False),
        row=1, col=3
    )
    
    # Row 2, Col 1: Put/Call Theta
    fig.add_trace(
        go.Bar(name='Put Theta', x=strikes_str, y=viz_df['Put_Theta'], 
               marker_color='red', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Call Theta', x=strikes_str, y=viz_df['Call_Theta'], 
               marker_color='blue', showlegend=False),
        row=2, col=1
    )
    
    # Row 2, Col 2: Put/Call Vega
    fig.add_trace(
        go.Bar(name='Put Vega', x=strikes_str, y=viz_df['Put_Vega'], 
               marker_color='red', showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(name='Call Vega', x=strikes_str, y=viz_df['Call_Vega'], 
               marker_color='blue', showlegend=False),
        row=2, col=2
    )
    
    # Row 2, Col 3: GEX
    fig.add_trace(
        go.Bar(name='Put GEX', x=strikes_str, y=viz_df['Put_GEX'], 
               marker_color='red', showlegend=False),
        row=2, col=3
    )
    fig.add_trace(
        go.Bar(name='Call GEX', x=strikes_str, y=viz_df['Call_GEX'], 
               marker_color='blue', showlegend=False),
        row=2, col=3
    )
    
    # Row 3, Col 1: Put GEX scatter
    fig.add_trace(
        go.Scatter(name='Put GEX', x=viz_df['Strike'], y=viz_df['Put_GEX'], 
                   mode='lines+markers', line=dict(color='red', width=2),
                   showlegend=False),
        row=3, col=1
    )
    
    # Row 3, Col 2: Call GEX scatter
    fig.add_trace(
        go.Scatter(name='Call GEX', x=viz_df['Strike'], y=viz_df['Call_GEX'], 
                   mode='lines+markers', line=dict(color='blue', width=2),
                   showlegend=False),
        row=3, col=2
    )
    
    # Row 3, Col 3: Total GEX Distribution (horizontal bar)
    viz_df['Total_GEX'] = viz_df['Call_GEX'] + viz_df['Put_GEX']
    colors = ['blue' if x > 0 else 'red' for x in viz_df['Total_GEX']]
    fig.add_trace(
        go.Bar(name='Total GEX', y=strikes_str, x=viz_df['Total_GEX'], 
               orientation='h', marker_color=colors, showlegend=False),
        row=3, col=3
    )
    
    # Add vertical line for spot price on scatter plots
    for row, col in [(3, 1), (3, 2)]:
        fig.add_vline(x=spot_price, line_dash="dash", line_color="green", 
                     row=row, col=col, annotation_text=f"Spot: {spot_price:.2f}")
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text=f"Options Chain Analysis - Spot: ${spot_price:.2f}",
        title_x=0.5,
        barmode='group'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Strike", row=3, col=1)
    fig.update_xaxes(title_text="Strike", row=3, col=2)
    fig.update_xaxes(title_text="GEX ($M)", row=3, col=3)
    fig.update_yaxes(title_text="Strike", row=3, col=3)
    
    return fig

# Main UI
st.title("📊 GEX Analysis - Options Chain with Gamma Exposure")

# Query Parameters at top of page
st.subheader("Query Parameters")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Symbol input
    symbol = st.text_input(
        "Symbol",
        value="SPY",
        help="Enter stock symbol. Use $SPX for SPX index"
    ).upper()

with col2:
    # Expiry date selection
    fridays = get_next_fridays(8)
    expiry_options = [f.strftime('%Y-%m-%d (%b %d)') for f in fridays]
    
    selected_expiry = st.selectbox(
        "Expiry Date",
        options=expiry_options,
        index=0
    )
    
    # Parse selected date
    expiry_date = datetime.strptime(selected_expiry.split(' ')[0], '%Y-%m-%d').date()

with col3:
    # Fetch button
    st.write("")  # Spacing
    st.write("")  # Spacing
    fetch_btn = st.button("🔄 Fetch Options Chain", type="primary", use_container_width=True)

st.divider()

# Sidebar info
with st.sidebar:
    st.header("About GEX Analysis")
    
    st.info("""
    **GEX (Gamma Exposure)**
    
    Measures the rate of change of delta. High GEX levels indicate:
    - **Positive GEX**: Market makers hedge by buying into rallies and selling into dips (stabilizing)
    - **Negative GEX**: Market makers hedge in the opposite direction (volatility)
    
    **Key Metrics**:
    - PCR: Put-Call Ratio
    - Greeks: Delta, Gamma, Theta, Vega
    - GEX: Gamma * OI * 100 * Spot / 1M
    """)

# Main content
if fetch_btn or 'chain_data' in st.session_state:
    
    if fetch_btn:
        with st.spinner(f"Fetching options chain for {symbol} expiring {expiry_date}..."):
            calls_df, puts_df, spot_price = fetch_options_chain(symbol, expiry_date)
            
            if calls_df is not None and puts_df is not None:
                st.session_state.chain_data = {
                    'calls': calls_df,
                    'puts': puts_df,
                    'spot': spot_price,
                    'symbol': symbol,
                    'expiry': expiry_date
                }
            else:
                st.error("Failed to fetch options chain data")
                st.stop()
    
    # Use cached data
    if 'chain_data' in st.session_state:
        data = st.session_state.chain_data
        calls_df = data['calls']
        puts_df = data['puts']
        spot_price = data['spot']
        symbol = data['symbol']
        expiry_date = data['expiry']
        
        # Display header metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Spot Price", f"${spot_price:,.2f}")
        with col3:
            total_call_oi = calls_df['openInterest'].sum()
            st.metric("Call OI", f"{total_call_oi:,.0f}")
        with col4:
            total_put_oi = puts_df['openInterest'].sum()
            st.metric("Put OI", f"{total_put_oi:,.0f}")
        with col5:
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            st.metric("PCR (OI)", f"{pcr:.2f}")
        with col6:
            st.metric("Expiry", expiry_date.strftime('%Y-%m-%d'))
        
        # Create combined table
        merged_df, atm_idx = create_chain_table(calls_df, puts_df, spot_price)
        
        # Calculate total GEX
        total_call_gex = merged_df['Call_GEX'].sum()
        total_put_gex = merged_df['Put_GEX'].sum()
        net_gex = total_call_gex + total_put_gex
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Call GEX", f"${total_call_gex:,.2f}M", 
                     delta="Positive" if total_call_gex > 0 else "Negative")
        with col2:
            st.metric("Total Put GEX", f"${total_put_gex:,.2f}M",
                     delta="Negative" if total_put_gex < 0 else "Positive")
        with col3:
            st.metric("Net GEX", f"${net_gex:,.2f}M",
                     delta="Stabilizing" if net_gex > 0 else "Volatile")
        
        st.divider()
        
        # Generate and display market summary
        market_summary = generate_market_summary(
            merged_df, spot_price, symbol, total_call_gex, total_put_gex, net_gex, pcr
        )
        
        with st.expander("📋 **Market Analysis & Trading Insights**", expanded=True):
            st.markdown(market_summary)
        
        st.divider()
        
        # Display visualizations
        with st.spinner("Creating visualizations..."):
            fig = create_visualizations(merged_df, spot_price)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Display data table
        st.subheader("📋 Options Chain Data")
        
        # Filter around ATM
        display_range = spot_price * 0.15
        display_df = merged_df[
            (merged_df['Strike'] >= spot_price - display_range) & 
            (merged_df['Strike'] <= spot_price + display_range)
        ].copy()
        
        # Format and reorder columns for display
        display_columns = [
            'Call_OI', 'Call_Vol', 'Call_IV', 'Call_Delta', 'Call_Gamma', 
            'Call_Theta', 'Call_Vega', 'Call_LTP', 'Call_GEX',
            'Strike',
            'Put_LTP', 'Put_GEX', 'Put_Vega', 'Put_Theta', 'Put_Gamma', 
            'Put_Delta', 'Put_IV', 'Put_Vol', 'Put_OI',
            'PCR_OI'
        ]
        
        display_df = display_df[display_columns]
        
        # Style the ATM row
        def highlight_atm(row):
            if abs(row['Strike'] - spot_price) < (spot_price * 0.01):  # Within 1%
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        # Format numbers
        format_dict = {
            'Strike': '{:.2f}',
            'Call_LTP': '{:.2f}',
            'Put_LTP': '{:.2f}',
            'Call_IV': '{:.1%}',
            'Put_IV': '{:.1%}',
            'Call_Delta': '{:.3f}',
            'Put_Delta': '{:.3f}',
            'Call_Gamma': '{:.4f}',
            'Put_Gamma': '{:.4f}',
            'Call_Theta': '{:.3f}',
            'Put_Theta': '{:.3f}',
            'Call_Vega': '{:.3f}',
            'Put_Vega': '{:.3f}',
            'Call_GEX': '{:.2f}',
            'Put_GEX': '{:.2f}',
            'PCR_OI': '{:.2f}',
            'Call_OI': '{:,.0f}',
            'Put_OI': '{:,.0f}',
            'Call_Vol': '{:,.0f}',
            'Put_Vol': '{:,.0f}'
        }
        
        styled_df = display_df.style.apply(highlight_atm, axis=1).format(format_dict)
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Download button
        csv = merged_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Chain (CSV)",
            data=csv,
            file_name=f"{symbol}_{expiry_date}_options_chain.csv",
            mime="text/csv"
        )

else:
    # Welcome message
    st.info("☝️ Enter a symbol and expiry date above, then click 'Fetch Options Chain' to begin analysis")
    
    # Example screenshots or descriptions
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Comprehensive Analysis**
        - Full options chain with all strikes
        - Real-time Greeks (Delta, Gamma, Theta, Vega)
        - Gamma Exposure (GEX) calculations
        - Put-Call Ratio (PCR) metrics
        - Open Interest and Volume analysis
        """)
    
    with col2:
        st.markdown("""
        **📈 Visualizations**
        - Open Interest distribution
        - Volume analysis
        - Greek distributions
        - GEX profiles by strike
        - Total GEX histogram
        - ATM strike highlighting
        """)
