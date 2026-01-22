#!/usr/bin/env python3
"""
Stock Deep Dive - Comprehensive Single Stock Analysis
Combines: Options Chain Analysis, Price Prediction, Fundamentals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client

# Page config
st.set_page_config(
    page_title="Stock Deep Dive",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }
    .bullish { border-left-color: #10b981; }
    .bearish { border-left-color: #ef4444; }
    .tab-content { padding: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ==================== OPTIONS CHAIN TAB ====================
@st.cache_data(ttl=300)
def fetch_options_chain_data(symbol, expiry_date=None):
    """Fetch options chain for symbol"""
    try:
        client = get_client()
        if not client:
            return None, None
        
        params = {'symbol': symbol, 'contract_type': 'ALL', 'strike_count': 50}
        if expiry_date:
            params['from_date'] = expiry_date
            params['to_date'] = expiry_date
        
        chain = client.get_options_chain(**params)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, None
        
        underlying_price = chain.get('underlyingPrice', 0)
        return chain, underlying_price
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return None, None


def parse_options_chain(chain, underlying_price):
    """Parse chain into calls and puts DataFrames"""
    calls_list, puts_list = [], []
    
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                calls_list.append({
                    'expiry': exp_key, 'strike': float(strike_str),
                    'bid': c.get('bid', 0), 'ask': c.get('ask', 0), 'mark': c.get('mark', 0),
                    'volume': c.get('totalVolume', 0), 'oi': c.get('openInterest', 0),
                    'delta': c.get('delta', 0), 'gamma': c.get('gamma', 0),
                    'iv': c.get('volatility', 0), 'dte': c.get('daysToExpiration', 0)
                })
    
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                puts_list.append({
                    'expiry': exp_key, 'strike': float(strike_str),
                    'bid': c.get('bid', 0), 'ask': c.get('ask', 0), 'mark': c.get('mark', 0),
                    'volume': c.get('totalVolume', 0), 'oi': c.get('openInterest', 0),
                    'delta': c.get('delta', 0), 'gamma': c.get('gamma', 0),
                    'iv': c.get('volatility', 0), 'dte': c.get('daysToExpiration', 0)
                })
    
    return pd.DataFrame(calls_list), pd.DataFrame(puts_list)


def render_options_tab(symbol, chain, underlying_price):
    """Render Options Chain Analysis tab"""
    if not chain:
        st.warning("No options data available")
        return
    
    calls_df, puts_df = parse_options_chain(chain, underlying_price)
    
    if calls_df.empty:
        st.warning("No options data available")
        return
    
    # Expiry selector
    expiries = sorted(calls_df['expiry'].unique())
    selected_expiry = st.selectbox("Select Expiry", expiries, key="opt_expiry")
    
    # Filter by expiry
    calls_exp = calls_df[calls_df['expiry'] == selected_expiry]
    puts_exp = puts_df[puts_df['expiry'] == selected_expiry]
    
    # OI Distribution Chart
    st.subheader("📊 Open Interest Distribution")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Call OI', 'Put OI'))
    
    # Filter to strikes within 15% of current price
    price_range = underlying_price * 0.15
    calls_filtered = calls_exp[(calls_exp['strike'] >= underlying_price - price_range) & 
                               (calls_exp['strike'] <= underlying_price + price_range)]
    puts_filtered = puts_exp[(puts_exp['strike'] >= underlying_price - price_range) & 
                              (puts_exp['strike'] <= underlying_price + price_range)]
    
    fig.add_trace(go.Bar(x=calls_filtered['strike'], y=calls_filtered['oi'], 
                         marker_color='#10b981', name='Call OI'), row=1, col=1)
    fig.add_trace(go.Bar(x=puts_filtered['strike'], y=puts_filtered['oi'], 
                         marker_color='#ef4444', name='Put OI'), row=1, col=2)
    
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow", row=1, col=1)
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="yellow", row=1, col=2)
    
    fig.update_layout(template='plotly_dark', height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Levels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Top Call Strikes (Resistance)**")
        top_calls = calls_exp[calls_exp['strike'] > underlying_price].nlargest(5, 'oi')
        for _, row in top_calls.iterrows():
            dist = ((row['strike'] - underlying_price) / underlying_price) * 100
            st.markdown(f"• ${row['strike']:.2f} (+{dist:.1f}%) - OI: {row['oi']:,}")
    
    with col2:
        st.markdown("**🔴 Top Put Strikes (Support)**")
        top_puts = puts_exp[puts_exp['strike'] < underlying_price].nlargest(5, 'oi')
        for _, row in top_puts.iterrows():
            dist = ((underlying_price - row['strike']) / underlying_price) * 100
            st.markdown(f"• ${row['strike']:.2f} (-{dist:.1f}%) - OI: {row['oi']:,}")
    
    # Full Chain Table
    with st.expander("📋 Full Options Chain"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Calls**")
            st.dataframe(calls_exp[['strike', 'bid', 'ask', 'volume', 'oi', 'delta', 'iv']].sort_values('strike'), 
                        use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Puts**")
            st.dataframe(puts_exp[['strike', 'bid', 'ask', 'volume', 'oi', 'delta', 'iv']].sort_values('strike'), 
                        use_container_width=True, hide_index=True)


# ==================== PRICE PREDICTION TAB ====================
def calculate_probability_zones(calls_df, puts_df, underlying_price):
    """Calculate probability zones from IV"""
    if calls_df.empty:
        return None
    
    nearest_exp = calls_df['expiry'].min()
    exp_calls = calls_df[calls_df['expiry'] == nearest_exp]
    exp_puts = puts_df[puts_df['expiry'] == nearest_exp]
    
    if exp_calls.empty:
        return None
    
    dte = exp_calls['dte'].iloc[0]
    atm_strike = exp_calls.iloc[(exp_calls['strike'] - underlying_price).abs().argsort()[:1]]['strike'].values[0]
    
    atm_call_iv = exp_calls[exp_calls['strike'] == atm_strike]['iv'].values
    atm_put_iv = exp_puts[exp_puts['strike'] == atm_strike]['iv'].values
    
    if len(atm_call_iv) == 0 or len(atm_put_iv) == 0:
        return None
    
    atm_iv = (atm_call_iv[0] + atm_put_iv[0]) / 2 / 100
    time_factor = np.sqrt(max(dte, 1) / 365)
    expected_move = underlying_price * atm_iv * time_factor
    
    return {
        'atm_iv': atm_iv * 100,
        'expected_move': expected_move,
        'expected_move_pct': (expected_move / underlying_price) * 100,
        'dte': dte,
        'expiry': nearest_exp,
        'upper_1sd': underlying_price + expected_move,
        'lower_1sd': underlying_price - expected_move,
        'upper_2sd': underlying_price + expected_move * 2,
        'lower_2sd': underlying_price - expected_move * 2
    }


def calculate_max_pain(calls_df, puts_df, underlying_price):
    """Calculate max pain strike"""
    nearest_exp = calls_df['expiry'].min()
    calls_exp = calls_df[calls_df['expiry'] == nearest_exp]
    puts_exp = puts_df[puts_df['expiry'] == nearest_exp]
    
    all_strikes = sorted(set(calls_exp['strike'].tolist() + puts_exp['strike'].tolist()))
    
    pain_by_strike = {}
    for test_price in all_strikes:
        call_pain = sum((test_price - s) * oi for s, oi in 
                       zip(calls_exp['strike'], calls_exp['oi']) if test_price > s)
        put_pain = sum((s - test_price) * oi for s, oi in 
                      zip(puts_exp['strike'], puts_exp['oi']) if test_price < s)
        pain_by_strike[test_price] = (call_pain + put_pain) * 100
    
    return min(pain_by_strike.keys(), key=lambda x: pain_by_strike[x])


def render_prediction_tab(symbol, chain, underlying_price):
    """Render Price Prediction tab"""
    if not chain:
        st.warning("No options data available")
        return
    
    calls_df, puts_df = parse_options_chain(chain, underlying_price)
    
    if calls_df.empty:
        st.warning("No options data available")
        return
    
    zones = calculate_probability_zones(calls_df, puts_df, underlying_price)
    
    if not zones:
        st.warning("Could not calculate probability zones")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${underlying_price:.2f}")
    col2.metric("ATM IV", f"{zones['atm_iv']:.1f}%")
    col3.metric("Expected Move", f"±${zones['expected_move']:.2f}", f"±{zones['expected_move_pct']:.1f}%")
    col4.metric("Days to Expiry", zones['dte'])
    
    st.success(f"""
    **68% Probability Range by {zones['expiry']}:** ${zones['lower_1sd']:.2f} - ${zones['upper_1sd']:.2f}
    """)
    
    # Max Pain
    max_pain = calculate_max_pain(calls_df, puts_df, underlying_price)
    mp_dist = ((max_pain - underlying_price) / underlying_price) * 100
    
    st.info(f"**Max Pain:** ${max_pain:.2f} ({mp_dist:+.1f}% from current)")
    
    # Probability Chart
    fig = go.Figure()
    
    # 2σ zone
    fig.add_shape(type="rect", x0=0, x1=1, y0=zones['lower_2sd'], y1=zones['upper_2sd'],
                  fillcolor="rgba(148, 163, 184, 0.1)", line=dict(width=0))
    # 1σ zone
    fig.add_shape(type="rect", x0=0, x1=1, y0=zones['lower_1sd'], y1=zones['upper_1sd'],
                  fillcolor="rgba(16, 185, 129, 0.3)", line=dict(width=0))
    
    fig.add_hline(y=underlying_price, line_color="#f59e0b", line_width=3,
                  annotation_text=f"Current: ${underlying_price:.2f}")
    fig.add_hline(y=max_pain, line_color="#8b5cf6", line_dash="dot",
                  annotation_text=f"Max Pain: ${max_pain:.2f}")
    
    fig.update_layout(
        title="Probability Zones",
        yaxis_title="Price ($)",
        xaxis=dict(showticklabels=False, showgrid=False),
        template='plotly_dark', height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================== FUNDAMENTALS TAB ====================
@st.cache_data(ttl=3600)
def fetch_fundamentals(symbol):
    """Fetch fundamental data using yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception as e:
        return None


def render_fundamentals_tab(symbol):
    """Render Fundamentals tab"""
    info = fetch_fundamentals(symbol)
    
    if not info:
        st.warning("Could not fetch fundamental data")
        return
    
    st.subheader(f"{info.get('longName', symbol)}")
    st.caption(f"{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A")
    col2.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}" if info.get('trailingPE') else "N/A")
    col3.metric("EPS", f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A")
    col4.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
    col2.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A")
    col3.metric("Beta", f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A")
    col4.metric("Avg Volume", f"{info.get('averageVolume', 0)/1e6:.1f}M" if info.get('averageVolume') else "N/A")
    
    # Business Summary
    with st.expander("📝 Business Summary"):
        st.write(info.get('longBusinessSummary', 'No summary available'))
    
    # Financials
    with st.expander("💰 Financial Metrics"):
        fin_col1, fin_col2 = st.columns(2)
        
        with fin_col1:
            st.markdown("**Valuation**")
            st.write(f"- Forward P/E: {info.get('forwardPE', 'N/A')}")
            st.write(f"- PEG Ratio: {info.get('pegRatio', 'N/A')}")
            st.write(f"- Price/Book: {info.get('priceToBook', 'N/A')}")
            st.write(f"- Price/Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}")
        
        with fin_col2:
            st.markdown("**Profitability**")
            st.write(f"- Profit Margin: {info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else "N/A")
            st.write(f"- Operating Margin: {info.get('operatingMargins', 0)*100:.1f}%" if info.get('operatingMargins') else "N/A")
            st.write(f"- ROE: {info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else "N/A")
            st.write(f"- ROA: {info.get('returnOnAssets', 0)*100:.1f}%" if info.get('returnOnAssets') else "N/A")


# ==================== MAIN APP ====================
def main():
    st.title("🔬 Stock Deep Dive")
    st.caption("Comprehensive single-stock analysis: Options, Predictions, Fundamentals")
    
    # Symbol input
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Symbol", value="SPY", key="deep_dive_symbol").upper().strip()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if not symbol:
        st.warning("Enter a symbol to begin")
        return
    
    # Fetch data
    with st.spinner(f"Loading {symbol} data..."):
        chain, underlying_price = fetch_options_chain_data(symbol)
    
    if underlying_price:
        st.header(f"📈 {symbol} @ ${underlying_price:.2f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Options Chain", "🔮 Price Analysis", "📋 Fundamentals"])
    
    with tab1:
        render_options_tab(symbol, chain, underlying_price)
    
    with tab2:
        render_prediction_tab(symbol, chain, underlying_price)
    
    with tab3:
        render_fundamentals_tab(symbol)


if __name__ == "__main__":
    main()
