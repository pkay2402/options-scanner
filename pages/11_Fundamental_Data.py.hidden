#!/usr/bin/env python3
"""
Fundamental Data
View comprehensive fundamental metrics for any stock using Schwab API
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Configure page
st.set_page_config(
    page_title="Fundamental Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #666;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.3em;
        font-weight: bold;
        color: #333;
        margin-top: 5px;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
        margin: 20px 0 15px 0;
    }
    
    .positive {
        color: #10b981;
        font-weight: bold;
    }
    
    .negative {
        color: #ef4444;
        font-weight: bold;
    }
    
    .neutral {
        color: #6b7280;
    }
    
    .dataframe {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>📊 Fundamental Data</h1><p>Comprehensive fundamental metrics powered by Schwab API</p></div>', unsafe_allow_html=True)

# Input section
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol_input = st.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        key="symbol_input",
        help="Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
    ).upper().strip()

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    fetch_button = st.button("🔍 Get Fundamental Data", type="primary", use_container_width=True)

with col3:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🔄 Clear", use_container_width=True):
        st.rerun()

# Initialize session state
if 'fundamental_data' not in st.session_state:
    st.session_state.fundamental_data = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

# Fetch data
if fetch_button and symbol_input:
    with st.spinner(f"Fetching fundamental data for {symbol_input}..."):
        try:
            client = SchwabClient()
            data = client.get_instrument_fundamental(symbol_input)
            
            if data:
                st.session_state.fundamental_data = data
                st.session_state.current_symbol = symbol_input
            else:
                st.error(f"Could not fetch data for {symbol_input}. Please check the symbol and try again.")
                st.session_state.fundamental_data = None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.session_state.fundamental_data = None

# Display data if available
if st.session_state.fundamental_data:
    data = st.session_state.fundamental_data
    symbol = st.session_state.current_symbol
    
    # Get fundamental data if available
    fundamental = data.get('fundamental', {})
    
    if not fundamental:
        st.warning(f"No fundamental data available for {symbol}")
    else:
        # Company Overview
        st.markdown(f'<div class="section-header">📈 {symbol} - Company Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            market_cap = fundamental.get('marketCap', 0)
            if market_cap:
                market_cap_b = market_cap / 1_000_000_000
                st.metric("Market Cap", f"${market_cap_b:.2f}B")
            else:
                st.metric("Market Cap", "N/A")
        
        with col2:
            pe_ratio = fundamental.get('peRatio', 0)
            st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        
        with col3:
            beta = fundamental.get('beta', 0)
            st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
        
        with col4:
            shares_outstanding = fundamental.get('sharesOutstanding', 0)
            if shares_outstanding:
                shares_m = shares_outstanding / 1_000_000
                st.metric("Shares Outstanding", f"{shares_m:.2f}M")
            else:
                st.metric("Shares Outstanding", "N/A")
        
        # Price Ranges
        st.markdown('<div class="section-header">💰 Price Ranges</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_52 = fundamental.get('high52', 0)
            st.metric("52-Week High", f"${high_52:.2f}" if high_52 else "N/A")
        
        with col2:
            low_52 = fundamental.get('low52', 0)
            st.metric("52-Week Low", f"${low_52:.2f}" if low_52 else "N/A")
        
        # Dividends
        st.markdown('<div class="section-header">💵 Dividend Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            div_amount = fundamental.get('dividendAmount', 0)
            st.metric("Dividend Amount", f"${div_amount:.2f}" if div_amount else "N/A")
        
        with col2:
            div_yield = fundamental.get('dividendYield', 0)
            st.metric("Dividend Yield", f"{div_yield:.2f}%" if div_yield else "N/A")
        
        with col3:
            div_date = fundamental.get('dividendDate', '')
            st.metric("Dividend Date", div_date if div_date else "N/A")
        
        with col4:
            div_pay_date = fundamental.get('dividendPayDate', '')
            st.metric("Payment Date", div_pay_date if div_pay_date else "N/A")
        
        # Valuation Ratios
        st.markdown('<div class="section-header">📊 Valuation Ratios</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            pe = fundamental.get('peRatio', 0)
            st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
        
        with col2:
            peg = fundamental.get('pegRatio', 0)
            st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
        
        with col3:
            pb = fundamental.get('pbRatio', 0)
            st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
        
        with col4:
            pr = fundamental.get('prRatio', 0)
            st.metric("P/R Ratio", f"{pr:.2f}" if pr else "N/A")
        
        with col5:
            pcf = fundamental.get('pcfRatio', 0)
            st.metric("P/CF Ratio", f"{pcf:.2f}" if pcf else "N/A")
        
        # Profitability Margins
        st.markdown('<div class="section-header">💹 Profitability Margins</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Gross Margin**")
            gross_ttm = fundamental.get('grossMarginTTM', 0)
            gross_mrq = fundamental.get('grossMarginMRQ', 0)
            st.metric("TTM", f"{gross_ttm:.2f}%" if gross_ttm else "N/A")
            st.metric("MRQ", f"{gross_mrq:.2f}%" if gross_mrq else "N/A")
        
        with col2:
            st.markdown("**Operating Margin**")
            op_ttm = fundamental.get('operatingMarginTTM', 0)
            op_mrq = fundamental.get('operatingMarginMRQ', 0)
            st.metric("TTM", f"{op_ttm:.2f}%" if op_ttm else "N/A")
            st.metric("MRQ", f"{op_mrq:.2f}%" if op_mrq else "N/A")
        
        with col3:
            st.markdown("**Net Profit Margin**")
            net_ttm = fundamental.get('netProfitMarginTTM', 0)
            net_mrq = fundamental.get('netProfitMarginMRQ', 0)
            st.metric("TTM", f"{net_ttm:.2f}%" if net_ttm else "N/A")
            st.metric("MRQ", f"{net_mrq:.2f}%" if net_mrq else "N/A")
        
        # Returns
        st.markdown('<div class="section-header">📈 Returns</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            roe = fundamental.get('returnOnEquity', 0)
            st.metric("Return on Equity", f"{roe:.2f}%" if roe else "N/A")
        
        with col2:
            roa = fundamental.get('returnOnAssets', 0)
            st.metric("Return on Assets", f"{roa:.2f}%" if roa else "N/A")
        
        with col3:
            roi = fundamental.get('returnOnInvestment', 0)
            st.metric("Return on Investment", f"{roi:.2f}%" if roi else "N/A")
        
        # Liquidity Ratios
        st.markdown('<div class="section-header">💧 Liquidity Ratios</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current = fundamental.get('currentRatio', 0)
            st.metric("Current Ratio", f"{current:.2f}" if current else "N/A")
        
        with col2:
            quick = fundamental.get('quickRatio', 0)
            st.metric("Quick Ratio", f"{quick:.2f}" if quick else "N/A")
        
        with col3:
            interest_cov = fundamental.get('interestCoverage', 0)
            st.metric("Interest Coverage", f"{interest_cov:.2f}" if interest_cov else "N/A")
        
        # Debt Ratios
        st.markdown('<div class="section-header">💳 Debt Ratios</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_debt = fundamental.get('totalDebtToCapital', 0)
            st.metric("Total Debt to Capital", f"{total_debt:.2f}%" if total_debt else "N/A")
        
        with col2:
            lt_debt = fundamental.get('ltDebtToEquity', 0)
            st.metric("LT Debt to Equity", f"{lt_debt:.2f}%" if lt_debt else "N/A")
        
        with col3:
            total_debt_eq = fundamental.get('totalDebtToEquity', 0)
            st.metric("Total Debt to Equity", f"{total_debt_eq:.2f}%" if total_debt_eq else "N/A")
        
        # EPS & Growth
        st.markdown('<div class="section-header">📊 Earnings Per Share (EPS)</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            eps_ttm = fundamental.get('epsTTM', 0)
            st.metric("EPS (TTM)", f"${eps_ttm:.2f}" if eps_ttm else "N/A")
        
        with col2:
            eps_change_pct = fundamental.get('epsChangePercentTTM', 0)
            delta_color = "normal" if eps_change_pct > 0 else "inverse"
            st.metric("EPS Change % (TTM)", f"{eps_change_pct:.2f}%" if eps_change_pct else "N/A", delta=f"{eps_change_pct:.2f}%")
        
        with col3:
            eps_change_year = fundamental.get('epsChangeYear', 0)
            st.metric("EPS Change (YoY)", f"{eps_change_year:.2f}%" if eps_change_year else "N/A")
        
        with col4:
            eps_change = fundamental.get('epsChange', 0)
            st.metric("EPS Change", f"${eps_change:.2f}" if eps_change else "N/A")
        
        # Revenue Growth
        st.markdown('<div class="section-header">💰 Revenue Growth</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rev_change_ttm = fundamental.get('revChangeTTM', 0)
            st.metric("Revenue Change (TTM)", f"{rev_change_ttm:.2f}%" if rev_change_ttm else "N/A")
        
        with col2:
            rev_change_year = fundamental.get('revChangeYear', 0)
            st.metric("Revenue Change (YoY)", f"{rev_change_year:.2f}%" if rev_change_year else "N/A")
        
        with col3:
            rev_change_in = fundamental.get('revChangeIn', 0)
            st.metric("Revenue Change (In)", f"{rev_change_in:.2f}%" if rev_change_in else "N/A")
        
        # Short Interest
        st.markdown('<div class="section-header">🔻 Short Interest</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            short_float = fundamental.get('shortIntToFloat', 0)
            st.metric("Short % of Float", f"{short_float:.2f}%" if short_float else "N/A")
        
        with col2:
            short_days = fundamental.get('shortIntDayToCover', 0)
            st.metric("Days to Cover", f"{short_days:.2f}" if short_days else "N/A")
        
        with col3:
            book_value = fundamental.get('bookValuePerShare', 0)
            st.metric("Book Value/Share", f"${book_value:.2f}" if book_value else "N/A")
        
        # Volume Metrics
        st.markdown('<div class="section-header">📊 Volume Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vol_1d = fundamental.get('vol1DayAvg', 0)
            if vol_1d:
                vol_1d_m = vol_1d / 1_000_000
                st.metric("1-Day Avg Volume", f"{vol_1d_m:.2f}M")
            else:
                st.metric("1-Day Avg Volume", "N/A")
        
        with col2:
            vol_10d = fundamental.get('vol10DayAvg', 0)
            if vol_10d:
                vol_10d_m = vol_10d / 1_000_000
                st.metric("10-Day Avg Volume", f"{vol_10d_m:.2f}M")
            else:
                st.metric("10-Day Avg Volume", "N/A")
        
        with col3:
            vol_3m = fundamental.get('vol3MonthAvg', 0)
            if vol_3m:
                vol_3m_m = vol_3m / 1_000_000
                st.metric("3-Month Avg Volume", f"{vol_3m_m:.2f}M")
            else:
                st.metric("3-Month Avg Volume", "N/A")
        
        # Raw Data Table
        st.markdown('<div class="section-header">📋 Complete Raw Data</div>', unsafe_allow_html=True)
        
        with st.expander("🔍 View All Raw Fundamental Data", expanded=False):
            # Convert to DataFrame for better display
            df = pd.DataFrame([fundamental]).T
            df.columns = ['Value']
            df.index.name = 'Metric'
            st.dataframe(df, use_container_width=True)
        
        # Export option
        st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # CSV export
            csv = pd.DataFrame([fundamental]).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_fundamental_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            import json
            json_str = json.dumps(fundamental, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{symbol}_fundamental_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # Instructions when no data is displayed
    st.info("👆 Enter a stock symbol above and click 'Get Fundamental Data' to view comprehensive fundamental metrics.")
    
    st.markdown("### 📚 Available Metrics")
    st.markdown("""
    This page provides access to **60+ fundamental metrics** including:
    
    - **📈 Valuation**: P/E, PEG, P/B, P/R, P/CF ratios
    - **💹 Profitability**: Gross, operating, and net profit margins
    - **📊 Returns**: ROE, ROA, ROI
    - **💧 Liquidity**: Current ratio, quick ratio, interest coverage
    - **💳 Debt**: Debt to capital, debt to equity ratios
    - **💰 Earnings**: EPS, EPS growth, revenue growth
    - **💵 Dividends**: Amount, yield, dates, growth rate
    - **🔻 Short Interest**: Short % of float, days to cover
    - **📊 Volume**: Average volumes across different timeframes
    - **📈 52-Week Range**: High and low prices
    - **🎯 Market Data**: Market cap, beta, shares outstanding
    """)
    
    st.markdown("### 🚀 Example Symbols")
    st.markdown("Try: `AAPL`, `MSFT`, `GOOGL`, `TSLA`, `NVDA`, `META`, `AMZN`, `SPY`, `QQQ`")
