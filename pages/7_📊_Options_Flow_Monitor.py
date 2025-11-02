"""
📊 Options Flow Monitor - Real-time options activity tracking
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.options_flow import OptionsFlowMonitor
from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="Options Flow Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Options Flow Monitor")
st.markdown("Real-time options flow tracking and big trades detection")

# Sidebar
st.sidebar.header("Monitor Settings")

symbols_input = st.sidebar.text_area(
    "Symbols (comma-separated)",
    value="SPY,QQQ,AAPL,NVDA,TSLA,META,MSFT,AMZN"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

min_premium = st.sidebar.number_input(
    "Min Premium ($)",
    min_value=10000,
    value=100000,
    step=10000
)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60)

# Initialize
if st.sidebar.button("🚀 Start Monitoring", type="primary"):
    try:
        # Test Schwab authentication first
        client = SchwabClient()
        if not client.authenticate():
            st.warning("⚠️ Schwab API authentication failed. Using fallback data sources.")
        
        # Initialize monitor (it creates its own client internally)
        monitor = OptionsFlowMonitor()
        
        st.success("✅ Options Flow Monitor initialized!")
        
        # Create placeholders
        summary_placeholder = st.empty()
        flow_placeholder = st.empty()
        
        # Monitor flow
        with st.spinner("Fetching options data..."):
            for symbol in symbols:
                st.subheader(f"📈 {symbol}")
                
                # Get options chain
                chain = client.get_options_chain(symbol)
                
                if not chain:
                    st.warning(f"No options chain data for {symbol}")
                    continue
                
                # Analyze flow
                flow_data = []
                
                # Process calls
                call_map = chain.get('callExpDateMap', {})
                for exp_date, strikes in call_map.items():
                    for strike, contracts in strikes.items():
                        if isinstance(contracts, list):
                            for contract in contracts:
                                volume = contract.get('totalVolume', 0)
                                oi = contract.get('openInterest', 0)
                                last = contract.get('last', 0)
                                
                                if volume > 0:
                                    premium = volume * last * 100
                                    
                                    if premium >= min_premium:
                                        flow_data.append({
                                            'Symbol': symbol,
                                            'Type': 'CALL',
                                            'Strike': strike,
                                            'Expiry': exp_date,
                                            'Volume': volume,
                                            'OI': oi,
                                            'Last': last,
                                            'Premium': premium,
                                            'Vol/OI': volume / oi if oi > 0 else 0
                                        })
                
                # Process puts
                put_map = chain.get('putExpDateMap', {})
                for exp_date, strikes in put_map.items():
                    for strike, contracts in strikes.items():
                        if isinstance(contracts, list):
                            for contract in contracts:
                                volume = contract.get('totalVolume', 0)
                                oi = contract.get('openInterest', 0)
                                last = contract.get('last', 0)
                                
                                if volume > 0:
                                    premium = volume * last * 100
                                    
                                    if premium >= min_premium:
                                        flow_data.append({
                                            'Symbol': symbol,
                                            'Type': 'PUT',
                                            'Strike': strike,
                                            'Expiry': exp_date,
                                            'Volume': volume,
                                            'OI': oi,
                                            'Last': last,
                                            'Premium': premium,
                                            'Vol/OI': volume / oi if oi > 0 else 0
                                        })
                
                if flow_data:
                    df = pd.DataFrame(flow_data)
                    df = df.sort_values('Premium', ascending=False)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trades", len(df))
                    with col2:
                        st.metric("Total Premium", f"${df['Premium'].sum():,.0f}")
                    with col3:
                        call_prem = df[df['Type']=='CALL']['Premium'].sum()
                        put_prem = df[df['Type']=='PUT']['Premium'].sum()
                        ratio = call_prem / put_prem if put_prem > 0 else 0
                        st.metric("C/P Ratio", f"{ratio:.2f}")
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No significant flow detected for {symbol}")
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Configure settings and click 'Start Monitoring' to begin")
    
    st.markdown("""
    ### 📊 What This Monitor Tracks:
    
    - **Large Trades**: Trades with premium > threshold
    - **Volume Analysis**: Volume vs Open Interest ratios
    - **Call/Put Flow**: Directional sentiment
    - **Real-time Updates**: Auto-refresh capability
    
    ### 🎯 Key Metrics:
    
    - **Premium**: Total $ value of contracts traded
    - **Vol/OI**: Volume to Open Interest ratio (>0.5 = unusual)
    - **C/P Ratio**: Call premium / Put premium (>1 = bullish)
    """)
