"""
🎯 Top Trading Opportunities - Live from Backend API
Displays pre-computed composite scores from droplet scanner
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Top Opportunities", page_icon="🎯", layout="wide")

# API Configuration
API_URL = "http://138.197.210.166:8000"

st.title("🎯 Top Trading Opportunities")
st.markdown("*Live data from backend scanner (updates every 5 minutes)*")

# Auto-refresh toggle in top right
col1, col2 = st.columns([4, 1])
with col2:
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

# Fetch top opportunities
st.markdown("### 🏆 Ranked Opportunities (Composite Score)")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    **Scoring:** Whale Flows (0-35) + Fresh OI (0-35) + Skew Alignment (0-30) = **Total 0-100**
    - **80-100**: 🔥 Extremely high conviction - ALL metrics align
    - **60-79**: ⚡ Strong setup - Most metrics confirm  
    - **40-59**: ✅ Moderate opportunity
    - **<40**: ⚠️ Low conviction
    """)

with col2:
    limit = st.number_input("Show top", min_value=10, max_value=100, value=50, step=10)

try:
    opportunities_response = requests.get(f"{API_URL}/api/top_opportunities?limit={limit}", timeout=5)
    
    if opportunities_response.status_code == 200:
        data = opportunities_response.json()
        opportunities = data.get('opportunities', [])
        
        if opportunities:
            df = pd.DataFrame(opportunities)
            
            # Add score emoji
            def score_emoji(score):
                if score >= 80:
                    return "🔥"
                elif score >= 60:
                    return "⚡"
                elif score >= 40:
                    return "✅"
                else:
                    return "⚠️"
            
            df['🎯'] = df['composite_score'].apply(score_emoji)
            
            # Format columns
            df['Composite Score'] = df['composite_score'].astype(int)
            df['Symbol'] = df['symbol']
            df['Strike'] = df['strike'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "N/A")
            df['Type'] = df['option_type']
            df['Expiry'] = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d')
            df['Whale Score'] = df['whale_score'].astype(int)
            df['Vol/OI Ratio'] = df['vol_oi_ratio'].round(2)
            df['Skew 25Δ'] = df['skew_25d'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            df['P/C Ratio'] = df['put_call_ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Display table
            display_df = df[['🎯', 'Composite Score', 'Symbol', 'Strike', 'Type', 'Expiry', 'Whale Score', 
                            'Vol/OI Ratio', 'Skew 25Δ', 'P/C Ratio']]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                column_config={
                    'Composite Score': st.column_config.ProgressColumn(
                        "Composite Score",
                        help="Combined score: 0-100",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
            
            st.caption(f"📅 Last updated: {data.get('timestamp', 'N/A')}")
            
            # Top 5 breakdown
            st.divider()
            st.markdown("### 🔍 Top 5 Breakdown")
            
            for idx, opp in enumerate(opportunities[:5], 1):
                with st.expander(f"#{idx} - {opp['symbol']} (Score: {opp['composite_score']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Whale Score", opp['whale_score'])
                        st.caption("High VALR institutional activity")
                    
                    with col2:
                        st.metric("Vol/OI Ratio", f"{opp['vol_oi_ratio']:.2f}x")
                        st.caption("Fresh positioning indicator")
                    
                    with col3:
                        skew = opp.get('skew_25d')
                        if skew:
                            st.metric("Put/Call Skew", f"{skew:.2f}%")
                            if skew > 5:
                                st.caption("🔴 EXTREME FEAR - Contrarian buy signal")
                            elif skew < -1:
                                st.caption("🟢 GREED - Contrarian sell signal")
                            else:
                                st.caption("⚪ Neutral skew")
                    
                    st.info(f"**Expiry:** {opp['expiry']}")
        else:
            st.info("⏳ No opportunities yet. Backend scanner runs every 5 minutes during market hours.")
    
    else:
        st.warning("Unable to fetch data from backend API")

except requests.exceptions.Timeout:
    st.error("⏱️ Request timed out. Backend may be busy.")
except requests.exceptions.ConnectionError:
    st.error("🔌 Cannot connect to backend API. Check if services are running on droplet.")
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(60)
    st.rerun()

st.divider()
st.caption("💡 **Tip:** This page shows pre-computed data from the backend scanner. No need to wait for scans!")
st.caption(f"🔗 **API Endpoint:** {API_URL}/api/top_opportunities")
