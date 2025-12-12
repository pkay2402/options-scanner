"""
Whale Flow History - View all historical whale flows with filtering and sorting
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Whale Flow History",
    page_icon="🐋",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .whale-card {
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .whale-card.call {
        border-left-color: #22c55e;
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.05) 0%, white 100%);
    }
    
    .whale-card.put {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.05) 0%, white 100%);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐋 Whale Flow History")
st.markdown("View all historical whale flows with advanced filtering and sorting")

# Filters
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])

with filter_col1:
    # Stock filter
    stock_filter = st.text_input("Filter by Symbol", placeholder="e.g., SPY, NVDA", help="Leave blank for all stocks")

with filter_col2:
    # Type filter
    type_filter = st.selectbox("Option Type", ["All", "CALL", "PUT"])

with filter_col3:
    # Time range filter
    time_range = st.selectbox(
        "Time Range", 
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        index=2  # Default to Last 24 Hours
    )

with filter_col4:
    # Refresh button
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.rerun()

st.markdown("---")

# Convert time range to hours
time_range_hours = {
    "Last Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
    "Last 30 Days": 720,
    "All Time": 999999
}
hours = time_range_hours[time_range]

# Fetch data from API
try:
    url = f"http://138.197.210.166:8000/api/whale_flows?sort_by=time&limit=1000&hours={hours}"
    
    with st.spinner("Loading whale flows..."):
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            flows = data.get('data', [])
            
            if flows:
                # Convert to DataFrame for easier filtering
                df = pd.DataFrame(flows)
                
                # Apply filters
                if stock_filter:
                    df = df[df['symbol'].str.upper() == stock_filter.upper()]
                
                if type_filter != "All":
                    df = df[df['type'] == type_filter]
                
                # Sort by detected_at descending (most recent first)
                df['detected_at'] = pd.to_datetime(df['detected_at'])
                df = df.sort_values('detected_at', ascending=False)
                
                # Deduplicate - keep only the most recent occurrence of each unique flow
                # Group by symbol, type, strike, expiry and keep the first (most recent)
                df = df.drop_duplicates(subset=['symbol', 'type', 'strike', 'expiry'], keep='first')
                
                # Display summary metrics
                summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
                
                with summary_col1:
                    st.metric("Total Flows", len(df))
                
                with summary_col2:
                    total_premium = df['premium'].sum()
                    st.metric("Total Premium", f"${total_premium:,.0f}")
                
                with summary_col3:
                    call_count = len(df[df['type'] == 'CALL'])
                    st.metric("Calls", call_count)
                
                with summary_col4:
                    put_count = len(df[df['type'] == 'PUT'])
                    st.metric("Puts", put_count)
                
                with summary_col5:
                    unique_symbols = df['symbol'].nunique()
                    st.metric("Unique Symbols", unique_symbols)
                
                st.markdown("---")
                
                # Display flows
                st.subheader(f"📊 Showing {len(df)} Whale Flows")
                
                # Add download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"whale_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.markdown("---")
                
                # Display each flow
                for idx, flow in df.iterrows():
                    card_class = 'call' if flow['type'] == 'CALL' else 'put'
                    
                    # Format detected time in ET
                    detected_time = flow['detected_at']
                    time_et = (detected_time - timedelta(hours=5)).strftime('%m/%d %I:%M%p ET').lower()
                    
                    # Format expiry
                    try:
                        if isinstance(flow['expiry'], str):
                            expiry_date = datetime.strptime(flow['expiry'], '%Y-%m-%d').date()
                        else:
                            expiry_date = flow['expiry']
                        dte = (expiry_date - datetime.now().date()).days
                        expiry_display = f"{expiry_date.strftime('%m/%d')} ({dte}DTE)"
                    except:
                        expiry_display = str(flow.get('expiry', 'N/A'))
                    
                    # Color for type
                    type_color = "#22c55e" if flow['type'] == 'CALL' else "#ef4444"
                    type_emoji = "📈" if flow['type'] == 'CALL' else "📉"
                    
                    html = f"""
                    <div class="whale-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                            <div>
                                <strong style="font-size: 16px;">{type_emoji} {flow['symbol']} {flow['type']}</strong>
                                <span style="margin-left: 12px; color: #6b7280; font-size: 12px;">{time_et}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="background: {type_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                                    Score: {int(flow['whale_score']):,}
                                </div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 12px;">
                            <div>
                                <span style="color: #6b7280;">Strike:</span> <strong>${flow['strike']:.2f}</strong>
                            </div>
                            <div>
                                <span style="color: #6b7280;">Volume:</span> <strong>{int(flow['volume']):,}</strong>
                            </div>
                            <div>
                                <span style="color: #6b7280;">Vol/OI:</span> <strong>{flow['vol_oi']:.1f}x</strong>
                            </div>
                            <div>
                                <span style="color: #6b7280;">Premium:</span> <strong>${flow['premium']:.2f}</strong>
                            </div>
                            <div>
                                <span style="color: #6b7280;">Delta:</span> <strong>{abs(flow['delta']):.3f}</strong>
                            </div>
                            <div>
                                <span style="color: #6b7280;">Expiry:</span> <strong>{expiry_display}</strong>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                
            else:
                st.info(f"No whale flows found for the selected filters in {time_range.lower()}")
        else:
            st.error(f"Failed to fetch data from API. Status code: {response.status_code}")
            
except requests.exceptions.Timeout:
    st.error("Request timed out. The API server might be busy. Please try again.")
except requests.exceptions.ConnectionError:
    st.error("Could not connect to API server. Please check if the service is running.")
except Exception as e:
    st.error(f"Error loading whale flows: {str(e)}")

# Footer with info
st.markdown("---")
st.caption("💡 **Tip**: Use the symbol filter to track specific stocks, or download the CSV for deeper analysis in Excel/Python")
