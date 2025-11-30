"""
Options Trading Platform - Main Dashboard
"""

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Options Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .core-tool-card {
        background: #1f2937;
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #374151;
        transition: all 0.3s ease;
    }
    .core-tool-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
        border-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><div class="main-title">📊 Options Trading Platform</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 👋 Welcome")
    st.write("Access your most frequently used trading tools with one click.")

with col2:
    st.info(f"📅 {datetime.now().strftime('%B %d, %Y')}\n🕐 {datetime.now().strftime('%I:%M %p ET')}")

st.markdown("---")
st.markdown("### ⭐ Frequently Used Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Stock Option Finder")
    st.write("Multi-symbol scanner for high gamma/delta opportunities")
    st.write("✓ Gamma/Delta analysis\n✓ Strike recommendations")
    if st.button("Launch Option Finder", key="finder", width="stretch"):
        st.switch_page("pages/1_Stock_Option_Finder.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Option Volume Walls")
    st.write("Volume-based support/resistance analysis")
    st.write("✓ Volume profile\n✓ Real-time tracking")
    if st.button("Launch Volume Walls", key="walls", width="stretch"):
        st.switch_page("pages/2_Option_Volume_Walls.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 0DTE Analysis")
    st.write("Same-day expiry options analysis")
    st.write("✓ Intraday levels\n✓ Quick decisions")
    if st.button("Launch 0DTE", key="dte", width="stretch"):
        st.switch_page("pages/3_0DTE.py")
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 0DTE by Index")
    st.write("SPY/QQQ/SPX levels and positioning")
    st.write("✓ Key levels\n✓ Market pulse")
    if st.button("Launch 0DTE Index", key="dte_index", width="stretch"):
        st.switch_page("pages/4_0DTE_by_Index.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🐋 Whale Flows")
    st.write("Track large institutional options orders")
    st.write("✓ Big trades\n✓ Smart money")
    if st.button("Launch Whale Flows", key="whale", width="stretch"):
        st.switch_page("pages/3_Whale_Flows.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🌊 Flow Scanner")
    st.write("Real-time unusual activity detector")
    st.write("✓ Block trades\n✓ Sentiment")
    if st.button("Launch Flow Scanner", key="flow", width="stretch"):
        st.switch_page("pages/4_Flow_Scanner.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🛠️ Additional Tools")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("☁️ EMA Cloud Scanner", width="stretch"):
        st.switch_page("pages/5_EMA_Cloud_Scanner.py")

with col2:
    if st.button("📡 Live Flow Feed", width="stretch"):
        st.switch_page("pages/6_Live_Flow_Feed.py")

with col3:
    if st.button("📈 Put/Call Ratio", width="stretch"):
        st.switch_page("pages/8_Put_Call_Ratio.py")

with col4:
    if st.button("📰 AI Stock Report", width="stretch"):
        st.switch_page("pages/11_AI_Stock_Report.py")

st.markdown("---")
st.caption("🚀 Options Trading Platform | Built for Professional Traders")
