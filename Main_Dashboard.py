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
    st.write("Focus on the three essential tools: Stock Option Finder, Volume Walls, and Flow Scanner.")

with col2:
    st.info(f"📅 {datetime.now().strftime('%B %d, %Y')}\n🕐 {datetime.now().strftime('%I:%M %p ET')}")

st.markdown("---")
st.markdown("### 🎯 Core Trading Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### �� Stock Option Finder")
    st.write("Multi-symbol scanner for high gamma/delta opportunities")
    st.write("✓ Gamma/Delta analysis\n✓ Strike recommendations")
    if st.button("Launch Option Finder", key="finder"):
        st.switch_page("pages/1_Stock_Option_Finder.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Option Volume Walls")
    st.write("Volume-based support/resistance with 4-corner command center")
    st.write("✓ Volume profile\n✓ Real-time tracking")
    if st.button("Launch Volume Walls", key="walls"):
        st.switch_page("pages/2_Option_Volume_Walls.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="core-tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🌊 Flow Scanner")
    st.write("Real-time unusual activity detector for institutional flow")
    st.write("✓ Block trades\n✓ Sentiment analysis")
    if st.button("Launch Flow Scanner", key="flow"):
        st.switch_page("pages/3_Flow_Scanner.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🛠️ Additional Tools")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌅 Morning Dashboard", use_container_width=True):
        st.switch_page("pages/4_Morning_Dashboard.py")
    if st.button("💎 Opportunity Scanner", use_container_width=True):
        st.switch_page("pages/7_Opportunity_Scanner.py")

with col2:
    if st.button("📍 Index Positioning", use_container_width=True):
        st.switch_page("pages/5_Index_Positioning.py")
    if st.button("📊 Options Flow Monitor", use_container_width=True):
        st.switch_page("pages/8_Options_Flow_Monitor.py")

with col3:
    if st.button("🎯 Boundary Scanner", use_container_width=True):
        st.switch_page("pages/6_Boundary_Scanner.py")
    if st.button("📰 Newsletter Generator", use_container_width=True):
        st.switch_page("pages/9_Newsletter_Generator.py")
