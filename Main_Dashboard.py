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
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        border: 1px solid #334155;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .hero-stats {
        display: flex;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10b981;
        display: block;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Professional Tool Cards */
    .tool-section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .tool-section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #10b981 0%, #3b82f6 100%);
        border-radius: 2px;
    }
    
    .pro-tool-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .pro-tool-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .pro-tool-card:hover {
        transform: translateY(-4px);
        border-color: #10b981;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15);
    }
    
    .pro-tool-card:hover::before {
        opacity: 1;
    }
    
    .tool-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .tool-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .tool-description {
        font-size: 0.875rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .tool-features {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.6;
    }
    
    /* Secondary Tool Grid */
    .secondary-tools {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    /* Professional Footer */
    .pro-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #334155;
        color: #64748b;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown(f"""
<div class="hero-section">
    <div class="hero-title">Options Trading Platform</div>
    <div class="hero-subtitle">Professional-grade analytics and real-time market intelligence</div>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-value">LIVE</span>
            <span class="stat-label">Market Data</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{datetime.now().strftime('%I:%M %p')}</span>
            <span class="stat-label">Eastern Time</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{datetime.now().strftime('%b %d')}</span>
            <span class="stat-label">Today</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Primary Tools Section
st.markdown('<div class="tool-section-title">⚡ Core Trading Tools</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">🔍</span>
        <div class="tool-title">Stock Option Finder</div>
        <div class="tool-description">Advanced multi-symbol scanner for identifying high-probability gamma and delta opportunities</div>
        <div class="tool-features">• Real-time Greek calculations<br>• Strike recommendations<br>• Risk-reward analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Finder →", key="finder", width="stretch", type="primary"):
        st.switch_page("pages/1_Stock_Option_Finder.py")

with col2:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">📊</span>
        <div class="tool-title">Option Volume Walls</div>
        <div class="tool-description">Professional volume-based support and resistance analysis with institutional flow tracking</div>
        <div class="tool-features">• Volume profile analysis<br>• Real-time wall detection<br>• Price level clustering</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Volume Walls →", key="walls", width="stretch", type="primary"):
        st.switch_page("pages/2_Option_Volume_Walls.py")

with col3:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">🎯</span>
        <div class="tool-title">0DTE Analysis</div>
        <div class="tool-description">Same-day expiry options analysis for precision intraday trading decisions</div>
        <div class="tool-features">• Intraday level detection<br>• Fast decision support<br>• Time-decay modeling</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch 0DTE →", key="dte", width="stretch", type="primary"):
        st.switch_page("pages/3_0DTE.py")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">📈</span>
        <div class="tool-title">0DTE by Index</div>
        <div class="tool-description">SPY, QQQ, and SPX same-day positioning with market pulse indicators</div>
        <div class="tool-features">• Index-specific levels<br>• Market sentiment gauges<br>• Correlation analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Index Analysis →", key="dte_index", width="stretch", type="primary"):
        st.switch_page("pages/4_0DTE_by_Index.py")

with col2:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">🐋</span>
        <div class="tool-title">Whale Flows</div>
        <div class="tool-description">Track large institutional options orders and smart money positioning</div>
        <div class="tool-features">• Big block detection<br>• Institutional signals<br>• Dark pool activity</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Whale Tracker →", key="whale", width="stretch", type="primary"):
        st.switch_page("pages/3_Whale_Flows.py")

with col3:
    st.markdown("""
    <div class="pro-tool-card">
        <span class="tool-icon">🌊</span>
        <div class="tool-title">Flow Scanner</div>
        <div class="tool-description">Real-time unusual options activity detector with sentiment analysis</div>
        <div class="tool-features">• Live flow monitoring<br>• Anomaly detection<br>• Sentiment scoring</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Flow Scanner →", key="flow", width="stretch", type="primary"):
        st.switch_page("pages/4_Flow_Scanner.py")


st.markdown("---")

# Additional Tools Section
st.markdown('<div class="tool-section-title">🛠️ Additional Tools</div>', unsafe_allow_html=True)

st.markdown('<div class="secondary-tools">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="pro-tool-card" style="height: 200px;">
        <span class="tool-icon" style="font-size: 2em;">☁️</span>
        <div class="tool-title" style="font-size: 1.1em;">EMA Cloud</div>
        <div class="tool-description" style="font-size: 0.85em;">Technical momentum analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch →", key="ema", width="stretch"):
        st.switch_page("pages/5_EMA_Cloud_Scanner.py")

with col2:
    st.markdown("""
    <div class="pro-tool-card" style="height: 200px;">
        <span class="tool-icon" style="font-size: 2em;">📡</span>
        <div class="tool-title" style="font-size: 1.1em;">Live Feed</div>
        <div class="tool-description" style="font-size: 0.85em;">Real-time streaming flow</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch →", key="feed", width="stretch"):
        st.switch_page("pages/6_Live_Flow_Feed.py")

with col3:
    st.markdown("""
    <div class="pro-tool-card" style="height: 200px;">
        <span class="tool-icon" style="font-size: 2em;">📖</span>
        <div class="tool-title" style="font-size: 1.1em;">Level 2</div>
        <div class="tool-description" style="font-size: 0.85em;">Order book depth analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch →", key="level2", width="stretch"):
        st.switch_page("pages/7_Level2_Book.py")

with col4:
    st.markdown("""
    <div class="pro-tool-card" style="height: 200px;">
        <span class="tool-icon" style="font-size: 2em;">⚖️</span>
        <div class="tool-title" style="font-size: 1.1em;">P/C Ratio</div>
        <div class="tool-description" style="font-size: 0.85em;">Put/Call market sentiment</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch →", key="pc", width="stretch"):
        st.switch_page("pages/8_Put_Call_Ratio.py")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="pro-footer">
    <div style="text-align: center; color: #64748b;">
        <p style="margin: 0 0 8px 0; font-size: 0.9em;">Professional Options Trading Platform</p>
        <p style="margin: 0; font-size: 0.8em; opacity: 0.7;">Real-time market analysis • Institutional-grade tools • Advanced analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("🚀 Options Trading Platform | Built for Professional Traders")
