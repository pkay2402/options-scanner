"""
📊 Options Trading Platform - Main Dashboard
Professional multi-page application for options analysis and trading signals
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Options Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors - softer, more professional */
    :root {
        --accent-green: #10b981;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --background-dark: #111827;
        --card-bg: #1f2937;
        --border-color: #374151;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: 0;
        font-weight: 400;
    }
    
    /* Feature cards */
    .feature-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        border-color: var(--accent-green);
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--accent-green);
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Stats cards */
    .stat-card {
        background: var(--card-bg);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--accent-blue);
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid var(--accent-green);
        background: var(--card-bg);
        color: var(--accent-green);
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background: var(--accent-green);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Info box styling */
    .stAlert {
        background: var(--card-bg) !important;
        border-left: 3px solid var(--accent-blue) !important;
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Options Trading Platform</div>
    <div class="main-subtitle">Professional Analysis Suite | Real-time Signals | Market Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Welcome message
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 👋 Welcome to Your Trading Command Center")
    st.markdown("""
    This comprehensive platform provides institutional-grade options analysis, 
    real-time market signals, and actionable trading insights. Navigate through 
    the sidebar to access powerful tools designed for professional traders.
    """)

with col2:
    st.markdown(f"""
    <div style="
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #374151;
    ">
        <div style="font-size: 1.1rem; font-weight: 600; color: #10b981; margin-bottom: 1rem;">
            System Status
        </div>
        <div style="color: #9ca3af; line-height: 1.8;">
            <div style="font-size: 1rem;">🟢 <span style="color: #10b981; font-weight: 500;">All Systems Operational</span></div>
            <div style="margin-top: 0.5rem;">📅 {datetime.now().strftime('%B %d, %Y')}</div>
            <div>🕐 {datetime.now().strftime('%I:%M %p')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Quick Stats
st.markdown("---")
st.markdown("### 📈 Platform Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">9</div>
        <div class="stat-label">Analysis Tools</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">6</div>
        <div class="stat-label">Signal Types</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">Real-time</div>
        <div class="stat-label">Market Data</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">25+</div>
        <div class="stat-label">Symbols Tracked</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">Live</div>
        <div class="stat-label">Flow Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

# Main Features Section
st.markdown("---")
st.markdown("### 🎯 Core Features")

# Feature Categories
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🔍 Signal Detection", "📊 Live Monitoring", "📝 Reports"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">� Stock Option Finder</div>
            <div class="feature-description">
                Discover which strikes and expiries have the most market-moving potential. 
                Analyze gamma exposure, EMA indicators, and real-time options flow for any stock.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Stock Option Finder", key="stock_finder"):
            st.switch_page("pages/1_🎯_Stock_Option_Finder.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📍 Index Positioning</div>
            <div class="feature-description">
                SPY/IWM/QQQ exclusive dashboard. Key levels, gamma walls, dealer positioning, 
                and support/resistance zones for major indices.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Index Positioning", key="index"):
            st.switch_page("pages/3_📍_Index_Positioning.py")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🚀 Advanced Max Gamma Scanner</div>
            <div class="feature-description">
                Advanced gamma analysis with heatmaps and detailed metrics. Track dealer positioning 
                across multiple expiries with institutional-grade analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Advanced Scanner", key="advanced"):
            st.switch_page("pages/2_🚀_Advanced_Max_Gamma_Scanner.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Boundary Scanner</div>
            <div class="feature-description">
                Milton Berg's "Reflecting Boundaries" methodology. Detect market turning points 
                using extreme price/volume conditions and breadth signals.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Boundary Scanner", key="boundary"):
            st.switch_page("pages/4_🎯_Boundary_Scanner.py")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🌊 Flow Scanner</div>
            <div class="feature-description">
                Unusual activity detector. Block trades, sweeps, and volume anomalies. 
                Real-time flow analysis with sentiment classification.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Flow Scanner", key="flow"):
            st.switch_page("pages/5_🌊_Flow_Scanner.py")
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">💎 Opportunity Scanner</div>
            <div class="feature-description">
                Systematic trade setup identification. Gamma squeeze, momentum flow, and volatility plays. 
                Scans liquid symbols automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Opportunity Scanner", key="opportunity"):
            st.switch_page("pages/6_💎_Opportunity_Scanner.py")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Options Flow Monitor</div>
            <div class="feature-description">
                Real-time options flow tracking. Big trades detection, premium analysis, 
                call/put ratios, and institutional activity monitoring.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Flow Monitor", key="monitor"):
            st.switch_page("pages/7_📊_Options_Flow_Monitor.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📰 Newsletter Generator</div>
            <div class="feature-description">
                Generate professional options analysis newsletters. Create Substack-style reports 
                from scanner results with charts and commentary.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Newsletter Generator", key="newsletter"):
            st.switch_page("pages/9_📰_Newsletter_Generator.py")

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📰 Newsletter Generator</div>
            <div class="feature-description">
                Substack-style HTML newsletter with comprehensive market summary. 
                Opportunity highlights, trade setups, and professional formatting.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Newsletter Generator", key="newsletter"):
            st.switch_page("pages/9_📰_Newsletter_Generator.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📄 Report Generator</div>
            <div class="feature-description">
                Custom HTML report generation. Export analysis, charts, and insights 
                in professional format for sharing or archival.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Report Generator", key="report"):
            st.switch_page("pages/10_📄_Report_Generator.py")

# Quick Links Section
st.markdown("---")
st.markdown("### 🚀 Quick Access")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎯 Find Trade Opportunities", use_container_width=True):
        st.switch_page("pages/6_💎_Opportunity_Scanner.py")

with col2:
    if st.button("📈 Check Index Levels", use_container_width=True):
        st.switch_page("pages/3_📍_Index_Positioning.py")

with col3:
    if st.button("🔍 Scan for Signals", use_container_width=True):
        st.switch_page("pages/4_🎯_Boundary_Scanner.py")

with col4:
    if st.button("📊 Monitor Live Flow", use_container_width=True):
        st.switch_page("pages/7_📊_Options_Flow_Monitor.py")

# Help & Documentation
st.markdown("---")
st.markdown("### 📚 Getting Started")

with st.expander("📖 How to Use This Platform"):
    st.markdown("""
    **Navigation:**
    - Use the **sidebar** (left) to access all tools
    - Each tool has its own dedicated page
    - Use the **Quick Access** buttons above for common tasks
    
    **Workflow Suggestions:**
    
    1. **Morning Routine:**
       - Check **Index Positioning** for key levels
       - Run **Opportunity Scanner** for trade setups
       - Monitor **Options Flow** for institutional activity
    
    2. **Signal Detection:**
       - Use **Boundary Scanner** for turning points
       - **Flow Scanner** for unusual activity
       - **Max Gamma** for dealer positioning
    
    3. **End of Day:**
       - Generate **Newsletter** with daily summary
       - Review **Reports** for performance tracking
       - Set up alerts for next session
    
    **Data Sources:**
    - Primary: Schwab API (when authenticated)
    - Fallback: yfinance (free, no auth required)
    - Historical data: Up to 10 years available
    
    **Tips:**
    - Combine multiple tools for confirmation
    - Focus on signals near key boundaries
    - Use backtesting to validate strategies
    - Set up alerts for important triggers
    """)

with st.expander("⚙️ System Configuration"):
    st.markdown("""
    **Schwab API Setup:**
    - Configure in Settings page
    - Tokens auto-refresh every 30 minutes
    - Manual re-auth required every 7 days
    
    **Alert System:**
    - Email notifications available
    - Webhook support for Discord/Slack
    - Console logging for development
    
    **Performance Optimization:**
    - Data caching enabled (60-300 seconds)
    - Parallel data fetching
    - Background process support
    """)

with st.expander("❓ FAQ & Troubleshooting"):
    st.markdown("""
    **Q: Why am I seeing yfinance instead of Schwab data?**
    - Check Schwab token expiration in Settings
    - Re-authenticate if expired
    - Ensure API credentials are correct
    
    **Q: How often should I run scans?**
    - Boundary Scanner: Once per day or after major moves
    - Flow Scanner: Real-time during market hours
    - Opportunity Scanner: Morning and end of day
    
    **Q: Can I customize the watchlists?**
    - Yes! Each tool allows custom symbol input
    - Defaults cover 25+ liquid symbols
    - Save preferences in Settings
    
    **Q: How do I export data?**
    - Use Report/Newsletter generators
    - Most tables have CSV export
    - Charts can be saved as PNG
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Options Trading Platform</strong> | Version 1.0 | Built with Streamlit</p>
    <p>⚠️ <em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
