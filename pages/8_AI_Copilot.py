#!/usr/bin/env python3
"""
AI Trading Copilot - Chat Interface
Powered by Groq's free Llama 3.1 API
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai_brain.copilot import TradingCopilot

# Page config
st.set_page_config(
    page_title="AI Trading Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Trading Copilot")
st.markdown("*Your AI-powered market analyst - synthesizes all your data into actionable insights*")

# Initialize copilot
@st.cache_resource
def get_copilot():
    # Check for API key in secrets or environment
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except:
        pass
    return TradingCopilot(api_key=api_key)

copilot = get_copilot()

# Sidebar - API Key setup
with st.sidebar:
    st.header("⚙️ Setup")
    
    if not copilot.is_available():
        st.warning("🔑 API Key Required")
        st.markdown("""
        **Get your FREE Groq API key:**
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up (free, no credit card)
        3. Create an API key
        4. Paste it below
        """)
        
        api_key = st.text_input("Groq API Key", type="password", key="api_key_input")
        if api_key:
            copilot = TradingCopilot(api_key=api_key)
            if copilot.is_available():
                st.success("✅ Connected!")
                st.rerun()
            else:
                st.error("Invalid API key")
    else:
        st.success("✅ AI Connected")
        st.caption(f"Model: {copilot.model}")
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    
    morning_brief = st.button("📰 Morning Brief", use_container_width=True)
    best_setups = st.button("🎯 Best Setups", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Analyze Stock")
    analyze_ticker = st.text_input("Ticker", placeholder="e.g., NVDA").upper()
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main content area
if not copilot.is_available():
    st.info("👈 Add your Groq API key in the sidebar to start chatting")
    
    # Show what the copilot can do
    st.markdown("### What can the AI Copilot do?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📰 Morning Brief**
        - Market sentiment overview
        - Top 3 trade setups
        - Stocks with improving momentum
        - Key risk factors
        
        **🎯 Find Best Setups**
        - Confluence detection
        - Multi-signal alignment
        - Risk/reward assessment
        """)
    
    with col2:
        st.markdown("""
        **🔍 Stock Analysis**
        - Historical score tracking
        - Momentum assessment
        - Support/resistance levels
        - Trade thesis generation
        
        **💬 Ask Anything**
        - "What's your market view?"
        - "Compare NVDA vs AMD"
        - "Which stocks are risky?"
        """)
    
    st.markdown("---")
    st.markdown("### 🆓 It's FREE!")
    st.markdown("Groq offers free API access to Llama 3.1 70B - one of the most capable open-source AI models.")

else:
    # Handle sidebar actions
    if morning_brief:
        with st.spinner("📰 Generating morning brief..."):
            response = copilot.generate_morning_brief()
        st.session_state.messages.append({"role": "assistant", "content": f"**📰 Morning Brief - {datetime.now().strftime('%B %d, %Y')}**\n\n{response}"})
    
    if best_setups:
        with st.spinner("🎯 Finding best setups..."):
            response = copilot.find_best_setups()
        st.session_state.messages.append({"role": "assistant", "content": f"**🎯 Top Trade Setups**\n\n{response}"})
    
    if analyze_btn and analyze_ticker:
        with st.spinner(f"🔍 Analyzing {analyze_ticker}..."):
            response = copilot.analyze_stock(analyze_ticker)
        st.session_state.messages.append({"role": "assistant", "content": f"**🔍 Analysis: {analyze_ticker}**\n\n{response}"})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the market..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = copilot.chat(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# Footer with data status
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    # Check for newsletter data
    history_file = project_root / "data" / "newsletter_scan_history.json"
    if history_file.exists():
        import json
        with open(history_file) as f:
            history = json.load(f)
        dates = sorted(history.keys(), reverse=True)
        st.caption(f"📊 Newsletter Data: {len(dates)} scan(s), Latest: {dates[0] if dates else 'None'}")
    else:
        st.caption("📊 Newsletter Data: No scans yet")

with col2:
    st.caption(f"🕐 Current Time: {datetime.now().strftime('%H:%M:%S')}")

with col3:
    st.caption("🤖 Powered by Llama 3.1 via Groq")
