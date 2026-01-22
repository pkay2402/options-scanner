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

# Get API key from secrets
api_key = None
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except:
    pass

# Initialize copilot
copilot = TradingCopilot(api_key=api_key)

# ==================== PAGE HEADER ====================
st.title("🤖 AI Trading Copilot")
st.markdown("*Your AI-powered market analyst - powered by Llama 3.1 via Groq (FREE)*")

# ==================== SETTINGS AT TOP ====================
with st.expander("⚙️ Settings & Connection", expanded=not copilot.is_available()):
    if copilot.is_available():
        st.success("✅ Connected to Groq AI")
        st.caption(f"Model: {copilot.model}")
    else:
        st.warning("🔑 API Key Required")
        st.markdown("""
**Get your FREE Groq API key:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Add to Streamlit secrets as `GROQ_API_KEY` or paste below
        """)
        
        manual_key = st.text_input("Or paste API Key here:", type="password")
        if manual_key:
            copilot = TradingCopilot(api_key=manual_key)
            if copilot.is_available():
                st.success("✅ Connected!")
                st.rerun()
            else:
                st.error("Invalid API key")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== QUICK ACTIONS ====================
if copilot.is_available():
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        morning_brief = st.button("📰 Morning Brief", use_container_width=True)
    with col2:
        best_setups = st.button("🎯 Best Setups", use_container_width=True)
    with col3:
        analyze_ticker = st.text_input("Ticker", placeholder="NVDA", label_visibility="collapsed")
    with col4:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    st.markdown("---")
    
    # Handle quick actions
    if morning_brief:
        st.session_state.messages.append({"role": "user", "content": "Generate a morning brief"})
        with st.spinner("📰 Generating morning brief..."):
            response = copilot.generate_morning_brief()
        st.session_state.messages.append({"role": "assistant", "content": f"**📰 Morning Brief - {datetime.now().strftime('%B %d, %Y')}**\n\n{response}"})
    
    if best_setups:
        st.session_state.messages.append({"role": "user", "content": "Find the best trade setups"})
        with st.spinner("🎯 Finding best setups..."):
            response = copilot.find_best_setups()
        st.session_state.messages.append({"role": "assistant", "content": f"**🎯 Top Trade Setups**\n\n{response}"})
    
    if analyze_btn and analyze_ticker:
        ticker = analyze_ticker.upper().strip()
        st.session_state.messages.append({"role": "user", "content": f"Analyze {ticker}"})
        with st.spinner(f"🔍 Analyzing {ticker}..."):
            response = copilot.analyze_stock(ticker)
        st.session_state.messages.append({"role": "assistant", "content": f"**🔍 Analysis: {ticker}**\n\n{response}"})
    
    # Display chat messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("👆 Click a quick action button or type a question below to get started!")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the market..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = copilot.chat(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

else:
    # Not connected - show info
    st.info("👆 Add your Groq API key above to start chatting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
### What can the AI Copilot do?

**📰 Morning Brief**
- Market sentiment overview
- Top 3 trade setups
- Stocks with improving momentum

**🎯 Find Best Setups**
- Confluence detection
- Multi-signal alignment
        """)
    
    with col2:
        st.markdown("""
### Features

**🔍 Stock Analysis**
- Historical score tracking
- Momentum assessment
- Trade thesis generation

**💬 Ask Anything**
- "What's your market view?"
- "Compare NVDA vs AMD"
        """)

# ==================== FOOTER ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    history_file = project_root / "data" / "newsletter_scan_history.json"
    if history_file.exists():
        import json
        with open(history_file) as f:
            history = json.load(f)
        dates = sorted(history.keys(), reverse=True)
        st.caption(f"📊 Newsletter Data: {len(dates)} scan(s)")
    else:
        st.caption("📊 No newsletter data yet")

with col2:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

with col3:
    st.caption("🤖 Llama 3.1 via Groq (FREE)")
