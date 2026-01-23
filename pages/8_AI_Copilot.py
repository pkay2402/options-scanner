#!/usr/bin/env python3
"""
AI Trading Copilot - Chat Interface
Powered by Groq's free Llama 3.1 API
"""

import streamlit as st
import sys
import requests
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

# Get API key from secrets (multiple methods)
import os
api_key = None
discord_webhook = None

# Get Discord webhook from secrets
try:
    discord_webhook = st.secrets["alerts"]["discord_webhook"]
except:
    pass

# Debug: show available secrets keys
# st.write("Available secrets:", list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else "none")

# Method 1: Direct access at top level
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    pass

# Method 2: Check inside [alerts] section
if not api_key:
    try:
        api_key = st.secrets["alerts"]["GROQ_API_KEY"]
    except:
        pass

# Method 3: Environment variable fallback
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

# Initialize copilot
copilot = TradingCopilot(api_key=api_key)

# ==================== PAGE HEADER ====================
st.title("🤖 AI Trading Copilot")
st.markdown("*Your AI-powered market analyst - powered by Llama 3.1 via Groq (FREE)*")

# ==================== NEWS SUMMARY AT TOP ====================
with st.expander("📰 Today's News Summary (Upgrades/Downgrades)", expanded=True):
    news_summary = copilot.get_news_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔼 Recent Upgrades")
        if news_summary['upgraded_tickers']:
            # Show tickers with upgrades
            for ticker, headlines in list(news_summary['upgraded_tickers'].items())[:8]:
                st.markdown(f"**{ticker}** - {headlines[0][:60]}...")
        else:
            st.caption("No recent upgrades found")
    
    with col2:
        st.markdown("### 🔽 Recent Downgrades")
        if news_summary['downgraded_tickers']:
            # Show tickers with downgrades
            for ticker, headlines in list(news_summary['downgraded_tickers'].items())[:8]:
                st.markdown(f"**{ticker}** - {headlines[0][:60]}...")
        else:
            st.caption("No recent downgrades found")
    
    # Quick summary
    st.caption(f"📊 Total: {news_summary['total_upgrades']} upgrades, {news_summary['total_downgrades']} downgrades from Google Alerts")

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

# Function to send to Discord
def send_to_discord(content: str, title: str = "AI Copilot Analysis"):
    """Send message to Discord webhook"""
    if not discord_webhook:
        return False, "Discord webhook not configured"
    
    # Discord has 2000 char limit per message, split if needed
    # Use embed for nicer formatting
    embed = {
        "title": f"🤖 {title}",
        "description": content[:4000],  # Discord embed description limit
        "color": 5814783,  # Blue color
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "AI Trading Copilot"}
    }
    
    payload = {"embeds": [embed]}
    
    try:
        response = requests.post(discord_webhook, json=payload, timeout=10)
        if response.status_code == 204:
            return True, "Sent to Discord!"
        else:
            return False, f"Discord error: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ==================== MAIN INTERFACE ====================
if copilot.is_available():
    # Chat input at top
    if prompt := st.chat_input("Ask me anything about the market..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = copilot.chat(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Analyze stock section
    st.markdown("### 🔍 Analyze Stock")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_ticker = st.text_input("Enter ticker symbol", placeholder="NVDA, AMD, AAPL...", label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    
    if analyze_btn and analyze_ticker:
        ticker = analyze_ticker.upper().strip()
        st.session_state.messages.append({"role": "user", "content": f"Analyze {ticker}"})
        with st.spinner(f"🔍 Analyzing {ticker} (fetching live data, options flow, IV, gamma walls, earnings)..."):
            response = copilot.analyze_stock(ticker)
        st.session_state.messages.append({"role": "assistant", "content": f"**🔍 Analysis: {ticker}**\n\n{response}"})
        st.rerun()
    
    st.markdown("---")
    
    # Display chat messages
    if st.session_state.messages:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Add Discord send button for assistant messages
                if message["role"] == "assistant" and discord_webhook:
                    if st.button("📤 Send to Discord", key=f"discord_{idx}"):
                        success, msg = send_to_discord(message["content"])
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Enter a ticker above or ask a question to get started")

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
