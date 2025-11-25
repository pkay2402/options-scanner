"""
Live Options Flow Feed Page
Real-time streaming options trades with whale detection
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
from src.streaming.schwab_streamer import SchwabStreamer
from src.streaming.models import OptionsFlow
import json

st.set_page_config(
    page_title="Live Options Flow Feed",
    page_icon="📡",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .flow-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00ff00;
    }
    .flow-card.bearish {
        border-left-color: #ff0000;
    }
    .whale-badge {
        background: #ffd700;
        color: #000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.75em;
    }
    .sweep-badge {
        background: #ff6b6b;
        color: #fff;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.75em;
    }
    .block-badge {
        background: #4ecdc4;
        color: #000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.75em;
    }
    .metric-box {
        background: #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #404040;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00ff00;
    }
    .metric-label {
        font-size: 0.9em;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 Live Options Flow Feed")
st.markdown("Real-time options trades with whale detection and audio alerts")

# Initialize session state
if 'streamer' not in st.session_state:
    st.session_state.streamer = None
if 'flows' not in st.session_state:
    st.session_state.flows = []
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

# Sidebar filters
with st.sidebar:
    st.header("🎛️ Filters")
    
    # Watchlist
    watchlist_input = st.text_area(
        "Watchlist (one per line)",
        value="SPY\nQQQ\nAAPL\nTSLA\nNVDA",
        height=150
    )
    watchlist = [s.strip().upper() for s in watchlist_input.split('\n') if s.strip()]
    
    st.divider()
    
    # Premium filters
    st.subheader("💰 Premium Filters")
    min_premium = st.number_input("Min Premium ($)", value=10000, step=1000)
    whale_threshold = st.number_input("Whale Threshold ($)", value=100000, step=10000)
    
    st.divider()
    
    # Trade type filters
    st.subheader("📊 Trade Types")
    show_sweeps = st.checkbox("Show Sweeps", value=True)
    show_blocks = st.checkbox("Show Blocks", value=True)
    show_splits = st.checkbox("Show Splits", value=False)
    
    st.divider()
    
    # Option filters
    st.subheader("🎯 Option Filters")
    option_types = st.multiselect(
        "Type",
        ["CALL", "PUT"],
        default=["CALL", "PUT"]
    )
    
    moneyness = st.multiselect(
        "Moneyness",
        ["ITM", "ATM", "OTM"],
        default=["ITM", "ATM", "OTM"]
    )
    
    max_dte = st.slider("Max DTE", 0, 90, 30)
    
    st.divider()
    
    # Audio alerts
    st.subheader("🔔 Audio Alerts")
    audio_enabled = st.checkbox("Enable Audio Alerts", value=False)
    alert_on_whale = st.checkbox("Alert on Whale Trades", value=True)
    alert_threshold = st.number_input("Alert Premium ($)", value=50000, step=5000)

# Main controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if not st.session_state.is_streaming:
        if st.button("🚀 Start Streaming", type="primary", use_container_width=True):
            # TODO: Initialize streamer with real credentials
            st.session_state.is_streaming = True
            st.success("✅ Streaming started")
            st.rerun()
    else:
        if st.button("⏹️ Stop Streaming", type="secondary", use_container_width=True):
            st.session_state.is_streaming = False
            if st.session_state.streamer:
                st.session_state.streamer.stop()
            st.info("⏸️ Streaming stopped")
            st.rerun()

with col2:
    if st.button("🗑️ Clear Feed", use_container_width=True):
        st.session_state.flows = []
        st.rerun()

with col3:
    auto_refresh = st.checkbox("Auto Refresh", value=True)

st.divider()

# Stats bar
if st.session_state.flows:
    recent_flows = [f for f in st.session_state.flows if 
                   (datetime.now() - f.timestamp).seconds < 300]  # Last 5 min
    
    total_premium = sum(f.premium for f in recent_flows)
    whale_trades = len([f for f in recent_flows if f.is_whale])
    call_volume = len([f for f in recent_flows if f.option_type == "CALL"])
    put_volume = len([f for f in recent_flows if f.option_type == "PUT"])
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(recent_flows)}</div>
            <div class="metric-label">Flows (5m)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #ffd700;">${total_premium/1_000_000:.2f}M</div>
            <div class="metric-label">Premium (5m)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #ffd700;">🐋 {whale_trades}</div>
            <div class="metric-label">Whale Trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0
        ratio_color = "#ff0000" if pc_ratio > 1 else "#00ff00"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {ratio_color};">{pc_ratio:.2f}</div>
            <div class="metric-label">P/C Ratio (5m)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

# Live feed
st.subheader("🔥 Live Options Flow")

# Filter flows
filtered_flows = st.session_state.flows

# Apply filters
filtered_flows = [f for f in filtered_flows if f.symbol in watchlist]
filtered_flows = [f for f in filtered_flows if f.premium >= min_premium]
filtered_flows = [f for f in filtered_flows if f.option_type in option_types]
filtered_flows = [f for f in filtered_flows if f.moneyness in moneyness]
filtered_flows = [f for f in filtered_flows if f.days_to_expiry <= max_dte]

# Trade type filter
if not show_sweeps:
    filtered_flows = [f for f in filtered_flows if f.trade_type != "SWEEP"]
if not show_blocks:
    filtered_flows = [f for f in filtered_flows if f.trade_type != "BLOCK"]
if not show_splits:
    filtered_flows = [f for f in filtered_flows if f.trade_type != "SPLIT"]

# Sort by timestamp (newest first)
filtered_flows = sorted(filtered_flows, key=lambda x: x.timestamp, reverse=True)

# Display flows
if filtered_flows:
    for flow in filtered_flows[:50]:  # Show last 50
        sentiment_class = "bearish" if flow.option_type == "PUT" else ""
        
        # Build badges
        badges = []
        if flow.is_whale:
            badges.append('<span class="whale-badge">🐋 WHALE</span>')
        if flow.trade_type == "SWEEP":
            badges.append('<span class="sweep-badge">SWEEP</span>')
        elif flow.trade_type == "BLOCK":
            badges.append('<span class="block-badge">BLOCK</span>')
        
        badges_html = " ".join(badges)
        
        # Flow card
        st.markdown(f"""
        <div class="flow-card {sentiment_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2em; font-weight: bold;">{flow.symbol}</span>
                    <span style="color: #888; margin-left: 10px;">{flow.timestamp.strftime('%H:%M:%S')}</span>
                    <span style="margin-left: 10px;">{badges_html}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5em; font-weight: bold; color: {'#ff0000' if flow.option_type == 'PUT' else '#00ff00'};">
                        {flow.premium_formatted}
                    </div>
                    <div style="color: #888; font-size: 0.9em;">
                        {flow.size} contracts @ ${flow.price:.2f}
                    </div>
                </div>
            </div>
            <div style="margin-top: 10px; color: #888;">
                <span>{flow.option_type}</span> • 
                <span>${flow.strike}</span> • 
                <span>{flow.expiry.strftime('%b %d')}</span> • 
                <span>{flow.days_to_expiry}DTE</span> • 
                <span>{flow.moneyness}</span> • 
                <span>Spot: ${flow.underlying_price:.2f}</span>
                {f' • IV: {flow.iv:.1f}%' if flow.iv else ''}
                {f' • Δ: {flow.delta:.2f}' if flow.delta else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    if st.session_state.is_streaming:
        st.info("🔍 Waiting for flows matching your filters...")
    else:
        st.warning("⏸️ Click 'Start Streaming' to begin receiving live options flow")

# Demo data for testing (remove when live)
if st.session_state.is_streaming and len(st.session_state.flows) < 10:
    with st.expander("🧪 Load Demo Data (Testing)"):
        if st.button("Generate Test Flows"):
            import random
            
            for _ in range(20):
                flow = OptionsFlow(
                    symbol=random.choice(watchlist),
                    option_symbol="",
                    strike=random.uniform(400, 600),
                    expiry=datetime.now() + timedelta(days=random.randint(0, 30)),
                    option_type=random.choice(["CALL", "PUT"]),
                    timestamp=datetime.now() - timedelta(seconds=random.randint(0, 300)),
                    price=random.uniform(1, 50),
                    size=random.randint(10, 500),
                    premium=random.uniform(10000, 500000),
                    bid=0,
                    ask=0,
                    underlying_price=random.uniform(450, 550),
                    iv=random.uniform(20, 80),
                    delta=random.uniform(0.1, 0.9),
                    trade_type=random.choice(["SWEEP", "BLOCK", "SINGLE"]),
                    sentiment=random.choice(["BULLISH", "BEARISH", "NEUTRAL"])
                )
                st.session_state.flows.append(flow)
            
            st.success("✅ Generated 20 test flows")
            st.rerun()

# Auto refresh
if auto_refresh and st.session_state.is_streaming:
    time.sleep(1)
    st.rerun()
