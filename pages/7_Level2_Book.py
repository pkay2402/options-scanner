"""
Level 2 Options Book Visualizer
Real-time order book depth for options contracts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from src.streaming.schwab_streamer import SchwabStreamer
from src.streaming.models import OptionsBook, OrderBookLevel

st.set_page_config(
    page_title="Level 2 Options Book",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .book-container {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .bid-row {
        background: rgba(0, 255, 0, 0.1);
        padding: 8px;
        margin: 2px 0;
        border-radius: 4px;
        display: flex;
        justify-content: space-between;
        border-left: 3px solid #00ff00;
    }
    .ask-row {
        background: rgba(255, 0, 0, 0.1);
        padding: 8px;
        margin: 2px 0;
        border-radius: 4px;
        display: flex;
        justify-content: space-between;
        border-left: 3px solid #ff0000;
    }
    .spread-indicator {
        background: #2d2d2d;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        text-align: center;
        border: 2px dashed #ffd700;
    }
    .book-metric {
        background: #2d2d2d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #404040;
    }
    .imbalance-bar {
        height: 20px;
        background: linear-gradient(to right, #00ff00 0%, #ff0000 100%);
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Level 2 Options Book")
st.markdown("Real-time order book depth and liquidity analysis")

# Initialize session state
if 'streamer' not in st.session_state:
    st.session_state.streamer = None
if 'books' not in st.session_state:
    st.session_state.books = {}
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

# Sidebar - Option selector
with st.sidebar:
    st.header("🎯 Option Selection")
    
    # Underlying symbol
    underlying = st.text_input("Underlying Symbol", value="SPY").upper()
    
    # Option type
    option_type = st.selectbox("Type", ["CALL", "PUT"])
    
    # Expiration
    expiry_date = st.date_input(
        "Expiration Date",
        value=datetime.now() + timedelta(days=7)
    )
    
    # Strike
    strike = st.number_input("Strike Price", value=450.0, step=1.0)
    
    st.divider()
    
    # Display settings
    st.header("⚙️ Display Settings")
    
    depth_levels = st.slider("Book Depth (levels)", 5, 50, 20)
    
    show_size = st.checkbox("Show Size", value=True)
    show_orders = st.checkbox("Show # Orders", value=True)
    show_notional = st.checkbox("Show Notional", value=True)
    
    st.divider()
    
    # Alerts
    st.header("🔔 Alerts")
    
    alert_on_imbalance = st.checkbox("Alert on Imbalance", value=False)
    imbalance_threshold = st.slider("Imbalance Ratio", 1.5, 5.0, 2.0, 0.1)
    
    alert_on_spread = st.checkbox("Alert on Wide Spread", value=False)
    spread_threshold = st.slider("Spread % Threshold", 1.0, 20.0, 5.0, 0.5)

# Main controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    option_symbol = f"{underlying}_{expiry_date.strftime('%y%m%d')}{'C' if option_type == 'CALL' else 'P'}{int(strike)}"
    st.info(f"📍 Monitoring: **{option_symbol}**")

with col2:
    if not st.session_state.is_streaming:
        if st.button("🚀 Start Streaming", type="primary", use_container_width=True):
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

with col3:
    auto_refresh = st.checkbox("Auto Refresh", value=True)

st.divider()

# Get current book
current_book = st.session_state.books.get(option_symbol)

if current_book:
    # Metrics row
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.markdown(f"""
        <div class="book-metric">
            <div style="font-size: 1.5em; font-weight: bold; color: #00ff00;">
                ${current_book.best_bid:.2f}
            </div>
            <div style="color: #888; font-size: 0.85em;">Best Bid</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="book-metric">
            <div style="font-size: 1.5em; font-weight: bold; color: #ff0000;">
                ${current_book.best_ask:.2f}
            </div>
            <div style="color: #888; font-size: 0.85em;">Best Ask</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        spread_color = "#ff0000" if current_book.spread_pct and current_book.spread_pct > spread_threshold else "#ffd700"
        st.markdown(f"""
        <div class="book-metric">
            <div style="font-size: 1.5em; font-weight: bold; color: {spread_color};">
                {current_book.spread_pct:.2f}%
            </div>
            <div style="color: #888; font-size: 0.85em;">Spread</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="book-metric">
            <div style="font-size: 1.5em; font-weight: bold; color: #00ff00;">
                {current_book.total_bid_size:,}
            </div>
            <div style="color: #888; font-size: 0.85em;">Total Bid Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col5:
        st.markdown(f"""
        <div class="book-metric">
            <div style="font-size: 1.5em; font-weight: bold; color: #ff0000;">
                {current_book.total_ask_size:,}
            </div>
            <div style="color: #888; font-size: 0.85em;">Total Ask Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Imbalance indicator
    if current_book.imbalance:
        imbalance = current_book.imbalance
        imbalance_pct = min(max((imbalance - 1) / 2, 0), 1) * 100  # Normalize to 0-100
        
        imbalance_text = f"{'BID HEAVY' if imbalance > 1 else 'ASK HEAVY'} ({imbalance:.2f}x)"
        imbalance_color = "#00ff00" if imbalance > 1 else "#ff0000"
        
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <div style="text-align: center; color: {imbalance_color}; font-weight: bold; margin-bottom: 10px;">
                {imbalance_text}
            </div>
            <div style="background: #2d2d2d; height: 30px; border-radius: 15px; overflow: hidden; position: relative;">
                <div style="position: absolute; left: {imbalance_pct}%; top: 0; bottom: 0; width: 4px; background: #ffd700; z-index: 2;"></div>
                <div style="background: linear-gradient(to right, #00ff00 0%, #2d2d2d 50%, #ff0000 100%); height: 100%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; color: #888; font-size: 0.85em;">
                <span>← More Bids</span>
                <span>More Asks →</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Order book visualization
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📗 Bid Side")
        
        # Get bid levels (limit to depth_levels)
        bid_levels = current_book.bids[:depth_levels]
        
        if bid_levels:
            # Create depth chart
            bid_prices = [level.price for level in bid_levels]
            bid_sizes = [level.size for level in bid_levels]
            
            # Cumulative size
            cumulative_bid = []
            total = 0
            for size in bid_sizes:
                total += size
                cumulative_bid.append(total)
            
            # Display book rows
            for level in bid_levels:
                # Size bar width
                max_size = max(bid_sizes) if bid_sizes else 1
                bar_width = (level.size / max_size) * 100
                
                st.markdown(f"""
                <div class="bid-row">
                    <div style="flex: 1;">
                        <div style="background: rgba(0, 255, 0, 0.2); height: 100%; width: {bar_width}%; position: absolute; left: 0; border-radius: 4px;"></div>
                        <span style="position: relative; font-weight: bold;">${level.price:.2f}</span>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        {f'{level.size:,}' if show_size else ''}
                    </div>
                    <div style="flex: 1; text-align: center; color: #888;">
                        {f'{level.num_orders}' if show_orders else ''}
                    </div>
                    <div style="flex: 1; text-align: right;">
                        {f'${level.notional:,.0f}' if show_notional else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Depth chart
            fig_bid = go.Figure()
            fig_bid.add_trace(go.Scatter(
                x=bid_prices,
                y=cumulative_bid,
                fill='tozeroy',
                line=dict(color='#00ff00', width=2),
                name='Cumulative Bid Size'
            ))
            fig_bid.update_layout(
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                title="Cumulative Bid Depth",
                xaxis_title="Price",
                yaxis_title="Cumulative Size",
                hovermode='x unified'
            )
            st.plotly_chart(fig_bid, use_container_width=True)
        else:
            st.warning("No bid levels available")
    
    with col_right:
        st.subheader("📕 Ask Side")
        
        # Get ask levels
        ask_levels = current_book.asks[:depth_levels]
        
        if ask_levels:
            # Create depth chart
            ask_prices = [level.price for level in ask_levels]
            ask_sizes = [level.size for level in ask_levels]
            
            # Cumulative size
            cumulative_ask = []
            total = 0
            for size in ask_sizes:
                total += size
                cumulative_ask.append(total)
            
            # Display book rows
            for level in ask_levels:
                # Size bar width
                max_size = max(ask_sizes) if ask_sizes else 1
                bar_width = (level.size / max_size) * 100
                
                st.markdown(f"""
                <div class="ask-row">
                    <div style="flex: 1;">
                        <div style="background: rgba(255, 0, 0, 0.2); height: 100%; width: {bar_width}%; position: absolute; left: 0; border-radius: 4px;"></div>
                        <span style="position: relative; font-weight: bold;">${level.price:.2f}</span>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        {f'{level.size:,}' if show_size else ''}
                    </div>
                    <div style="flex: 1; text-align: center; color: #888;">
                        {f'{level.num_orders}' if show_orders else ''}
                    </div>
                    <div style="flex: 1; text-align: right;">
                        {f'${level.notional:,.0f}' if show_notional else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Depth chart
            fig_ask = go.Figure()
            fig_ask.add_trace(go.Scatter(
                x=ask_prices,
                y=cumulative_ask,
                fill='tozeroy',
                line=dict(color='#ff0000', width=2),
                name='Cumulative Ask Size'
            ))
            fig_ask.update_layout(
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                title="Cumulative Ask Depth",
                xaxis_title="Price",
                yaxis_title="Cumulative Size",
                hovermode='x unified'
            )
            st.plotly_chart(fig_ask, use_container_width=True)
        else:
            st.warning("No ask levels available")
    
    # Combined depth chart
    st.subheader("📊 Full Book Depth")
    
    if bid_levels and ask_levels:
        fig_combined = go.Figure()
        
        # Bid side
        bid_prices = [level.price for level in bid_levels]
        bid_sizes = [level.size for level in bid_levels]
        cumulative_bid = []
        total = 0
        for size in reversed(bid_sizes):
            total += size
            cumulative_bid.append(total)
        cumulative_bid = list(reversed(cumulative_bid))
        
        fig_combined.add_trace(go.Scatter(
            x=bid_prices,
            y=cumulative_bid,
            fill='tozeroy',
            line=dict(color='#00ff00', width=2),
            name='Bids',
            hovertemplate='Price: $%{x:.2f}<br>Size: %{y:,}<extra></extra>'
        ))
        
        # Ask side
        ask_prices = [level.price for level in ask_levels]
        ask_sizes = [level.size for level in ask_levels]
        cumulative_ask = []
        total = 0
        for size in ask_sizes:
            total += size
            cumulative_ask.append(total)
        
        fig_combined.add_trace(go.Scatter(
            x=ask_prices,
            y=cumulative_ask,
            fill='tozeroy',
            line=dict(color='#ff0000', width=2),
            name='Asks',
            hovertemplate='Price: $%{x:.2f}<br>Size: %{y:,}<extra></extra>'
        ))
        
        # Mark spread
        if current_book.best_bid and current_book.best_ask:
            mid = (current_book.best_bid + current_book.best_ask) / 2
            fig_combined.add_vline(
                x=mid,
                line_dash="dash",
                line_color="#ffd700",
                annotation_text=f"Mid: ${mid:.2f}",
                annotation_position="top"
            )
        
        fig_combined.update_layout(
            template='plotly_dark',
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Price",
            yaxis_title="Cumulative Size",
            hovermode='x unified',
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    # Timestamp
    st.caption(f"Last updated: {current_book.timestamp.strftime('%H:%M:%S.%f')[:-3]}")

else:
    if st.session_state.is_streaming:
        st.info("🔍 Waiting for order book data...")
        
        # Demo data for testing
        with st.expander("🧪 Load Demo Data (Testing)"):
            if st.button("Generate Test Book"):
                import random
                
                # Generate demo bid levels
                bids = []
                base_bid = strike - 0.50
                for i in range(20):
                    level = OrderBookLevel(
                        price=base_bid - (i * 0.05),
                        size=random.randint(10, 200),
                        num_orders=random.randint(1, 10),
                        side='BID'
                    )
                    bids.append(level)
                
                # Generate demo ask levels
                asks = []
                base_ask = strike + 0.50
                for i in range(20):
                    level = OrderBookLevel(
                        price=base_ask + (i * 0.05),
                        size=random.randint(10, 200),
                        num_orders=random.randint(1, 10),
                        side='ASK'
                    )
                    asks.append(level)
                
                # Create book
                book = OptionsBook(
                    option_symbol=option_symbol,
                    symbol=underlying,
                    strike=strike,
                    expiry=datetime.combine(expiry_date, datetime.min.time()),
                    option_type=option_type,
                    timestamp=datetime.now(),
                    bids=bids,
                    asks=asks
                )
                
                st.session_state.books[option_symbol] = book
                st.success("✅ Generated demo order book")
                st.rerun()
    else:
        st.warning("⏸️ Click 'Start Streaming' to view live order book data")

# Auto refresh
if auto_refresh and st.session_state.is_streaming:
    time.sleep(0.5)
    st.rerun()
