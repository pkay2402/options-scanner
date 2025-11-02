#!/usr/bin/env python3
"""
Morning Trading Dashboard
Smart overview of your watchlist ranked by technical and intraday strength
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Configure Streamlit page
st.set_page_config(
    page_title="Morning Dashboard",
    page_icon="🌅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .strength-strong {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .strength-moderate {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .strength-weak {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .metric-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 0.5rem;
    }
    .bullish { color: #10b981; font-weight: 600; }
    .bearish { color: #ef4444; font-weight: 600; }
    .neutral { color: #f59e0b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    return prices.ewm(span=period, adjust=False).mean().iloc[-1]


def get_technical_strength(symbol):
    """
    Calculate technical strength score (0-100) based on:
    - EMA alignment (8, 21, 50, 200)
    - Price vs EMAs
    - Recent momentum
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            return None
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate EMAs
        ema_8 = calculate_ema(hist['Close'], 8)
        ema_21 = calculate_ema(hist['Close'], 21)
        ema_50 = calculate_ema(hist['Close'], 50)
        ema_200 = calculate_ema(hist['Close'], 200)
        
        score = 0
        signals = []
        
        # Price above EMAs (25 points each, 100 total)
        if current_price > ema_8:
            score += 25
            signals.append("Above EMA-8")
        if current_price > ema_21:
            score += 25
            signals.append("Above EMA-21")
        if current_price > ema_50:
            score += 25
            signals.append("Above EMA-50")
        if current_price > ema_200:
            score += 25
            signals.append("Above EMA-200")
        
        # EMA alignment bonus (perfect order = +20)
        if ema_8 > ema_21 > ema_50 > ema_200:
            score += 20
            signals.append("Perfect EMA alignment")
        
        # Momentum (recent 5 days)
        momentum = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
        
        return {
            'score': min(score, 100),
            'current_price': current_price,
            'ema_8': ema_8,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'ema_200': ema_200,
            'momentum_5d': momentum,
            'signals': signals
        }
    except Exception as e:
        st.error(f"Error calculating technical for {symbol}: {e}")
        return None


def get_intraday_strength(symbol):
    """
    Calculate intraday strength based on:
    - Premarket/current movement
    - Volume vs average
    - Volatility
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Get today's data
        today = ticker.history(period="1d", interval="1m")
        if today.empty:
            return None
        
        # Get recent data for comparison
        recent = ticker.history(period="5d")
        if len(recent) < 2:
            return None
        
        current_price = today['Close'].iloc[-1]
        open_price = today['Open'].iloc[0]
        prev_close = recent['Close'].iloc[-2]
        
        # Intraday change
        intraday_change = ((current_price - open_price) / open_price) * 100
        
        # Gap %
        gap = ((open_price - prev_close) / prev_close) * 100
        
        # Volume vs average
        avg_volume = recent['Volume'].mean()
        current_volume = today['Volume'].sum()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate intraday score
        score = 50  # Base score
        
        # Positive movement
        if intraday_change > 0:
            score += min(intraday_change * 10, 30)  # Up to +30
        else:
            score += max(intraday_change * 10, -30)  # Down to -30
        
        # Volume bonus
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.0:
            score += 10
        
        return {
            'score': max(0, min(score, 100)),
            'intraday_change': intraday_change,
            'gap': gap,
            'volume_ratio': volume_ratio,
            'current_price': current_price
        }
    except Exception as e:
        st.error(f"Error calculating intraday for {symbol}: {e}")
        return None


def get_strength_label(score):
    """Get strength label and color based on score"""
    if score >= 75:
        return "🟢 STRONG", "strength-strong"
    elif score >= 50:
        return "🟡 MODERATE", "strength-moderate"
    else:
        return "🔴 WEAK", "strength-weak"


def main():
    st.title("🌅 Morning Trading Dashboard")
    st.caption("Your watchlist ranked by technical and intraday strength")
    
    # Default watchlist
    default_symbols = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD"
    
    # Settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbols_input = st.text_input(
            "📝 Your Watchlist (comma-separated)",
            value=default_symbols,
            help="Add or remove symbols from your watchlist"
        )
    
    with col2:
        if st.button("🔄 Refresh All", use_container_width=True, type="primary"):
            st.rerun()
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    if not symbols:
        st.warning("Please enter at least one symbol")
        return
    
    st.markdown("---")
    
    # Analyze all symbols
    with st.spinner(f"Analyzing {len(symbols)} symbols..."):
        results = []
        
        progress_bar = st.progress(0)
        for idx, symbol in enumerate(symbols):
            technical = get_technical_strength(symbol)
            intraday = get_intraday_strength(symbol)
            
            if technical and intraday:
                # Combined score (60% technical, 40% intraday)
                combined_score = (technical['score'] * 0.6) + (intraday['score'] * 0.4)
                
                results.append({
                    'symbol': symbol,
                    'combined_score': combined_score,
                    'technical_score': technical['score'],
                    'intraday_score': intraday['score'],
                    'price': technical['current_price'],
                    'intraday_change': intraday['intraday_change'],
                    'gap': intraday['gap'],
                    'volume_ratio': intraday['volume_ratio'],
                    'momentum_5d': technical['momentum_5d'],
                    'signals': technical['signals'],
                    'ema_8': technical['ema_8'],
                    'ema_21': technical['ema_21'],
                    'ema_50': technical['ema_50'],
                    'ema_200': technical['ema_200']
                })
            
            progress_bar.progress((idx + 1) / len(symbols))
        
        progress_bar.empty()
    
    if not results:
        st.error("No data available for the selected symbols")
        return
    
    # Sort by combined score
    results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    # Display results
    st.subheader("📊 Ranked by Overall Strength")
    
    for idx, data in enumerate(results):
        with st.expander(
            f"#{idx+1} {data['symbol']} - ${data['price']:.2f} ({data['intraday_change']:+.2f}%) | "
            f"Overall: {data['combined_score']:.0f}/100",
            expanded=(idx < 3)  # Auto-expand top 3
        ):
            # Top metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                tech_label, tech_class = get_strength_label(data['technical_score'])
                st.markdown(f'<div class="{tech_class}">Technical: {tech_label}</div>', unsafe_allow_html=True)
            
            with metric_cols[1]:
                intra_label, intra_class = get_strength_label(data['intraday_score'])
                st.markdown(f'<div class="{intra_class}">Intraday: {intra_label}</div>', unsafe_allow_html=True)
            
            with metric_cols[2]:
                momentum_class = "bullish" if data['momentum_5d'] > 0 else "bearish"
                st.markdown(f'<div class="metric-card">5D Momentum<br><span class="{momentum_class}">{data["momentum_5d"]:+.2f}%</span></div>', unsafe_allow_html=True)
            
            with metric_cols[3]:
                vol_class = "bullish" if data['volume_ratio'] > 1.2 else "neutral"
                st.markdown(f'<div class="metric-card">Volume<br><span class="{vol_class}">{data["volume_ratio"]:.2f}x avg</span></div>', unsafe_allow_html=True)
            
            # Technical signals
            if data['signals']:
                st.markdown("**✅ Technical Signals:**")
                st.write(", ".join(data['signals']))
            
            # EMA levels
            st.markdown("**📈 EMA Levels:**")
            ema_cols = st.columns(4)
            for i, (period, value) in enumerate([
                ("EMA-8", data['ema_8']),
                ("EMA-21", data['ema_21']),
                ("EMA-50", data['ema_50']),
                ("EMA-200", data['ema_200'])
            ]):
                with ema_cols[i]:
                    above = data['price'] > value
                    color = "bullish" if above else "bearish"
                    st.markdown(f'<span class="{color}">{period}: ${value:.2f}</span>', unsafe_allow_html=True)
            
            # Action buttons
            st.markdown("---")
            action_cols = st.columns(4)
            
            with action_cols[0]:
                if st.button(f"📊 View Options Flow", key=f"flow_{data['symbol']}"):
                    st.info(f"Opening Flow Scanner for {data['symbol']}...")
                    # Could add navigation here
            
            with action_cols[1]:
                if st.button(f"⚡ Gamma Levels", key=f"gamma_{data['symbol']}"):
                    st.info(f"Opening Stock Option Finder for {data['symbol']}...")
                    # Could add navigation here
            
            with action_cols[2]:
                if st.button(f"🎯 Find Opportunities", key=f"opp_{data['symbol']}"):
                    st.info(f"Scanning opportunities for {data['symbol']}...")
            
            with action_cols[3]:
                if st.button(f"📍 Full Analysis", key=f"full_{data['symbol']}"):
                    st.info(f"Loading complete analysis for {data['symbol']}...")
    
    # Summary stats
    st.markdown("---")
    st.subheader("📈 Market Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        strong_count = sum(1 for r in results if r['combined_score'] >= 75)
        st.metric("Strong Setups", f"{strong_count}/{len(results)}")
    
    with summary_cols[1]:
        bullish_count = sum(1 for r in results if r['intraday_change'] > 0)
        st.metric("Bullish Stocks", f"{bullish_count}/{len(results)}")
    
    with summary_cols[2]:
        high_volume = sum(1 for r in results if r['volume_ratio'] > 1.5)
        st.metric("High Volume", f"{high_volume}/{len(results)}")
    
    with summary_cols[3]:
        avg_momentum = np.mean([r['momentum_5d'] for r in results])
        st.metric("Avg 5D Momentum", f"{avg_momentum:+.2f}%")
    
    # Footer
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
