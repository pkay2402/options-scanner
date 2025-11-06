import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(page_title="EMA Cloud Swing Scanner", layout="wide")

# Stock universe
STOCKS = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'ADBE', 'CRM', 'ORCL'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'DHR', 'BMY'],
    'AI': ['NVDA', 'PLTR', 'NBIS', 'SNOW', 'NET', 'DDOG', 'PANW', 'CRWD', 'ZS', 'CRWV','COIN','APP','RDDT']
}

ALL_STOCKS = list(set([stock for category in STOCKS.values() for stock in category]))

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def get_ema_cloud_data(ticker, days_back=100):
    """Fetch data and calculate all EMA clouds"""
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get different timeframes
        data_1h = stock.history(period='60d', interval='1h')
        data_4h = stock.history(period='60d', interval='1h')  # We'll resample
        data_daily = stock.history(period='1y', interval='1d')
        
        if data_daily.empty:
            return None
        
        # Calculate EMAs for daily (our main swing trading timeframe)
        data_daily['EMA_5'] = calculate_ema(data_daily, 5)
        data_daily['EMA_12'] = calculate_ema(data_daily, 12)
        data_daily['EMA_34'] = calculate_ema(data_daily, 34)
        data_daily['EMA_50'] = calculate_ema(data_daily, 50)
        
        # Calculate 1H EMAs if available
        if not data_1h.empty and len(data_1h) > 50:
            data_1h['EMA_5'] = calculate_ema(data_1h, 5)
            data_1h['EMA_12'] = calculate_ema(data_1h, 12)
            data_1h['EMA_34'] = calculate_ema(data_1h, 34)
            data_1h['EMA_50'] = calculate_ema(data_1h, 50)
        
        return {
            'daily': data_daily,
            'hourly': data_1h,
            'ticker': ticker
        }
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return None

def analyze_cloud_position(data_daily, data_hourly):
    """Analyze EMA cloud positions and generate signals"""
    
    current_price = data_daily['Close'].iloc[-1]
    prev_price = data_daily['Close'].iloc[-2] if len(data_daily) > 1 else current_price
    
    # Daily EMAs (Ranks 9-10: Macro direction)
    daily_ema_5_12_cloud_top = max(data_daily['EMA_5'].iloc[-1], data_daily['EMA_12'].iloc[-1])
    daily_ema_5_12_cloud_bottom = min(data_daily['EMA_5'].iloc[-1], data_daily['EMA_12'].iloc[-1])
    
    daily_ema_34_50_cloud_top = max(data_daily['EMA_34'].iloc[-1], data_daily['EMA_50'].iloc[-1])
    daily_ema_34_50_cloud_bottom = min(data_daily['EMA_34'].iloc[-1], data_daily['EMA_50'].iloc[-1])
    
    # Determine cloud alignment
    daily_5_12_bullish = data_daily['EMA_5'].iloc[-1] > data_daily['EMA_12'].iloc[-1]
    daily_34_50_bullish = data_daily['EMA_34'].iloc[-1] > data_daily['EMA_50'].iloc[-1]
    
    # Price position relative to clouds
    above_weak_cloud = current_price > daily_ema_5_12_cloud_top
    above_strong_cloud = current_price > daily_ema_34_50_cloud_top
    below_weak_cloud = current_price < daily_ema_5_12_cloud_bottom
    below_strong_cloud = current_price < daily_ema_34_50_cloud_bottom
    
    # Calculate cloud strength score
    cloud_alignment_score = 0
    if daily_5_12_bullish:
        cloud_alignment_score += 1
    if daily_34_50_bullish:
        cloud_alignment_score += 2
    if above_weak_cloud:
        cloud_alignment_score += 1
    if above_strong_cloud:
        cloud_alignment_score += 2
        
    # Bearish scoring
    if not daily_5_12_bullish:
        cloud_alignment_score -= 1
    if not daily_34_50_bullish:
        cloud_alignment_score -= 2
    if below_weak_cloud:
        cloud_alignment_score -= 1
    if below_strong_cloud:
        cloud_alignment_score -= 2
    
    signal = None
    reasons = []
    targets = {'target1': None, 'target2': None}
    stop_loss = None
    
    # LONG SIGNALS (Score >= 4)
    if cloud_alignment_score >= 4:
        signal = 'LONG'
        
        if above_strong_cloud and daily_34_50_bullish:
            reasons.append("✅ Price above strong Rank 10 cloud (34-50 daily)")
        if above_weak_cloud and daily_5_12_bullish:
            reasons.append("✅ Price above Rank 9 cloud (5-12 daily) - momentum accelerator")
        if daily_5_12_bullish and daily_34_50_bullish:
            reasons.append("✅ Full cloud alignment bullish (swing trader's dream)")
        if current_price > prev_price:
            reasons.append("✅ Price making higher highs")
            
        # Calculate targets (using ATR-like approach)
        atr = data_daily['High'].rolling(14).max() - data_daily['Low'].rolling(14).min()
        avg_atr = atr.iloc[-14:].mean()
        
        targets['target1'] = current_price + (avg_atr * 1.5)
        targets['target2'] = current_price + (avg_atr * 3.0)
        
        # Stop loss: Low of last 5 days (tighter for options trading)
        stop_loss = data_daily['Low'].iloc[-5:].min()
        
    # SHORT SIGNALS (Score <= -4)
    elif cloud_alignment_score <= -4:
        signal = 'SHORT'
        
        if below_strong_cloud and not daily_34_50_bullish:
            reasons.append("🔻 Price below strong Rank 10 cloud (34-50 daily)")
        if below_weak_cloud and not daily_5_12_bullish:
            reasons.append("🔻 Price below Rank 9 cloud (5-12 daily) - momentum breakdown")
        if not daily_5_12_bullish and not daily_34_50_bullish:
            reasons.append("🔻 Full cloud alignment bearish")
        if current_price < prev_price:
            reasons.append("🔻 Price making lower lows")
            
        # Calculate targets
        atr = data_daily['High'].rolling(14).max() - data_daily['Low'].rolling(14).min()
        avg_atr = atr.iloc[-14:].mean()
        
        targets['target1'] = current_price - (avg_atr * 1.5)
        targets['target2'] = current_price - (avg_atr * 3.0)
        
        # Stop loss: High of last 5 days (tighter for options trading)
        stop_loss = data_daily['High'].iloc[-5:].max()
    
    # NEUTRAL (waiting for setup)
    else:
        reasons.append("⏸️ No clear swing trade setup - clouds mixed or price consolidating")
        if -2 <= cloud_alignment_score <= 2:
            reasons.append("⚠️ Price in 'no man's land' between Rank 9 and 10 clouds")
    
    return {
        'signal': signal,
        'reasons': reasons,
        'current_price': current_price,
        'targets': targets,
        'stop_loss': stop_loss,
        'score': cloud_alignment_score,
        'ema_5_12_cloud': (daily_ema_5_12_cloud_bottom, daily_ema_5_12_cloud_top),
        'ema_34_50_cloud': (daily_ema_34_50_cloud_bottom, daily_ema_34_50_cloud_top)
    }

# Streamlit UI
st.title("📊 EMA Cloud Swing Trading Scanner")
st.markdown("### Based on Daily EMA Clouds (Ranks 9-10) for Swing Traders")

# Sidebar
with st.sidebar:
    st.header("Scanner Settings")
    selected_sectors = st.multiselect(
        "Select Sectors",
        options=list(STOCKS.keys()),
        default=list(STOCKS.keys())
    )
    
    min_score = st.slider("Minimum Signal Strength", -6, 6, 4)
    
    scan_button = st.button("🔍 Run Scan", type="primary")

if scan_button:
    # Get stocks from selected sectors
    stocks_to_scan = list(set([
        stock for sector in selected_sectors 
        for stock in STOCKS[sector]
    ]))
    
    st.info(f"Scanning {len(stocks_to_scan)} stocks...")
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(stocks_to_scan):
        progress_bar.progress((idx + 1) / len(stocks_to_scan))
        
        data = get_ema_cloud_data(ticker)
        
        if data is not None:
            analysis = analyze_cloud_position(data['daily'], data['hourly'])
            
            if analysis['signal'] and abs(analysis['score']) >= min_score:
                sector = next((s for s, stocks in STOCKS.items() if ticker in stocks), 'Unknown')
                results.append({
                    'Ticker': ticker,
                    'Sector': sector,
                    'Signal': analysis['signal'],
                    'Score': analysis['score'],
                    'Price': f"${analysis['current_price']:.2f}",
                    'Target 1': f"${analysis['targets']['target1']:.2f}" if analysis['targets']['target1'] else 'N/A',
                    'Target 2': f"${analysis['targets']['target2']:.2f}" if analysis['targets']['target2'] else 'N/A',
                    'Stop Loss': f"${analysis['stop_loss']:.2f}" if analysis['stop_loss'] else 'N/A',
                    'Reasons': analysis['reasons'],
                    'Analysis': analysis
                })
        
        time.sleep(0.1)  # Rate limiting
    
    progress_bar.empty()
    
    # Display results
    if results:
        st.success(f"Found {len(results)} trading opportunities!")
        
        # Separate LONG and SHORT
        long_trades = [r for r in results if r['Signal'] == 'LONG']
        short_trades = [r for r in results if r['Signal'] == 'SHORT']
        
        # LONG OPPORTUNITIES
        if long_trades:
            st.markdown("## 🟢 LONG Opportunities")
            for trade in sorted(long_trades, key=lambda x: x['Score'], reverse=True):
                with st.expander(f"{trade['Ticker']} - {trade['Sector']} | Score: {trade['Score']}/6"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", trade['Price'])
                        st.metric("Stop Loss", trade['Stop Loss'])
                    
                    with col2:
                        st.metric("Target 1", trade['Target 1'])
                        potential_gain_1 = ((float(trade['Target 1'].replace('$','')) / float(trade['Price'].replace('$',''))) - 1) * 100
                        st.caption(f"Potential: +{potential_gain_1:.1f}%")
                    
                    with col3:
                        st.metric("Target 2", trade['Target 2'])
                        potential_gain_2 = ((float(trade['Target 2'].replace('$','')) / float(trade['Price'].replace('$',''))) - 1) * 100
                        st.caption(f"Potential: +{potential_gain_2:.1f}%")
                    
                    st.markdown("**Why LONG:**")
                    for reason in trade['Reasons']:
                        st.markdown(f"- {reason}")
                    
                    # Cloud levels
                    st.markdown("**Key Levels:**")
                    st.markdown(f"- Rank 9 Cloud (5-12): ${trade['Analysis']['ema_5_12_cloud'][0]:.2f} - ${trade['Analysis']['ema_5_12_cloud'][1]:.2f}")
                    st.markdown(f"- Rank 10 Cloud (34-50): ${trade['Analysis']['ema_34_50_cloud'][0]:.2f} - ${trade['Analysis']['ema_34_50_cloud'][1]:.2f}")
        
        # SHORT OPPORTUNITIES
        if short_trades:
            st.markdown("## 🔴 SHORT Opportunities")
            for trade in sorted(short_trades, key=lambda x: x['Score']):
                with st.expander(f"{trade['Ticker']} - {trade['Sector']} | Score: {trade['Score']}/6"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", trade['Price'])
                        st.metric("Stop Loss", trade['Stop Loss'])
                    
                    with col2:
                        st.metric("Target 1", trade['Target 1'])
                        potential_gain_1 = ((1 - float(trade['Target 1'].replace('$','')) / float(trade['Price'].replace('$',''))) ) * 100
                        st.caption(f"Potential: +{potential_gain_1:.1f}%")
                    
                    with col3:
                        st.metric("Target 2", trade['Target 2'])
                        potential_gain_2 = ((1 - float(trade['Target 2'].replace('$','')) / float(trade['Price'].replace('$',''))) ) * 100
                        st.caption(f"Potential: +{potential_gain_2:.1f}%")
                    
                    st.markdown("**Why SHORT:**")
                    for reason in trade['Reasons']:
                        st.markdown(f"- {reason}")
                    
                    # Cloud levels
                    st.markdown("**Key Levels:**")
                    st.markdown(f"- Rank 9 Cloud (5-12): ${trade['Analysis']['ema_5_12_cloud'][0]:.2f} - ${trade['Analysis']['ema_5_12_cloud'][1]:.2f}")
                    st.markdown(f"- Rank 10 Cloud (34-50): ${trade['Analysis']['ema_34_50_cloud'][0]:.2f} - ${trade['Analysis']['ema_34_50_cloud'][1]:.2f}")
    
    else:
        st.warning("No trading opportunities found with current criteria. Try adjusting the minimum signal strength.")

else:
    st.info("👈 Configure your settings and click 'Run Scan' to find swing trading opportunities")
    
    st.markdown("""
    ### How This Scanner Works:
    
    **EMA Cloud Hierarchy for Swing Trading:**
    - **Rank 9 (Daily 5-12 cloud)**: Macro direction accelerator - momentum confirmer
    - **Rank 10 (Daily 34-50 cloud)**: The Granddaddy cloud - major support/resistance
    
    **Signal Scoring System:**
    - +6 to +4: Strong LONG (all clouds aligned bullish, price above)
    - -4 to -6: Strong SHORT (all clouds aligned bearish, price below)
    - -3 to +3: No clear setup (wait for alignment)
    
    **Targets Calculated Using:**
    - 14-day Average True Range (ATR)
    - Target 1: 1.5x ATR move
    - Target 2: 3.0x ATR move
    - Stop Loss: Opposite side of Rank 10 cloud (strongest level)
    """)
