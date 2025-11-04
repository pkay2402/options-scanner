"""
Power Hour Predictor - Historical Pattern Analysis
Analyzes last 5 trading days to predict next-day movement based on power hour signals
Run 10 mins before close to make buy/sell decisions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

st.set_page_config(
    page_title="Power Hour Predictor",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .buy-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .sell-signal {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
    .hold-signal {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
    }
    .accuracy-high {
        color: #28a745;
        font-weight: bold;
    }
    .accuracy-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .accuracy-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stock-row {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Power Hour Predictor")
st.markdown("**Predict tomorrow's movement based on power hour patterns - Run 10 mins before close for best results**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Mode",
        options=["Single Stock Analysis", "Batch Watchlist Scanner"],
        help="Single stock for detailed analysis, or batch mode for your watchlist"
    )
    
    if mode == "Single Stock Analysis":
        symbol = st.text_input("Stock Symbol", value="SPY").upper()
        symbols = [symbol]
    else:
        symbols_input = st.text_area(
            "Watchlist (one per line or comma-separated)",
            value="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX",
            height=150,
            help="Enter your top stocks to scan"
        )
        # Parse symbols
        symbols = []
        for line in symbols_input.split('\n'):
            symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
    
    lookback_days = st.slider(
        "Historical Days to Analyze",
        min_value=3,
        max_value=10,
        value=5,
        help="More days = better pattern accuracy"
    )
    
    power_hour_start = st.selectbox(
        "Power Hour Start",
        options=["2:00 PM", "2:30 PM", "3:00 PM"],
        index=2
    )
    
    min_accuracy = st.slider(
        "Minimum Accuracy Threshold",
        min_value=50,
        max_value=90,
        value=70,
        help="Only show predictions with this accuracy or higher"
    )
    
    analyze_button = st.button("🔍 Analyze & Predict", type="primary", use_container_width=True)

def get_power_hour_start_hour(selection):
    """Convert selection to hour"""
    mapping = {"2:00 PM": 14, "2:30 PM": 14, "3:00 PM": 15}
    return mapping[selection]

def analyze_day_power_hour(candles_df, power_hour_start_hour=15):
    """Analyze single day's power hour and return metrics"""
    try:
        if candles_df.empty:
            return None
        
        # Convert to ET
        candles_df['datetime'] = pd.to_datetime(candles_df['datetime'], unit='ms', utc=True)
        candles_df['datetime'] = candles_df['datetime'].dt.tz_convert('America/New_York')
        candles_df['hour'] = candles_df['datetime'].dt.hour
        candles_df['minute'] = candles_df['datetime'].dt.minute
        
        # Filter to market hours
        candles_df = candles_df[
            ((candles_df['hour'] == 9) & (candles_df['minute'] >= 30)) |
            ((candles_df['hour'] >= 10) & (candles_df['hour'] < 16))
        ].copy()
        
        if candles_df.empty:
            return None
        
        # Power hour analysis
        power_hour = candles_df[candles_df['hour'] >= power_hour_start_hour].copy()
        
        if power_hour.empty:
            return None
        
        # Calculate metrics
        open_price = candles_df.iloc[0]['open']
        close_price = candles_df.iloc[-1]['close']
        day_change_pct = ((close_price - open_price) / open_price * 100)
        
        power_hour_open = power_hour.iloc[0]['open']
        power_hour_close = power_hour.iloc[-1]['close']
        power_hour_change_pct = ((power_hour_close - power_hour_open) / power_hour_open * 100)
        
        total_volume = candles_df['volume'].sum()
        power_hour_volume = power_hour['volume'].sum()
        volume_concentration = (power_hour_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Calculate momentum score
        momentum_score = 0
        
        # Power hour driving the day
        if abs(power_hour_change_pct) > abs(day_change_pct) * 0.5:
            momentum_score += 2
        
        # Volume concentration
        if volume_concentration > 35:
            momentum_score += 2
        elif volume_concentration > 25:
            momentum_score += 1
        
        # Direction
        if power_hour_change_pct > 0.5:
            momentum_score += 1
        elif power_hour_change_pct < -0.5:
            momentum_score -= 1
        
        # VWAP
        power_hour['vwap'] = (power_hour['volume'] * (power_hour['high'] + power_hour['low'] + power_hour['close']) / 3).cumsum() / power_hour['volume'].cumsum()
        final_vwap = power_hour.iloc[-1]['vwap']
        price_vs_vwap = ((power_hour_close - final_vwap) / final_vwap * 100)
        
        if price_vs_vwap > 0.2:
            momentum_score += 1
        elif price_vs_vwap < -0.2:
            momentum_score -= 1
        
        # Determine signal
        if momentum_score >= 3:
            signal = "STRONG_BUY"
        elif momentum_score >= 1:
            signal = "BUY"
        elif momentum_score <= -3:
            signal = "STRONG_SELL"
        elif momentum_score <= -1:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            'close_price': close_price,
            'day_change_pct': day_change_pct,
            'power_hour_change_pct': power_hour_change_pct,
            'volume_concentration': volume_concentration,
            'momentum_score': momentum_score,
            'signal': signal
        }
        
    except Exception as e:
        return None

def get_next_day_outcome(client, symbol, trading_date, current_close):
    """Get next trading day's performance"""
    try:
        # Get data for next few days to find next trading day
        start = int((trading_date + timedelta(days=1)).timestamp() * 1000)
        end = int((trading_date + timedelta(days=5)).timestamp() * 1000)
        
        history = client.get_price_history(
            symbol=symbol,
            frequency_type='minute',
            frequency=1,
            start_date=start,
            end_date=end,
            need_extended_hours=False
        )
        
        if 'candles' in history and history['candles']:
            df = pd.DataFrame(history['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
            df['date'] = df['datetime'].dt.date
            
            # Get first trading day after
            unique_dates = sorted(df['date'].unique())
            if len(unique_dates) > 0:
                next_day = unique_dates[0]
                next_day_data = df[df['date'] == next_day]
                
                if not next_day_data.empty:
                    next_open = next_day_data.iloc[0]['open']
                    next_high = next_day_data['high'].max()
                    next_low = next_day_data['low'].min()
                    next_close = next_day_data.iloc[-1]['close']
                    
                    gap_pct = ((next_open - current_close) / current_close * 100)
                    day_change_pct = ((next_close - next_open) / next_open * 100)
                    total_return = ((next_close - current_close) / current_close * 100)
                    
                    return {
                        'next_date': next_day,
                        'next_open': next_open,
                        'next_high': next_high,
                        'next_low': next_low,
                        'next_close': next_close,
                        'gap_pct': gap_pct,
                        'day_change_pct': day_change_pct,
                        'total_return': total_return,
                        'success': True
                    }
        
        return None
        
    except Exception as e:
        return None

def calculate_prediction_accuracy(historical_results):
    """Calculate how accurate the signal was historically"""
    if not historical_results:
        return 0, "N/A"
    
    correct = 0
    total = 0
    
    for result in historical_results:
        if result['next_day'] is None:
            continue
        
        total += 1
        signal = result['signal']
        next_return = result['next_day']['total_return']
        
        # Check if signal was correct
        if signal in ['STRONG_BUY', 'BUY'] and next_return > 0:
            correct += 1
        elif signal in ['STRONG_SELL', 'SELL'] and next_return < 0:
            correct += 1
        elif signal == 'HOLD' and abs(next_return) < 0.5:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    if accuracy >= 70:
        rating = "HIGH"
    elif accuracy >= 50:
        rating = "MEDIUM"
    else:
        rating = "LOW"
    
    return accuracy, rating

def analyze_historical_patterns(client, symbol, lookback_days, power_hour_start_hour):
    """Analyze last N trading days"""
    try:
        # Get historical data
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=lookback_days * 2)).timestamp() * 1000)
        
        history = client.get_price_history(
            symbol=symbol,
            frequency_type='minute',
            frequency=1,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        if 'candles' not in history or not history['candles']:
            return None
        
        df = pd.DataFrame(history['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        df['date'] = df['datetime'].dt.date
        
        # Group by date
        trading_dates = sorted(df['date'].unique())[-lookback_days:]
        
        results = []
        for i, date in enumerate(trading_dates):
            day_data = df[df['date'] == date]
            
            if day_data.empty:
                continue
            
            # Analyze this day's power hour
            analysis = analyze_day_power_hour(day_data, power_hour_start_hour)
            
            if analysis:
                # Get next day outcome (if not the most recent day)
                next_day = None
                if i < len(trading_dates) - 1:
                    trading_datetime = datetime.combine(date, datetime.min.time())
                    next_day = get_next_day_outcome(client, symbol, trading_datetime, analysis['close_price'])
                
                results.append({
                    'date': date,
                    'signal': analysis['signal'],
                    'momentum_score': analysis['momentum_score'],
                    'power_hour_change': analysis['power_hour_change_pct'],
                    'volume_concentration': analysis['volume_concentration'],
                    'close_price': analysis['close_price'],
                    'next_day': next_day
                })
        
        return results
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# Main analysis
if analyze_button:
    power_hour_start_hour = get_power_hour_start_hour(power_hour_start)
    
    with st.spinner("🔄 Analyzing historical patterns and making predictions..."):
        try:
            client = SchwabClient()
            if not client.authenticate():
                st.error("Failed to authenticate with Schwab API")
                st.stop()
            
            all_predictions = []
            
            for symbol in symbols:
                results = analyze_historical_patterns(client, symbol, lookback_days, power_hour_start_hour)
                
                if results and len(results) > 0:
                    # Calculate accuracy
                    accuracy, rating = calculate_prediction_accuracy(results)
                    
                    # Get today's signal (most recent)
                    today_signal = results[-1]['signal']
                    today_score = results[-1]['momentum_score']
                    
                    all_predictions.append({
                        'symbol': symbol,
                        'signal': today_signal,
                        'score': today_score,
                        'accuracy': accuracy,
                        'rating': rating,
                        'results': results
                    })
            
            if mode == "Single Stock Analysis":
                # Detailed single stock view
                if all_predictions:
                    pred = all_predictions[0]
                    results = pred['results']
                    
                    st.markdown(f"## 📊 {pred['symbol']} - Historical Pattern Analysis")
                    
                    # Show accuracy and prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        acc_class = "accuracy-high" if pred['rating'] == "HIGH" else "accuracy-medium" if pred['rating'] == "MEDIUM" else "accuracy-low"
                        st.markdown(f"### Prediction Accuracy")
                        st.markdown(f"<h1 class='{acc_class}'>{pred['accuracy']:.1f}%</h1>", unsafe_allow_html=True)
                        st.caption(f"Rating: {pred['rating']}")
                    
                    with col2:
                        st.markdown(f"### Today's Signal")
                        signal_class = "buy-signal" if 'BUY' in pred['signal'] else "sell-signal" if 'SELL' in pred['signal'] else "hold-signal"
                        st.markdown(f"<div class='prediction-card {signal_class}'><h2>{pred['signal']}</h2></div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"### Momentum Score")
                        st.markdown(f"<h1>{pred['score']:+d}</h1>", unsafe_allow_html=True)
                        st.caption("Range: -5 to +5")
                    
                    # Historical results table
                    st.markdown("### 📋 Last 5 Trading Days Analysis")
                    
                    history_data = []
                    for r in results:
                        row = {
                            'Date': r['date'].strftime('%Y-%m-%d'),
                            'Signal': r['signal'],
                            'Score': r['momentum_score'],
                            'Power Hour Δ': f"{r['power_hour_change']:.2f}%",
                            'Vol Conc.': f"{r['volume_concentration']:.1f}%",
                            'Close': f"${r['close_price']:.2f}"
                        }
                        
                        if r['next_day']:
                            row['Next Day Gap'] = f"{r['next_day']['gap_pct']:.2f}%"
                            row['Next Day Δ'] = f"{r['next_day']['day_change_pct']:.2f}%"
                            row['Total Return'] = f"{r['next_day']['total_return']:.2f}%"
                            row['✓'] = '✅' if (
                                ('BUY' in r['signal'] and r['next_day']['total_return'] > 0) or
                                ('SELL' in r['signal'] and r['next_day']['total_return'] < 0) or
                                (r['signal'] == 'HOLD' and abs(r['next_day']['total_return']) < 0.5)
                            ) else '❌'
                        else:
                            row['Next Day Gap'] = 'Today'
                            row['Next Day Δ'] = '-'
                            row['Total Return'] = '-'
                            row['✓'] = '⏳'
                        
                        history_data.append(row)
                    
                    df_history = pd.DataFrame(history_data)
                    st.dataframe(df_history, use_container_width=True, hide_index=True)
                    
                    # Chart showing signal vs outcome
                    fig = go.Figure()
                    
                    dates = [r['date'].strftime('%m/%d') for r in results[:-1]]
                    next_returns = [r['next_day']['total_return'] if r['next_day'] else 0 for r in results[:-1]]
                    colors = ['green' if r > 0 else 'red' for r in next_returns]
                    
                    fig.add_trace(go.Bar(
                        x=dates,
                        y=next_returns,
                        name='Next Day Return',
                        marker_color=colors,
                        text=[f"{r:.2f}%" for r in next_returns],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Historical Signal Performance (Next Day Returns)",
                        xaxis_title="Date",
                        yaxis_title="Return %",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendation for Tomorrow")
                    if pred['accuracy'] >= min_accuracy:
                        if pred['signal'] in ['STRONG_BUY', 'BUY']:
                            st.success(f"✅ **{pred['signal']}** - Pattern shows {pred['accuracy']:.1f}% accuracy. Consider buying for next-day trade.")
                        elif pred['signal'] in ['STRONG_SELL', 'SELL']:
                            st.error(f"❌ **{pred['signal']}** - Pattern shows {pred['accuracy']:.1f}% accuracy. Consider selling or shorting for next-day trade.")
                        else:
                            st.info(f"⏸️ **{pred['signal']}** - No clear pattern. Consider staying flat.")
                    else:
                        st.warning(f"⚠️ Accuracy ({pred['accuracy']:.1f}%) below threshold ({min_accuracy}%). Signal may not be reliable.")
                
            else:
                # Batch watchlist scanner
                st.markdown("## 🎯 Watchlist Predictions")
                
                # Filter by accuracy
                filtered = [p for p in all_predictions if p['accuracy'] >= min_accuracy]
                
                # Sort by signal strength
                buy_signals = [p for p in filtered if 'BUY' in p['signal']]
                sell_signals = [p for p in filtered if 'SELL' in p['signal']]
                
                buy_signals.sort(key=lambda x: (x['score'], x['accuracy']), reverse=True)
                sell_signals.sort(key=lambda x: (abs(x['score']), x['accuracy']), reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 🟢 BUY Signals ({len(buy_signals)})")
                    for p in buy_signals[:10]:  # Top 10
                        st.markdown(f"""
                        <div class="stock-row">
                            <strong>{p['symbol']}</strong> - {p['signal']} 
                            <span style="color: #28a745;">Score: {p['score']:+d}</span> | 
                            Accuracy: <span class="accuracy-{'high' if p['rating']=='HIGH' else 'medium' if p['rating']=='MEDIUM' else 'low'}">{p['accuracy']:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if not buy_signals:
                        st.info("No buy signals above accuracy threshold")
                
                with col2:
                    st.markdown(f"### 🔴 SELL Signals ({len(sell_signals)})")
                    for p in sell_signals[:10]:  # Top 10
                        st.markdown(f"""
                        <div class="stock-row">
                            <strong>{p['symbol']}</strong> - {p['signal']} 
                            <span style="color: #dc3545;">Score: {p['score']:+d}</span> | 
                            Accuracy: <span class="accuracy-{'high' if p['rating']=='HIGH' else 'medium' if p['rating']=='MEDIUM' else 'low'}">{p['accuracy']:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if not sell_signals:
                        st.info("No sell signals above accuracy threshold")
                
                # Summary stats
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analyzed", len(all_predictions))
                with col2:
                    st.metric("Above Threshold", len(filtered))
                with col3:
                    st.metric("Buy Signals", len(buy_signals))
                with col4:
                    st.metric("Sell Signals", len(sell_signals))
                
                # Download results
                st.markdown("### 📥 Export Results")
                export_data = []
                for p in filtered:
                    export_data.append({
                        'Symbol': p['symbol'],
                        'Signal': p['signal'],
                        'Score': p['score'],
                        'Accuracy': f"{p['accuracy']:.1f}%",
                        'Rating': p['rating']
                    })
                
                if export_data:
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"power_hour_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())

else:
    st.info("""
    ### 🎯 How To Use This Predictor
    
    **Perfect for end-of-day decision making:**
    
    1. **Single Stock Mode** - Detailed analysis showing:
       - Historical accuracy of power hour signals
       - Last 5 days: signal → next day outcome
       - Today's prediction with confidence level
       - Visual performance tracking
    
    2. **Batch Scanner Mode** - Scan your entire watchlist:
       - Analyzes all stocks simultaneously
       - Shows only high-accuracy signals
       - Sorted by signal strength and accuracy
       - Export results for your trading platform
    
    ### 📊 Signal Accuracy
    
    - **HIGH (70%+)**: Strong predictive pattern - high confidence
    - **MEDIUM (50-70%)**: Moderate pattern - use with caution
    - **LOW (<50%)**: Weak pattern - skip these signals
    
    ### ⏰ Best Time to Run
    
    **3:50 PM ET** - 10 minutes before market close:
    - Power hour signal is nearly complete
    - Still time to place orders for next day
    - Get overnight/pre-market advantage
    
    ### 💡 Strategy
    
    1. Run batch scanner at 3:50 PM
    2. Filter for HIGH accuracy signals only
    3. Buy strongest BUY signals after hours or at pre-market
    4. Short/avoid strongest SELL signals
    5. Check results next day to refine accuracy
    
    **Configure settings in sidebar and click 'Analyze & Predict'**
    """)
