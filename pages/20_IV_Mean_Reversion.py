#!/usr/bin/env python3
"""
IV Mean Reversion Scanner
Based on Cracking Markets strategy: https://www.crackingmarkets.com/iv-mean-reversion/

Core Logic:
- Daily IV = Annual IV / √252
- BUY when: Daily price drop > Daily IV (market moved more than expected)
- Filters: Russell 1000, volume > 200K, price > $20, uptrend (C > MA200), Daily IV > 0.55
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import concurrent.futures

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Page config
st.set_page_config(
    page_title="IV Mean Reversion Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .signal-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 6px solid;
    }
    
    .signal-card.strong {
        border-left-color: #22c55e;
        background: linear-gradient(to right, #f0fdf4 0%, white 100%);
    }
    
    .signal-card.moderate {
        border-left-color: #fbbf24;
        background: linear-gradient(to right, #fffbeb 0%, white 100%);
    }
    
    .signal-card.weak {
        border-left-color: #94a3b8;
        background: linear-gradient(to right, #f8fafc 0%, white 100%);
    }
    
    .iv-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
        margin: 10px 5px;
    }
    
    .badge-strong {
        background: #22c55e;
        color: white;
    }
    
    .badge-moderate {
        background: #fbbf24;
        color: white;
    }
    
    .badge-weak {
        background: #94a3b8;
        color: white;
    }
    
    .stMetric {
        background: #f9fafb;
        padding: 15px;
        border-radius: 8px;
    }
    
    .info-box {
        background: #eff6ff;
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Russell 1000 watchlist (subset - expand as needed)
RUSSELL_1000 = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'TTD', 'PNC',
    'JNJ', 'UNH', 'PFE', 'LLY', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK',
    'NOW', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    'WMT', 'HD', 'COST', 'TGT', 'LOW', 'NKE', 'SBUX', 'MCD',
    'BA', 'CAT', 'GE', 'HON', 'MMM', 'LMT', 'RTX', 'UPS',
    'DIS', 'ZS', 'PANW', 'VZ', 'CHTR', 'CRWD',
    'V', 'MA', 'PYPL', 'AXP', 'XYZ',
    'MSTR', 'COIN', 'RIOT', 'MARA', 'PLTR', 'SNOW', 'RBLX'
]


def calculate_daily_iv(annual_iv: float) -> float:
    """
    Convert annual IV to daily IV
    Formula: Daily IV = Annual IV / sqrt(252)
    """
    return annual_iv / np.sqrt(252)


def get_stock_data(symbol: str, days: int = 210) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def get_implied_volatility(symbol: str, client: SchwabClient) -> float:
    """
    Get implied volatility from ATM options
    Uses 30-day expiry options similar to VIX calculation
    """
    try:
        # Get quote for current price
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
            
        current_price = quote[symbol].get('quote', {}).get('lastPrice', 0)
        if not current_price:
            return None
        
        # Get options chain
        chain = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=10
        )
        
        if not chain or 'callExpDateMap' not in chain:
            return None
        
        # Find options closest to 30 DTE
        target_dte = 30
        best_expiry = None
        best_dte_diff = float('inf')
        
        for exp_date in chain['callExpDateMap'].keys():
            dte = int(exp_date.split(':')[1])
            dte_diff = abs(dte - target_dte)
            if dte_diff < best_dte_diff:
                best_dte_diff = dte_diff
                best_expiry = exp_date
        
        if not best_expiry:
            return None
        
        # Get ATM call option IV
        strikes = chain['callExpDateMap'][best_expiry]
        atm_strike = min(strikes.keys(), key=lambda x: abs(float(x) - current_price))
        
        contracts = strikes[atm_strike]
        if contracts and len(contracts) > 0:
            # Return theoretical volatility (more stable than market IV)
            iv = contracts[0].get('theoreticalVolatility', contracts[0].get('volatility', 0))
            return iv if iv > 0 else None
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting IV for {symbol}: {e}")
        return None


def analyze_stock_iv(symbol: str, client: SchwabClient) -> dict:
    """
    Analyze a single stock for IV mean reversion opportunity
    """
    try:
        # Get historical price data
        df = get_stock_data(symbol, days=210)
        if df is None or len(df) < 200:
            return None
        
        # Current price and previous close
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        
        # Price filters
        if current_price < 20:
            return None
        
        # Volume filter
        avg_volume = df['Volume'].iloc[-20:].mean()
        if avg_volume < 200_000:
            return None
        
        # Uptrend filter: Close > MA(200)
        ma_200 = df['Close'].rolling(200).mean().iloc[-1]
        in_uptrend = current_price > ma_200
        
        # Get implied volatility
        annual_iv = get_implied_volatility(symbol, client)
        if annual_iv is None or annual_iv < 0.55:
            return None
        
        # Calculate daily IV
        daily_iv = calculate_daily_iv(annual_iv)
        
        # Calculate today's move (percentage)
        daily_move_pct = ((current_price - prev_close) / prev_close) * 100
        
        # Check if move is significant (exceeds daily IV expectation)
        iv_multiple = abs(daily_move_pct) / daily_iv if daily_iv > 0 else 0
        
        # Check for previous low break (bearish setup for buy)
        prev_low = df['Low'].iloc[-2]
        closed_below_prev_low = current_price < prev_low
        
        # Signal strength
        signal = None
        if daily_move_pct < 0 and iv_multiple > 1.0 and in_uptrend:
            if iv_multiple > 2.0 and closed_below_prev_low:
                signal = "STRONG"
            elif iv_multiple > 1.5:
                signal = "MODERATE"
            elif iv_multiple > 1.0:
                signal = "WEAK"
        
        if signal is None:
            return None
        
        return {
            'symbol': symbol,
            'price': current_price,
            'daily_move_pct': daily_move_pct,
            'annual_iv': annual_iv,
            'daily_iv': daily_iv,
            'iv_multiple': iv_multiple,
            'signal': signal,
            'in_uptrend': in_uptrend,
            'closed_below_prev_low': closed_below_prev_low,
            'avg_volume': avg_volume,
            'ma_200': ma_200,
            'distance_from_ma200': ((current_price - ma_200) / ma_200) * 100
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


@st.cache_data(ttl=300)
def scan_watchlist(symbols: list) -> pd.DataFrame:
    """
    Scan entire watchlist for IV mean reversion opportunities
    """
    client = SchwabClient()
    if not client.authenticate():
        st.error("Failed to authenticate with Schwab API")
        return pd.DataFrame()
    
    results = []
    
    # Use threading for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {
            executor.submit(analyze_stock_iv, symbol, client): symbol 
            for symbol in symbols
        }
        
        progress_bar = st.progress(0)
        completed = 0
        total = len(symbols)
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            completed += 1
            progress_bar.progress(completed / total)
            
            result = future.result()
            if result:
                results.append(result)
    
    progress_bar.empty()
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Sort by signal strength and IV multiple
    signal_order = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}
    df['signal_rank'] = df['signal'].map(signal_order)
    df = df.sort_values(['signal_rank', 'iv_multiple'], ascending=[False, False])
    df = df.drop('signal_rank', axis=1)
    
    return df


def display_opportunity(row: pd.Series):
    """
    Display a single IV mean reversion opportunity
    """
    signal_class = row['signal'].lower()
    
    st.markdown(f"""
    <div class="signal-card {signal_class}">
        <h2 style="margin: 0 0 15px 0;">
            {row['symbol']} 
            <span class="iv-badge badge-{signal_class}">{row['signal']} SIGNAL</span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"${row['price']:.2f}")
    
    with col2:
        st.metric("Daily Move", f"{row['daily_move_pct']:.2f}%", 
                 delta=f"{row['daily_move_pct']:.2f}%")
    
    with col3:
        st.metric("Annual IV", f"{row['annual_iv']:.1f}%")
    
    with col4:
        st.metric("Daily IV", f"{row['daily_iv']:.2f}%")
    
    with col5:
        st.metric("IV Multiple", f"{row['iv_multiple']:.2f}x",
                 help="How much today's move exceeded expected daily volatility")
    
    # Additional details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_emoji = "✅" if row['in_uptrend'] else "❌"
        st.write(f"{trend_emoji} **Uptrend:** {row['in_uptrend']}")
        st.write(f"📊 **MA(200):** ${row['ma_200']:.2f}")
    
    with col2:
        low_break_emoji = "✅" if row['closed_below_prev_low'] else "❌"
        st.write(f"{low_break_emoji} **Closed Below Prev Low:** {row['closed_below_prev_low']}")
        st.write(f"📈 **Distance from MA(200):** {row['distance_from_ma200']:.1f}%")
    
    with col3:
        st.write(f"📊 **Avg Volume:** {row['avg_volume']:,.0f}")
    
    # Entry strategy
    with st.expander("📝 Entry Strategy"):
        entry_price = row['price'] * 0.995  # 0.5% below current
        stop_loss = row['ma_200'] * 0.98  # 2% below MA200
        profit_target = row['price'] * 1.03  # 3% profit target
        
        st.markdown(f"""
        **Suggested Entry Plan:**
        - 🎯 **Limit Order:** ${entry_price:.2f} (0.5% below current)
        - 🛑 **Stop Loss:** ${stop_loss:.2f} (2% below MA200)
        - 💰 **Profit Target:** ${profit_target:.2f} (3% gain)
        - **Risk/Reward:** {((profit_target - entry_price) / (entry_price - stop_loss)):.2f}:1
        
        **Why This Signal:**
        - Stock dropped {abs(row['daily_move_pct']):.2f}% today
        - This is {row['iv_multiple']:.2f}x the expected daily move ({row['daily_iv']:.2f}%)
        - Market was NOT expecting this large of a move = potential overreaction
        - In uptrend (above MA200), so likely to bounce
        """)
    
    st.markdown("---")


def main():
    st.title("📊 IV Mean Reversion Scanner")
    
    
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Scanner Settings")
        
        st.subheader("Filters")
        st.write("✅ Price > $20")
        st.write("✅ Avg Volume > 200K")
        st.write("✅ Close > MA(200) - Uptrend")
        st.write("✅ Daily IV ≥ 0.55%")
        st.write("✅ Daily Move > Daily IV")
        
        st.markdown("---")
        
        custom_symbols = st.text_area(
            "Custom Symbols (comma-separated)",
            help="Add your own symbols to scan",
            placeholder="AAPL, TSLA, NVDA"
        )
        
        if custom_symbols:
            custom_list = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]
            scan_symbols = list(set(RUSSELL_1000 + custom_list))
        else:
            scan_symbols = RUSSELL_1000
        
        st.info(f"Scanning {len(scan_symbols)} symbols")
        
        if st.button("🔄 Refresh Scan", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main scan
    with st.spinner(f"🔍 Scanning {len(scan_symbols)} stocks for IV mean reversion opportunities..."):
        results_df = scan_watchlist(scan_symbols)
    
    if results_df.empty:
        st.warning("⚠️ No IV mean reversion signals found at this time.")
        st.info("💡 **What this means:** Current market conditions aren't showing oversold opportunities that exceed expected volatility. This is actually a good sign - the strategy avoids forcing trades.")
        return
    
    # Summary metrics
    st.subheader("📈 Scan Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(results_df))
    
    with col2:
        strong_count = len(results_df[results_df['signal'] == 'STRONG'])
        st.metric("Strong Signals", strong_count)
    
    with col3:
        moderate_count = len(results_df[results_df['signal'] == 'MODERATE'])
        st.metric("Moderate Signals", moderate_count)
    
    with col4:
        avg_iv_multiple = results_df['iv_multiple'].mean()
        st.metric("Avg IV Multiple", f"{avg_iv_multiple:.2f}x")
    
    st.markdown("---")
    
    # Display tabs for filtering
    tab1, tab2, tab3, tab4 = st.tabs(["🟢 All Signals", "💪 Strong Only", "⚡ Moderate+", "📊 Data Table"])
    
    with tab1:
        st.subheader(f"All Signals ({len(results_df)})")
        for _, row in results_df.iterrows():
            display_opportunity(row)
    
    with tab2:
        strong_df = results_df[results_df['signal'] == 'STRONG']
        st.subheader(f"Strong Signals Only ({len(strong_df)})")
        if strong_df.empty:
            st.info("No strong signals at this time")
        else:
            for _, row in strong_df.iterrows():
                display_opportunity(row)
    
    with tab3:
        moderate_plus = results_df[results_df['signal'].isin(['STRONG', 'MODERATE'])]
        st.subheader(f"Moderate+ Signals ({len(moderate_plus)})")
        if moderate_plus.empty:
            st.info("No moderate or strong signals")
        else:
            for _, row in moderate_plus.iterrows():
                display_opportunity(row)
    
    with tab4:
        st.subheader("Data Table")
        st.dataframe(
            results_df.style.format({
                'price': '${:.2f}',
                'daily_move_pct': '{:.2f}%',
                'annual_iv': '{:.1f}%',
                'daily_iv': '{:.2f}%',
                'iv_multiple': '{:.2f}x',
                'ma_200': '${:.2f}',
                'distance_from_ma200': '{:.1f}%',
                'avg_volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"iv_mean_reversion_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
