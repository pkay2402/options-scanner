"""
ETF Volume Pattern Signals
Identifies leveraged ETFs with volume patterns indicating 10%+ moves
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ETF Volume Signals",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ETF Volume Pattern Signals")
st.markdown("**Actionable volume patterns for leveraged ETF trading (Last 30 Days)**")

# Define regular ETFs by volatility category
LOW_VOL_ETFS = ['SPY', 'IWM', 'DIA', 'XLU', 'XLP']  # Stable, large-cap
MED_VOL_ETFS = ['QQQ', 'XLK', 'XLF', 'XLC', 'XLI', 'XLY', 'XLV', 'XLB', 'XHB', 'XRT', 'MAGS']  # Moderate volatility
HIGH_VOL_ETFS = ['SMH', 'XBI', 'GLD', 'SLVP', 'GDX', 'XME', 'XLE']  # Higher volatility sectors

# Combined regular ETFs list
REGULAR_ETFS = LOW_VOL_ETFS + MED_VOL_ETFS + HIGH_VOL_ETFS

# Scan criteria thresholds by ETF type
LEVERAGED_THRESHOLDS = {
    'price_move': 10.0,       # 10%+ price move
    'explosive_surge': 100,   # >100% volume surge
    'strong_surge': 50,       # 50-100% volume surge
    'momentum_trend': 20,     # 20%+ volume trend
}

# Tiered thresholds for regular ETFs based on volatility
LOW_VOL_THRESHOLDS = {
    'price_move': 1.5,        # SPY rarely moves >2%
    'explosive_surge': 50,    # >50% volume surge (rare for SPY)
    'strong_surge': 30,       # 30-50% volume surge
    'momentum_trend': 20,
}

MED_VOL_THRESHOLDS = {
    'price_move': 2.0,        # QQQ/XLK can move 2-3%
    'explosive_surge': 50,    # >50% volume surge
    'strong_surge': 30,       # 30-50% volume surge
    'momentum_trend': 20,
}

HIGH_VOL_THRESHOLDS = {
    'price_move': 2.5,        # SMH/XBI more volatile
    'explosive_surge': 60,    # >60% volume surge
    'strong_surge': 40,       # 40-60% volume surge
    'momentum_trend': 20,
}

def get_etf_thresholds(symbol: str) -> dict:
    """Get appropriate thresholds based on ETF volatility category."""
    if symbol in LOW_VOL_ETFS:
        return LOW_VOL_THRESHOLDS
    elif symbol in MED_VOL_ETFS:
        return MED_VOL_THRESHOLDS
    elif symbol in HIGH_VOL_ETFS:
        return HIGH_VOL_THRESHOLDS
    else:
        return LEVERAGED_THRESHOLDS  # Default for leveraged ETFs

# =====================================================
# VOLUME PATTERN SCANNER (Integrated from analyzer)
# =====================================================

def load_symbols() -> list:
    """Load ETF symbols from CSV."""
    csv_path = Path(__file__).parent.parent / 'extracted_symbols.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df['Symbol'].tolist()
    return []

def fetch_etf_data(symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch historical data for a symbol."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty or len(data) < 10:
            return pd.DataFrame()
        return data
    except:
        return pd.DataFrame()

def identify_big_moves(data: pd.DataFrame, threshold: float = 10.0) -> list:
    """Identify days with significant price moves."""
    moves = []
    data['Return'] = data['Close'].pct_change() * 100
    
    # Single day moves
    for idx in range(1, len(data)):
        daily_return = data['Return'].iloc[idx]
        if abs(daily_return) >= threshold:
            moves.append({
                'date': data.index[idx],
                'return': daily_return,
                'direction': 'UP' if daily_return > 0 else 'DOWN',
                'volume': data['Volume'].iloc[idx],
                'move_type': 'single_day'
            })
    
    # 2-day moves
    for idx in range(2, len(data)):
        two_day_return = ((data['Close'].iloc[idx] / data['Close'].iloc[idx-2]) - 1) * 100
        if abs(two_day_return) >= threshold and abs(data['Return'].iloc[idx]) < threshold:
            moves.append({
                'date': data.index[idx],
                'return': two_day_return,
                'direction': 'UP' if two_day_return > 0 else 'DOWN',
                'volume': data['Volume'].iloc[idx],
                'move_type': '2_day'
            })
    return moves

def analyze_volume_pattern(data: pd.DataFrame, move_date, thresholds: dict, lookback: int = 5) -> dict:
    """Analyze volume pattern before a big move with configurable thresholds."""
    try:
        move_idx = data.index.get_loc(move_date)
        if move_idx < lookback:
            return {}
        
        pre_move_volume = data['Volume'].iloc[move_idx-lookback:move_idx].values
        move_volume = data['Volume'].iloc[move_idx]
        avg_volume = np.mean(pre_move_volume)
        volume_surge = (move_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
        
        volume_increases = sum(1 for i in range(len(pre_move_volume) - 1) 
                              if pre_move_volume[i+1] > pre_move_volume[i])
        
        recent_3d_avg = np.mean(pre_move_volume[-3:]) if len(pre_move_volume) >= 3 else avg_volume
        earlier_vol_avg = np.mean(pre_move_volume[:-3]) if len(pre_move_volume) > 3 else avg_volume
        volume_trend = (recent_3d_avg / earlier_vol_avg - 1) * 100 if earlier_vol_avg > 0 else 0
        
        # Classify pattern using configurable thresholds
        if volume_surge > thresholds['explosive_surge']:
            pattern = "EXPLOSIVE_SURGE"
        elif volume_surge > thresholds['strong_surge']:
            pattern = "STRONG_SURGE"
        elif volume_increases >= 3 and volume_trend > thresholds['momentum_trend']:
            pattern = "BUILDING_MOMENTUM"
        else:
            pattern = "NORMAL"
        
        return {
            'avg_volume_5d': int(avg_volume),
            'move_volume': int(move_volume),
            'volume_surge_pct': round(volume_surge, 1),
            'consecutive_increases': volume_increases,
            'volume_trend_pct': round(volume_trend, 1),
            'volume_pattern': pattern
        }
    except:
        return {}

def run_volume_scan(progress_callback=None) -> pd.DataFrame:
    """Run full volume pattern scan on all ETFs with appropriate thresholds."""
    symbols = load_symbols()
    if not symbols:
        return pd.DataFrame()
    
    # Add regular ETFs to scan list if not already included
    all_symbols = list(set(symbols + REGULAR_ETFS))
    
    results = []
    total = len(all_symbols)
    
    for i, symbol in enumerate(all_symbols):
        if progress_callback:
            progress_callback((i + 1) / total, f"Scanning {symbol}... ({i+1}/{total})")
        
        data = fetch_etf_data(symbol)
        if data.empty:
            continue
        
        # Get appropriate thresholds for this ETF
        thresholds = get_etf_thresholds(symbol)
        is_regular = symbol in REGULAR_ETFS
        
        moves = identify_big_moves(data, threshold=thresholds['price_move'])
        for move in moves:
            vol_analysis = analyze_volume_pattern(data, move['date'], thresholds)
            if vol_analysis and vol_analysis.get('volume_pattern') != 'NORMAL':
                results.append({
                    'symbol': symbol,
                    'date': move['date'].strftime('%Y-%m-%d'),
                    'return': round(move['return'], 2),
                    'direction': move['direction'],
                    'move_type': move['move_type'],
                    'etf_type': 'regular' if is_regular else 'leveraged',
                    **vol_analysis
                })
    
    if results:
        df = pd.DataFrame(results)
        # Save to CSV
        output_path = Path(__file__).parent.parent / 'etf_volume_analysis_results.csv'
        df.to_csv(output_path, index=False)
        return df
    return pd.DataFrame()

# =====================================================
# SCAN BUTTON & DATA LOADING
# =====================================================

# Scan controls in sidebar or header
col_scan, col_info = st.columns([1, 3])
with col_scan:
    if st.button("🔄 Scan Now", type="primary", use_container_width=True):
        with st.spinner("Scanning ETFs for volume patterns..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, text):
                progress_bar.progress(pct)
                status_text.text(text)
            
            df_new = run_volume_scan(progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if not df_new.empty:
                st.success(f"✅ Scan complete! Found {len(df_new)} patterns across {df_new['symbol'].nunique()} ETFs")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("No significant volume patterns found")

with col_info:
    csv_path = Path(__file__).parent.parent / 'etf_volume_analysis_results.csv'
    if csv_path.exists():
        mod_time = datetime.fromtimestamp(csv_path.stat().st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        if age_hours < 1:
            st.info(f"📊 Data last updated: {mod_time.strftime('%I:%M %p')} (fresh)")
        elif age_hours < 24:
            st.info(f"📊 Data last updated: {mod_time.strftime('%I:%M %p')} ({age_hours:.1f}h ago)")
        else:
            st.warning(f"⚠️ Data is {age_hours/24:.1f} days old - consider scanning")
    else:
        st.warning("⚠️ No data found - click Scan Now")

# Load the analysis results
@st.cache_data(ttl=300)
def load_volume_analysis():
    """Load the volume pattern analysis results."""
    csv_path = Path(__file__).parent.parent / 'etf_volume_analysis_results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

df = load_volume_analysis()

if df.empty:
    st.warning("⚠️ No volume analysis data found. Run analyze_etf_volume_patterns.py first.")
    st.stop()

# AI Top Picks Section
st.markdown("### 🤖 AI Top 3 Long Picks")
st.markdown("*Based on recent volume patterns with highest upside probability*")

# Get most recent patterns (last 3 days)
recent_cutoff = datetime.now() - timedelta(days=3)
recent_patterns = df[df['date'] >= recent_cutoff].copy()

# Score each ETF based on pattern strength
def calculate_signal_score(row):
    """Calculate a score for bullish potential."""
    score = 0
    
    # Pattern weight
    if row['volume_pattern'] == 'EXPLOSIVE_SURGE':
        score += 50
    elif row['volume_pattern'] == 'BUILDING_MOMENTUM':
        score += 40
    elif row['volume_pattern'] == 'STRONG_SURGE':
        score += 30
    
    # Volume surge weight
    if row['volume_surge_pct'] > 150:
        score += 30
    elif row['volume_surge_pct'] > 100:
        score += 20
    elif row['volume_surge_pct'] > 50:
        score += 10
    
    # Consecutive increases weight
    score += row['consecutive_increases'] * 5
    
    # Recency weight (more recent = higher score)
    days_old = (datetime.now() - row['date']).days
    recency_bonus = max(0, 20 - (days_old * 5))
    score += recency_bonus
    
    return score

if not recent_patterns.empty:
    recent_patterns['signal_score'] = recent_patterns.apply(calculate_signal_score, axis=1)
    
    # Get unique ETFs with highest scores
    top_picks = recent_patterns.sort_values('signal_score', ascending=False).drop_duplicates('symbol').head(3)
    
    if len(top_picks) > 0:
        cols = st.columns(3)
        for idx, (_, pick) in enumerate(top_picks.iterrows()):
            with cols[idx]:
                # Determine color and icon based on pattern
                if pick['volume_pattern'] == 'EXPLOSIVE_SURGE':
                    icon = "🚀"
                    color = "red"
                elif pick['volume_pattern'] == 'BUILDING_MOMENTUM':
                    icon = "🔥"
                    color = "orange"
                else:
                    icon = "💪"
                    color = "blue"
                
                st.markdown(f"""
                <div style='background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 4px solid {color};'>
                    <h2 style='margin: 0; color: {color};'>{icon} {pick['symbol']}</h2>
                    <p style='margin: 5px 0; font-size: 14px;'>
                        <b>Signal:</b> {pick['volume_pattern'].replace('_', ' ').title()}<br>
                        <b>Vol Surge:</b> {pick['volume_surge_pct']:.0f}%<br>
                        <b>Date:</b> {pick['date'].strftime('%m/%d')}<br>
                        <b>Score:</b> {pick['signal_score']:.0f}/100
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No strong signals detected in last 3 days. Check individual tabs for historical patterns.")
else:
    st.info("No recent signals detected in last 3 days. Run analyze_etf_volume_patterns.py to refresh data.")

st.markdown("---")

# Key Metrics at top
col1, col2, col3, col4 = st.columns(4)
with col1:
    explosive_count = len(df[df['volume_pattern'] == 'EXPLOSIVE_SURGE'])
    st.metric("🚀 Explosive Surge", explosive_count, help="Volume surge >100%")
with col2:
    momentum_count = len(df[df['volume_pattern'] == 'BUILDING_MOMENTUM'])
    st.metric("🔥 Building Momentum", momentum_count, help="3+ consecutive volume increases")
with col3:
    strong_count = len(df[df['volume_pattern'] == 'STRONG_SURGE'])
    st.metric("💪 Strong Surge", strong_count, help="Volume surge 50-100%")
with col4:
    active_etfs = df['symbol'].nunique()
    st.metric("📊 Active ETFs", active_etfs, help="ETFs with 10%+ moves")

# Create 3 focused tabs
tab1, tab2, tab3 = st.tabs([
    "🚀 EXPLOSIVE SURGE (>100% Volume)", 
    "🔥 BUILDING MOMENTUM (3+ Days)", 
    "💪 STRONG SURGE (50-100% Volume)"
])

# Tab 1: Explosive Surge Signal
with tab1:
    st.header("🚀 EXPLOSIVE SURGE SIGNAL (Highest Win Rate)")
    
    with st.expander("📖 Strategy Guide", expanded=False):
        st.markdown("""
        **Leveraged ETFs:** Volume surge >100% | Price move >10%
        **Regular ETFs (tiered by volatility):**
        - 🔵 Low Vol (SPY, IWM, DIA): >50% vol surge | >1.5% move
        - 🟡 Med Vol (QQQ, XLK, etc.): >50% vol surge | >2% move  
        - 🟠 High Vol (SMH, XBI, GLD): >60% vol surge | >2.5% move
        
        - **Entry:** Trade immediately when pattern detected
        - **Stop Loss:** -5% (leveraged), -1.5% to -2% (regular)
        - **Take Profit:** Scale out at +10%, +15% (leveraged) or +2%, +4% (regular)
        """)
    
    # Filter data
    explosive_df = df[df['volume_pattern'] == 'EXPLOSIVE_SURGE'].sort_values('date', ascending=False)
    leveraged_explosive_df = explosive_df[~explosive_df['symbol'].isin(REGULAR_ETFS)]
    regular_explosive_df = explosive_df[explosive_df['symbol'].isin(REGULAR_ETFS)]
    
    # Sub-tabs for Leveraged vs Regular
    subtab1, subtab2 = st.tabs([f"⚡ Leveraged ETFs ({len(leveraged_explosive_df)})", f"📈 Regular ETFs ({len(regular_explosive_df)})"])
    
    with subtab1:
        if leveraged_explosive_df.empty:
            st.info("No explosive surge patterns found for leveraged ETFs in the last 30 days")
        else:
            # Statistics in expander
            with st.expander("📊 Statistics", expanded=False):
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                up_moves = leveraged_explosive_df[leveraged_explosive_df['direction'] == 'UP']
                down_moves = leveraged_explosive_df[leveraged_explosive_df['direction'] == 'DOWN']
                
                with stat_col1:
                    st.metric("Upward Moves", len(up_moves))
                    st.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                with stat_col2:
                    st.metric("Downward Moves", len(down_moves))
                    st.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                with stat_col3:
                    st.metric("Best Return", f"{leveraged_explosive_df['return'].max():.2f}%")
                    st.metric("Max Vol Surge", f"{leveraged_explosive_df['volume_surge_pct'].max():.1f}%")
                with stat_col4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = leveraged_explosive_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        st.write(f"**{symbol}**: {count}x")
            
            st.markdown(f"**{leveraged_explosive_df['symbol'].nunique()} unique leveraged ETFs** showing this pattern:")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            
            display_df = leveraged_explosive_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                    'Move Volume', '5D Avg Volume', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns, 
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Surge %': '{:.1f}%',
                'Move Volume': '{:,.0f}',
                '5D Avg Volume': '{:,.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)
    
    with subtab2:
        if regular_explosive_df.empty:
            st.info("No explosive surge patterns found for regular ETFs in the last 30 days")
        else:
            # Statistics in expander
            with st.expander("📊 Statistics", expanded=False):
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                up_moves = regular_explosive_df[regular_explosive_df['direction'] == 'UP']
                down_moves = regular_explosive_df[regular_explosive_df['direction'] == 'DOWN']
                
                with stat_col1:
                    st.metric("Upward Moves", len(up_moves))
                    st.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                with stat_col2:
                    st.metric("Downward Moves", len(down_moves))
                    st.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                with stat_col3:
                    st.metric("Best Return", f"{regular_explosive_df['return'].max():.2f}%")
                    st.metric("Max Vol Surge", f"{regular_explosive_df['volume_surge_pct'].max():.1f}%")
                with stat_col4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = regular_explosive_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        st.write(f"**{symbol}**: {count}x")
            
            st.markdown(f"**{regular_explosive_df['symbol'].nunique()} unique regular ETFs** showing this pattern:")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            
            display_df = regular_explosive_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                    'Move Volume', '5D Avg Volume', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns, 
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Surge %': '{:.1f}%',
                'Move Volume': '{:,.0f}',
                '5D Avg Volume': '{:,.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)

# Tab 2: Building Momentum Signal
with tab2:
    st.header("🔥 BUILDING MOMENTUM SIGNAL (Early Entry)")
    
    with st.expander("📖 Strategy Guide", expanded=False):
        st.markdown("""
        - **Signal:** 3+ consecutive days of volume increases + volume trend >20%
        - **Expected Return:** **16.50%** average absolute return
        - **Entry:** Enter on 3rd day of consecutive volume increase
        - **Add to Position:** If volume continues increasing
        - **Stop Loss:** -3%
        - **Take Profit:** Scale out at +12%, +18%
        """)
    
    # Filter data for both ETF types
    momentum_df = df[df['volume_pattern'] == 'BUILDING_MOMENTUM'].sort_values('date', ascending=False)
    leveraged_momentum_df = momentum_df[~momentum_df['symbol'].isin(REGULAR_ETFS)]
    regular_momentum_df = momentum_df[momentum_df['symbol'].isin(REGULAR_ETFS)]
    
    # Sub-tabs for Leveraged vs Regular
    subtab1, subtab2 = st.tabs([
        f"⚡ Leveraged ETFs ({len(leveraged_momentum_df)})", 
        f"📈 Regular ETFs ({len(regular_momentum_df)})"
    ])
    
    with subtab1:
        if leveraged_momentum_df.empty:
            st.info("No building momentum patterns found for leveraged ETFs")
        else:
            # Collapsible statistics
            with st.expander("📊 Leveraged ETF Statistics", expanded=False):
                up_moves = leveraged_momentum_df[leveraged_momentum_df['direction'] == 'UP']
                down_moves = leveraged_momentum_df[leveraged_momentum_df['direction'] == 'DOWN']
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Upward Moves", len(up_moves))
                c1.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                c2.metric("Downward Moves", len(down_moves))
                c2.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                c3.metric("Avg Abs Return", f"{leveraged_momentum_df['return'].abs().mean():.2f}%")
                c3.metric("Max Consec Days", int(leveraged_momentum_df['consecutive_increases'].max()))
                
                # Top ETFs
                with c4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = leveraged_momentum_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        avg_return = leveraged_momentum_df[leveraged_momentum_df['symbol'] == symbol]['return'].mean()
                        st.write(f"**{symbol}**: {count}x ({avg_return:+.1f}%)")
            
            st.markdown(f"**{len(leveraged_momentum_df)} occurrences** across **{leveraged_momentum_df['symbol'].nunique()} ETFs**")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'consecutive_increases', 
                           'volume_trend_pct', 'volume_surge_pct', 'move_type']
            display_df = leveraged_momentum_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 
                                    'Consec Days', 'Vol Trend %', 'Vol Surge %', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns,
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Trend %': '{:.1f}%',
                'Vol Surge %': '{:.1f}%',
                'Consec Days': '{:.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)
    
    with subtab2:
        if regular_momentum_df.empty:
            st.info("No building momentum patterns found for regular ETFs")
        else:
            # Collapsible statistics
            with st.expander("📊 Regular ETF Statistics", expanded=False):
                up_moves = regular_momentum_df[regular_momentum_df['direction'] == 'UP']
                down_moves = regular_momentum_df[regular_momentum_df['direction'] == 'DOWN']
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Upward Moves", len(up_moves))
                c1.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                c2.metric("Downward Moves", len(down_moves))
                c2.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                c3.metric("Avg Abs Return", f"{regular_momentum_df['return'].abs().mean():.2f}%")
                c3.metric("Max Consec Days", int(regular_momentum_df['consecutive_increases'].max()))
                
                # Top ETFs
                with c4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = regular_momentum_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        avg_return = regular_momentum_df[regular_momentum_df['symbol'] == symbol]['return'].mean()
                        st.write(f"**{symbol}**: {count}x ({avg_return:+.1f}%)")
            
            st.markdown(f"**{len(regular_momentum_df)} occurrences** across **{regular_momentum_df['symbol'].nunique()} ETFs**")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'consecutive_increases', 
                           'volume_trend_pct', 'volume_surge_pct', 'move_type']
            display_df = regular_momentum_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 
                                    'Consec Days', 'Vol Trend %', 'Vol Surge %', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns,
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Trend %': '{:.1f}%',
                'Vol Surge %': '{:.1f}%',
                'Consec Days': '{:.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)

# Tab 3: Strong Surge Signal
with tab3:
    st.header("💪 STRONG SURGE SIGNAL (Good Risk/Reward)")
    
    with st.expander("📖 Strategy Guide", expanded=False):
        st.markdown("""
        **Leveraged ETFs:** Volume surge 50-100% | Price move >10%
        **Regular ETFs (tiered by volatility):**
        - 🔵 Low Vol (SPY, IWM, DIA): 30-50% vol surge | >1.5% move
        - 🟡 Med Vol (QQQ, XLK, etc.): 30-50% vol surge | >2% move  
        - 🟠 High Vol (SMH, XBI, GLD): 40-60% vol surge | >2.5% move
        
        - **Entry:** Moderate risk entry point
        - **Stop Loss:** -3% (leveraged), -1% to -1.5% (regular)
        - **Take Profit:** Scale out at +8%, +15% (leveraged) or +2%, +3% (regular)
        """)
    
    # Filter data for both ETF types
    strong_df = df[df['volume_pattern'] == 'STRONG_SURGE'].sort_values('date', ascending=False)
    leveraged_strong_df = strong_df[~strong_df['symbol'].isin(REGULAR_ETFS)]
    regular_strong_df = strong_df[strong_df['symbol'].isin(REGULAR_ETFS)]
    
    # Sub-tabs for Leveraged vs Regular
    subtab1, subtab2 = st.tabs([
        f"⚡ Leveraged ETFs ({len(leveraged_strong_df)})", 
        f"📈 Regular ETFs ({len(regular_strong_df)})"
    ])
    
    with subtab1:
        if leveraged_strong_df.empty:
            st.info("No strong surge patterns found for leveraged ETFs")
        else:
            # Collapsible statistics
            with st.expander("📊 Leveraged ETF Statistics", expanded=False):
                up_moves = leveraged_strong_df[leveraged_strong_df['direction'] == 'UP']
                down_moves = leveraged_strong_df[leveraged_strong_df['direction'] == 'DOWN']
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Upward Moves", len(up_moves))
                c1.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                c2.metric("Downward Moves", len(down_moves))
                c2.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                c3.metric("Best Return", f"{leveraged_strong_df['return'].max():.2f}%")
                c3.metric("Avg Vol Surge", f"{leveraged_strong_df['volume_surge_pct'].mean():.1f}%")
                
                # Top ETFs
                with c4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = leveraged_strong_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        avg_return = leveraged_strong_df[leveraged_strong_df['symbol'] == symbol]['return'].mean()
                        st.write(f"**{symbol}**: {count}x ({avg_return:+.1f}%)")
            
            st.markdown(f"**{len(leveraged_strong_df)} occurrences** across **{leveraged_strong_df['symbol'].nunique()} ETFs**")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            display_df = leveraged_strong_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                    'Move Volume', '5D Avg Volume', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns, 
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Surge %': '{:.1f}%',
                'Move Volume': '{:,.0f}',
                '5D Avg Volume': '{:,.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)
    
    with subtab2:
        if regular_strong_df.empty:
            st.info("No strong surge patterns found for regular ETFs")
        else:
            # Collapsible statistics
            with st.expander("📊 Regular ETF Statistics", expanded=False):
                up_moves = regular_strong_df[regular_strong_df['direction'] == 'UP']
                down_moves = regular_strong_df[regular_strong_df['direction'] == 'DOWN']
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Upward Moves", len(up_moves))
                c1.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%" if len(up_moves) > 0 else "N/A")
                c2.metric("Downward Moves", len(down_moves))
                c2.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%" if len(down_moves) > 0 else "N/A")
                c3.metric("Best Return", f"{regular_strong_df['return'].max():.2f}%")
                c3.metric("Avg Vol Surge", f"{regular_strong_df['volume_surge_pct'].mean():.1f}%")
                
                # Top ETFs
                with c4:
                    st.markdown("**Most Frequent:**")
                    top_etfs = regular_strong_df['symbol'].value_counts().head(5)
                    for symbol, count in top_etfs.items():
                        avg_return = regular_strong_df[regular_strong_df['symbol'] == symbol]['return'].mean()
                        st.write(f"**{symbol}**: {count}x ({avg_return:+.1f}%)")
            
            st.markdown(f"**{len(regular_strong_df)} occurrences** across **{regular_strong_df['symbol'].nunique()} ETFs**")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            display_df = regular_strong_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                    'Move Volume', '5D Avg Volume', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.map(
                color_returns, 
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Surge %': '{:.1f}%',
                'Move Volume': '{:,.0f}',
                '5D Avg Volume': '{:,.0f}'
            })
            
            st.dataframe(styled_df, height=500, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Run analyze_etf_volume_patterns.py to refresh data | Last 30 days analysis</p>
    <p>⚠️ Past performance does not guarantee future results. Trade at your own risk.</p>
    <p>📝 Move Type: 'single_day' = 1-day move | '2_day' = 2-day cumulative move ending on date shown</p>
</div>
""", unsafe_allow_html=True)
