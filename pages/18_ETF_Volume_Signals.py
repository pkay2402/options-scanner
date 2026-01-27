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

st.set_page_config(
    page_title="ETF Volume Signals",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ETF Volume Pattern Signals")
st.markdown("**Actionable volume patterns for leveraged ETF trading (Last 30 Days)**")

# Define regular ETFs list
REGULAR_ETFS = ['XRT', 'XLY', 'XLV', 'XLU', 'XLP', 'XLK', 'XLI', 'XLF', 'XLE', 'XLC', 
                'XLB', 'XHB', 'XBI', 'GDX', 'MAGS', 'XME', 'GLD', 'SLVP', 'QQQ', 'SPY', 'IWM', 'DIA']

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
        - **Signal:** Volume surge >100% above 5-day average
        - **Expected Return:** **21.59%** average on upward moves
        - **Entry:** Trade immediately when pattern detected
        - **Stop Loss:** -5%
        - **Take Profit:** Scale out at +10%, +15%, let rest run
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
        - **Signal:** Volume surge 50-100% above 5-day average
        - **Expected Return:** **~17%** average on upward moves
        - **Entry:** Moderate risk entry point
        - **Stop Loss:** -3%
        - **Take Profit:** Scale out at +8%, +15%
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
