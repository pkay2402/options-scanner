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
    
    st.markdown("""
    ### Strategy:
    - **Signal:** Volume surge >100% above 5-day average
    - **Expected Return:** **21.59%** average on upward moves
    - **Entry:** Trade immediately when pattern detected
    - **Stop Loss:** -5%
    - **Take Profit:** Scale out at +10%, +15%, let rest run
    
    ---
    """)
    
    # Filter for explosive surges
    explosive_df = df[df['volume_pattern'] == 'EXPLOSIVE_SURGE'].sort_values('date', ascending=False)
    
    if explosive_df.empty:
        st.info("No explosive surge patterns found in the last 30 days")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"ETFs with Explosive Surge Pattern ({len(explosive_df)} occurrences)")
            
            # Show unique ETFs
            unique_etfs = explosive_df['symbol'].unique()
            st.markdown(f"**{len(unique_etfs)} unique ETFs** showing this pattern:")
            
            # Display detailed table
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            
            display_df = explosive_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                    'Move Volume', '5D Avg Volume', 'Move Type']
            
            # Color code
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
            
            st.dataframe(styled_df, height=600)
        
        with col2:
            st.subheader("Statistics")
            
            up_moves = explosive_df[explosive_df['direction'] == 'UP']
            down_moves = explosive_df[explosive_df['direction'] == 'DOWN']
            
            st.metric("Upward Moves", len(up_moves))
            st.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%")
            st.metric("Downward Moves", len(down_moves))
            st.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%")
            st.metric("Best Return", f"{explosive_df['return'].max():.2f}%")
            st.metric("Max Vol Surge", f"{explosive_df['volume_surge_pct'].max():.1f}%")
            
            # Top ETFs by frequency
            st.markdown("---")
            st.markdown("**Most Frequent:**")
            top_etfs = explosive_df['symbol'].value_counts().head(10)
            for symbol, count in top_etfs.items():
                avg_return = explosive_df[explosive_df['symbol'] == symbol]['return'].mean()
                st.write(f"**{symbol}**: {count}x (avg: {avg_return:+.1f}%)")
            
            # Direction pie chart
            st.markdown("---")
            direction_counts = explosive_df['direction'].value_counts()
            fig = px.pie(
                values=direction_counts.values,
                names=direction_counts.index,
                title="Direction Split",
                color=direction_counts.index,
                color_discrete_map={'UP': 'green', 'DOWN': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Building Momentum Signal
with tab2:
    st.header("🔥 BUILDING MOMENTUM SIGNAL (Early Entry)")
    
    st.markdown("""
    ### Strategy:
    - **Signal:** 3+ consecutive days of volume increases + volume trend >20%
    - **Expected Return:** **16.50%** average absolute return
    - **Entry:** Enter on 3rd day of consecutive volume increase
    - **Add to Position:** If volume continues increasing
    - **Stop Loss:** -3%
    - **Take Profit:** Scale out at +12%, +18%
    
    ---
    """)
    
    # Filter for building momentum
    momentum_df = df[df['volume_pattern'] == 'BUILDING_MOMENTUM'].sort_values('date', ascending=False)
    
    if momentum_df.empty:
        st.info("No building momentum patterns found in the last 30 days")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"ETFs with Building Momentum ({len(momentum_df)} occurrences)")
            
            # Show unique ETFs
            unique_etfs = momentum_df['symbol'].unique()
            st.markdown(f"**{len(unique_etfs)} unique ETFs** showing this pattern:")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'consecutive_increases', 
                           'volume_trend_pct', 'volume_surge_pct', 'move_type']
            
            display_df = momentum_df[display_cols].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Symbol', 'Date', 'Return %', 'Direction', 
                                    'Consec Days', 'Vol Trend %', 'Vol Surge %', 'Move Type']
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_momentum = display_df.style.map(
                color_returns,
                subset=['Return %']
            ).format({
                'Return %': '{:.2f}%',
                'Vol Trend %': '{:.1f}%',
                'Vol Surge %': '{:.1f}%',
                'Consec Days': '{:.0f}'
            })
            
            st.dataframe(styled_momentum, height=600)
        
        with col2:
            st.subheader("Statistics")
            
            up_moves = momentum_df[momentum_df['direction'] == 'UP']
            down_moves = momentum_df[momentum_df['direction'] == 'DOWN']
            
            st.metric("Upward Moves", len(up_moves))
            st.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%")
            st.metric("Downward Moves", len(down_moves))
            st.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%")
            st.metric("Avg Abs Return", f"{momentum_df['return'].abs().mean():.2f}%")
            st.metric("Max Consec Days", int(momentum_df['consecutive_increases'].max()))
            
            # Top ETFs
            st.markdown("---")
            st.markdown("**Most Frequent:**")
            top_etfs = momentum_df['symbol'].value_counts().head(10)
            for symbol, count in top_etfs.items():
                avg_return = momentum_df[momentum_df['symbol'] == symbol]['return'].mean()
                st.write(f"**{symbol}**: {count}x (avg: {avg_return:+.1f}%)")
            
            # Consecutive days distribution
            st.markdown("---")
            consec_dist = momentum_df['consecutive_increases'].value_counts().sort_index()
            fig = go.Figure(data=[go.Bar(
                x=consec_dist.index.astype(str), 
                y=consec_dist.values,
                marker_color='orange'
            )])
            fig.update_layout(
                title="Consecutive Volume Days",
                xaxis_title="Days",
                yaxis_title="Count",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Strong Surge Signal
with tab3:
    st.header("💪 STRONG SURGE SIGNAL (Good Risk/Reward)")
    
    st.markdown("""
    ### Strategy:
    - **Signal:** Volume surge 50-100% above 5-day average
    - **Expected Return:** **~17%** average on upward moves
    - **Entry:** Moderate risk entry point
    - **Stop Loss:** -3%
    - **Take Profit:** Scale out at +8%, +15%
    
    ---
    """)
    
    # Filter for strong surges
    strong_df = df[df['volume_pattern'] == 'STRONG_SURGE'].sort_values('date', ascending=False)
    
    if strong_df.empty:
        st.info("No strong surge patterns found in the last 30 days")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"ETFs with Strong Surge Pattern ({len(strong_df)} occurrences)")
            
            # Show unique ETFs
            unique_etfs = strong_df['symbol'].unique()
            st.markdown(f"**{len(unique_etfs)} unique ETFs** showing this pattern:")
            
            display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                           'move_volume', 'avg_volume_5d', 'move_type']
            
            display_df = strong_df[display_cols].copy()
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
            
            st.dataframe(styled_df, height=600)
        
        with col2:
            st.subheader("Statistics")
            
            up_moves = strong_df[strong_df['direction'] == 'UP']
            down_moves = strong_df[strong_df['direction'] == 'DOWN']
            
            st.metric("Upward Moves", len(up_moves))
            st.metric("Avg UP Return", f"+{up_moves['return'].mean():.2f}%")
            st.metric("Downward Moves", len(down_moves))
            st.metric("Avg DOWN Return", f"{down_moves['return'].mean():.2f}%")
            st.metric("Best Return", f"{strong_df['return'].max():.2f}%")
            st.metric("Avg Vol Surge", f"{strong_df['volume_surge_pct'].mean():.1f}%")
            
            # Top ETFs
            st.markdown("---")
            st.markdown("**Most Frequent:**")
            top_etfs = strong_df['symbol'].value_counts().head(10)
            for symbol, count in top_etfs.items():
                avg_return = strong_df[strong_df['symbol'] == symbol]['return'].mean()
                st.write(f"**{symbol}**: {count}x (avg: {avg_return:+.1f}%)")
            
            # Volume surge distribution
            st.markdown("---")
            fig = go.Figure(data=[go.Histogram(
                x=strong_df['volume_surge_pct'],
                nbinsx=20,
                marker_color='steelblue'
            )])
            fig.update_layout(
                title="Volume Surge Distribution",
                xaxis_title="Vol Surge %",
                yaxis_title="Count",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Run analyze_etf_volume_patterns.py to refresh data | Last 30 days analysis</p>
    <p>⚠️ Past performance does not guarantee future results. Trade at your own risk.</p>
    <p>📝 Move Type: 'single_day' = 1-day move | '2_day' = 2-day cumulative move ending on date shown</p>
</div>
""", unsafe_allow_html=True)
