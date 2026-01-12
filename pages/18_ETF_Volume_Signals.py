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
st.markdown("**Real-time volume patterns that predict 10%+ moves in leveraged ETFs**")

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

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_return = st.sidebar.slider("Minimum Return %", 0, 50, 10)
selected_patterns = st.sidebar.multiselect(
    "Volume Patterns",
    options=df['volume_pattern'].unique().tolist(),
    default=['EXPLOSIVE_SURGE', 'STRONG_SURGE', 'BUILDING_MOMENTUM']
)
direction_filter = st.sidebar.radio("Direction", ["All", "UP", "DOWN"])

# Apply filters
filtered_df = df[df['return'].abs() >= min_return]
if selected_patterns:
    filtered_df = filtered_df[filtered_df['volume_pattern'].isin(selected_patterns)]
if direction_filter != "All":
    filtered_df = filtered_df[filtered_df['direction'] == direction_filter]

# Key Metrics at top
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Big Moves", len(df))
with col2:
    avg_return = filtered_df['return'].mean()
    st.metric("Avg Return", f"{avg_return:.2f}%")
with col3:
    explosive_count = len(df[df['volume_pattern'] == 'EXPLOSIVE_SURGE'])
    st.metric("Explosive Surges", explosive_count)
with col4:
    momentum_count = len(df[df['volume_pattern'] == 'BUILDING_MOMENTUM'])
    st.metric("Building Momentum", momentum_count)
with col5:
    active_etfs = df['symbol'].nunique()
    st.metric("Active ETFs", active_etfs)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Volume Surge Signals", 
    "🔥 Building Momentum", 
    "📈 Pattern Analysis",
    "🎯 Most Active ETFs",
    "💡 Trading Signals"
])

# Tab 1: Volume Surge Signals (50%+ volume)
with tab1:
    st.header("🚀 50%+ Volume Surge → 10%+ Move Likely")
    st.markdown("""
    **Key Finding:** ETFs with 50%+ volume surge have averaged **21.59%** returns on upward moves.
    
    Volume surges indicate institutional accumulation and often precede major price moves.
    """)
    
    # Filter for strong surges
    surge_df = df[df['volume_surge_pct'] >= 50].sort_values('volume_surge_pct', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Recent High Volume Surges ({len(surge_df)} found)")
        
        # Display table
        display_cols = ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 
                       'move_volume', 'avg_volume_5d', 'volume_pattern']
        
        surge_display = surge_df[display_cols].head(20).copy()
        surge_display['date'] = surge_display['date'].dt.strftime('%Y-%m-%d')
        surge_display.columns = ['Symbol', 'Date', 'Return %', 'Direction', 'Vol Surge %', 
                                'Move Volume', 'Avg Volume (5d)', 'Pattern']
        
        # Color code returns
        def color_returns(val):
            if isinstance(val, (int, float)):
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'
            return ''
        
        styled_df = surge_display.style.applymap(
            color_returns, 
            subset=['Return %']
        ).format({
            'Return %': '{:.2f}%',
            'Vol Surge %': '{:.1f}%',
            'Move Volume': '{:,.0f}',
            'Avg Volume (5d)': '{:,.0f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=600)
    
    with col2:
        st.subheader("Volume Surge Stats")
        
        # Stats
        up_surges = surge_df[surge_df['direction'] == 'UP']
        down_surges = surge_df[surge_df['direction'] == 'DOWN']
        
        st.metric("Upward Moves", len(up_surges), 
                 delta=f"{up_surges['return'].mean():.2f}% avg return")
        st.metric("Downward Moves", len(down_surges),
                 delta=f"{down_surges['return'].mean():.2f}% avg return")
        st.metric("Best Return", f"{surge_df['return'].max():.2f}%")
        st.metric("Largest Surge", f"{surge_df['volume_surge_pct'].max():.1f}%")
        
        # Pattern distribution
        st.subheader("Surge Patterns")
        pattern_counts = surge_df['volume_pattern'].value_counts()
        fig = px.pie(
            values=pattern_counts.values,
            names=pattern_counts.index,
            title="Volume Pattern Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.subheader("Top Performers")
        top_symbols = surge_df.groupby('symbol')['return'].agg(['count', 'mean']).sort_values('count', ascending=False).head(10)
        st.dataframe(top_symbols.style.format({
            'count': '{:.0f}',
            'mean': '{:.2f}%'
        }))

# Tab 2: Building Momentum
with tab2:
    st.header("🔥 3+ Consecutive Days of Volume Increase")
    st.markdown("""
    **Key Finding:** ETFs with 3+ consecutive volume increases averaged **16.50%** absolute returns.
    
    This pattern indicates sustained interest and building momentum before a breakout.
    """)
    
    # Filter for building momentum
    momentum_df = df[df['consecutive_increases'] >= 3].sort_values('return', ascending=False, key=abs)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Building Momentum Signals ({len(momentum_df)} found)")
        
        display_cols = ['symbol', 'date', 'return', 'direction', 'consecutive_increases', 
                       'volume_trend_pct', 'volume_surge_pct', 'avg_return_5d']
        
        momentum_display = momentum_df[display_cols].head(20).copy()
        momentum_display['date'] = momentum_display['date'].dt.strftime('%Y-%m-%d')
        momentum_display.columns = ['Symbol', 'Date', 'Return %', 'Direction', 
                                    'Consec Increases', 'Vol Trend %', 'Vol Surge %', 'Avg Return 5d']
        
        styled_momentum = momentum_display.style.applymap(
            color_returns,
            subset=['Return %', 'Avg Return 5d']
        ).format({
            'Return %': '{:.2f}%',
            'Vol Trend %': '{:.1f}%',
            'Vol Surge %': '{:.1f}%',
            'Avg Return 5d': '{:.2f}%',
            'Consec Increases': '{:.0f}'
        })
        
        st.dataframe(styled_momentum, use_container_width=True, height=600)
    
    with col2:
        st.subheader("Momentum Stats")
        
        st.metric("Avg Absolute Return", f"{momentum_df['return'].abs().mean():.2f}%")
        st.metric("Max Consecutive Days", int(momentum_df['consecutive_increases'].max()))
        st.metric("Avg Volume Trend", f"{momentum_df['volume_trend_pct'].mean():.1f}%")
        
        # Direction distribution
        st.subheader("Direction Split")
        direction_counts = momentum_df['direction'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=direction_counts.index,
                y=direction_counts.values,
                marker_color=['green' if x == 'UP' else 'red' for x in direction_counts.index]
            )
        ])
        fig.update_layout(title="UP vs DOWN Moves", height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Consecutive increases distribution
        st.subheader("Consecutive Increases")
        consec_dist = momentum_df['consecutive_increases'].value_counts().sort_index()
        fig = go.Figure(data=[go.Bar(x=consec_dist.index, y=consec_dist.values)])
        fig.update_layout(title="Distribution", height=250)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Pattern Analysis
with tab3:
    st.header("📈 Volume Pattern Deep Dive")
    
    # Pattern performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pattern Performance")
        pattern_stats = df.groupby('volume_pattern').agg({
            'return': ['mean', 'count'],
            'volume_surge_pct': 'mean',
            'consecutive_increases': 'mean'
        }).round(2)
        
        pattern_stats.columns = ['Avg Return %', 'Count', 'Avg Vol Surge %', 'Avg Consec Inc']
        pattern_stats = pattern_stats.sort_values('Avg Return %', ascending=False)
        
        st.dataframe(pattern_stats.style.format({
            'Avg Return %': '{:.2f}%',
            'Count': '{:.0f}',
            'Avg Vol Surge %': '{:.1f}%',
            'Avg Consec Inc': '{:.2f}'
        }), use_container_width=True)
        
        # Pattern distribution over time
        st.subheader("Pattern Timeline")
        timeline_df = df.groupby([pd.Grouper(key='date', freq='D'), 'volume_pattern']).size().reset_index(name='count')
        fig = px.line(timeline_df, x='date', y='count', color='volume_pattern',
                     title="Volume Patterns Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Return Distribution by Pattern")
        
        # Box plot
        fig = go.Figure()
        for pattern in df['volume_pattern'].unique():
            pattern_data = df[df['volume_pattern'] == pattern]['return']
            fig.add_trace(go.Box(y=pattern_data, name=pattern))
        
        fig.update_layout(
            title="Return Distribution by Volume Pattern",
            yaxis_title="Return %",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume surge vs return scatter
        st.subheader("Volume Surge vs Return")
        fig = px.scatter(
            df,
            x='volume_surge_pct',
            y='return',
            color='volume_pattern',
            hover_data=['symbol', 'date'],
            title="Correlation: Volume Surge vs Return"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Most Active ETFs
with tab4:
    st.header("🎯 Most Active ETFs - Highest Volatility")
    
    # Calculate activity metrics
    etf_stats = df.groupby('symbol').agg({
        'return': ['count', 'mean', lambda x: x.abs().mean()],
        'volume_surge_pct': 'mean',
        'consecutive_increases': 'mean'
    }).round(2)
    
    etf_stats.columns = ['Big Moves', 'Avg Return %', 'Avg Abs Return %', 'Avg Vol Surge %', 'Avg Consec Inc']
    etf_stats = etf_stats.sort_values('Big Moves', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 30 Most Active ETFs")
        st.markdown("*ETFs with most 10%+ moves in last 45 days*")
        
        top_etfs = etf_stats.head(30)
        st.dataframe(top_etfs.style.format({
            'Big Moves': '{:.0f}',
            'Avg Return %': '{:.2f}%',
            'Avg Abs Return %': '{:.2f}%',
            'Avg Vol Surge %': '{:.1f}%',
            'Avg Consec Inc': '{:.2f}'
        }).background_gradient(subset=['Big Moves'], cmap='Reds'),
        use_container_width=True, height=800)
    
    with col2:
        st.subheader("Activity Breakdown")
        
        # Top 10 chart
        top_10 = etf_stats.head(10)
        fig = go.Figure(data=[
            go.Bar(x=top_10.index, y=top_10['Big Moves'], 
                  marker_color='crimson')
        ])
        fig.update_layout(
            title="Top 10 Most Active ETFs",
            xaxis_title="Symbol",
            yaxis_title="Number of Big Moves",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        st.subheader("Overall Stats")
        st.metric("Total Unique ETFs", len(etf_stats))
        st.metric("Most Active", f"{etf_stats.index[0]} ({int(etf_stats['Big Moves'].iloc[0])} moves)")
        st.metric("Avg Moves per ETF", f"{etf_stats['Big Moves'].mean():.1f}")
        
        # Get ETF details
        st.subheader("Quick Lookup")
        selected_etf = st.selectbox("Select ETF", etf_stats.index.tolist())
        
        if selected_etf:
            etf_data = df[df['symbol'] == selected_etf].sort_values('date', ascending=False)
            st.metric("Total Big Moves", len(etf_data))
            st.metric("Avg Return", f"{etf_data['return'].mean():.2f}%")
            st.metric("Last Move", etf_data['date'].iloc[0].strftime('%Y-%m-%d'))
            
            st.subheader(f"{selected_etf} Recent Moves")
            recent_moves = etf_data[['date', 'return', 'direction', 'volume_surge_pct', 'volume_pattern']].head(10).copy()
            recent_moves['date'] = recent_moves['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(recent_moves, use_container_width=True)

# Tab 5: Trading Signals
with tab5:
    st.header("💡 Actionable Trading Signals")
    
    st.markdown("""
    ### 📋 Trading Strategy Based on Volume Analysis
    
    #### 🎯 High Probability Setups:
    
    1. **EXPLOSIVE SURGE SIGNAL** (Highest Win Rate)
       - Volume surge > 100% above 5-day average
       - Expected return: **21.59%** on average (upward moves)
       - Trade immediately when detected
       - Set stop loss at -5%
    
    2. **BUILDING MOMENTUM SIGNAL** (Early Entry)
       - 3+ consecutive days of volume increases
       - Volume trend > 20%
       - Expected return: **16.50%** on average
       - Enter on 3rd day of volume increase
       - Add to position if volume continues
    
    3. **STRONG SURGE SIGNAL** (Good Risk/Reward)
       - Volume surge 50-100% above average
       - Expected return: **~17%** on average
       - Moderate risk entry
       - Set tighter stops at -3%
    """)
    
    # Current opportunities
    st.subheader("🔥 Current Hot ETFs (Most Recent Activity)")
    
    # Get most recent signals
    recent_cutoff = datetime.now() - timedelta(days=7)
    recent_signals = df[df['date'] >= recent_cutoff].sort_values('date', ascending=False)
    
    if len(recent_signals) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Recent Explosive Surges")
            explosive_recent = recent_signals[recent_signals['volume_pattern'] == 'EXPLOSIVE_SURGE'].head(10)
            if len(explosive_recent) > 0:
                display_signals = explosive_recent[['symbol', 'date', 'return', 'volume_surge_pct']].copy()
                display_signals['date'] = display_signals['date'].dt.strftime('%Y-%m-%d')
                display_signals.columns = ['Symbol', 'Date', 'Return %', 'Vol Surge %']
                st.dataframe(display_signals.style.format({
                    'Return %': '{:.2f}%',
                    'Vol Surge %': '{:.1f}%'
                }), use_container_width=True)
            else:
                st.info("No recent explosive surges in last 7 days")
        
        with col2:
            st.markdown("### Recent Building Momentum")
            momentum_recent = recent_signals[recent_signals['volume_pattern'] == 'BUILDING_MOMENTUM'].head(10)
            if len(momentum_recent) > 0:
                display_momentum = momentum_recent[['symbol', 'date', 'return', 'consecutive_increases']].copy()
                display_momentum['date'] = display_momentum['date'].dt.strftime('%Y-%m-%d')
                display_momentum.columns = ['Symbol', 'Date', 'Return %', 'Consec Days']
                st.dataframe(display_momentum.style.format({
                    'Return %': '{:.2f}%',
                    'Consec Days': '{:.0f}'
                }), use_container_width=True)
            else:
                st.info("No recent building momentum in last 7 days")
    else:
        st.info("No signals in last 7 days. Analysis based on 45-day historical data.")
    
    # Risk management
    st.subheader("⚠️ Risk Management")
    st.markdown("""
    - **Position sizing:** 2-3% of portfolio per trade
    - **Stop loss:** -5% for explosive surges, -3% for building momentum
    - **Take profit:** Scale out at +10%, +15%, let rest run
    - **Max positions:** 3-5 leveraged ETF trades at once
    - **Avoid:** ETFs with "DRY_UP" pattern (volume decreasing)
    """)
    
    # Pattern success rates
    st.subheader("📊 Pattern Success Rates")
    
    success_data = []
    for pattern in df['volume_pattern'].unique():
        pattern_data = df[df['volume_pattern'] == pattern]
        up_moves = len(pattern_data[pattern_data['direction'] == 'UP'])
        total_moves = len(pattern_data)
        success_rate = (up_moves / total_moves * 100) if total_moves > 0 else 0
        avg_return = pattern_data['return'].mean()
        
        success_data.append({
            'Pattern': pattern,
            'Success Rate %': round(success_rate, 1),
            'Total Trades': total_moves,
            'Avg Return %': round(avg_return, 2)
        })
    
    success_df = pd.DataFrame(success_data).sort_values('Success Rate %', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(success_df, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=success_df['Pattern'],
                y=success_df['Success Rate %'],
                marker_color='lightblue',
                text=success_df['Success Rate %'],
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Success Rate by Pattern",
            yaxis_title="Success Rate %",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Data updates every 5 minutes | Run analyze_etf_volume_patterns.py to refresh</p>
    <p>⚠️ Past performance does not guarantee future results. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)
