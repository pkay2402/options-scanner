"""
Relative Rotation Graph (RRG)
Shows relative strength and momentum of sectors/stocks vs benchmark
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Relative Rotation Graph", page_icon="🔄", layout="wide")

# ============================================
# CONFIGURATION
# ============================================

# Sector ETFs and their top holdings
SECTOR_ETFS = {
    'XLK': {'name': 'Technology', 'color': '#3b82f6', 'stocks': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM']},
    'XLF': {'name': 'Financials', 'color': '#22c55e', 'stocks': ['JPM', 'V', 'MA', 'BAC', 'GS']},
    'XLE': {'name': 'Energy', 'color': '#f97316', 'stocks': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']},
    'XLV': {'name': 'Healthcare', 'color': '#ec4899', 'stocks': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV']},
    'XLY': {'name': 'Consumer Disc', 'color': '#a855f7', 'stocks': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE']},
    'XLP': {'name': 'Consumer Staples', 'color': '#14b8a6', 'stocks': ['PG', 'KO', 'PEP', 'COST', 'WMT']},
    'XLI': {'name': 'Industrials', 'color': '#eab308', 'stocks': ['CAT', 'GE', 'UNP', 'BA', 'HON']},
    'XLB': {'name': 'Materials', 'color': '#78716c', 'stocks': ['LIN', 'APD', 'SHW', 'ECL', 'NEM']},
    'XLU': {'name': 'Utilities', 'color': '#06b6d4', 'stocks': ['NEE', 'SO', 'DUK', 'SRE', 'AEP']},
    'XLRE': {'name': 'Real Estate', 'color': '#8b5cf6', 'stocks': ['PLD', 'AMT', 'EQIX', 'SPG', 'PSA']},
    'XLC': {'name': 'Communication', 'color': '#f43f5e', 'stocks': ['META', 'GOOGL', 'NFLX', 'DIS', 'VZ']},
}

BENCHMARK = 'SPY'
LOOKBACK_DAYS = 90  # For calculating RS-Ratio
MOMENTUM_PERIOD = 14  # For RS-Momentum calculation
TRAIL_LENGTH = 3  # Number of historical points to show (reduced for cleaner chart)

# ============================================
# RRG CALCULATION FUNCTIONS
# ============================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch daily price history for a symbol using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='3mo')  # 3 months of data
        
        if df.empty:
            return pd.DataFrame()
        
        # Return just the close prices with symbol as column name
        result = df[['Close']].copy()
        result.columns = [symbol]
        return result
    except Exception as e:
        return pd.DataFrame()


def calculate_rs_ratio(stock_prices: pd.Series, benchmark_prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate JdK RS-Ratio (relative strength vs benchmark)
    
    RS-Ratio = 100 + ((Stock/Benchmark - SMA(Stock/Benchmark)) / StdDev) * 10
    """
    # Calculate relative performance
    relative = stock_prices / benchmark_prices
    
    # Normalize using rolling mean and std
    rolling_mean = relative.rolling(window=period).mean()
    rolling_std = relative.rolling(window=period).std()
    
    # RS-Ratio centered at 100
    rs_ratio = 100 + ((relative - rolling_mean) / rolling_std) * 10
    
    return rs_ratio


def calculate_rs_momentum(rs_ratio: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate JdK RS-Momentum (rate of change of RS-Ratio)
    
    RS-Momentum = 100 + ((RS-Ratio - SMA(RS-Ratio)) / StdDev) * 10
    """
    rolling_mean = rs_ratio.rolling(window=period).mean()
    rolling_std = rs_ratio.rolling(window=period).std()
    
    rs_momentum = 100 + ((rs_ratio - rolling_mean) / rolling_std) * 10
    
    return rs_momentum


def get_quadrant(rs_ratio: float, rs_momentum: float) -> tuple:
    """Determine which quadrant a security is in"""
    if rs_ratio >= 100 and rs_momentum >= 100:
        return 'Leading', '#22c55e'
    elif rs_ratio >= 100 and rs_momentum < 100:
        return 'Weakening', '#eab308'
    elif rs_ratio < 100 and rs_momentum < 100:
        return 'Lagging', '#ef4444'
    else:  # rs_ratio < 100 and rs_momentum >= 100
        return 'Improving', '#3b82f6'


@st.cache_data(ttl=300, show_spinner="Calculating relative rotation...")
def calculate_rrg_data(symbols: list, benchmark: str = 'SPY', lookback: int = 90) -> pd.DataFrame:
    """Calculate RRG data for a list of symbols"""
    
    # Fetch benchmark data
    benchmark_df = fetch_price_history(benchmark, lookback + 30)
    if benchmark_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for symbol in symbols:
        try:
            # Fetch symbol data
            symbol_df = fetch_price_history(symbol, lookback + 30)
            if symbol_df.empty:
                continue
            
            # Align dates
            combined = pd.concat([symbol_df, benchmark_df], axis=1).dropna()
            if len(combined) < 30:
                continue
            
            stock_prices = combined[symbol]
            bench_prices = combined[benchmark]
            
            # Calculate RS-Ratio and RS-Momentum
            rs_ratio = calculate_rs_ratio(stock_prices, bench_prices)
            rs_momentum = calculate_rs_momentum(rs_ratio)
            
            # Get last N values for trail
            trail_data = []
            for i in range(TRAIL_LENGTH, 0, -1):
                if len(rs_ratio) >= i and len(rs_momentum) >= i:
                    trail_data.append({
                        'rs_ratio': rs_ratio.iloc[-i],
                        'rs_momentum': rs_momentum.iloc[-i],
                        'age': i - 1  # 0 = current, higher = older
                    })
            
            if not trail_data:
                continue
            
            current = trail_data[-1]
            quadrant, color = get_quadrant(current['rs_ratio'], current['rs_momentum'])
            
            # Calculate direction (momentum of movement)
            if len(trail_data) >= 2:
                prev = trail_data[-2]
                direction = 'improving' if current['rs_ratio'] > prev['rs_ratio'] else 'declining'
            else:
                direction = 'neutral'
            
            results.append({
                'symbol': symbol,
                'rs_ratio': current['rs_ratio'],
                'rs_momentum': current['rs_momentum'],
                'quadrant': quadrant,
                'color': color,
                'direction': direction,
                'trail': trail_data
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)


def create_rrg_chart(df: pd.DataFrame, title: str, show_trails: bool = True, 
                     symbol_colors: dict = None, height: int = 600) -> go.Figure:
    """Create the RRG scatter plot with quadrants - improved styling"""
    
    fig = go.Figure()
    
    # Add quadrant backgrounds with more visible colors
    # Leading (top-right) - green
    fig.add_shape(type="rect", x0=100, y0=100, x1=120, y1=120,
                  fillcolor="rgba(34, 197, 94, 0.15)", line=dict(width=0))
    # Weakening (bottom-right) - yellow/orange
    fig.add_shape(type="rect", x0=100, y0=80, x1=120, y1=100,
                  fillcolor="rgba(251, 191, 36, 0.15)", line=dict(width=0))
    # Lagging (bottom-left) - red
    fig.add_shape(type="rect", x0=80, y0=80, x1=100, y1=100,
                  fillcolor="rgba(239, 68, 68, 0.15)", line=dict(width=0))
    # Improving (top-left) - blue
    fig.add_shape(type="rect", x0=80, y0=100, x1=100, y1=120,
                  fillcolor="rgba(59, 130, 246, 0.15)", line=dict(width=0))
    
    # Add quadrant labels in corners (less intrusive)
    fig.add_annotation(x=118, y=118, text="LEADING", showarrow=False,
                       font=dict(size=12, color='#16a34a', family='Arial Black'), opacity=0.6)
    fig.add_annotation(x=118, y=82, text="WEAKENING", showarrow=False,
                       font=dict(size=12, color='#d97706', family='Arial Black'), opacity=0.6)
    fig.add_annotation(x=82, y=82, text="LAGGING", showarrow=False,
                       font=dict(size=12, color='#dc2626', family='Arial Black'), opacity=0.6)
    fig.add_annotation(x=82, y=118, text="IMPROVING", showarrow=False,
                       font=dict(size=12, color='#2563eb', family='Arial Black'), opacity=0.6)
    
    # Add center lines - more prominent
    fig.add_hline(y=100, line_dash="dash", line_color="#6b7280", line_width=2, opacity=0.7)
    fig.add_vline(x=100, line_dash="dash", line_color="#6b7280", line_width=2, opacity=0.7)
    
    # Plot each symbol
    for _, row in df.iterrows():
        symbol = row['symbol']
        color = symbol_colors.get(symbol, row['color']) if symbol_colors else row['color']
        
        # Draw trail if enabled (with arrow showing direction)
        if show_trails and 'trail' in row and row['trail'] and len(row['trail']) >= 2:
            trail = row['trail']
            trail_x = [t['rs_ratio'] for t in trail]
            trail_y = [t['rs_momentum'] for t in trail]
            
            # Trail line with gradient effect
            fig.add_trace(go.Scatter(
                x=trail_x, y=trail_y,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=[4, 5, 6, 7, 8][:len(trail)], color=color, opacity=0.6),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Current position - larger marker with white border
        fig.add_trace(go.Scatter(
            x=[row['rs_ratio']],
            y=[row['rs_momentum']],
            mode='markers',
            marker=dict(
                size=18, 
                color=color, 
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            name=symbol,
            hovertemplate=(
                f"<b>{symbol}</b><br>"
                f"RS-Ratio: %{{x:.1f}}<br>"
                f"RS-Momentum: %{{y:.1f}}<br>"
                f"Quadrant: {row['quadrant']}<br>"
                f"<extra></extra>"
            )
        ))
        
        # Add label slightly offset to avoid overlap
        label_offset_x = 0.5
        label_offset_y = 2.5
        fig.add_annotation(
            x=row['rs_ratio'] + label_offset_x,
            y=row['rs_momentum'] + label_offset_y,
            text=f"<b>{symbol}</b>",
            showarrow=False,
            font=dict(size=11, color=color, family='Arial'),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=2
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family='Arial Black'), x=0.5),
        xaxis=dict(
            title=dict(text="RS-Ratio (Relative Strength)", font=dict(size=14)),
            range=[78, 122],
            dtick=5,
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=False,
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="RS-Momentum", font=dict(size=14)),
            range=[78, 122],
            dtick=5,
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=False,
            showgrid=True
        ),
        height=height,
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig


def create_quadrant_summary(df: pd.DataFrame) -> dict:
    """Summarize symbols by quadrant"""
    summary = {
        'Leading': [],
        'Improving': [],
        'Weakening': [],
        'Lagging': []
    }
    
    for _, row in df.iterrows():
        summary[row['quadrant']].append({
            'symbol': row['symbol'],
            'rs_ratio': row['rs_ratio'],
            'rs_momentum': row['rs_momentum'],
            'direction': row['direction']
        })
    
    return summary


# ============================================
# MAIN UI
# ============================================

st.markdown("## 🔄 Relative Rotation Graph")
st.caption("Compare relative strength & momentum vs SPY benchmark")

# Controls
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    show_trails = st.checkbox("Show Rotation Trails", value=True, help="Show historical movement path")

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col3:
    st.caption(f"📊 Benchmark: {BENCHMARK} | Updated: {datetime.now().strftime('%I:%M %p')}")

st.divider()

# ============================================
# SECTOR ETFs RRG
# ============================================

st.markdown("### 📊 Sector ETFs vs SPY")

with st.spinner("Calculating sector rotations..."):
    sector_symbols = list(SECTOR_ETFS.keys())
    sector_df = calculate_rrg_data(sector_symbols, BENCHMARK, LOOKBACK_DAYS)

if not sector_df.empty:
    # Create color map for sectors
    sector_colors = {etf: info['color'] for etf, info in SECTOR_ETFS.items()}
    
    # Create RRG chart
    sector_chart = create_rrg_chart(
        sector_df, 
        "Sector Rotation vs SPY",
        show_trails=show_trails,
        symbol_colors=sector_colors,
        height=500
    )
    st.plotly_chart(sector_chart, use_container_width=True)
    
    # Quadrant summary
    sector_summary = create_quadrant_summary(sector_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🟢 Leading")
        for item in sector_summary['Leading']:
            name = SECTOR_ETFS.get(item['symbol'], {}).get('name', item['symbol'])
            arrow = "↗️" if item['direction'] == 'improving' else "↘️"
            st.markdown(f"**{item['symbol']}** {name} {arrow}")
    
    with col2:
        st.markdown("#### 🟡 Weakening")
        for item in sector_summary['Weakening']:
            name = SECTOR_ETFS.get(item['symbol'], {}).get('name', item['symbol'])
            arrow = "↗️" if item['direction'] == 'improving' else "↘️"
            st.markdown(f"**{item['symbol']}** {name} {arrow}")
    
    with col3:
        st.markdown("#### 🔴 Lagging")
        for item in sector_summary['Lagging']:
            name = SECTOR_ETFS.get(item['symbol'], {}).get('name', item['symbol'])
            arrow = "↗️" if item['direction'] == 'improving' else "↘️"
            st.markdown(f"**{item['symbol']}** {name} {arrow}")
    
    with col4:
        st.markdown("#### 🔵 Improving")
        for item in sector_summary['Improving']:
            name = SECTOR_ETFS.get(item['symbol'], {}).get('name', item['symbol'])
            arrow = "↗️" if item['direction'] == 'improving' else "↘️"
            st.markdown(f"**{item['symbol']}** {name} {arrow}")

else:
    st.warning("Could not load sector data")

st.divider()

# ============================================
# STOCKS BY SECTOR
# ============================================

st.markdown("### 📈 Stocks by Sector")

# Sector selector
selected_sectors = st.multiselect(
    "Select sectors to analyze:",
    options=list(SECTOR_ETFS.keys()),
    default=['XLK', 'XLF', 'XLE', 'XLV'],
    format_func=lambda x: f"{x} - {SECTOR_ETFS[x]['name']}"
)

if selected_sectors:
    # Create tabs for each sector
    tabs = st.tabs([f"{etf} - {SECTOR_ETFS[etf]['name']}" for etf in selected_sectors])
    
    for i, etf in enumerate(selected_sectors):
        with tabs[i]:
            sector_info = SECTOR_ETFS[etf]
            stocks = sector_info['stocks']
            
            with st.spinner(f"Analyzing {sector_info['name']} stocks..."):
                stock_df = calculate_rrg_data(stocks, BENCHMARK, LOOKBACK_DAYS)
            
            if not stock_df.empty:
                # Create chart
                stock_chart = create_rrg_chart(
                    stock_df,
                    f"{sector_info['name']} Stocks vs SPY",
                    show_trails=show_trails,
                    symbol_colors={s: sector_info['color'] for s in stocks},
                    height=450
                )
                st.plotly_chart(stock_chart, use_container_width=True)
                
                # Summary table
                summary_data = []
                for _, row in stock_df.iterrows():
                    summary_data.append({
                        'Symbol': row['symbol'],
                        'RS-Ratio': f"{row['rs_ratio']:.1f}",
                        'RS-Momentum': f"{row['rs_momentum']:.1f}",
                        'Quadrant': row['quadrant'],
                        'Trend': '📈' if row['direction'] == 'improving' else '📉'
                    })
                
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"Could not load data for {sector_info['name']} stocks")

# ============================================
# INTERPRETATION GUIDE
# ============================================

with st.expander("📖 How to Read RRG"):
    st.markdown("""
    ### Understanding Relative Rotation Graphs
    
    **The 4 Quadrants:**
    
    | Quadrant | Location | Meaning | Action |
    |----------|----------|---------|--------|
    | 🟢 **Leading** | Top-Right | Outperforming with strong momentum | **Hold/Buy** - Strongest performers |
    | 🟡 **Weakening** | Bottom-Right | Still outperforming but losing steam | **Watch** - Consider taking profits |
    | 🔴 **Lagging** | Bottom-Left | Underperforming with weak momentum | **Avoid/Sell** - Weakest performers |
    | 🔵 **Improving** | Top-Left | Underperforming but gaining momentum | **Watch** - Potential entry point |
    
    **Rotation Pattern:**
    - Securities typically rotate **clockwise**: Leading → Weakening → Lagging → Improving → Leading
    - This rotation reflects the natural cycle of relative strength
    
    **Key Signals:**
    - **Entering Leading from Improving**: Strong buy signal
    - **Entering Lagging from Weakening**: Strong sell signal
    - **Distance from center (100,100)**: Further = stronger signal
    
    **Trading Strategy:**
    1. **Go long** sectors/stocks in Leading or entering Leading from Improving
    2. **Avoid/short** sectors/stocks in Lagging or entering Lagging from Weakening
    3. **Watch** securities near quadrant boundaries for rotation signals
    
    **Time Frame:**
    - This RRG uses {LOOKBACK_DAYS}-day lookback for trend analysis
    - Best suited for swing trading (days to weeks)
    """)
