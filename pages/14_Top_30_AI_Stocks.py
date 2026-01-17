"""
Top 30 AI Stocks Tracker
Track performance and options flow for the top AI stocks
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.schwab_client import SchwabClient
from src.utils.cached_client import get_client

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Top 30 AI Stocks",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .theme-overview-card {
        background: white;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .theme-overview-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .theme-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stock-metric {
        background: white;
        color: #333;
        padding: 8px;
        border-radius: 6px;
        margin: 3px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .heatmap-cell {
        padding: 6px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        margin: 2px;
    }
    
    .view-toggle {
        background: #f3f4f6;
        padding: 4px;
        border-radius: 8px;
        display: inline-flex;
        gap: 4px;
    }
    
    .quick-stat {
        background: rgba(255,255,255,0.9);
        padding: 5px 8px;
        border-radius: 4px;
        margin: 3px 0;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# AI Stocks organized by theme
AI_THEMES = [
    {
        "name": "🏗️ AI Infrastructure Leaders",
        "description": "Core semiconductor and hardware companies powering AI revolution",
        "stocks": ["NVDA", "AMD", "AVGO", "TSM", "QCOM"],
        "color": "#22c55e"
    },
    {
        "name": "☁️ Cloud AI Giants",
        "description": "Hyperscalers deploying massive AI compute and services",
        "stocks": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
        "color": "#3b82f6"
    },
    {
        "name": "🧠 AI Software & Platforms",
        "description": "Pure-play AI software companies and platforms",
        "stocks": ["PLTR", "AI", "SNOW", "DDOG", "CRWD"],
        "color": "#8b5cf6"
    },
    {
        "name": "⚡ AI Chip Designers",
        "description": "Companies designing next-gen AI accelerators",
        "stocks": ["ARM", "MRVL", "MU", "SMCI", "AMAT"],
        "color": "#f59e0b"
    },
    {
        "name": "🔬 AI Tools & Enterprise",
        "description": "Enterprise AI tools, design automation, and infrastructure",
        "stocks": ["NOW", "CRM", "ADBE", "SNPS", "CDNS"],
        "color": "#ec4899"
    },
    {
        "name": "🚀 Emerging AI Players",
        "description": "High-growth AI enablers and emerging opportunities",
        "stocks": ["SOUN", "UPST", "PATH", "NET", "ZS"],
        "color": "#06b6d4"
    }
]

def get_next_monthly_expiries(n=4):
    """Get next N monthly expiries (3rd Friday of each month)"""
    expiries = []
    today = datetime.now().date()
    current_month = today.month
    current_year = today.year
    
    for i in range(n + 2):
        year = current_year + (current_month + i - 1) // 12
        month = ((current_month + i - 1) % 12) + 1
        
        first_day = datetime(year, month, 1).date()
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(days=14)
        
        if third_friday > today:
            expiries.append(third_friday)
    
    return expiries[:n]

@st.cache_data(ttl=300)
def get_stock_data(symbol: str):
    """Get current stock price and basic info"""
    client = get_client()
    
    if not client:
        return None
    
    try:
        quote = client.get_quote(symbol)
        if quote and symbol in quote:
            data = quote[symbol]['quote']
            return {
                'price': data['lastPrice'],
                'change': data['netChange'],
                'change_pct': data['netPercentChange'],
                'volume': data.get('totalVolume', 0),
                'prev_close': data.get('closePrice', data['lastPrice']),
                'high': data.get('highPrice', data['lastPrice']),
                'low': data.get('lowPrice', data['lastPrice']),
                'market_cap': data.get('marketCap', 0)
            }
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    
    return None

@st.cache_data(ttl=300)
def scan_options_flow(symbol: str, expiry_dates: list):
    """Scan options flow across multiple expiries for unusual activity"""
    client = get_client()
    
    if not client:
        return None
    
    try:
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
        
        price = quote[symbol]['quote']['lastPrice']
        flows = []
        
        for expiry in expiry_dates:
            exp_str = expiry.strftime('%Y-%m-%d')
            
            options = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                from_date=exp_str,
                to_date=exp_str
            )
            
            if not options:
                continue
            
            dte = (expiry - datetime.now().date()).days
            
            # Scan calls
            if 'callExpDateMap' in options:
                for exp_date, strikes in options['callExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        strike = float(strike_str)
                        
                        if abs(strike - price) / price > 0.15:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1)
                            
                            if volume == 0 or oi == 0:
                                continue
                            
                            vol_oi = volume / oi if oi > 0 else 0
                            
                            if vol_oi > 2.0 or volume > 1000:
                                flows.append({
                                    'type': 'CALL',
                                    'strike': strike,
                                    'expiry': expiry,
                                    'dte': dte,
                                    'volume': volume,
                                    'oi': oi,
                                    'vol_oi': vol_oi,
                                    'premium': contract.get('mark', 0),
                                    'delta': contract.get('delta', 0),
                                    'iv': contract.get('volatility', 0)
                                })
            
            # Scan puts
            if 'putExpDateMap' in options:
                for exp_date, strikes in options['putExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        strike = float(strike_str)
                        
                        if abs(strike - price) / price > 0.15:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1)
                            
                            if volume == 0 or oi == 0:
                                continue
                            
                            vol_oi = volume / oi if oi > 0 else 0
                            
                            if vol_oi > 2.0 or volume > 1000:
                                flows.append({
                                    'type': 'PUT',
                                    'strike': strike,
                                    'expiry': expiry,
                                    'dte': dte,
                                    'volume': volume,
                                    'oi': oi,
                                    'vol_oi': vol_oi,
                                    'premium': contract.get('mark', 0),
                                    'delta': contract.get('delta', 0),
                                    'iv': contract.get('volatility', 0)
                                })
        
        return {
            'symbol': symbol,
            'price': price,
            'flows': sorted(flows, key=lambda x: x['volume'], reverse=True)[:10]
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None

def generate_theme_insights(theme, stock_data_list, all_patterns):
    """Generate AI insights for entire theme"""
    if not stock_data_list or not all_patterns:
        return None
    
    sentiment_map = {
        'BULLISH': 1, 'BULLISH_CAPPED': 0.7, 'BULLISH_HEDGED': 0.5,
        'BEARISH': -1, 'VOLATILE': 0, 'SCALPING': 0,
        'LONG_TERM': 0.8, 'UNUSUAL': 0.5
    }
    
    total_sentiment = 0
    pattern_count = 0
    
    for patterns in all_patterns.values():
        if patterns:
            for pattern in patterns:
                sentiment_value = sentiment_map.get(pattern['sentiment'], 0)
                confidence_weight = pattern['confidence'] / 100
                total_sentiment += sentiment_value * confidence_weight
                pattern_count += 1
    
    if pattern_count == 0:
        return None
    
    avg_sentiment = total_sentiment / pattern_count
    avg_performance = sum([s['change_pct'] for s in stock_data_list]) / len(stock_data_list)
    
    if avg_sentiment > 0.4:
        sentiment_emoji = "🚀"
        sentiment_text = "BULLISH"
        sentiment_color = "#22c55e"
        insight = f"Strong bullish flow detected across {len(stock_data_list)} AI stocks. Options positioning suggests institutional accumulation and upside expectations."
    elif avg_sentiment < -0.4:
        sentiment_emoji = "🔻"
        sentiment_text = "BEARISH"
        sentiment_color = "#ef4444"
        insight = f"Bearish flow dominates. Heavy put buying or call selling indicates defensive positioning or profit-taking in AI sector."
    elif abs(avg_sentiment) <= 0.4 and pattern_count > 3:
        sentiment_emoji = "💥"
        sentiment_text = "VOLATILE"
        sentiment_color = "#f59e0b"
        insight = f"Mixed signals with balanced flow. Market expects volatility or big move but direction unclear."
    else:
        sentiment_emoji = "➡️"
        sentiment_text = "NEUTRAL"
        sentiment_color = "#6b7280"
        insight = f"Neutral positioning. Limited unusual activity detected. Theme may be consolidating."
    
    if avg_performance > 2:
        insight += f" Price action confirms strength (+{avg_performance:.1f}% avg)."
    elif avg_performance < -2:
        insight += f" Price weakness ({avg_performance:.1f}% avg) aligns with bearish flows."
    
    return {
        'emoji': sentiment_emoji,
        'sentiment': sentiment_text,
        'color': sentiment_color,
        'confidence': int(min(95, pattern_count * 10 + 50)),
        'insight': insight,
        'pattern_count': pattern_count
    }

@st.cache_data(ttl=300)
def analyze_flow_patterns(flows_data, current_price):
    """AI-powered pattern recognition for options flows"""
    # Convert tuple back to DataFrame for caching compatibility
    if not flows_data:
        return None
    
    flows_df = pd.DataFrame(list(flows_data))
    if flows_df.empty:
        return None
    
    patterns = []
    
    call_vol = flows_df[flows_df['type'] == 'CALL']['volume'].sum()
    put_vol = flows_df[flows_df['type'] == 'PUT']['volume'].sum()
    total_vol = call_vol + put_vol
    
    call_premium = flows_df[flows_df['type'] == 'CALL']['volume'].sum() * flows_df[flows_df['type'] == 'CALL']['premium'].mean()
    put_premium = flows_df[flows_df['type'] == 'PUT']['volume'].sum() * flows_df[flows_df['type'] == 'PUT']['premium'].mean()
    
    # Pattern 1: Heavy Call Buying (Bullish)
    if call_vol > put_vol * 2 and call_vol > 5000:
        patterns.append({
            'name': '🚀 Heavy Call Buying',
            'sentiment': 'BULLISH',
            'confidence': min(95, int((call_vol / (call_vol + put_vol)) * 100)),
            'description': f'Strong call volume ({call_vol:,} vs {put_vol:,} puts) suggests bullish positioning. Large players betting on AI upside.',
            'color': '#22c55e'
        })
    
    # Pattern 2: Heavy Put Buying (Bearish)
    elif put_vol > call_vol * 2 and put_vol > 5000:
        patterns.append({
            'name': '🔻 Heavy Put Buying',
            'sentiment': 'BEARISH',
            'confidence': min(95, int((put_vol / (call_vol + put_vol)) * 100)),
            'description': f'Strong put volume ({put_vol:,} vs {call_vol:,} calls) suggests bearish positioning or hedging activity.',
            'color': '#ef4444'
        })
    
    # Pattern 3: Straddle/Strangle (Big Move Expected)
    near_atm_calls = flows_df[(flows_df['type'] == 'CALL') & (abs(flows_df['strike'] - current_price) / current_price < 0.03)]
    near_atm_puts = flows_df[(flows_df['type'] == 'PUT') & (abs(flows_df['strike'] - current_price) / current_price < 0.03)]
    
    if not near_atm_calls.empty and not near_atm_puts.empty:
        atm_call_vol = near_atm_calls['volume'].sum()
        atm_put_vol = near_atm_puts['volume'].sum()
        
        if abs(atm_call_vol - atm_put_vol) / max(atm_call_vol, atm_put_vol) < 0.3:
            patterns.append({
                'name': '💥 Straddle/Strangle Play',
                'sentiment': 'VOLATILE',
                'confidence': 75,
                'description': f'Balanced ATM call/put buying suggests expectation of large move. Premium: ${(call_premium + put_premium)/1000:.1f}K',
                'color': '#f59e0b'
            })
    
    # Pattern 4: Call Ratio Spread
    otm_calls = flows_df[(flows_df['type'] == 'CALL') & (flows_df['strike'] > current_price * 1.05)]
    atm_calls = flows_df[(flows_df['type'] == 'CALL') & (abs(flows_df['strike'] - current_price) / current_price < 0.05)]
    
    if not otm_calls.empty and not atm_calls.empty:
        otm_call_vol = otm_calls['volume'].sum()
        atm_call_vol = atm_calls['volume'].sum()
        
        if otm_call_vol > atm_call_vol * 1.5:
            patterns.append({
                'name': '📊 Call Ratio Spread',
                'sentiment': 'BULLISH_CAPPED',
                'confidence': 70,
                'description': f'Heavy OTM call selling vs ATM buying suggests bullish but capped upside. Target: ${otm_calls["strike"].min():.2f}',
                'color': '#10b981'
            })
    
    # Pattern 5: Protective Put Buying
    otm_puts = flows_df[(flows_df['type'] == 'PUT') & (flows_df['strike'] < current_price * 0.95)]
    
    if not otm_puts.empty and call_vol > put_vol:
        otm_put_vol = otm_puts['volume'].sum()
        
        if otm_put_vol > 3000:
            patterns.append({
                'name': '🛡️ Protective Put Buying',
                'sentiment': 'BULLISH_HEDGED',
                'confidence': 65,
                'description': f'OTM puts ({otm_put_vol:,}) with call dominance suggests bullish thesis with risk management. Floor: ${otm_puts["strike"].max():.2f}',
                'color': '#3b82f6'
            })
    
    # Pattern 6: Short-term Gamma Scalp
    short_dte = flows_df[flows_df['dte'] <= 7]
    
    if not short_dte.empty and short_dte['volume'].sum() > total_vol * 0.6:
        patterns.append({
            'name': '⚡ Short-term Gamma Play',
            'sentiment': 'SCALPING',
            'confidence': 80,
            'description': f'{short_dte["volume"].sum() / total_vol * 100:.0f}% volume in <7 DTE suggests gamma scalping or event-driven trade.',
            'color': '#8b5cf6'
        })
    
    # Pattern 7: LEAPS Accumulation
    long_dte = flows_df[flows_df['dte'] > 60]
    
    if not long_dte.empty and long_dte['volume'].sum() > 2000:
        long_sentiment = 'CALL' if long_dte[long_dte['type'] == 'CALL']['volume'].sum() > long_dte[long_dte['type'] == 'PUT']['volume'].sum() else 'PUT'
        patterns.append({
            'name': '🎯 LEAPS Accumulation',
            'sentiment': 'LONG_TERM',
            'confidence': 85,
            'description': f'Significant {long_sentiment.lower()} volume in long-dated options shows institutional conviction. Avg DTE: {long_dte["dte"].mean():.0f}',
            'color': '#6366f1'
        })
    
    # Pattern 8: Smart Money Flow
    high_ratio = flows_df[flows_df['vol_oi'] > 5]
    
    if not high_ratio.empty:
        avg_ratio = high_ratio['vol_oi'].mean()
        patterns.append({
            'name': '🧠 Smart Money Flow',
            'sentiment': 'UNUSUAL',
            'confidence': 90,
            'description': f'{len(high_ratio)} contracts with Vol/OI > 5 (avg {avg_ratio:.1f}x). New positions opening.',
            'color': '#ec4899'
        })
    
    return patterns

def create_performance_chart(theme_data, theme_name):
    """Create performance chart for theme stocks"""
    if not theme_data:
        return None
    
    fig = go.Figure()
    
    for stock_data in theme_data:
        if stock_data:
            fig.add_trace(go.Bar(
                x=[stock_data['symbol']],
                y=[stock_data.get('change_pct', 0)],
                text=[f"{stock_data.get('change_pct', 0):.2f}%"],
                textposition='auto',
                marker_color='green' if stock_data.get('change_pct', 0) > 0 else 'red',
                name=stock_data['symbol']
            ))
    
    fig.update_layout(
        title=f"{theme_name} Performance",
        yaxis_title="Change %",
        showlegend=False,
        height=300,
        template='plotly_white'
    )
    
    return fig

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1 style="font-size: 3em; margin-bottom: 10px;">🤖 Top 30 AI Stocks Tracker</h1>
    <p style="font-size: 1.2em; opacity: 0.9;">Real-time performance & options flow analysis for leading AI companies</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'overview'
if 'selected_theme_idx' not in st.session_state:
    st.session_state.selected_theme_idx = None

# Controls
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

with col1:
    view_mode = st.segmented_control(
        "View Mode",
        options=["Overview", "Heatmap", "Detailed"],
        default="Overview",
        key='view_selector'
    )
    st.session_state.view_mode = view_mode.lower() if view_mode else 'overview'

with col2:
    sentiment_filter = st.selectbox(
        "Sentiment Filter",
        options=["All", "Bullish", "Bearish", "Volatile"],
        help="Filter by market sentiment"
    )

with col3:
    min_volume = st.slider(
        "Min Options Volume",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Minimum option volume to show"
    )

with col4:
    if st.button("🔄", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Get monthly expiries
monthly_expiries = get_next_monthly_expiries(4)

# Fetch all stock data first
@st.cache_data(ttl=300)
def fetch_all_stock_data():
    """Fetch data for all 30 AI stocks"""
    all_data = {}
    for theme in AI_THEMES:
        for symbol in theme['stocks']:
            if symbol not in all_data:
                stock_info = get_stock_data(symbol)
                if stock_info:
                    stock_info['symbol'] = symbol
                    all_data[symbol] = stock_info
    return all_data

with st.spinner("Loading AI sector data..."):
    all_stock_data = fetch_all_stock_data()

# Calculate theme summaries
theme_summaries = []
for theme in AI_THEMES:
    stocks_data = [all_stock_data[s] for s in theme['stocks'] if s in all_stock_data]
    if stocks_data:
        avg_change = sum([s['change_pct'] for s in stocks_data]) / len(stocks_data)
        winners = len([s for s in stocks_data if s['change_pct'] > 0])
        best_stock = max(stocks_data, key=lambda x: x['change_pct'])
        
        theme_summaries.append({
            'theme': theme,
            'avg_change': avg_change,
            'winners': winners,
            'total': len(stocks_data),
            'best_stock': best_stock,
            'stocks_data': stocks_data
        })

# ===== VIEW MODE: OVERVIEW =====
if st.session_state.view_mode == 'overview':
    st.markdown("### 📊 AI Themes Overview")
    st.caption(f"📅 Options scanning: {', '.join([e.strftime('%b %d') for e in monthly_expiries])}")
    
    # Display theme cards in grid
    for idx in range(0, len(theme_summaries), 3):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            if idx + col_idx < len(theme_summaries):
                summary = theme_summaries[idx + col_idx]
                theme = summary['theme']
                
                with col:
                    # Determine sentiment color
                    if summary['avg_change'] > 1:
                        border_color = "#22c55e"
                        sentiment_emoji = "🚀"
                    elif summary['avg_change'] < -1:
                        border_color = "#ef4444"
                        sentiment_emoji = "🔻"
                    else:
                        border_color = "#f59e0b"
                        sentiment_emoji = "➡️"
                    
                    st.markdown(f"""
                    <div class="theme-overview-card" style="border-left-color: {border_color};">
                        <div style="font-size: 20px; margin-bottom: 5px;">{sentiment_emoji}</div>
                        <div style="font-size: 14px; font-weight: 700; color: #1f2937; margin-bottom: 5px;">
                            {theme['name'].split(' ', 1)[1]}
                        </div>
                        <div style="font-size: 11px; color: #6b7280; margin-bottom: 8px; line-height: 1.3;">
                            {theme['description'][:55]}...
                        </div>
                        <div class="quick-stat" style="background: {border_color}22; color: {border_color}; font-weight: 700;">
                            Avg: {summary['avg_change']:+.2f}%
                        </div>
                        <div class="quick-stat">
                            Winners: {summary['winners']}/{summary['total']}
                        </div>
                        <div class="quick-stat">
                            Top: {summary['best_stock']['symbol']} ({summary['best_stock']['change_pct']:+.1f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("View Details", key=f"view_theme_{idx + col_idx}", use_container_width=True):
                        st.session_state.selected_theme_idx = idx + col_idx
                        st.session_state.view_mode = 'detailed'
                        st.rerun()

# ===== VIEW MODE: HEATMAP =====
elif st.session_state.view_mode == 'heatmap':
    st.markdown("### 🔥 AI Sector Heatmap")
    st.caption("Visual overview of all 30 stocks - Click any cell for details")
    
    # Create heatmap
    for theme_summary in theme_summaries:
        theme = theme_summary['theme']
        st.markdown(f"**{theme['name']}**")
        
        # Display stocks in grid
        cols = st.columns(5)
        for idx, stock_data in enumerate(theme_summary['stocks_data']):
            with cols[idx % 5]:
                change = stock_data['change_pct']
                
                # Color based on performance
                if change > 3:
                    bg_color = "#22c55e"
                    text_color = "white"
                elif change > 0:
                    bg_color = "#86efac"
                    text_color = "#166534"
                elif change > -3:
                    bg_color = "#fca5a5"
                    text_color = "#991b1b"
                else:
                    bg_color = "#ef4444"
                    text_color = "white"
                
                st.markdown(f"""
                <div class="heatmap-cell" style="background: {bg_color}; color: {text_color};">
                    <div style="font-size: 12px; font-weight: 700;">{stock_data['symbol']}</div>
                    <div style="font-size: 14px; font-weight: 700;">{change:+.1f}%</div>
                    <div style="font-size: 10px; opacity: 0.9;">${stock_data['price']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

# ===== VIEW MODE: DETAILED =====
elif st.session_state.view_mode == 'detailed':
    # Show selected theme or first theme
    if st.session_state.selected_theme_idx is not None:
        selected_summary = theme_summaries[st.session_state.selected_theme_idx]
    else:
        selected_summary = theme_summaries[0]
    
    theme = selected_summary['theme']
    theme_idx = st.session_state.selected_theme_idx or 0
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Overview"):
            st.session_state.view_mode = 'overview'
            st.rerun()
    
    # Theme selector
    with col2:
        theme_names = [t['theme']['name'] for t in theme_summaries]
        selected_theme_name = st.selectbox(
            "Select Theme",
            options=theme_names,
            index=theme_idx,
            label_visibility="collapsed"
        )
        # Update index if changed
        new_idx = theme_names.index(selected_theme_name)
        if new_idx != theme_idx:
            st.session_state.selected_theme_idx = new_idx
            st.rerun()
    
    st.markdown("---")
    
    # Process detailed view for selected theme
    theme_to_show = theme_summaries[st.session_state.selected_theme_idx or 0]
    theme = theme_to_show['theme']
    stock_data_list = theme_to_show['stocks_data']
    
    # Theme description
    st.markdown(f"""
    <div class="theme-card">
        <div style="font-size: 20px; font-weight: 700; margin-bottom: 10px;">{theme['name']}</div>
        <div class="theme-description">{theme['description']}</div>
        <div style="font-size: 12px; opacity: 0.8;">
            📊 Stocks: {', '.join(theme['stocks'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance overview
    if stock_data_list:
        perf_col1, perf_col2 = st.columns([1, 2])
        
        with perf_col1:
            for stock in stock_data_list:
                change_color = "🟢" if stock['change_pct'] > 0 else "🔴"
                st.markdown(f"""
                <div class="stock-metric">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 14px;">{stock['symbol']}</strong>
                            <div style="font-size: 16px; font-weight: 700; color: #333;">
                                ${stock['price']:.2f}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14px; font-weight: 600; color: {'green' if stock['change_pct'] > 0 else 'red'};">
                                {change_color} {stock['change_pct']:.2f}%
                            </div>
                            <div style="font-size: 11px; color: #666;">
                                Vol: {stock['volume']:,}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with perf_col2:
            chart = create_performance_chart(stock_data_list, theme['name'])
            if chart:
                st.plotly_chart(chart, use_container_width=True, key=f"ai_perf_chart_{theme_idx}")
    
    # Options flow analysis
    st.markdown("### 📊 Options Flow Analysis")
    st.caption(f"Scanning {', '.join([e.strftime('%b %d') for e in monthly_expiries])}")
    
    # Pre-fetch ALL flow data for all symbols before creating tabs
    # This way tab switching is instant - data is already cached
    with st.spinner(f"Loading options flow for {len(theme['stocks'])} stocks..."):
        all_flow_data = {}
        all_patterns = {}
        
        for symbol in theme['stocks']:
            flow_data = scan_options_flow(symbol, monthly_expiries)
            if flow_data and flow_data['flows']:
                flows_df = pd.DataFrame(flow_data['flows'])
                flows_df = flows_df[flows_df['volume'] >= min_volume]
                
                if not flows_df.empty:
                    # Convert DataFrame to tuple for caching
                    flows_tuple = tuple(flows_df.to_dict('records'))
                    patterns = analyze_flow_patterns(flows_tuple, flow_data['price'])
                    
                    all_flow_data[symbol] = {
                        'flow_data': flow_data,
                        'flows_df': flows_df,
                        'patterns': patterns
                    }
                    all_patterns[symbol] = patterns
    
    theme_insights_placeholder = st.empty()
    
    # Now create tabs and display pre-fetched data (instant switching)
    flow_tabs = st.tabs(theme['stocks'])
    
    for idx, symbol in enumerate(theme['stocks']):
        with flow_tabs[idx]:
            if symbol in all_flow_data:
                cached_data = all_flow_data[symbol]
                flow_data = cached_data['flow_data']
                flows_df = cached_data['flows_df']
                patterns = cached_data['patterns']
                
                st.success(f"Found {len(flow_data['flows'])} unusual options flows")
                
                # AI Pattern Recognition
                st.markdown("#### 🤖 AI Pattern Recognition")
                
                if patterns:
                    pattern_cols = st.columns(min(3, len(patterns)))
                    
                    for p_idx, pattern in enumerate(patterns[:3]):
                        with pattern_cols[p_idx]:
                            st.markdown(f"""
                            <div style="
                                background: {pattern['color']};
                                color: white;
                                padding: 15px;
                                border-radius: 10px;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                                margin-bottom: 10px;
                            ">
                                <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                                    {pattern['name']}
                                </div>
                                <div style="font-size: 12px; opacity: 0.9; margin-bottom: 8px;">
                                    {pattern['sentiment']} • {pattern['confidence']}% confidence
                                </div>
                                <div style="font-size: 13px; line-height: 1.4;">
                                    {pattern['description']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if len(patterns) > 3:
                        with st.expander(f"View all {len(patterns)} patterns"):
                            for pattern in patterns[3:]:
                                st.markdown(f"""
                                **{pattern['name']}** ({pattern['confidence']}% confidence)  
                                *{pattern['sentiment']}*: {pattern['description']}
                                """)
                else:
                    st.info("No significant patterns detected")
                
                st.markdown("---")
                
                # Summary metrics
                call_vol = flows_df[flows_df['type'] == 'CALL']['volume'].sum()
                put_vol = flows_df[flows_df['type'] == 'PUT']['volume'].sum()
                pc_ratio = put_vol / call_vol if call_vol > 0 else 0
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Total Call Volume", f"{call_vol:,.0f}")
                metric_col2.metric("Total Put Volume", f"{put_vol:,.0f}")
                metric_col3.metric("P/C Ratio", f"{pc_ratio:.2f}")
                
                # Top flows table
                st.markdown("**Top Unusual Flows:**")
                display_df = flows_df.copy()
                display_df['expiry'] = display_df['expiry'].astype(str)
                display_df = display_df[['type', 'strike', 'expiry', 'dte', 'volume', 'oi', 'vol_oi', 'premium', 'iv']]
                display_df.columns = ['Type', 'Strike', 'Expiry', 'DTE', 'Volume', 'OI', 'Vol/OI', 'Premium', 'IV']
                
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Volume": st.column_config.NumberColumn(format="%d"),
                        "Premium": st.column_config.NumberColumn(format="$%.2f"),
                        "Vol/OI": st.column_config.NumberColumn(format="%.2f"),
                        "IV": st.column_config.NumberColumn(format="%.1f%%")
                    }
                )
                
                # Volume by expiry chart
                exp_vol = flows_df.groupby(['expiry', 'type'])['volume'].sum().reset_index()
                fig = px.bar(
                    exp_vol,
                    x='expiry',
                    y='volume',
                    color='type',
                    title=f"{symbol} - Option Volume by Expiry",
                    color_discrete_map={'CALL': '#22c55e', 'PUT': '#ef4444'}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"ai_vol_chart_{theme_idx}_{symbol}")
            else:
                st.warning(f"No unusual options activity found for {symbol}")
    
    # Theme-level insights
    if all_patterns and stock_data_list:
        theme_summary = generate_theme_insights(theme, stock_data_list, all_patterns)
        
        if theme_summary:
            with theme_insights_placeholder:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {theme_summary['color']}22 0%, {theme_summary['color']}11 100%);
                    border-left: 5px solid {theme_summary['color']};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <div style="font-size: 32px; margin-right: 15px;">{theme_summary['emoji']}</div>
                        <div>
                            <div style="font-size: 20px; font-weight: 700; color: {theme_summary['color']};">
                                Theme Sentiment: {theme_summary['sentiment']}
                            </div>
                            <div style="font-size: 13px; color: #666;">
                                AI Confidence: {theme_summary['confidence']}% • Based on {theme_summary['pattern_count']} patterns
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 15px; line-height: 1.6; color: #333;">
                        {theme_summary['insight']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")

# Summary statistics
st.markdown("## 📈 AI Sector Summary")

all_stocks = []
for theme_summary in theme_summaries:
    all_stocks.extend(theme_summary['theme']['stocks'])

all_stocks = list(set(all_stocks))

if all_stocks:
    summary_data = []
    
    with st.spinner("Calculating AI sector metrics..."):
        for symbol in all_stocks:
            stock_info = get_stock_data(symbol)
            if stock_info:
                summary_data.append({
                    'Symbol': symbol,
                    'Price': stock_info['price'],
                    'Change %': stock_info['change_pct'],
                    'Volume': stock_info['volume']
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        with sum_col1:
            winners = len(summary_df[summary_df['Change %'] > 0])
            st.metric("Winners", f"{winners}/{len(summary_df)}")
        
        with sum_col2:
            avg_change = summary_df['Change %'].mean()
            st.metric("Avg Change", f"{avg_change:.2f}%")
        
        with sum_col3:
            best = summary_df.nlargest(1, 'Change %').iloc[0]
            st.metric("Best Performer", f"{best['Symbol']}", f"+{best['Change %']:.2f}%")
        
        with sum_col4:
            worst = summary_df.nsmallest(1, 'Change %').iloc[0]
            st.metric("Worst Performer", f"{worst['Symbol']}", f"{worst['Change %']:.2f}%")
        
        # Full table
        st.dataframe(
            summary_df.sort_values('Change %', ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                "Volume": st.column_config.NumberColumn(format="%d")
            }
        )

st.caption("💡 Data refreshes every 5 minutes. Click Refresh to update immediately.")
