"""
Trade Ideas 2026 Tracker
Track performance and options flow for the 26 trade ideas
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

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Trade Ideas 2026",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'overview'
if 'selected_theme_idx' not in st.session_state:
    st.session_state.selected_theme_idx = None

# Custom CSS
st.markdown("""
<style>
    .theme-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .theme-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .theme-description {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    .stock-metric {
        background: white;
        color: #333;
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .flow-card {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffd700;
    }
    
    .theme-overview-card {
        background: white;
        border-radius: 6px;
        padding: 8px 10px;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: transform 0.15s, box-shadow 0.15s;
        cursor: pointer;
        margin-bottom: 2px;
    }
    
    .theme-overview-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }
    
    .theme-overview-card:hover .view-icon {
        opacity: 1 !important;
    }
    
    .heatmap-cell {
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin: 4px 0;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .quick-stat {
        padding: 2px 6px;
        background: #f3f4f6;
        border-radius: 3px;
        margin: 2px 0;
        font-size: 10px;
    }
    
    .view-toggle {
        background: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Trade Ideas Data Structure
TRADE_IDEAS = [
    {
        "number": 1,
        "title": "Bullshit Jobs",
        "description": "Companies with high bureaucracy scores and low margins that can benefit from AI-driven efficiency gains",
        "stocks": ["CHRW", "ACN"],
        "catalyst": "AI adoption"
    },
    {
        "number": 2,
        "title": "Post-Traumatic Supply Disorder",
        "description": "Oversupplied sectors that have rationalized capacity and now face structural tailwinds",
        "stocks": ["MU", "WDC", "GEV"],
        "catalyst": "Supply/Demand rebalance"
    },
    {
        "number": 3,
        "title": "Inference on Device",
        "description": "Next wave of AI monetization through on-device inference",
        "stocks": ["AVGO", "AMD"],
        "catalyst": "Edge AI deployment"
    },
    {
        "number": 4,
        "title": "Advanced Packaging",
        "description": "Critical enabler of next-gen AI chips through chiplets and 3D stacking",
        "stocks": ["AMKR", "ATAT", "KLIC"],
        "catalyst": "AI chip complexity"
    },
    {
        "number": 5,
        "title": "AI Materials",
        "description": "Copper miners benefiting from massive AI data center buildout",
        "stocks": ["FCX", "SCCO"],
        "catalyst": "Data center demand"
    },
    {
        "number": 6,
        "title": "GOP Loses House",
        "description": "Medicaid-focused healthcare plays positioned for 2026 midterms",
        "stocks": ["MOH", "CNC", "ELV"],
        "catalyst": "2026 Midterms"
    },
    {
        "number": 7,
        "title": "Midterm Media Spend",
        "description": "Political advertising surge in 2026 midterm election year",
        "stocks": ["LYV", "PARA"],
        "catalyst": "Political ad spend"
    },
    {
        "number": 8,
        "title": "Shipbuilding",
        "description": "Defense contractors focused on naval vessels",
        "stocks": ["HII", "GD", "NOC", "LHX"],
        "catalyst": "Defense spending"
    },
    {
        "number": 9,
        "title": "Bread & Circuses",
        "description": "Entertainment and leisure companies capitalizing on experience economy",
        "stocks": ["LYV", "PLAY", "FUN"],
        "catalyst": "Experience economy"
    },
    {
        "number": 10,
        "title": "Earned Wage Access",
        "description": "Fintech platforms providing workers instant access to earned wages",
        "stocks": ["DAVE", "OPFI"],
        "catalyst": "Financial inclusion"
    },
    {
        "number": 11,
        "title": "Rate Sensitive Regionals",
        "description": "Regional banks positioned to benefit from rate cuts",
        "stocks": ["OPBK", "WSBF", "FCFS"],
        "catalyst": "Fed rate cuts"
    },
    {
        "number": 12,
        "title": "Insurance Marketing",
        "description": "Digital insurance comparison platforms capturing lead generation",
        "stocks": ["NRDS", "EVER"],
        "catalyst": "Digital distribution"
    },
    {
        "number": 13,
        "title": "World Cup 2026",
        "description": "Entertainment and hospitality plays benefiting from World Cup",
        "stocks": ["LYV"],
        "catalyst": "FIFA World Cup"
    },
    {
        "number": 14,
        "title": "Geopolitical Special Situations",
        "description": "Opportunistic plays on geopolitical shifts",
        "stocks": ["JMIA"],
        "catalyst": "Policy changes"
    },
    {
        "number": 15,
        "title": "China Plays",
        "description": "Semiconductor equipment benefiting from China's domestic chip push",
        "stocks": ["ACMR"],
        "catalyst": "China stimulus"
    },
    {
        "number": 16,
        "title": "European Utilities",
        "description": "European power companies benefiting from energy transition",
        "stocks": [],  # EOAN GR, ENR GR - not tradable on Schwab
        "catalyst": "Energy demand"
    },
    {
        "number": 17,
        "title": "Robotics & Automation",
        "description": "Industrial robotics benefiting from labor shortage",
        "stocks": ["TXT", "SNBR", "EVLV"],
        "catalyst": "Labor automation"
    },
    {
        "number": 18,
        "title": "Biotherapeutics",
        "description": "Fat redistribution treatments and novel oncology therapies",
        "stocks": ["ALGN", "OSCR"],
        "catalyst": "GLP-1 adoption"
    },
    {
        "number": 19,
        "title": "Macro Plays",
        "description": "Lithium plays on battery demand",
        "stocks": ["SQM", "ALB"],
        "catalyst": "EV adoption"
    },
    {
        "number": 20,
        "title": "The Girlfriend Index",
        "description": "Consumer brands favored by young women driving cultural trends",
        "stocks": ["CAVA", "YETI", "BOOT"],
        "catalyst": "Consumer trends"
    },
    {
        "number": 21,
        "title": "Jumia",
        "description": "African e-commerce leader positioned for continent's digital commerce explosion",
        "stocks": ["JMIA"],
        "catalyst": "African e-commerce"
    },
    {
        "number": 22,
        "title": "Synopsys",
        "description": "EDA software essential for chip design",
        "stocks": ["SNPS", "CDNS", "ANSS"],
        "catalyst": "Chip complexity"
    },
    {
        "number": 23,
        "title": "Long Boeing / Short Airbus",
        "description": "Contrarian play on Boeing turnaround",
        "stocks": ["BA"],
        "catalyst": "Production ramp"
    },
    {
        "number": 24,
        "title": "Long Bitcoin / Short MSTR",
        "description": "Capture Bitcoin upside while shorting MSTR premium",
        "stocks": ["MSTR"],
        "catalyst": "Crypto volatility"
    },
    {
        "number": 25,
        "title": "WPP - Not Dead Yet",
        "description": "Contrarian value play on advertising giant",
        "stocks": [],  # WPP LN - not tradable on Schwab
        "catalyst": "AI margin expansion"
    },
    {
        "number": 26,
        "title": "Orbital Manufacturing",
        "description": "Space-based manufacturing moonshot",
        "stocks": ["RKLB"],
        "catalyst": "Space economy"
    }
]

def get_next_monthly_expiries(n=4):
    """Get next N monthly expiries (3rd Friday of each month)"""
    expiries = []
    today = datetime.now().date()
    current_month = today.month
    current_year = today.year
    
    for i in range(n + 2):  # Get extra to ensure we have enough
        # Calculate 3rd Friday of target month
        year = current_year + (current_month + i - 1) // 12
        month = ((current_month + i - 1) % 12) + 1
        
        # Find first day of month
        first_day = datetime(year, month, 1).date()
        # Find first Friday (weekday 4)
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)
        
        # Only include if in the future
        if third_friday > today:
            expiries.append(third_friday)
    
    return expiries[:n]

@st.cache_data(ttl=300)
def get_stock_data(symbol: str):
    """Get current stock price and basic info"""
    client = SchwabClient()
    
    if not client.authenticate():
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
                'prev_close': data.get('closePrice', data['lastPrice'])
            }
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    
    return None

@st.cache_data(ttl=300)
def scan_options_flow(symbol: str, expiry_dates: list):
    """Scan options flow across multiple expiries for unusual activity"""
    client = SchwabClient()
    
    if not client.authenticate():
        return None
    
    try:
        # Get current price
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
        
        price = quote[symbol]['quote']['lastPrice']
        flows = []
        
        for expiry in expiry_dates:
            exp_str = expiry.strftime('%Y-%m-%d')
            
            # Get options chain
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
                        
                        # Focus on ATM ±15%
                        if abs(strike - price) / price > 0.15:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1)
                            
                            if volume == 0 or oi == 0:
                                continue
                            
                            vol_oi = volume / oi if oi > 0 else 0
                            
                            # Look for unusual activity
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
                        
                        # Focus on ATM ±15%
                        if abs(strike - price) / price > 0.15:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1)
                            
                            if volume == 0 or oi == 0:
                                continue
                            
                            vol_oi = volume / oi if oi > 0 else 0
                            
                            # Look for unusual activity
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
            'flows': sorted(flows, key=lambda x: x['volume'], reverse=True)[:10]  # Top 10
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None

def generate_theme_insights(theme, stock_data_list, all_patterns):
    """Generate AI insights for entire theme"""
    if not stock_data_list or not all_patterns:
        return None
    
    # Aggregate sentiment
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
    
    # Average performance
    avg_performance = sum([s['change_pct'] for s in stock_data_list]) / len(stock_data_list)
    
    # Generate insight
    if avg_sentiment > 0.4:
        sentiment_emoji = "🚀"
        sentiment_text = "BULLISH"
        sentiment_color = "#22c55e"
        insight = f"Strong bullish flow detected across {len(stock_data_list)} stocks. Options positioning suggests institutional accumulation and upside expectations."
    elif avg_sentiment < -0.4:
        sentiment_emoji = "🔻"
        sentiment_text = "BEARISH"
        sentiment_color = "#ef4444"
        insight = f"Bearish flow dominates. Heavy put buying or call selling indicates defensive positioning or profit-taking."
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
    
    # Add performance context
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

def analyze_flow_patterns(flows_df, current_price):
    """AI-powered pattern recognition for options flows"""
    if flows_df.empty:
        return None
    
    patterns = []
    
    # Calculate aggregate metrics
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
            'description': f'Strong call volume ({call_vol:,} vs {put_vol:,} puts) suggests bullish positioning. Large players may be betting on upside.',
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
                'description': f'Balanced ATM call/put buying suggests expectation of large move in either direction. Premium: ${(call_premium + put_premium)/1000:.1f}K',
                'color': '#f59e0b'
            })
    
    # Pattern 4: Call Ratio Spread (Bullish but Capped)
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
                'description': f'Heavy OTM call selling vs ATM buying suggests bullish but capped upside expectations. Target: ${otm_calls["strike"].min():.2f}',
                'color': '#10b981'
            })
    
    # Pattern 5: Protective Put Buying (Bullish but Hedged)
    otm_puts = flows_df[(flows_df['type'] == 'PUT') & (flows_df['strike'] < current_price * 0.95)]
    
    if not otm_puts.empty and call_vol > put_vol:
        otm_put_vol = otm_puts['volume'].sum()
        
        if otm_put_vol > 3000:
            patterns.append({
                'name': '🛡️ Protective Put Buying',
                'sentiment': 'BULLISH_HEDGED',
                'confidence': 65,
                'description': f'OTM puts ({otm_put_vol:,}) with call dominance suggests bullish thesis but risk management. Floor: ${otm_puts["strike"].max():.2f}',
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
    
    # Pattern 7: LEAPS Accumulation (Long-term Conviction)
    long_dte = flows_df[flows_df['dte'] > 60]
    
    if not long_dte.empty and long_dte['volume'].sum() > 2000:
        long_sentiment = 'CALL' if long_dte[long_dte['type'] == 'CALL']['volume'].sum() > long_dte[long_dte['type'] == 'PUT']['volume'].sum() else 'PUT'
        patterns.append({
            'name': '🎯 LEAPS Accumulation',
            'sentiment': 'LONG_TERM',
            'confidence': 85,
            'description': f'Significant {long_sentiment.lower()} volume in long-dated options suggests institutional conviction. Avg DTE: {long_dte["dte"].mean():.0f}',
            'color': '#6366f1'
        })
    
    # Pattern 8: Unusual Vol/OI Ratio (Smart Money)
    high_ratio = flows_df[flows_df['vol_oi'] > 5]
    
    if not high_ratio.empty:
        avg_ratio = high_ratio['vol_oi'].mean()
        patterns.append({
            'name': '🧠 Smart Money Flow',
            'sentiment': 'UNUSUAL',
            'confidence': 90,
            'description': f'{len(high_ratio)} contracts with Vol/OI > 5 (avg {avg_ratio:.1f}x). New positions opening, not unwinding.',
            'color': '#ec4899'
        })
    
    return patterns

def create_performance_chart(theme_data):
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
        title="Stock Performance",
        yaxis_title="Change %",
        showlegend=False,
        height=300,
        template='plotly_white'
    )
    
    return fig

# Title
st.title("🚀 Trade Ideas 2026 Tracker")
st.markdown("Track performance and options flow for the 26 trade ideas")

# View mode selector
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    view_mode = st.segmented_control(
        "View",
        options=['overview', 'heatmap', 'detailed'],
        format_func=lambda x: {'overview': '📋 Overview', 'heatmap': '🗺️ Heatmap', 'detailed': '🔍 Detailed'}[x],
        selection_mode='single',
        default=st.session_state.view_mode,
        label_visibility='collapsed'
    )
    if view_mode and view_mode != st.session_state.view_mode:
        st.session_state.view_mode = view_mode
        st.rerun()

with col2:
    min_volume = st.number_input(
        "Min Volume",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        label_visibility='collapsed'
    )

with col3:
    sentiment_filter = st.selectbox(
        "Sentiment",
        options=['All', 'Bullish', 'Bearish', 'Neutral'],
        index=0,
        label_visibility='collapsed'
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
    """Fetch data for all stocks across all trade ideas"""
    all_data = {}
    for theme in TRADE_IDEAS:
        for symbol in theme['stocks']:
            if symbol and symbol not in all_data:
                stock_info = get_stock_data(symbol)
                if stock_info:
                    stock_info['symbol'] = symbol
                    all_data[symbol] = stock_info
    return all_data

with st.spinner("Loading trade ideas data..."):
    all_stock_data = fetch_all_stock_data()

# Calculate theme summaries
theme_summaries = []
for theme in TRADE_IDEAS:
    if not theme['stocks']:  # Skip themes with no stocks
        continue
    
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
    st.markdown("### 📊 Trade Ideas Overview")
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
                    
                    # Create clickable card with icon - wrap in container for button
                    with st.container():
                        card_html = f"""
                        <div class="theme-overview-card" style="border-left-color: {border_color}; position: relative;">
                            <div class="view-icon" style="position: absolute; top: 6px; right: 8px; font-size: 14px; opacity: 0.5; cursor: pointer; transition: opacity 0.2s;">
                                👁️
                            </div>
                            <div style="font-size: 11px; font-weight: 700; color: #1f2937; margin-bottom: 3px; display: flex; align-items: center; gap: 6px; padding-right: 25px;">
                                <span>{sentiment_emoji}</span>
                                <span>#{theme['number']}: {theme['title']}</span>
                            </div>
                            <div style="display: flex; gap: 4px; margin-bottom: 2px;">
                                <div class="quick-stat" style="background: {border_color}22; color: {border_color}; font-weight: 700; flex: 1;">
                                    Avg: {summary['avg_change']:+.1f}%
                                </div>
                                <div class="quick-stat" style="flex: 1;">
                                    {summary['winners']}/{summary['total']} 🏆
                                </div>
                            </div>
                            <div style="font-size: 9px; color: #9ca3af;">
                                💡 {theme['catalyst']}
                            </div>
                        </div>
                        """
                        
                        if st.button("", key=f"view_theme_{idx + col_idx}", use_container_width=True, help="Click to view details"):
                            st.session_state.selected_theme_idx = idx + col_idx
                            st.session_state.view_mode = 'detailed'
                            st.rerun()
                        
                        st.markdown(card_html, unsafe_allow_html=True)

# ===== VIEW MODE: HEATMAP =====
elif st.session_state.view_mode == 'heatmap':
    st.markdown("### 🔥 Trade Ideas Heatmap")
    st.caption("Visual overview of all stocks - Click any cell for details")
    
    # Create heatmap by theme
    for theme_summary in theme_summaries:
        theme = theme_summary['theme']
        st.markdown(f"**#{theme['number']}: {theme['title']}**")
        
        # Display stocks in grid
        if theme_summary['stocks_data']:
            cols = st.columns(min(5, len(theme_summary['stocks_data'])))
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
                        <div style="font-size: 13px; font-weight: 700;">{stock_data['symbol']}</div>
                        <div style="font-size: 16px; font-weight: 700;">{change:+.1f}%</div>
                        <div style="font-size: 11px; opacity: 0.9;">${stock_data['price']:.2f}</div>
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
        theme_names = [f"#{t['theme']['number']}: {t['theme']['title']}" for t in theme_summaries]
        selected_theme_name = st.selectbox(
            "Select Trade Idea",
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
        <div style="font-size: 20px; font-weight: 700; margin-bottom: 10px;">#{theme['number']}: {theme['title']}</div>
        <div class="theme-description">{theme['description']}</div>
        <div style="font-size: 12px; opacity: 0.8;">
            💡 Catalyst: {theme['catalyst']} | 📊 Stocks: {', '.join(theme['stocks'])}
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
                            <strong style="font-size: 16px;">{stock['symbol']}</strong>
                            <div style="font-size: 20px; font-weight: 700; color: #333;">
                                ${stock['price']:.2f}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 16px; font-weight: 600; color: {'green' if stock['change_pct'] > 0 else 'red'};">
                                {change_color} {stock['change_pct']:.2f}%
                            </div>
                            <div style="font-size: 12px; color: #666;">
                                Vol: {stock['volume']:,}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with perf_col2:
            chart = create_performance_chart(stock_data_list)
            if chart:
                st.plotly_chart(chart, use_container_width=True, key=f"trade_perf_chart_{theme_idx}")
    
    # Options flow analysis
    st.markdown("### 📊 Options Flow Analysis")
    st.caption(f"Scanning {', '.join([e.strftime('%b %d') for e in monthly_expiries])}")
    
    theme_insights_placeholder = st.empty()
    
    flow_tabs = st.tabs(theme['stocks'])
    
    all_patterns = {}
    
    for idx, symbol in enumerate(theme['stocks']):
        with flow_tabs[idx]:
            with st.spinner(f"Scanning options for {symbol}..."):
                flow_data = scan_options_flow(symbol, monthly_expiries)
            
            if flow_data and flow_data['flows']:
                st.success(f"Found {len(flow_data['flows'])} unusual options flows")
                
                # Create flow dataframe
                flows_df = pd.DataFrame(flow_data['flows'])
                
                # Filter by volume
                flows_df = flows_df[flows_df['volume'] >= min_volume]
                
                if not flows_df.empty:
                    # AI Pattern Recognition
                    st.markdown("#### 🤖 AI Pattern Recognition")
                    patterns = analyze_flow_patterns(flows_df, flow_data['price'])
                    all_patterns[symbol] = patterns  # Store for theme insights
                    
                    if patterns:
                        pattern_cols = st.columns(min(3, len(patterns)))
                        
                        for p_idx, pattern in enumerate(patterns[:3]):  # Show top 3 patterns
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
                        
                        # Show all patterns in expander
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
                    st.plotly_chart(fig, use_container_width=True, key=f"trade_vol_chart_{theme_idx}_{symbol}")
                else:
                    st.info(f"No flows above {min_volume:,} volume threshold")
            else:
                st.warning(f"No unusual options activity found for {symbol}")
    
    # Generate and display theme-level insights
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
st.markdown("## 📈 Portfolio Summary")

all_stocks = []
for theme_summary in theme_summaries:
    all_stocks.extend(theme_summary['theme']['stocks'])

all_stocks = list(set(all_stocks))  # Remove duplicates

if all_stocks:
    summary_data = []
    
    with st.spinner("Calculating portfolio metrics..."):
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
