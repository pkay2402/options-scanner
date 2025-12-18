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

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Trade Ideas 2026",
    page_icon="🚀",
    layout="wide"
)

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

# Controls
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_themes = st.multiselect(
        "Filter Themes",
        options=[f"#{t['number']}: {t['title']}" for t in TRADE_IDEAS],
        default=[],
        help="Select specific themes to analyze"
    )

with col2:
    min_volume = st.slider(
        "Min Options Volume",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Minimum option volume to show"
    )

with col3:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Get monthly expiries
monthly_expiries = get_next_monthly_expiries(4)
st.caption(f"📅 Scanning next 4 monthly expiries: {', '.join([e.strftime('%b %d') for e in monthly_expiries])}")

# Filter themes if selected
if selected_themes:
    theme_numbers = [int(t.split(':')[0].replace('#', '')) for t in selected_themes]
    themes_to_show = [t for t in TRADE_IDEAS if t['number'] in theme_numbers]
else:
    themes_to_show = TRADE_IDEAS

# Process each theme
for theme in themes_to_show:
    if not theme['stocks']:
        continue
    
    with st.expander(f"**#{theme['number']}: {theme['title']}** ({len(theme['stocks'])} stocks)", expanded=False):
        # Theme description
        st.markdown(f"""
        <div class="theme-card">
            <div class="theme-description">{theme['description']}</div>
            <div style="font-size: 12px; opacity: 0.8;">
                💡 Catalyst: {theme['catalyst']} | 📊 Stocks: {', '.join(theme['stocks'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fetch stock data
        with st.spinner(f"Loading data for {', '.join(theme['stocks'])}..."):
            stock_data_list = []
            
            for symbol in theme['stocks']:
                stock_info = get_stock_data(symbol)
                if stock_info:
                    stock_info['symbol'] = symbol
                    stock_data_list.append(stock_info)
        
        # Performance overview
        if stock_data_list:
            perf_col1, perf_col2 = st.columns([1, 2])
            
            with perf_col1:
                # Metrics
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
                # Performance chart
                chart = create_performance_chart(stock_data_list)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        
        # Options flow analysis
        st.markdown("### 📊 Options Flow Analysis")
        
        flow_tabs = st.tabs(theme['stocks'])
        
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
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No flows above {min_volume:,} volume threshold")
                else:
                    st.warning(f"No unusual options activity found for {symbol}")
        
        st.markdown("---")

# Summary statistics
st.markdown("## 📈 Portfolio Summary")

all_stocks = []
for theme in themes_to_show:
    all_stocks.extend(theme['stocks'])

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
