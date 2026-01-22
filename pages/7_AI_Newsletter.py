#!/usr/bin/env python3
"""
Streamlit AI Stock Newsletter Generator
Interactive web app to generate and visualize stock opportunities with options flow data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.ai_stock_screener import StockScreener
from src.theme_tracker import THEMES
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import options functions
try:
    from src.ai_options import calculate_flow_score, fetch_all_options_data
    OPTIONS_AVAILABLE = True
except:
    OPTIONS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AI Stock Newsletter Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .score-high {
        background-color: #00ff00;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .score-medium {
        background-color: #ffff00;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .score-low {
        background-color: #ff9900;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stock-card {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_screening(min_score):
    """Run AI screening with caching"""
    screener = StockScreener()
    opportunities = screener.screen_themes(min_score=min_score)
    opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return opportunities  # Only return opportunities, not the screener object

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_options_data_for_symbols(symbols):
    """Fetch options data for given symbols"""
    if not OPTIONS_AVAILABLE:
        return {}
    
    try:
        df = fetch_all_options_data()
        if df.empty:
            return {}
        
        # Filter for our symbols
        df = df[df['Symbol'].isin(symbols)]
        if df.empty:
            return {}
        
        # Calculate metrics for each symbol
        options_metrics = {}
        for symbol in symbols:
            symbol_df = df[df['Symbol'] == symbol]
            if symbol_df.empty:
                continue
            
            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.fast_info.last_price
            except:
                current_price = None
            
            if current_price is None:
                continue
            
            # Calculate flow score and metrics
            symbol_df['Days_to_Expiry'] = (symbol_df['Expiration'] - datetime.now()).dt.days
            symbol_df['Premium'] = symbol_df['Volume'] * symbol_df['Last Price'] * 100
            
            flow_result = calculate_flow_score(symbol_df, current_price)
            
            if flow_result['score'] > 0:
                details = flow_result['details']
                
                # Calculate call/put ratio
                call_vol = symbol_df[symbol_df['Call/Put'] == 'C']['Volume'].sum()
                put_vol = symbol_df[symbol_df['Call/Put'] == 'P']['Volume'].sum()
                cp_ratio = call_vol / put_vol if put_vol > 0 else float('inf')
                
                # Identify large trades (adaptive threshold)
                total_premium = details['total_premium']
                avg_premium_per_contract = total_premium / symbol_df['Volume'].sum() if symbol_df['Volume'].sum() > 0 else 0
                
                # Filter for OTM options only (use 'Strike Price' column name from CBOE data)
                strike_col = 'Strike Price' if 'Strike Price' in symbol_df.columns else 'Strike'
                otm_df = symbol_df.copy()
                otm_df = otm_df[
                    ((otm_df['Call/Put'] == 'C') & (otm_df[strike_col] > current_price)) |  # Calls: strike > price
                    ((otm_df['Call/Put'] == 'P') & (otm_df[strike_col] < current_price))    # Puts: strike < price
                ]
                
                # Dynamic threshold: Large trades are those > 80th percentile or > $100k (OTM only)
                if not otm_df.empty:
                    large_threshold = max(100000, otm_df['Premium'].quantile(0.8))
                    large_trades = otm_df[otm_df['Premium'] > large_threshold].copy()
                    large_trades = large_trades.nlargest(5, 'Premium')
                else:
                    large_trades = pd.DataFrame()
                
                options_metrics[symbol] = {
                    'cp_ratio': cp_ratio,
                    'call_volume': int(call_vol),
                    'put_volume': int(put_vol),
                    'total_premium': total_premium,
                    'call_premium': details['call_premium'],
                    'put_premium': details['put_premium'],
                    'sentiment': details.get('sentiment', 'MIXED'),
                    'bias_strength': details.get('bias_strength', 50),
                    'large_trades': large_trades.to_dict('records') if not large_trades.empty else [],
                    'block_trades': details.get('block_trades', []),
                    'flow_score': flow_result['score']
                }
        
        return options_metrics
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return {}

def get_stock_chart(ticker, period='6mo'):
    """Get price chart for a stock"""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker
        ))
        
        # Add moving averages
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], 
                                name='50-day MA', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], 
                                name='200-day MA', line=dict(color='red', width=1)))
        
        fig.update_layout(
            title=f'{ticker} Price Chart',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=400,
            template='plotly_white'
        )
        
        return fig
    except:
        return None

def calculate_rsi(ticker, period=14):
    """Calculate RSI for a stock"""
    try:
        df = yf.download(ticker, period='3mo', progress=False)
        if df.empty:
            return None
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Calculate RSI using Wilder's smoothing method
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # First value is simple average
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Apply Wilder's smoothing: (previous_avg * (period-1) + current) / period
        for i in range(period, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else None
    except:
        return None

def find_reversal_candidates():
    """Find stocks showing potential bullish or bearish reversal patterns"""
    bullish_reversals = []
    bearish_reversals = []
    
    all_tickers = []
    for theme, stocks in THEMES.items():
        for ticker in stocks.keys():
            all_tickers.append((ticker, theme, stocks[ticker]))
    
    for ticker, theme, description in all_tickers:
        try:
            # Get stock data
            df = yf.download(ticker, period='3mo', progress=False)
            if df.empty or len(df) < 50:
                continue
            
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Calculate technical indicators
            current_price = df['Close'].iloc[-1]
            week_ago = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
            month_ago = df['Close'].iloc[-21] if len(df) >= 21 else df['Close'].iloc[0]
            
            week_return = ((current_price / week_ago) - 1) * 100
            month_return = ((current_price / month_ago) - 1) * 100
            
            # RSI
            rsi = calculate_rsi(ticker)
            if rsi is None:
                continue
            
            # Volume analysis
            avg_volume = df['Volume'].iloc[-20:].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            volume_surge = (recent_volume / avg_volume) if avg_volume > 0 else 1
            
            # Moving averages
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
            
            # Bullish Reversal Criteria:
            # - RSI < 40 (oversold or weak, but showing potential)
            # - Recent positive momentum (week_return > 0)
            # - Below MA50 but starting to turn up
            # - Volume increasing
            if (rsi < 40 and week_return > 0 and 
                current_price < ma50 * 1.05 and volume_surge > 1.1):
                
                reversal_score = 50
                reversal_score += min((40 - rsi), 20)  # More oversold = higher score
                reversal_score += min(week_return * 2, 20)  # Positive momentum
                reversal_score += min((volume_surge - 1) * 10, 10)  # Volume surge
                
                bullish_reversals.append({
                    'ticker': ticker,
                    'theme': theme,
                    'description': description,
                    'current_price': current_price,
                    'rsi': rsi,
                    'week_return': week_return,
                    'month_return': month_return,
                    'volume_surge': volume_surge,
                    'reversal_score': min(reversal_score, 100),
                    'distance_from_ma50': ((current_price / ma50) - 1) * 100
                })
            
            # Bearish Reversal Criteria:
            # - RSI > 65 (overbought)
            # - Recent negative momentum (week_return < 0)
            # - Above MA50 but starting to roll over
            # - Volume increasing (distribution)
            if (rsi > 65 and week_return < 0 and 
                current_price > ma50 * 0.95 and volume_surge > 1.1):
                
                reversal_score = 50
                reversal_score += min((rsi - 65), 20)  # More overbought = higher score
                reversal_score += min(abs(week_return) * 2, 20)  # Negative momentum
                reversal_score += min((volume_surge - 1) * 10, 10)  # Volume surge
                
                bearish_reversals.append({
                    'ticker': ticker,
                    'theme': theme,
                    'description': description,
                    'current_price': current_price,
                    'rsi': rsi,
                    'week_return': week_return,
                    'month_return': month_return,
                    'volume_surge': volume_surge,
                    'reversal_score': min(reversal_score, 100),
                    'distance_from_ma50': ((current_price / ma50) - 1) * 100
                })
                
        except Exception as e:
            continue
    
    # Sort by reversal score
    bullish_reversals.sort(key=lambda x: x['reversal_score'], reverse=True)
    bearish_reversals.sort(key=lambda x: x['reversal_score'], reverse=True)
    
    return bullish_reversals, bearish_reversals

def generate_enhanced_newsletter(opportunities, top_n, options_data):
    """Generate enhanced newsletter content with specific details and options data"""
    content = f"""## 🎯 AI-Identified Stock Opportunities

*Using technical momentum, volume analysis, fundamental catalysts, and options flow to identify potential movers*
n i8l
This week's AI screening identified **{len(opportunities)} high-probability setups** across our thematic baskets. Here are the top opportunities:

"""
    
    for i, opp in enumerate(opportunities[:top_n], 1):
        ticker = opp['ticker']
        signals = opp['technical_signals']
        
        content += f"""### {i}. **{ticker}** - {opp['description']}
**Theme**: {opp['theme']} | **Score**: {opp['opportunity_score']}/100 | **Price**: ${opp['current_price']:.2f}

**Performance**: 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%

"""
        
        # Add options data if available
        if ticker in options_data:
            opt = options_data[ticker]
            cp_ratio = opt['cp_ratio']
            
            if cp_ratio > 2:
                cp_signal = "🟢 BULLISH"
            elif cp_ratio < 0.5:
                cp_signal = "🔴 BEARISH"
            else:
                cp_signal = "🟡 NEUTRAL"
            
            content += f"""**Options Flow**: C/P Ratio: {cp_ratio:.2f}x ({cp_signal}) | Sentiment: {opt['sentiment']} ({opt['bias_strength']:.0f}% conviction) | Premium: ${opt['total_premium']/1000000:.2f}M

"""
        
        # Technical Setup with RSI
        content += "**Technical Setup**:\n"
        
        # Calculate and add RSI
        rsi = calculate_rsi(ticker)
        if rsi is not None:
            if rsi > 70:
                rsi_signal = "⚠️ Overbought"
            elif rsi < 30:
                rsi_signal = "🟢 Oversold"
            elif rsi > 60:
                rsi_signal = "💪 Strong"
            elif rsi < 40:
                rsi_signal = "📉 Weak"
            else:
                rsi_signal = "🟡 Neutral"
            content += f"- 📊 **RSI(14)**: {rsi:.1f} ({rsi_signal})\n"
        
        # Specific trend information
        trend_info = signals.get('trend', '')
        if 'Strong uptrend' in trend_info:
            content += f"- ✅ **Trend**: Price trading above all major moving averages (20/50/200-day)\n"
        elif 'Moderate uptrend' in trend_info:
            content += f"- ⚠️ **Trend**: Price above 50-day MA but testing resistance\n"
        else:
            content += f"- ⚠️ **Trend**: {trend_info}\n"
        
        # Specific momentum information
        week_ret = opp['week_return']
        month_ret = opp['month_return']
        if week_ret > 5:
            content += f"- 🚀 **Momentum**: Strong upward momentum +{week_ret:.1f}% weekly\n"
        elif week_ret > 2:
            content += f"- ✅ **Momentum**: Positive momentum (+{week_ret:.1f}%)\n"
        elif week_ret < -2:
            content += f"- 📉 **Momentum**: Recent weakness ({week_ret:.1f}% weekly)\n"
        else:
            content += f"- 📊 **Momentum**: Consolidating after {month_ret:+.1f}% monthly move\n"
        
        # Specific volume information
        vol_info = signals.get('volume', '')
        if 'surge' in vol_info.lower():
            content += f"- 💥 **Volume**: Institutional volume surge detected\n"
        elif 'above average' in vol_info.lower():
            content += f"- ✅ **Volume**: Above-average volume\n"
        else:
            content += f"- 📊 **Volume**: Normal volume\n"
        
        # Volatility/breakout information
        vol_sig = signals.get('volatility', '')
        breakout_sig = signals.get('breakout', '')
        
        if 'compression' in vol_sig.lower():
            content += f"- ⚡ **Volatility**: Compression detected (breakout setup)\n"
        
        if 'Near 52-week high' in breakout_sig:
            content += f"- 🎯 **Level**: Near 52-week highs, minimal overhead resistance\n"
        elif 'Testing resistance' in breakout_sig:
            content += f"- 🎯 **Level**: {breakout_sig}\n"
        
        # Top 2 options flows if available
        if ticker in options_data and options_data[ticker].get('large_trades'):
            large_trades = options_data[ticker]['large_trades'][:2]
            if large_trades:
                content += "\n**Top Options Flows**:\n"
                for trade in large_trades:
                    trade_type = "CALL" if trade['Call/Put'] == 'C' else "PUT"
                    trade_emoji = "📞" if trade['Call/Put'] == 'C' else "📉"
                    expiry_date = pd.to_datetime(trade['Expiration']).strftime('%m/%d')
                    premium = trade['Premium']
                    volume = trade['Volume']
                    strike = trade['Strike Price']
                    
                    content += f"- {trade_emoji} {trade_type} ${strike:.0f} exp {expiry_date}: {volume:,} contracts, ${premium/1000000:.2f}M premium\n"
        
        content += "\n---\n\n"
    
    content += """
*Note: These are technical setups with quantified conviction levels, not buy recommendations. Options flow data provides additional institutional sentiment context. Always do your own research, consider your risk tolerance, and use proper position sizing.*
"""
    
    return content

def format_opportunity_card(opp, rank):
    """Format opportunity as HTML card"""
    score = opp['opportunity_score']
    if score >= 80:
        score_class = "score-high"
        conviction = "🔥 High Conviction"
    elif score >= 70:
        score_class = "score-medium"
        conviction = "💪 Strong Setup"
    else:
        score_class = "score-low"
        conviction = "👀 Speculative"
    
    card_html = f"""
    <div class="stock-card">
        <h3>#{rank} - {opp['ticker']} - {opp['description']}</h3>
        <p><strong>Theme:</strong> {opp['theme']} | <strong>Score:</strong> <span class="{score_class}">{score}/100</span></p>
        <p><strong>Price:</strong> ${opp['current_price']:.2f} | <strong>1W:</strong> {opp['week_return']:+.1f}% | <strong>1M:</strong> {opp['month_return']:+.1f}%</p>
        <p><strong>{conviction}</strong></p>
    </div>
    """
    return card_html

def generate_markdown_content(opportunities, top_n):
    """Generate markdown content for newsletter"""
    content = f"""## 🎯 AI-Identified Stock Opportunities

*Using technical momentum, volume analysis, and fundamental catalysts to identify potential movers*

This week's AI screening identified **{len(opportunities)} high-probability setups** across our thematic baskets. Here are the top opportunities:

"""
    
    for i, opp in enumerate(opportunities[:top_n], 1):
        content += f"""### {i}. **{opp['ticker']}** - {opp['description']}
**Theme**: {opp['theme']} | **Score**: {opp['opportunity_score']}/100 | **Price**: ${opp['current_price']:.2f}

**Performance**: 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%

**Technical Setup**:
"""
        # Add key signals
        if 'Strong uptrend' in opp['technical_signals'].get('trend', ''):
            content += f"- ✅ {opp['technical_signals']['trend']}\n"
        if opp['week_return'] > 3:
            content += f"- ✅ {opp['technical_signals']['momentum']}\n"
        if 'surge' in opp['technical_signals'].get('volume', '').lower():
            content += f"- ✅ {opp['technical_signals']['volume']}\n"
        if 'compression' in opp['technical_signals'].get('volatility', '').lower():
            content += f"- ✅ {opp['technical_signals']['volatility']}\n"
        
        content += f"\n**Why It Could Move**: "
        
        if opp['opportunity_score'] >= 80:
            content += f"Multiple bullish signals converging - technical momentum, "
            if 'surge' in opp['technical_signals'].get('volume', '').lower():
                content += "unusual volume, "
            if opp['catalysts']['has_catalyst']:
                content += f"plus {', '.join(opp['catalysts']['catalyst_type'][:2])} catalyst(s)"
            else:
                content += "strong chart setup"
        else:
            content += "Emerging setup with positive momentum"
        
        content += f"\n\n**Risk/Reward**: {'High conviction' if opp['opportunity_score'] >= 80 else 'Speculative'} - {'Attractive risk/reward' if opp['opportunity_score'] >= 75 else 'Monitor closely'}\n\n---\n\n"
    
    content += """
*Note: These are technical setups, not buy recommendations. Always do your own research, consider your risk tolerance, and use proper position sizing.*
"""
    
    return content

# Main app
def main():
    st.title("🎯 AI Stock Newsletter Generator")
    st.markdown("*Intelligent stock screening for your weekly newsletter*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    min_score = st.sidebar.slider(
        "Minimum Opportunity Score",
        min_value=50,
        max_value=90,
        value=60,
        step=5,
        help="Lower = more opportunities, Higher = only best setups"
    )
    
    top_n = st.sidebar.slider(
        "Number of Stocks to Show",
        min_value=3,
        max_value=40,
        value=20,
        step=1
    )
    
    show_charts = st.sidebar.checkbox("Show Stock Charts", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Scan Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown(f"**Total Themes:** {len(THEMES)}")
    st.sidebar.markdown(f"**Total Stocks:** {sum(len(stocks) for stocks in THEMES.values())}")
    
    # Run button
    if st.sidebar.button("🔍 Run Screening", type="primary"):
        st.session_state.screening_done = True
    
    # Run screening
    if 'screening_done' not in st.session_state:
        st.session_state.screening_done = False
    
    if st.session_state.screening_done or st.sidebar.button("🔄 Refresh Data"):
        with st.spinner("🔍 Scanning all themes for opportunities..."):
            opportunities = run_screening(min_score)
        
        if not opportunities:
            st.warning(f"⚠️ No opportunities found with score >= {min_score}. Try lowering the minimum score.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Opportunities", len(opportunities))
        
        with col2:
            high_conviction = len([o for o in opportunities if o['opportunity_score'] >= 80])
            st.metric("High Conviction (80+)", high_conviction)
        
        with col3:
            avg_score = sum(o['opportunity_score'] for o in opportunities) / len(opportunities)
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col4:
            best_theme = max(set(o['theme'] for o in opportunities),
                           key=lambda t: len([o for o in opportunities if o['theme'] == t]))
            st.metric("Top Theme", best_theme)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 Detailed List", "📈 Charts", "📧 Newsletter", "🔄 Reversal Candidates"])
        
        # Tab 1: Overview with visualizations
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                st.subheader("Score Distribution")
                score_bins = pd.cut([o['opportunity_score'] for o in opportunities],
                                  bins=[0, 70, 80, 90, 100],
                                  labels=['60-70 (Speculative)', '70-80 (Strong)', '80-90 (High)', '90-100 (Exceptional)'])
                score_df = pd.DataFrame({'Score Range': score_bins})
                fig1 = px.histogram(score_df, x='Score Range', 
                                   color='Score Range',
                                   color_discrete_map={
                                       '60-70 (Speculative)': '#ff9900',
                                       '70-80 (Strong)': '#ffff00',
                                       '80-90 (High)': '#00ff00',
                                       '90-100 (Exceptional)': '#00cc00'
                                   })
                fig1.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig1, width='stretch', key="score_distribution_chart")
            
            with col2:
                # Theme breakdown
                st.subheader("Opportunities by Theme")
                theme_counts = {}
                for opp in opportunities:
                    theme = opp['theme']
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                
                theme_df = pd.DataFrame(list(theme_counts.items()), 
                                       columns=['Theme', 'Count']).sort_values('Count', ascending=True)
                fig2 = px.bar(theme_df, x='Count', y='Theme', orientation='h',
                             color='Count', color_continuous_scale='viridis')
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, width='stretch', key="theme_breakdown_chart")
            
            # Top opportunities cards
            st.subheader(f"🔥 Top {min(20, len(opportunities))} Opportunities")
            for i, opp in enumerate(opportunities[:20], 1):
                st.markdown(format_opportunity_card(opp, i), unsafe_allow_html=True)
        
        # Tab 2: Detailed list
        with tab2:
            st.subheader(f"All Opportunities (Top {min(top_n, len(opportunities))})")
            
            # Fetch options data for all opportunity symbols
            if OPTIONS_AVAILABLE:
                with st.spinner("📊 Fetching options flow data..."):
                    symbols = [opp['ticker'] for opp in opportunities[:top_n]]
                    options_data = get_options_data_for_symbols(symbols)
            else:
                options_data = {}
            
            for i, opp in enumerate(opportunities[:top_n], 1):
                ticker = opp['ticker']
                has_options = ticker in options_data
                
                with st.expander(f"#{i} - {ticker} ({opp['theme']}) - Score: {opp['opportunity_score']}/100 {'📊' if has_options else ''}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{opp['description']}**")
                        st.markdown(f"**Price:** ${opp['current_price']:.2f}")
                        st.markdown(f"**Performance:** 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%")
                        
                        st.markdown("**Technical Setup:**")
                        for key, value in opp['technical_signals'].items():
                            if key not in ['score', 'current_price', 'week_return', 'month_return']:
                                st.markdown(f"- {key.title()}: {value}")
                        
                        if opp['catalysts']['catalyst_type']:
                            st.markdown("**Catalysts:**")
                            for cat_type, desc in zip(opp['catalysts']['catalyst_type'], 
                                                     opp['catalysts']['description']):
                                st.markdown(f"- {cat_type}: {desc}")
                        
                        # Options flow data
                        if has_options:
                            st.markdown("---")
                            st.markdown("### 📊 Options Flow Analysis")
                            
                            opt_data = options_data[ticker]
                            
                            # Call/Put Ratio
                            cp_ratio = opt_data['cp_ratio']
                            if cp_ratio > 2:
                                cp_emoji = "🟢"
                                cp_signal = "BULLISH"
                            elif cp_ratio < 0.5:
                                cp_emoji = "🔴"
                                cp_signal = "BEARISH"
                            else:
                                cp_emoji = "🟡"
                                cp_signal = "NEUTRAL"
                            
                            st.markdown(f"**Call/Put Ratio:** {cp_emoji} {cp_ratio:.2f}x ({cp_signal})")
                            st.markdown(f"**Flow Sentiment:** {opt_data['sentiment']} ({opt_data['bias_strength']:.0f}% conviction)")
                            st.markdown(f"**Total Premium:** ${opt_data['total_premium']/1000000:.2f}M")
                            
                            # Volume breakdown
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("📞 Call Volume", f"{opt_data['call_volume']:,}")
                                st.metric("💵 Call Premium", f"${opt_data['call_premium']/1000000:.2f}M")
                            with col_b:
                                st.metric("📉 Put Volume", f"{opt_data['put_volume']:,}")
                                st.metric("💵 Put Premium", f"${opt_data['put_premium']/1000000:.2f}M")
                            
                            # Large trades
                            if opt_data['large_trades']:
                                st.markdown("**🔥 Large Trades (Unusual Activity):**")
                                for trade in opt_data['large_trades'][:5]:
                                    trade_type = "CALL" if trade['Call/Put'] == 'C' else "PUT"
                                    trade_emoji = "📞" if trade['Call/Put'] == 'C' else "📉"
                                    expiry_date = pd.to_datetime(trade['Expiration']).strftime('%Y-%m-%d')
                                    premium = trade['Premium']
                                    volume = trade['Volume']
                                    strike = trade['Strike Price']
                                    
                                    st.markdown(
                                        f"- {trade_emoji} **{trade_type}** ${strike:.0f} exp {expiry_date}: "
                                        f"{volume:,} contracts @ ${premium/1000000:.2f}M"
                                    )
                    
                    with col2:
                        # Score gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=opp['opportunity_score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 70], 'color': "lightgray"},
                                    {'range': [70, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_gauge, width='stretch', key=f"gauge_{ticker}_{i}")
                        
                        # Options flow gauge if available
                        if has_options:
                            opt_data = options_data[ticker]
                            cp_ratio = min(opt_data['cp_ratio'], 5)  # Cap at 5 for display
                            
                            fig_cp = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=cp_ratio,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "C/P Ratio"},
                                delta={'reference': 1, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                gauge={
                                    'axis': {'range': [0, 5]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.5], 'color': "lightcoral"},
                                        {'range': [0.5, 2], 'color': "lightyellow"},
                                        {'range': [2, 5], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 2},
                                        'thickness': 0.75,
                                        'value': 1
                                    }
                                }
                            ))
                            fig_cp.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_cp, width='stretch', key=f"cp_ratio_{ticker}_{i}")
        
        # Tab 3: Charts
        with tab3:
            if show_charts:
                st.subheader("📈 Stock Price Charts (Top 5)")
                
                for i, opp in enumerate(opportunities[:5], 1):
                    st.markdown(f"### {i}. {opp['ticker']} - {opp['description']}")
                    
                    fig = get_stock_chart(opp['ticker'])
                    if fig:
                        st.plotly_chart(fig, width='stretch', key=f"chart_{opp['ticker']}_{i}")
                    else:
                        st.info(f"Chart not available for {opp['ticker']}")
                    
                    st.markdown("---")
            else:
                st.info("Enable 'Show Stock Charts' in the sidebar to view charts")
        
        # Tab 4: Newsletter
        with tab4:
            st.subheader("📧 Newsletter-Ready Content")
            st.markdown("*Copy the content below directly into your Substack or blog*")
            
            # Fetch options data for newsletter
            if OPTIONS_AVAILABLE:
                with st.spinner("📊 Fetching options data for newsletter..."):
                    symbols = [opp['ticker'] for opp in opportunities[:top_n]]
                    options_data = get_options_data_for_symbols(symbols)
            else:
                options_data = {}
            
            # Generate enhanced newsletter content with options data
            newsletter_content = generate_enhanced_newsletter(opportunities, top_n, options_data)
            
            # Display in a nice container
            st.markdown(newsletter_content)
            
            # Copy button
            st.download_button(
                label="📥 Download Newsletter Content",
                data=newsletter_content,
                file_name=f"ai_newsletter_{datetime.now().strftime('%Y-%m-%d')}.md",
                mime="text/markdown"
            )
            
            # Also show raw markdown for copy
            with st.expander("📋 View Raw Markdown (for copying)"):
                st.code(newsletter_content, language="markdown")
        
        # Tab 5: Reversal Candidates
        with tab5:
            st.subheader("🔄 Reversal Candidates")
            st.markdown("*Stocks showing potential to reverse trend based on RSI, momentum, and volume*")
            
            with st.spinner("🔍 Scanning for reversal patterns..."):
                bullish_reversals, bearish_reversals = find_reversal_candidates()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Bullish Reversal Candidates")
                st.markdown("*Oversold stocks showing early signs of turning positive*")
                
                if bullish_reversals:
                    st.metric("Total Bullish Candidates", len(bullish_reversals))
                    
                    for i, rev in enumerate(bullish_reversals[:15], 1):
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 2px solid #00cc00; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f0fff0;">
                                <h4 style="color: #006600; margin-top: 0;">{i}. {rev['ticker']} - {rev['description'][:50]}...</h4>
                                <p><strong>Theme:</strong> {rev['theme']}</p>
                                <p><strong>Reversal Score:</strong> <span style="background-color: #00cc00; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold;">{rev['reversal_score']:.0f}</span></p>
                                <p><strong>Current Price:</strong> ${rev['current_price']:.2f}</p>
                                <p><strong>RSI:</strong> {rev['rsi']:.1f} (Oversold territory)</p>
                                <p><strong>Weekly Return:</strong> <span style="color: {'green' if rev['week_return'] > 0 else 'red'};">{rev['week_return']:+.2f}%</span></p>
                                <p><strong>Monthly Return:</strong> <span style="color: {'green' if rev['month_return'] > 0 else 'red'};">{rev['month_return']:+.2f}%</span></p>
                                <p><strong>Volume Surge:</strong> {rev['volume_surge']:.2f}x</p>
                                <p><strong>Distance from 50-MA:</strong> {rev['distance_from_ma50']:+.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No bullish reversal candidates found at this time.")
            
            with col2:
                st.markdown("### 🔴 Bearish Reversal Candidates")
                st.markdown("*Overbought stocks showing early signs of turning negative*")
                
                if bearish_reversals:
                    st.metric("Total Bearish Candidates", len(bearish_reversals))
                    
                    for i, rev in enumerate(bearish_reversals[:15], 1):
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 2px solid #cc0000; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #fff0f0;">
                                <h4 style="color: #660000; margin-top: 0;">{i}. {rev['ticker']} - {rev['description'][:50]}...</h4>
                                <p><strong>Theme:</strong> {rev['theme']}</p>
                                <p><strong>Reversal Score:</strong> <span style="background-color: #cc0000; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold;">{rev['reversal_score']:.0f}</span></p>
                                <p><strong>Current Price:</strong> ${rev['current_price']:.2f}</p>
                                <p><strong>RSI:</strong> {rev['rsi']:.1f} (Overbought territory)</p>
                                <p><strong>Weekly Return:</strong> <span style="color: {'green' if rev['week_return'] > 0 else 'red'};">{rev['week_return']:+.2f}%</span></p>
                                <p><strong>Monthly Return:</strong> <span style="color: {'green' if rev['month_return'] > 0 else 'red'};">{rev['month_return']:+.2f}%</span></p>
                                <p><strong>Volume Surge:</strong> {rev['volume_surge']:.2f}x</p>
                                <p><strong>Distance from 50-MA:</strong> {rev['distance_from_ma50']:+.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No bearish reversal candidates found at this time.")
        
        # Summary statistics at bottom
        st.markdown("---")
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**By Conviction Level:**")
            high_conviction = len([o for o in opportunities if o['opportunity_score'] >= 80])
            strong_setups = len([o for o in opportunities if 70 <= o['opportunity_score'] < 80])
            speculative = len([o for o in opportunities if 60 <= o['opportunity_score'] < 70])
            
            st.markdown(f"- 🔥 High Conviction (80-100): **{high_conviction}**")
            st.markdown(f"- 💪 Strong Setups (70-79): **{strong_setups}**")
            st.markdown(f"- 👀 Speculative (60-69): **{speculative}**")
        
        with col2:
            st.markdown("**Top 5 Themes:**")
            theme_counts = {}
            for opp in opportunities:
                theme = opp['theme']
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.markdown(f"- {theme}: **{count}**")
        
        with col3:
            st.markdown("**Performance Range:**")
            weekly_returns = [o['week_return'] for o in opportunities]
            monthly_returns = [o['month_return'] for o in opportunities]
            
            st.markdown(f"- Weekly: **{min(weekly_returns):.1f}%** to **{max(weekly_returns):.1f}%**")
            st.markdown(f"- Monthly: **{min(monthly_returns):.1f}%** to **{max(monthly_returns):.1f}%**")
            st.markdown(f"- Avg Weekly: **{sum(weekly_returns)/len(weekly_returns):.1f}%**")
    
    else:
        st.info("👈 Click **'Run Screening'** in the sidebar to start")
        
        # Show universe stats
        st.markdown("### 📚 Stock Universe")
        st.markdown(f"**Total Themes:** {len(THEMES)}")
        st.markdown(f"**Total Stocks:** {sum(len(stocks) for stocks in THEMES.values())}")
        
        # Show theme breakdown
        theme_df = pd.DataFrame([
            {'Theme': theme, 'Stock Count': len(stocks)}
            for theme, stocks in THEMES.items()
        ]).sort_values('Stock Count', ascending=False)
        
        st.dataframe(theme_df, width='stretch')


if __name__ == "__main__":
    main()
