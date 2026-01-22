#!/usr/bin/env python3
"""
Streamlit AI Stock Newsletter Generator
Interactive web app to generate and visualize stock opportunities with options flow data
OPTIMIZED VERSION - Parallel fetching, caching, progress bars
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.ai_stock_screener import StockScreener
from src.theme_tracker import THEMES
import yfinance as yf

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
.big-font { font-size:20px !important; font-weight: bold; }
.score-high { background-color: #00ff00; color: black; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
.score-medium { background-color: #ffff00; color: black; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
.score-low { background-color: #ff9900; color: black; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
.stock-card { border: 2px solid #1f77b4; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)


# ==================== CACHED DATA FETCHING ====================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data_batch(tickers: tuple, period: str = "6mo") -> dict:
    """Fetch stock data for multiple tickers in parallel"""
    results = {}
    
    def fetch_single(ticker):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return ticker, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return ticker, df
        except:
            return ticker, None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                results[ticker] = df
    
    return results


@st.cache_data(ttl=300, show_spinner=False)
def run_screening(min_score: int):
    """Run AI screening with caching"""
    screener = StockScreener()
    opportunities = screener.screen_themes(min_score=min_score)
    opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return opportunities


@st.cache_data(ttl=600, show_spinner=False)
def get_options_data_for_symbols(symbols: tuple):
    """Fetch options data for given symbols - CACHED"""
    if not OPTIONS_AVAILABLE:
        return {}
    
    try:
        df = fetch_all_options_data()
        if df.empty:
            return {}
        
        df = df[df['Symbol'].isin(symbols)]
        if df.empty:
            return {}
        
        options_metrics = {}
        for symbol in symbols:
            symbol_df = df[df['Symbol'] == symbol]
            if symbol_df.empty:
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.fast_info.last_price
            except:
                continue
            
            if current_price is None:
                continue
            
            symbol_df = symbol_df.copy()
            symbol_df['Days_to_Expiry'] = (symbol_df['Expiration'] - datetime.now()).dt.days
            symbol_df['Premium'] = symbol_df['Volume'] * symbol_df['Last Price'] * 100
            
            flow_result = calculate_flow_score(symbol_df, current_price)
            
            if flow_result['score'] > 0:
                details = flow_result['details']
                call_vol = symbol_df[symbol_df['Call/Put'] == 'C']['Volume'].sum()
                put_vol = symbol_df[symbol_df['Call/Put'] == 'P']['Volume'].sum()
                cp_ratio = call_vol / put_vol if put_vol > 0 else float('inf')
                
                strike_col = 'Strike Price' if 'Strike Price' in symbol_df.columns else 'Strike'
                otm_df = symbol_df[
                    ((symbol_df['Call/Put'] == 'C') & (symbol_df[strike_col] > current_price)) |
                    ((symbol_df['Call/Put'] == 'P') & (symbol_df[strike_col] < current_price))
                ]
                
                large_trades = pd.DataFrame()
                if not otm_df.empty:
                    large_threshold = max(100000, otm_df['Premium'].quantile(0.8))
                    large_trades = otm_df[otm_df['Premium'] > large_threshold].nlargest(5, 'Premium')
                
                options_metrics[symbol] = {
                    'cp_ratio': cp_ratio,
                    'call_volume': int(call_vol),
                    'put_volume': int(put_vol),
                    'total_premium': details['total_premium'],
                    'call_premium': details['call_premium'],
                    'put_premium': details['put_premium'],
                    'sentiment': details.get('sentiment', 'MIXED'),
                    'bias_strength': details.get('bias_strength', 50),
                    'large_trades': large_trades.to_dict('records') if not large_trades.empty else [],
                    'flow_score': flow_result['score']
                }
        
        return options_metrics
    except Exception as e:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def find_reversal_candidates():
    """Find reversal candidates - CACHED with parallel processing"""
    bullish_reversals = []
    bearish_reversals = []
    
    all_tickers = []
    for theme, stocks in THEMES.items():
        for ticker in stocks.keys():
            all_tickers.append((ticker, theme, stocks[ticker]))
    
    # Fetch all data in parallel
    ticker_list = tuple(t[0] for t in all_tickers)
    stock_data = fetch_stock_data_batch(ticker_list, period='3mo')
    
    for ticker, theme, description in all_tickers:
        try:
            df = stock_data.get(ticker)
            if df is None or len(df) < 50:
                continue
            
            current_price = df['Close'].iloc[-1]
            week_ago = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
            month_ago = df['Close'].iloc[-21] if len(df) >= 21 else df['Close'].iloc[0]
            
            week_return = ((current_price / week_ago) - 1) * 100
            month_return = ((current_price / month_ago) - 1) * 100
            
            # RSI calculation inline
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            if pd.isna(rsi):
                continue
            
            avg_volume = df['Volume'].iloc[-20:].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            volume_surge = (recent_volume / avg_volume) if avg_volume > 0 else 1
            
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
            
            # Bullish Reversal
            if rsi < 40 and week_return > 0 and current_price < ma50 * 1.05 and volume_surge > 1.1:
                reversal_score = 50 + min((40 - rsi), 20) + min(week_return * 2, 20) + min((volume_surge - 1) * 10, 10)
                bullish_reversals.append({
                    'ticker': ticker, 'theme': theme, 'description': description,
                    'current_price': current_price, 'rsi': rsi,
                    'week_return': week_return, 'month_return': month_return,
                    'volume_surge': volume_surge, 'reversal_score': min(reversal_score, 100),
                    'distance_from_ma50': ((current_price / ma50) - 1) * 100
                })
            
            # Bearish Reversal
            if rsi > 65 and week_return < 0 and current_price > ma50 * 0.95 and volume_surge > 1.1:
                reversal_score = 50 + min((rsi - 65), 20) + min(abs(week_return) * 2, 20) + min((volume_surge - 1) * 10, 10)
                bearish_reversals.append({
                    'ticker': ticker, 'theme': theme, 'description': description,
                    'current_price': current_price, 'rsi': rsi,
                    'week_return': week_return, 'month_return': month_return,
                    'volume_surge': volume_surge, 'reversal_score': min(reversal_score, 100),
                    'distance_from_ma50': ((current_price / ma50) - 1) * 100
                })
                
        except Exception:
            continue
    
    bullish_reversals.sort(key=lambda x: x['reversal_score'], reverse=True)
    bearish_reversals.sort(key=lambda x: x['reversal_score'], reverse=True)
    
    return bullish_reversals, bearish_reversals


def get_stock_chart(ticker, period='6mo'):
    """Get price chart for a stock"""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name=ticker
        ))
        
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200-day MA', line=dict(color='red', width=1)))
        
        fig.update_layout(title=f'{ticker} Price Chart', yaxis_title='Price ($)', height=400, template='plotly_white')
        return fig
    except:
        return None


def calculate_rsi_from_data(df, period=14):
    """Calculate RSI from existing dataframe"""
    if df is None or len(df) < period + 1:
        return None
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def format_opportunity_card(opp, rank):
    """Format opportunity as HTML card"""
    score = opp['opportunity_score']
    if score >= 80:
        score_class, conviction = "score-high", "🔥 High Conviction"
    elif score >= 70:
        score_class, conviction = "score-medium", "💪 Strong Setup"
    else:
        score_class, conviction = "score-low", "👀 Speculative"
    
    return f"""
    <div class="stock-card">
        <h3>#{rank} - {opp['ticker']} - {opp['description']}</h3>
        <p><strong>Theme:</strong> {opp['theme']} | <strong>Score:</strong> <span class="{score_class}">{score}/100</span></p>
        <p><strong>Price:</strong> ${opp['current_price']:.2f} | <strong>1W:</strong> {opp['week_return']:+.1f}% | <strong>1M:</strong> {opp['month_return']:+.1f}%</p>
        <p><strong>{conviction}</strong></p>
    </div>
    """


def generate_enhanced_newsletter(opportunities, top_n, options_data, stock_data_cache):
    """Generate enhanced newsletter content"""
    content = f"""## 🎯 AI-Identified Stock Opportunities

*Using technical momentum, volume analysis, fundamental catalysts, and options flow*

This week's AI screening identified **{len(opportunities)} high-probability setups**. Top opportunities:

"""
    
    for i, opp in enumerate(opportunities[:top_n], 1):
        ticker = opp['ticker']
        signals = opp['technical_signals']
        
        content += f"""### {i}. **{ticker}** - {opp['description']}
**Theme**: {opp['theme']} | **Score**: {opp['opportunity_score']}/100 | **Price**: ${opp['current_price']:.2f}

**Performance**: 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%

"""
        
        if ticker in options_data:
            opt = options_data[ticker]
            cp_ratio = opt['cp_ratio']
            cp_signal = "🟢 BULLISH" if cp_ratio > 2 else ("🔴 BEARISH" if cp_ratio < 0.5 else "🟡 NEUTRAL")
            content += f"""**Options Flow**: C/P Ratio: {cp_ratio:.2f}x ({cp_signal}) | Sentiment: {opt['sentiment']} ({opt['bias_strength']:.0f}% conviction)

"""
        
        # RSI from cached data
        df = stock_data_cache.get(ticker)
        rsi = calculate_rsi_from_data(df) if df is not None else None
        if rsi is not None:
            rsi_signal = "⚠️ Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "🟡 Neutral")
            content += f"- 📊 **RSI(14)**: {rsi:.1f} ({rsi_signal})\n"
        
        content += f"- **Trend**: {signals.get('trend', 'N/A')}\n"
        content += f"- **Volume**: {signals.get('volume', 'N/A')}\n\n---\n\n"
    
    content += "\n*Note: Technical setups, not recommendations. Always do your own research.*\n"
    return content


# ==================== MAIN APP ====================
def main():
    st.title("🎯 AI Stock Newsletter Generator")
    st.markdown("*Intelligent stock screening for your weekly newsletter*")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    min_score = st.sidebar.slider("Minimum Opportunity Score", 50, 90, 60, 5)
    top_n = st.sidebar.slider("Number of Stocks to Show", 3, 40, 20)
    show_charts = st.sidebar.checkbox("Show Stock Charts", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Scan Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown(f"**Total Themes:** {len(THEMES)}")
    st.sidebar.markdown(f"**Total Stocks:** {sum(len(stocks) for stocks in THEMES.values())}")
    
    run_scan = st.sidebar.button("🔍 Run Screening", type="primary")
    
    if 'screening_done' not in st.session_state:
        st.session_state.screening_done = False
    
    if run_scan:
        st.session_state.screening_done = True
    
    if st.session_state.screening_done:
        # Run screening with progress
        with st.spinner("🔍 Scanning all themes for opportunities..."):
            opportunities = run_screening(min_score)
        
        if not opportunities:
            st.warning(f"⚠️ No opportunities found with score >= {min_score}")
            return
        
        # Pre-fetch all data once
        symbols = tuple(opp['ticker'] for opp in opportunities[:top_n])
        
        with st.spinner("📊 Loading stock data..."):
            stock_data_cache = fetch_stock_data_batch(symbols)
        
        with st.spinner("�� Loading options flow data..."):
            options_data = get_options_data_for_symbols(symbols) if OPTIONS_AVAILABLE else {}
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Opportunities", len(opportunities))
        col2.metric("High Conviction (80+)", len([o for o in opportunities if o['opportunity_score'] >= 80]))
        col3.metric("Average Score", f"{sum(o['opportunity_score'] for o in opportunities) / len(opportunities):.1f}")
        col4.metric("Top Theme", max(set(o['theme'] for o in opportunities), key=lambda t: len([o for o in opportunities if o['theme'] == t])))
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 Detailed List", "📈 Charts", "📧 Newsletter", "🔄 Reversals"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Score Distribution")
                score_bins = pd.cut([o['opportunity_score'] for o in opportunities],
                                  bins=[0, 70, 80, 90, 100],
                                  labels=['60-70', '70-80', '80-90', '90-100'])
                fig1 = px.histogram(pd.DataFrame({'Score': score_bins}), x='Score')
                fig1.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig1, use_container_width=True, key="score_histogram")
            
            with col2:
                st.subheader("By Theme")
                theme_counts = {}
                for opp in opportunities:
                    theme_counts[opp['theme']] = theme_counts.get(opp['theme'], 0) + 1
                theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Count']).sort_values('Count', ascending=True)
                fig2 = px.bar(theme_df, x='Count', y='Theme', orientation='h')
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True, key="theme_bar_chart")
            
            st.subheader(f"🔥 Top {min(20, len(opportunities))} Opportunities")
            for i, opp in enumerate(opportunities[:20], 1):
                st.markdown(format_opportunity_card(opp, i), unsafe_allow_html=True)
        
        with tab2:
            st.subheader(f"All Opportunities (Top {min(top_n, len(opportunities))})")
            
            for i, opp in enumerate(opportunities[:top_n], 1):
                ticker = opp['ticker']
                has_options = ticker in options_data
                
                with st.expander(f"#{i} - {ticker} ({opp['theme']}) - Score: {opp['opportunity_score']}/100 {'📊' if has_options else ''}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{opp['description']}**")
                        st.markdown(f"**Price:** ${opp['current_price']:.2f} | **1W:** {opp['week_return']:+.1f}% | **1M:** {opp['month_return']:+.1f}%")
                        
                        st.markdown("**Technical Setup:**")
                        for key, value in opp['technical_signals'].items():
                            if key not in ['score', 'current_price', 'week_return', 'month_return']:
                                st.markdown(f"- {key.title()}: {value}")
                        
                        if has_options:
                            st.markdown("---")
                            st.markdown("### �� Options Flow")
                            opt = options_data[ticker]
                            cp_ratio = opt['cp_ratio']
                            cp_emoji = "🟢" if cp_ratio > 2 else ("🔴" if cp_ratio < 0.5 else "🟡")
                            st.markdown(f"**C/P Ratio:** {cp_emoji} {cp_ratio:.2f}x | **Sentiment:** {opt['sentiment']}")
                            st.markdown(f"**Total Premium:** ${opt['total_premium']/1e6:.2f}M")
                            
                            if opt['large_trades']:
                                st.markdown("**🔥 Large Trades:**")
                                for trade in opt['large_trades'][:3]:
                                    t_type = "CALL" if trade['Call/Put'] == 'C' else "PUT"
                                    st.markdown(f"- {t_type} ${trade['Strike Price']:.0f}: ${trade['Premium']/1e6:.2f}M")
                    
                    with col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=opp['opportunity_score'],
                            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                                   'steps': [{'range': [0, 70], 'color': "lightgray"},
                                            {'range': [70, 80], 'color': "yellow"},
                                            {'range': [80, 100], 'color': "lightgreen"}]}
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{ticker}_{i}")
        
        with tab3:
            if show_charts:
                st.subheader("📈 Stock Charts (Top 5)")
                for i, opp in enumerate(opportunities[:5], 1):
                    st.markdown(f"### {i}. {opp['ticker']}")
                    fig = get_stock_chart(opp['ticker'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{opp['ticker']}")
                    st.markdown("---")
            else:
                st.info("Enable 'Show Stock Charts' in sidebar")
        
        with tab4:
            st.subheader("📧 Newsletter-Ready Content")
            newsletter_content = generate_enhanced_newsletter(opportunities, top_n, options_data, stock_data_cache)
            st.markdown(newsletter_content)
            
            st.download_button("📥 Download Newsletter", newsletter_content,
                             f"ai_newsletter_{datetime.now().strftime('%Y-%m-%d')}.md", "text/markdown")
            
            with st.expander("📋 Raw Markdown"):
                st.code(newsletter_content, language="markdown")
        
        with tab5:
            st.subheader("🔄 Reversal Candidates")
            
            with st.spinner("🔍 Scanning for reversal patterns..."):
                bullish_reversals, bearish_reversals = find_reversal_candidates()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Bullish Reversals")
                st.metric("Candidates", len(bullish_reversals))
                for rev in bullish_reversals[:10]:
                    st.markdown(f"""
                    **{rev['ticker']}** - Score: {rev['reversal_score']:.0f}  
                    RSI: {rev['rsi']:.1f} | Week: {rev['week_return']:+.1f}% | Vol: {rev['volume_surge']:.1f}x
                    """)
                    st.divider()
            
            with col2:
                st.markdown("### 🔴 Bearish Reversals")
                st.metric("Candidates", len(bearish_reversals))
                for rev in bearish_reversals[:10]:
                    st.markdown(f"""
                    **{rev['ticker']}** - Score: {rev['reversal_score']:.0f}  
                    RSI: {rev['rsi']:.1f} | Week: {rev['week_return']:+.1f}% | Vol: {rev['volume_surge']:.1f}x
                    """)
                    st.divider()
        
        # Summary
        st.markdown("---")
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**By Conviction:**")
            st.markdown(f"- 🔥 High (80+): **{len([o for o in opportunities if o['opportunity_score'] >= 80])}**")
            st.markdown(f"- 💪 Strong (70-79): **{len([o for o in opportunities if 70 <= o['opportunity_score'] < 80])}**")
            st.markdown(f"- 👀 Speculative: **{len([o for o in opportunities if o['opportunity_score'] < 70])}**")
        
        with col2:
            st.markdown("**Top 5 Themes:**")
            theme_counts = {}
            for opp in opportunities:
                theme_counts[opp['theme']] = theme_counts.get(opp['theme'], 0) + 1
            for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.markdown(f"- {theme}: **{count}**")
        
        with col3:
            weekly = [o['week_return'] for o in opportunities]
            st.markdown("**Performance:**")
            st.markdown(f"- Weekly Range: **{min(weekly):.1f}%** to **{max(weekly):.1f}%**")
            st.markdown(f"- Avg Weekly: **{sum(weekly)/len(weekly):.1f}%**")
    
    else:
        st.info("👈 Click **'Run Screening'** to start")
        
        st.markdown("### 📚 Stock Universe")
        theme_df = pd.DataFrame([
            {'Theme': theme, 'Count': len(stocks)}
            for theme, stocks in THEMES.items()
        ]).sort_values('Count', ascending=False)
        st.dataframe(theme_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
