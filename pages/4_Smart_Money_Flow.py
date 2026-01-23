#!/usr/bin/env python3
"""
Smart Money Flow - Consolidated Options Flow Analysis
Combines: Whale Flows, Flow Scanner, Options Flow Dashboard, CBOE Flow Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client

# Try to import yfinance for CBOE scanner
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Smart Money Flow",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Watchlists
TOP_STOCKS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 
              'AMD', 'INTC', 'PLTR', 'COIN', 'MSTR', 'HOOD', 'SOFI', 'MU', 'PYPL',
              'XLF', 'XLE', 'XLK', 'XLV', 'IWM', 'TLT', 'GLD', 'SLV', 'USO', 'DIA']

# CSS
st.markdown("""
<style>
    .whale-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
    }
    .bullish-card { border-left: 4px solid #10b981; }
    .bearish-card { border-left: 4px solid #ef4444; }
    .whale-premium { font-size: 1.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==================== CBOE DATA SOURCE HELPERS ====================
@st.cache_data(ttl=600)
def fetch_cboe_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch options data from CBOE URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        required_columns = ['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Last Price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return None
        df = df.dropna(subset=['Symbol', 'Expiration', 'Strike Price', 'Call/Put'])
        df = df[df['Volume'] >= 50].copy()
        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
        df = df.dropna(subset=['Expiration'])
        df = df[df['Expiration'].dt.date >= datetime.now().date()]
        return df
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_all_cboe_options_data() -> pd.DataFrame:
    """Fetch options data from all CBOE exchanges"""
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    data_frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_cboe_data_from_url, url) for url in urls]
        for future in futures:
            df = future.result()
            if df is not None and not df.empty:
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


@st.cache_data(ttl=300)
def get_stock_price_yf(symbol: str) -> Optional[float]:
    """Get stock price using yfinance"""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        if symbol.startswith('$') or len(symbol) > 5:
            return None
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.fast_info
            price = info.get('last_price')
            if price and price > 0:
                return float(price)
        except:
            pass
        try:
            info = ticker.info
            price = (info.get('currentPrice') or 
                    info.get('regularMarketPrice') or 
                    info.get('previousClose'))
            if price and price > 0:
                return float(price)
        except:
            pass
        return None
    except Exception:
        return None


# ==================== DATA FETCHING ====================
@st.cache_data(ttl=120)
def fetch_symbol_flow(symbol):
    """Fetch options flow data for a symbol"""
    try:
        client = get_client()
        if not client:
            return None, 0
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=30)
        if not chain or chain.get('status') != 'SUCCESS':
            return None, 0
        
        underlying_price = chain.get('underlyingPrice', 0)
        return chain, underlying_price
    except Exception as e:
        return None, 0


def analyze_flow(chain, underlying_price, symbol):
    """Analyze options flow for whale activity and sentiment"""
    if not chain:
        return None
    
    calls_flow, puts_flow = [], []
    total_call_vol, total_put_vol = 0, 0
    total_call_premium, total_put_premium = 0, 0
    
    for exp_date, strikes in chain.get('callExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                vol = c.get('totalVolume', 0)
                oi = c.get('openInterest', 0)
                mark = c.get('mark', 0)
                premium = vol * mark * 100
                
                total_call_vol += vol
                total_call_premium += premium
                
                # Whale criteria: premium > $100K or vol/oi > 3
                vol_oi_ratio = vol / oi if oi > 0 else 0
                if premium > 100000 or (vol_oi_ratio > 3 and premium > 25000):
                    calls_flow.append({
                        'type': 'CALL', 'symbol': symbol,
                        'strike': float(strike_str), 'expiry': exp_key,
                        'volume': vol, 'oi': oi, 'premium': premium,
                        'vol_oi': vol_oi_ratio, 'mark': mark,
                        'delta': c.get('delta', 0), 'iv': c.get('volatility', 0)
                    })
    
    for exp_date, strikes in chain.get('putExpDateMap', {}).items():
        exp_key = exp_date.split(':')[0]
        for strike_str, contracts in strikes.items():
            if contracts:
                c = contracts[0]
                vol = c.get('totalVolume', 0)
                oi = c.get('openInterest', 0)
                mark = c.get('mark', 0)
                premium = vol * mark * 100
                
                total_put_vol += vol
                total_put_premium += premium
                
                vol_oi_ratio = vol / oi if oi > 0 else 0
                if premium > 100000 or (vol_oi_ratio > 3 and premium > 25000):
                    puts_flow.append({
                        'type': 'PUT', 'symbol': symbol,
                        'strike': float(strike_str), 'expiry': exp_key,
                        'volume': vol, 'oi': oi, 'premium': premium,
                        'vol_oi': vol_oi_ratio, 'mark': mark,
                        'delta': c.get('delta', 0), 'iv': c.get('volatility', 0)
                    })
    
    # Calculate metrics
    total_vol = total_call_vol + total_put_vol
    pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    net_premium = total_call_premium - total_put_premium
    
    return {
        'symbol': symbol,
        'price': underlying_price,
        'call_vol': total_call_vol,
        'put_vol': total_put_vol,
        'pc_ratio': pc_ratio,
        'call_premium': total_call_premium,
        'put_premium': total_put_premium,
        'net_premium': net_premium,
        'whale_calls': sorted(calls_flow, key=lambda x: x['premium'], reverse=True)[:10],
        'whale_puts': sorted(puts_flow, key=lambda x: x['premium'], reverse=True)[:10]
    }


# ==================== WHALE SCANNER TAB ====================
def render_whale_scanner_tab(symbols):
    """Scan multiple symbols for whale activity"""
    st.subheader("🐋 Whale Flow Scanner")
    st.caption(f"Scanning {len(symbols)} symbols for unusual options activity")
    
    # On-demand scan button
    if 'whale_scan_results' not in st.session_state:
        st.session_state.whale_scan_results = None
    
    col1, col2 = st.columns([1, 4])
    with col1:
        scan_button = st.button("🔍 Scan Now", key="whale_scan_btn", type="primary", use_container_width=True)
    with col2:
        st.caption("Click to scan for whale trades (large premium > $100K or unusual vol/OI)")
    
    if scan_button:
        progress = st.progress(0)
        all_whales = []
        summaries = []
        
        for i, symbol in enumerate(symbols):
            chain, price = fetch_symbol_flow(symbol)
            if chain:
                flow = analyze_flow(chain, price, symbol)
                if flow:
                    summaries.append(flow)
                    all_whales.extend(flow['whale_calls'])
                    all_whales.extend(flow['whale_puts'])
            progress.progress((i + 1) / len(symbols))
        
        progress.empty()
        
        # Sort all whales by premium and store in session state
        all_whales = sorted(all_whales, key=lambda x: x['premium'], reverse=True)[:30]
        st.session_state.whale_scan_results = all_whales
    
    # Display results if available
    all_whales = st.session_state.whale_scan_results
    
    if all_whales is None:
        st.info("👆 Click 'Scan Now' to detect whale trades")
        return
    
    if not all_whales:
        st.info("No whale trades detected in current scan")
        return
    
    # Display whale trades
    st.markdown(f"**Found {len(all_whales)} whale trades across {len(symbols)} symbols**")
    
    for whale in all_whales:
        is_call = whale['type'] == 'CALL'
        card_class = 'bullish-card' if is_call else 'bearish-card'
        emoji = '🟢' if is_call else '🔴'
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{emoji} {whale['symbol']}** {whale['type']}")
            st.caption(f"Strike: ${whale['strike']:.0f} | Exp: {whale['expiry']}")
        
        with col2:
            st.metric("Premium", f"${whale['premium']/1000:.0f}K")
        
        with col3:
            st.metric("Volume", f"{whale['volume']:,}")
            st.caption(f"Vol/OI: {whale['vol_oi']:.1f}x")
        
        with col4:
            st.metric("IV", f"{whale['iv']:.0f}%")
        
        st.divider()


# ==================== FLOW SUMMARY TAB ====================
def render_flow_summary_tab(symbols):
    """Show aggregated flow summary"""
    st.subheader("📊 Market Flow Summary")
    st.caption(f"Aggregate options flow data for {len(symbols)} symbols")
    
    # On-demand scan button
    if 'flow_summary_results' not in st.session_state:
        st.session_state.flow_summary_results = None
    
    col1, col2 = st.columns([1, 4])
    with col1:
        scan_button = st.button("📊 Analyze Flow", key="flow_summary_btn", type="primary", use_container_width=True)
    with col2:
        st.caption("Click to analyze call/put volume and premium flow")
    
    if scan_button:
        progress = st.progress(0)
        summaries = []
        
        for i, symbol in enumerate(symbols):
            chain, price = fetch_symbol_flow(symbol)
            if chain:
                flow = analyze_flow(chain, price, symbol)
                if flow:
                    summaries.append(flow)
            progress.progress((i + 1) / len(symbols))
        
        progress.empty()
        st.session_state.flow_summary_results = summaries
    
    # Display results if available
    summaries = st.session_state.flow_summary_results
    
    if summaries is None:
        st.info("👆 Click 'Analyze Flow' to see market flow summary")
        return
    
    if not summaries:
        st.info("No flow data available")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame([{
        'Symbol': s['symbol'],
        'Price': s['price'],
        'Call Vol': s['call_vol'],
        'Put Vol': s['put_vol'],
        'P/C Ratio': s['pc_ratio'],
        'Call Premium': s['call_premium'],
        'Put Premium': s['put_premium'],
        'Net Premium': s['net_premium'],
        'Sentiment': 'BULLISH' if s['net_premium'] > 0 else 'BEARISH'
    } for s in summaries])
    
    # Metrics
    total_call_prem = df['Call Premium'].sum()
    total_put_prem = df['Put Premium'].sum()
    net_market = total_call_prem - total_put_prem
    bullish_count = (df['Net Premium'] > 0).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Market Flow", f"${net_market/1e6:.1f}M", 
                "Bullish" if net_market > 0 else "Bearish")
    col2.metric("Total Call Premium", f"${total_call_prem/1e6:.1f}M")
    col3.metric("Total Put Premium", f"${total_put_prem/1e6:.1f}M")
    col4.metric("Bullish Symbols", f"{bullish_count}/{len(df)}")
    
    # Flow Chart
    fig = go.Figure()
    
    df_sorted = df.sort_values('Net Premium', ascending=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['Net Premium']]
    
    fig.add_trace(go.Bar(
        y=df_sorted['Symbol'],
        x=df_sorted['Net Premium'] / 1e6,
        orientation='h',
        marker_color=colors,
        text=[f"${x/1e6:.1f}M" for x in df_sorted['Net Premium']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Net Premium Flow by Symbol",
        xaxis_title="Net Premium (Millions)",
        template='plotly_dark',
        height=max(400, len(df) * 25)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown("**Detailed Flow Data**")
    df_display = df.copy()
    df_display['Call Premium'] = df_display['Call Premium'].apply(lambda x: f"${x/1000:.0f}K")
    df_display['Put Premium'] = df_display['Put Premium'].apply(lambda x: f"${x/1000:.0f}K")
    df_display['Net Premium'] = df_display['Net Premium'].apply(lambda x: f"${x/1000:+.0f}K")
    df_display['P/C Ratio'] = df_display['P/C Ratio'].apply(lambda x: f"{x:.2f}")
    df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ==================== SINGLE SYMBOL TAB ====================
def render_single_symbol_tab():
    """Deep dive into single symbol flow"""
    st.subheader("🔍 Single Symbol Analysis")
    
    symbol = st.text_input("Enter Symbol", value="SPY", key="flow_symbol").upper().strip()
    
    if not symbol:
        return
    
    chain, price = fetch_symbol_flow(symbol)
    
    if not chain:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    flow = analyze_flow(chain, price, symbol)
    
    if not flow:
        st.warning("No flow data available")
        return
    
    # Header
    st.header(f"📈 {symbol} @ ${price:.2f}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Call Volume", f"{flow['call_vol']:,}")
    col2.metric("Put Volume", f"{flow['put_vol']:,}")
    col3.metric("P/C Ratio", f"{flow['pc_ratio']:.2f}")
    col4.metric("Net Premium", f"${flow['net_premium']/1000:+.0f}K",
                "Bullish" if flow['net_premium'] > 0 else "Bearish")
    
    # Premium breakdown
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                        subplot_titles=('Premium Split', 'Volume by Type'))
    
    fig.add_trace(go.Pie(
        values=[flow['call_premium'], flow['put_premium']],
        labels=['Calls', 'Puts'],
        marker_colors=['#10b981', '#ef4444'],
        hole=0.4
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=['Calls', 'Puts'],
        y=[flow['call_vol'], flow['put_vol']],
        marker_color=['#10b981', '#ef4444']
    ), row=1, col=2)
    
    fig.update_layout(template='plotly_dark', height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Whale trades
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Top Call Flow**")
        for whale in flow['whale_calls'][:5]:
            st.markdown(f"""
            **${whale['strike']:.0f} {whale['expiry']}**  
            Premium: ${whale['premium']/1000:.0f}K | Vol: {whale['volume']:,} | IV: {whale['iv']:.0f}%
            """)
    
    with col2:
        st.markdown("**🔴 Top Put Flow**")
        for whale in flow['whale_puts'][:5]:
            st.markdown(f"""
            **${whale['strike']:.0f} {whale['expiry']}**  
            Premium: ${whale['premium']/1000:.0f}K | Vol: {whale['volume']:,} | IV: {whale['iv']:.0f}%
            """)


# ==================== CBOE FLOW SCANNER TAB ====================
def render_cboe_flow_tab():
    """CBOE-based options flow scanner - scans all market activity"""
    st.subheader("🌊 CBOE Flow Scanner")
    st.caption("Real-time options flow from CBOE exchanges (no API limits)")
    
    # Session state for results
    if 'cboe_flow_results' not in st.session_state:
        st.session_state.cboe_flow_results = None
    
    # Filters
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        min_premium = st.number_input(
            "Min Premium ($)",
            min_value=10000,
            max_value=5000000,
            value=150000,
            step=10000,
            key="cboe_min_premium"
        )
    with col2:
        min_volume = st.number_input(
            "Min Volume",
            min_value=50,
            max_value=10000,
            value=500,
            step=50,
            key="cboe_min_volume"
        )
    with col3:
        scan_button = st.button("🔍 Scan CBOE Flow", key="cboe_scan_btn", type="primary", use_container_width=True)
    
    if scan_button:
        with st.spinner("Fetching CBOE options data..."):
            df = fetch_all_cboe_options_data()
        
        if df.empty:
            st.warning("No CBOE data available. Try again later.")
            return
        
        # Process data
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        df['Last Price'] = pd.to_numeric(df.get('Last Price', df.get('LastPrice', pd.Series([0]*len(df)))), errors='coerce').fillna(0)
        df['premium'] = df['Volume'] * df['Last Price'] * 100
        df['volume'] = df['Volume']
        
        # Filter by premium and volume
        df_flows = df[(df['premium'] >= min_premium) & (df['volume'] >= min_volume)].copy()
        
        if df_flows.empty:
            st.info("No flows match the filters. Try lowering the minimum premium.")
            return
        
        # Normalize columns
        df_flows['symbol'] = df_flows['Symbol'].astype(str).str.upper()
        df_flows['type'] = df_flows['Call/Put'].apply(lambda x: 'CALL' if str(x).strip().upper().startswith('C') else 'PUT')
        df_flows['strike'] = pd.to_numeric(df_flows.get('Strike Price', pd.Series([0]*len(df_flows))), errors='coerce')
        df_flows['Expiration'] = pd.to_datetime(df_flows['Expiration'], errors='coerce')
        df_flows['expiry'] = df_flows['Expiration'].dt.strftime('%Y-%m-%d')
        df_flows['days_to_exp'] = (df_flows['Expiration'] - pd.Timestamp.now()).dt.days
        df_flows['price'] = df_flows['Last Price']
        
        # Exclude index options (SPX, VIX, etc.)
        excluded = {'SPX', 'SPXW', 'VIX', 'VIXW', 'NDX', 'RUT', 'RUTW', 'SPXQ', 'XSP'}
        df_flows = df_flows[~df_flows['symbol'].isin(excluded)]
        
        # Fetch underlying prices
        unique_symbols = df_flows['symbol'].unique().tolist()[:50]  # Limit to 50 symbols
        
        def fetch_price_safe(sym):
            try:
                return (sym, get_stock_price_yf(sym) or 0)
            except:
                return (sym, 0)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            price_results = list(executor.map(fetch_price_safe, unique_symbols))
        
        price_map = dict(price_results)
        df_flows['underlying_price'] = df_flows['symbol'].map(price_map).fillna(0)
        
        # Filter OTM only (where we have price data)
        sp = df_flows['underlying_price'].fillna(0).astype(float)
        strikes = df_flows['strike'].fillna(0).astype(float)
        types = df_flows['type'].astype(str).str.upper()
        
        is_call_otm = (types == 'CALL') & (strikes > sp) & (sp > 0)
        is_put_otm = (types == 'PUT') & (strikes < sp) & (sp > 0)
        
        # Always include key symbols
        priority_symbols = {'SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA'}
        is_priority = df_flows['symbol'].isin(priority_symbols)
        
        df_flows = df_flows[is_call_otm | is_put_otm | is_priority]
        
        # Sort by premium
        df_flows = df_flows.sort_values('premium', ascending=False).reset_index(drop=True)
        
        st.session_state.cboe_flow_results = df_flows
    
    # Display results
    df_flows = st.session_state.cboe_flow_results
    
    if df_flows is None:
        st.info("👆 Click 'Scan CBOE Flow' to detect unusual options activity across all symbols")
        return
    
    if df_flows.empty:
        st.info("No flows found matching filters")
        return
    
    # Summary metrics
    total_premium = df_flows['premium'].sum()
    call_premium = df_flows[df_flows['type'] == 'CALL']['premium'].sum()
    put_premium = df_flows[df_flows['type'] == 'PUT']['premium'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flows", f"{len(df_flows)}")
    col2.metric("Total Premium", f"${total_premium/1e6:.1f}M")
    col3.metric("Call Premium", f"${call_premium/1e6:.1f}M")
    col4.metric("Put Premium", f"${put_premium/1e6:.1f}M")
    
    # Categorize
    index_symbols = {'SPY', 'QQQ', 'IWM', 'DIA'}
    mag7_symbols = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA'}
    
    index_plays = df_flows[df_flows['symbol'].isin(index_symbols)].head(20)
    mag7_plays = df_flows[df_flows['symbol'].isin(mag7_symbols)].head(20)
    other_plays = df_flows[~df_flows['symbol'].isin(index_symbols | mag7_symbols)].head(20)
    
    tab_idx, tab_mag, tab_other = st.tabs(["📈 Index ETFs", "🚀 Mag 7", "📋 Other Stocks"])
    
    with tab_idx:
        if index_plays.empty:
            st.info("No index ETF flows")
        else:
            for i, row in index_plays.iterrows():
                _display_cboe_flow_row(row, i)
    
    with tab_mag:
        if mag7_plays.empty:
            st.info("No Mag 7 flows")
        else:
            for i, row in mag7_plays.iterrows():
                _display_cboe_flow_row(row, i)
    
    with tab_other:
        if other_plays.empty:
            st.info("No other stock flows")
        else:
            for i, row in other_plays.iterrows():
                _display_cboe_flow_row(row, i)


def _display_cboe_flow_row(row, idx):
    """Display a single CBOE flow row"""
    sym = row.get('symbol', '')
    strike = row.get('strike', 0)
    opt_type = row.get('type', 'CALL')
    expiry = row.get('expiry', '')
    prem = float(row.get('premium', 0) or 0)
    vol = int(row.get('volume', 0) or 0)
    price = float(row.get('price', 0) or 0)
    days = int(row.get('days_to_exp', 0) or 0)
    
    emoji = "🟢" if opt_type == 'CALL' else "🔴"
    leg = f"{strike:.0f}{'C' if opt_type == 'CALL' else 'P'}"
    
    prem_str = f"${prem/1e6:.2f}M" if prem >= 1e6 else f"${prem/1000:.0f}K"
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**{emoji} {sym}** {leg} exp {expiry}")
    with col2:
        st.metric("Premium", prem_str, label_visibility="collapsed")
    with col3:
        st.metric("Volume", f"{vol:,}", label_visibility="collapsed")
    with col4:
        st.metric("DTE", f"{days}d", label_visibility="collapsed")


# ==================== MAIN APP ====================
def main():
    st.title("🐋 Smart Money Flow")
    st.caption("Track institutional and whale options activity in real-time")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        watchlist = st.multiselect(
            "Select Symbols",
            options=TOP_STOCKS,
            default=['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'AMD', 'META'],
            key="flow_watchlist"
        )
        
        custom = st.text_input("Add Custom Symbols (comma-separated)")
        if custom:
            watchlist.extend([s.strip().upper() for s in custom.split(',')])
        
        if st.button("🔄 Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🐋 Whale Scanner", "📊 Flow Summary", "🔍 Single Symbol", "🌊 CBOE Scanner"])
    
    with tab1:
        render_whale_scanner_tab(watchlist)
    
    with tab2:
        render_flow_summary_tab(watchlist)
    
    with tab3:
        render_single_symbol_tab()
    
    with tab4:
        render_cboe_flow_tab()


if __name__ == "__main__":
    main()
