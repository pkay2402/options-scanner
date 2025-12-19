"""
SPX Market Intelligence
Professional market maker view of SPX options positioning and expectations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)

# Initialize session state for auto-refresh
if 'auto_refresh_spx' not in st.session_state:
    st.session_state.auto_refresh_spx = True
if 'last_refresh_spx' not in st.session_state:
    st.session_state.last_refresh_spx = datetime.now()

st.set_page_config(
    page_title="SPX Market Intelligence",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .gamma-positive {
        background: linear-gradient(135deg, #22c55e22 0%, #22c55e11 100%);
        border-left-color: #22c55e;
    }
    
    .gamma-negative {
        background: linear-gradient(135deg, #ef444422 0%, #ef444411 100%);
        border-left-color: #ef4444;
    }
    
    .level-card {
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
    }
    
    .alert-bullish {
        background: #22c55e22;
        border-left-color: #22c55e;
        color: #166534;
    }
    
    .alert-bearish {
        background: #ef444422;
        border-left-color: #ef4444;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎯 SPX Market Intelligence")
st.markdown("Professional market maker view of SPX options positioning")

# Auto-refresh controls
col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 2, 3])

with col_refresh1:
    st.session_state.auto_refresh_spx = st.checkbox(
        "🔄 Auto-Refresh (60s)",
        value=st.session_state.auto_refresh_spx,
        help="Automatically refresh data every 60 seconds"
    )

with col_refresh2:
    if st.button("🔃 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh_spx = datetime.now()
        st.rerun()

with col_refresh3:
    if st.session_state.auto_refresh_spx:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh_spx).seconds
        time_until_next = max(0, 60 - time_since_refresh)
        st.info(f"⏱️ Next refresh in: {time_until_next}s")
    else:
        st.caption(f"Last updated: {st.session_state.last_refresh_spx.strftime('%I:%M:%S %p')}")

st.markdown("---")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.markdown("**$SPX Options Analysis**")

with col2:
    strike_range = st.slider(
        "Strike Range %",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Limit strikes to ±X% from current price"
    )

with col3:
    min_volume = st.number_input(
        "Min Volume",
        min_value=10,
        max_value=1000,
        value=50,
        step=10
    )

with col4:
    if st.button("🔄", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

@st.cache_data(ttl=30)  # Cache for 30 seconds (streaming updates)
def get_spx_price():
    """Get current SPX price"""
    client = SchwabClient()
    if not client.authenticate():
        return None
    
    try:
        quote = client.get_quote("$SPX")
        if quote and "$SPX" in quote:
            data = quote["$SPX"]["quote"]
            return {
                'price': data['lastPrice'],
                'change': data['netChange'],
                'change_pct': data['netPercentChange'],
                'open': data.get('openPrice', data['lastPrice']),
                'high': data.get('highPrice', data['lastPrice']),
                'low': data.get('lowPrice', data['lastPrice']),
                'prev_close': data.get('closePrice', data['lastPrice'])
            }
    except Exception as e:
        logger.error(f"Error fetching SPX: {e}")
    
    return None

@st.cache_data(ttl=30)
def get_vix_price():
    """Get current VIX price"""
    client = SchwabClient()
    if not client.authenticate():
        return None
    
    try:
        quote = client.get_quote("$VIX.X")
        if quote and "$VIX.X" in quote:
            data = quote["$VIX.X"]["quote"]
            return {
                'price': data['lastPrice'],
                'change': data['netChange'],
                'change_pct': data['netPercentChange']
            }
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
    
    return None

def get_expiry_dates():
    """Get key expiry dates (0DTE, 1DTE, weekly, monthly)"""
    today = datetime.now().date()
    dates = []
    
    # 0DTE (if market is open and before 4pm)
    current_hour = datetime.now().hour
    if today.weekday() < 5 and current_hour < 16:  # Mon-Fri before 4pm
        dates.append(('0DTE', today))
    
    # Next trading day (1DTE)
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip weekend
        next_day += timedelta(days=1)
    dates.append(('1DTE', next_day))
    
    # Next weekly expiry (Friday)
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and current_hour >= 16:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday or 7)
    dates.append(('Weekly', next_friday))
    
    # Next monthly expiry (3rd Friday)
    current_month = today.month
    current_year = today.year
    
    # Find 3rd Friday of next month
    for month_offset in range(1, 3):
        year = current_year + (current_month + month_offset - 1) // 12
        month = ((current_month + month_offset - 1) % 12) + 1
        
        first_day = datetime(year, month, 1).date()
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(days=14)
        
        if third_friday > today:
            dates.append(('Monthly', third_friday))
            break
    
    return dates

@st.cache_data(ttl=30)
def get_spx_options_chain(expiry_date, current_price, strike_range_pct=2):
    """Get SPX options chain for specific expiry, limited to strike range"""
    client = SchwabClient()
    if not client.authenticate():
        return None
    
    try:
        exp_str = expiry_date.strftime('%Y-%m-%d')
        
        # Calculate strike range
        lower_strike = current_price * (1 - strike_range_pct/100)
        upper_strike = current_price * (1 + strike_range_pct/100)
        
        options = client.get_options_chain(
            symbol="$SPX",
            contract_type='ALL',
            from_date=exp_str,
            to_date=exp_str,
            strike_count=100  # Get plenty of strikes
        )
        
        if not options:
            return None
        
        chain_data = []
        
        # Process calls
        if 'callExpDateMap' in options:
            for exp_date, strikes in options['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    
                    # Filter by strike range
                    if strike < lower_strike or strike > upper_strike:
                        continue
                    
                    for contract in contracts:
                        chain_data.append({
                            'strike': strike,
                            'type': 'CALL',
                            'bid': contract.get('bid', 0),
                            'ask': contract.get('ask', 0),
                            'last': contract.get('last', 0),
                            'volume': contract.get('totalVolume', 0),
                            'openInterest': contract.get('openInterest', 0),
                            'delta': contract.get('delta', 0),
                            'gamma': contract.get('gamma', 0),
                            'theta': contract.get('theta', 0),
                            'vega': contract.get('vega', 0),
                            'impliedVolatility': contract.get('volatility', 0) * 100,
                            'dte': (expiry_date - datetime.now().date()).days
                        })
        
        # Process puts
        if 'putExpDateMap' in options:
            for exp_date, strikes in options['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    
                    # Filter by strike range
                    if strike < lower_strike or strike > upper_strike:
                        continue
                    
                    for contract in contracts:
                        chain_data.append({
                            'strike': strike,
                            'type': 'PUT',
                            'bid': contract.get('bid', 0),
                            'ask': contract.get('ask', 0),
                            'last': contract.get('last', 0),
                            'volume': contract.get('totalVolume', 0),
                            'openInterest': contract.get('openInterest', 0),
                            'delta': contract.get('delta', 0),
                            'gamma': contract.get('gamma', 0),
                            'theta': contract.get('theta', 0),
                            'vega': contract.get('vega', 0),
                            'impliedVolatility': contract.get('volatility', 0) * 100,
                            'dte': (expiry_date - datetime.now().date()).days
                        })
        
        return pd.DataFrame(chain_data)
    
    except Exception as e:
        logger.error(f"Error fetching options chain: {e}")
        return None

def calculate_expected_move(chain_df, current_price):
    """Calculate expected move from ATM straddle"""
    if chain_df is None or chain_df.empty:
        return None
    
    # Find ATM strike
    atm_strike = chain_df['strike'].iloc[(chain_df['strike'] - current_price).abs().argsort()[0]]
    
    # Get ATM call and put
    atm_call = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['type'] == 'CALL')]
    atm_put = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['type'] == 'PUT')]
    
    if atm_call.empty or atm_put.empty:
        return None
    
    # Straddle price (use mid price)
    call_mid = (atm_call.iloc[0]['bid'] + atm_call.iloc[0]['ask']) / 2
    put_mid = (atm_put.iloc[0]['bid'] + atm_put.iloc[0]['ask']) / 2
    straddle_price = call_mid + put_mid
    
    # Expected move is approximately straddle price * 0.85 (1 standard deviation)
    expected_move = straddle_price * 0.85
    expected_move_pct = (expected_move / current_price) * 100
    
    return {
        'atm_strike': atm_strike,
        'straddle_price': straddle_price,
        'expected_move': expected_move,
        'expected_move_pct': expected_move_pct,
        'upper_level': current_price + expected_move,
        'lower_level': current_price - expected_move,
        'atm_iv': atm_call.iloc[0]['impliedVolatility']
    }

def calculate_gamma_exposure(chain_df, current_price):
    """Calculate dealer gamma exposure by strike"""
    if chain_df is None or chain_df.empty:
        return None
    
    gex_by_strike = []
    
    for strike in chain_df['strike'].unique():
        strike_data = chain_df[chain_df['strike'] == strike]
        
        # Calls: dealers short = negative gamma
        calls = strike_data[strike_data['type'] == 'CALL']
        call_gex = 0
        if not calls.empty:
            # Gamma exposure = gamma * open interest * 100 (contract multiplier) * spot^2 / 100
            call_gex = -calls['gamma'].sum() * calls['openInterest'].sum() * 100 * current_price * current_price / 100
        
        # Puts: dealers short = positive gamma
        puts = strike_data[strike_data['type'] == 'PUT']
        put_gex = 0
        if not puts.empty:
            put_gex = puts['gamma'].sum() * puts['openInterest'].sum() * 100 * current_price * current_price / 100
        
        total_gex = call_gex + put_gex
        
        gex_by_strike.append({
            'strike': strike,
            'call_gex': call_gex / 1e9,  # Convert to billions
            'put_gex': put_gex / 1e9,
            'total_gex': total_gex / 1e9,
            'call_oi': calls['openInterest'].sum() if not calls.empty else 0,
            'put_oi': puts['openInterest'].sum() if not puts.empty else 0
        })
    
    return pd.DataFrame(gex_by_strike).sort_values('strike')

def find_gamma_levels(gex_df, current_price):
    """Find key gamma levels"""
    if gex_df is None or gex_df.empty:
        return None
    
    # Find zero gamma level (where total GEX crosses zero)
    positive_gex = gex_df[gex_df['total_gex'] > 0]
    negative_gex = gex_df[gex_df['total_gex'] < 0]
    
    zero_gamma_level = None
    if not positive_gex.empty and not negative_gex.empty:
        # Find the strike closest to where it crosses zero
        above_current = gex_df[gex_df['strike'] > current_price]
        below_current = gex_df[gex_df['strike'] < current_price]
        
        if not above_current.empty and not below_current.empty:
            # Simple approximation - find where sign changes
            for i in range(len(gex_df) - 1):
                if gex_df.iloc[i]['total_gex'] * gex_df.iloc[i+1]['total_gex'] < 0:
                    zero_gamma_level = gex_df.iloc[i]['strike']
                    break
    
    # Max positive GEX (support)
    max_pos_gex = gex_df.loc[gex_df['total_gex'].idxmax()]
    
    # Max negative GEX (resistance)
    max_neg_gex = gex_df.loc[gex_df['total_gex'].idxmin()]
    
    # Total gamma exposure
    total_gex = gex_df['total_gex'].sum()
    
    return {
        'zero_gamma_level': zero_gamma_level,
        'max_positive_gex_strike': max_pos_gex['strike'],
        'max_positive_gex_value': max_pos_gex['total_gex'],
        'max_negative_gex_strike': max_neg_gex['strike'],
        'max_negative_gex_value': max_neg_gex['total_gex'],
        'total_gex': total_gex,
        'gamma_regime': 'Positive' if total_gex > 0 else 'Negative'
    }

def calculate_max_pain(chain_df):
    """Calculate max pain level"""
    if chain_df is None or chain_df.empty:
        return None
    
    strikes = sorted(chain_df['strike'].unique())
    max_pain_data = []
    
    for strike in strikes:
        # Calculate total value of all ITM options at this strike
        calls_itm = chain_df[(chain_df['type'] == 'CALL') & (chain_df['strike'] < strike)]
        puts_itm = chain_df[(chain_df['type'] == 'PUT') & (chain_df['strike'] > strike)]
        
        call_pain = ((strike - calls_itm['strike']) * calls_itm['openInterest']).sum()
        put_pain = ((puts_itm['strike'] - strike) * puts_itm['openInterest']).sum()
        
        total_pain = call_pain + put_pain
        
        max_pain_data.append({
            'strike': strike,
            'total_pain': total_pain
        })
    
    max_pain_df = pd.DataFrame(max_pain_data)
    max_pain_strike = max_pain_df.loc[max_pain_df['total_pain'].idxmin(), 'strike']
    
    return max_pain_strike

def find_call_put_walls(chain_df, current_price):
    """Find call and put walls (high OI concentration)"""
    if chain_df is None or chain_df.empty:
        return None
    
    # Group by strike and type
    oi_by_strike = chain_df.groupby(['strike', 'type'])['openInterest'].sum().reset_index()
    
    # Find call wall (above current price)
    calls_above = oi_by_strike[(oi_by_strike['type'] == 'CALL') & (oi_by_strike['strike'] > current_price)]
    call_wall = None
    if not calls_above.empty:
        call_wall = calls_above.loc[calls_above['openInterest'].idxmax()]
    
    # Find put wall (below current price)
    puts_below = oi_by_strike[(oi_by_strike['type'] == 'PUT') & (oi_by_strike['strike'] < current_price)]
    put_wall = None
    if not puts_below.empty:
        put_wall = puts_below.loc[puts_below['openInterest'].idxmax()]
    
    return {
        'call_wall': call_wall,
        'put_wall': put_wall
    }

# Fetch current SPX data
with st.spinner("Loading SPX data..."):
    spx_data = get_spx_price()
    vix_data = get_vix_price()

if not spx_data:
    st.error("Failed to fetch SPX data. Please try again.")
    st.stop()

current_price = spx_data['price']

# Display current market conditions
st.markdown("### 📊 Current Market Conditions")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    change_color = "🟢" if spx_data['change'] > 0 else "🔴"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">SPX Price</div>
        <div style="font-size: 28px; font-weight: 700;">${current_price:.2f}</div>
        <div style="font-size: 14px; color: {'green' if spx_data['change'] > 0 else 'red'};">
            {change_color} {spx_data['change']:+.2f} ({spx_data['change_pct']:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Day Range</div>
        <div style="font-size: 20px; font-weight: 700;">${spx_data['low']:.2f}</div>
        <div style="font-size: 14px; color: #666;">to ${spx_data['high']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if vix_data:
        vix_color = "🔴" if vix_data['change'] > 0 else "🟢"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">VIX</div>
            <div style="font-size: 28px; font-weight: 700;">{vix_data['price']:.2f}</div>
            <div style="font-size: 14px; color: {'red' if vix_data['change'] > 0 else 'green'};">
                {vix_color} {vix_data['change']:+.2f} ({vix_data['change_pct']:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    strike_lower = current_price * (1 - strike_range/100)
    strike_upper = current_price * (1 + strike_range/100)
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Strike Range</div>
        <div style="font-size: 18px; font-weight: 700;">${strike_lower:.0f}</div>
        <div style="font-size: 14px; color: #666;">to ${strike_upper:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Analysis Range</div>
        <div style="font-size: 28px; font-weight: 700;">±{strike_range}%</div>
        <div style="font-size: 14px; color: #666;">{int(strike_upper - strike_lower)} points</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Get expiry dates
expiry_dates = get_expiry_dates()

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Expected Moves",
    "🎯 Gamma Exposure",
    "💎 Strike Intelligence",
    "📊 Flow Analysis"
])

# Tab 1: Expected Moves
with tab1:
    st.markdown("### 📈 Expected Moves (from ATM Straddles)")
    st.caption("Market-implied move for different time horizons")
    
    move_data = []
    
    with st.spinner("Calculating expected moves..."):
        for label, exp_date in expiry_dates:
            chain = get_spx_options_chain(exp_date, current_price, strike_range)
            
            if chain is not None and not chain.empty:
                move = calculate_expected_move(chain, current_price)
                
                if move:
                    dte = (exp_date - datetime.now().date()).days
                    move_data.append({
                        'Expiry': label,
                        'Date': exp_date.strftime('%m/%d'),
                        'DTE': dte,
                        'ATM IV': move['atm_iv'],
                        'Expected Move': move['expected_move'],
                        'Move %': move['expected_move_pct'],
                        'Upper': move['upper_level'],
                        'Lower': move['lower_level']
                    })
    
    if move_data:
        move_df = pd.DataFrame(move_data)
        
        # Display cards
        cols = st.columns(len(move_df))
        for idx, row in move_df.iterrows():
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 16px; font-weight: 700; color: #667eea; margin-bottom: 8px;">
                        {row['Expiry']}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
                        {row['Date']} ({row['DTE']} days)
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #333; margin-bottom: 5px;">
                        ±{row['Expected Move']:.2f}
                    </div>
                    <div style="font-size: 18px; font-weight: 600; color: #667eea; margin-bottom: 10px;">
                        ±{row['Move %']:.2f}%
                    </div>
                    <div style="font-size: 12px; color: #666; padding: 8px; background: #f1f5f9; border-radius: 6px;">
                        <div>Upper: <strong>${row['Upper']:.2f}</strong></div>
                        <div>Lower: <strong>${row['Lower']:.2f}</strong></div>
                        <div style="margin-top: 5px;">IV: <strong>{row['ATM IV']:.1f}%</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chart of expected moves
        st.markdown("#### Expected Move by Expiry")
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=move_df['Expiry'],
            y=move_df['Move %'],
            text=move_df['Move %'].apply(lambda x: f'{x:.2f}%'),
            textposition='auto',
            marker_color='#667eea',
            name='Expected Move %'
        ))
        
        fig.update_layout(
            yaxis_title="Expected Move %",
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Gamma Exposure
with tab2:
    st.markdown("### 🎯 Dealer Gamma Exposure (GEX)")
    st.caption("Understanding dealer hedging flows and market inflection points")
    
    # Current SPX price display
    col_spx1, col_spx2 = st.columns([1, 3])
    with col_spx1:
        change_color = "🟢" if spx_data['change'] > 0 else "🔴"
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">SPX Current</div>
            <div style="font-size: 32px; font-weight: 700;">${current_price:.2f}</div>
            <div style="font-size: 16px; color: {'green' if spx_data['change'] > 0 else 'red'}; font-weight: 600;">
                {change_color} {spx_data['change']:+.2f} ({spx_data['change_pct']:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_spx2:
        # Select expiry for GEX analysis
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        gex_expiry_label = st.selectbox(
            "Select Expiry for GEX Analysis",
            options=[label for label, _ in expiry_dates],
            index=0
        )
    
    selected_exp = [exp_date for label, exp_date in expiry_dates if label == gex_expiry_label][0]
    
    with st.spinner(f"Calculating gamma exposure for {gex_expiry_label}..."):
        chain = get_spx_options_chain(selected_exp, current_price, strike_range)
        
        if chain is not None and not chain.empty:
            gex_df = calculate_gamma_exposure(chain, current_price)
            gamma_levels = find_gamma_levels(gex_df, current_price)
            
            if gex_df is not None and gamma_levels:
                # Display key levels
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    regime_class = "gamma-positive" if gamma_levels['gamma_regime'] == 'Positive' else "gamma-negative"
                    st.markdown(f"""
                    <div class="metric-card {regime_class}">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Gamma Regime</div>
                        <div style="font-size: 24px; font-weight: 700;">{gamma_levels['gamma_regime']}</div>
                        <div style="font-size: 14px; color: #666;">Total: {gamma_levels['total_gex']:.2f}B</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if gamma_levels['zero_gamma_level']:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Zero Gamma Level</div>
                            <div style="font-size: 24px; font-weight: 700;">${gamma_levels['zero_gamma_level']:.0f}</div>
                            <div style="font-size: 12px; color: #666;">Inflection Point</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card gamma-positive">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Max Positive GEX</div>
                        <div style="font-size: 24px; font-weight: 700;">${gamma_levels['max_positive_gex_strike']:.0f}</div>
                        <div style="font-size: 12px; color: #16a34a;">{gamma_levels['max_positive_gex_value']:.2f}B (Support)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card gamma-negative">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Max Negative GEX</div>
                        <div style="font-size: 24px; font-weight: 700;">${gamma_levels['max_negative_gex_strike']:.0f}</div>
                        <div style="font-size: 12px; color: #dc2626;">{gamma_levels['max_negative_gex_value']:.2f}B (Resistance)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # GEX Chart
                st.markdown("#### Gamma Exposure Profile")
                
                fig = go.Figure()
                
                # Total GEX
                fig.add_trace(go.Bar(
                    x=gex_df['strike'],
                    y=gex_df['total_gex'],
                    name='Total GEX',
                    marker_color=gex_df['total_gex'].apply(lambda x: '#22c55e' if x > 0 else '#ef4444'),
                    text=gex_df['total_gex'].apply(lambda x: f'{x:.2f}B'),
                    textposition='outside'
                ))
                
                # Current price line
                fig.add_vline(x=current_price, line_dash="dash", line_color="blue", 
                             annotation_text=f"SPX: ${current_price:.0f}")
                
                # Zero gamma level
                if gamma_levels['zero_gamma_level']:
                    fig.add_vline(x=gamma_levels['zero_gamma_level'], line_dash="dot", line_color="orange",
                                 annotation_text=f"Zero Gamma: ${gamma_levels['zero_gamma_level']:.0f}")
                
                fig.update_layout(
                    xaxis_title="Strike",
                    yaxis_title="Gamma Exposure (Billions)",
                    showlegend=False,
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                st.info("""
                **Dealer Gamma Exposure Explained:**
                - **Positive GEX (Green)**: Dealers are long gamma → they sell into rallies and buy dips → **dampens volatility** → acts as **support**
                - **Negative GEX (Red)**: Dealers are short gamma → they buy rallies and sell dips → **amplifies moves** → acts as **resistance**
                - **Zero Gamma Level**: Inflection point where dealer hedging behavior changes
                - **Current Regime**: {}  
                  {} 
                """.format(
                    gamma_levels['gamma_regime'],
                    "Market likely to be more stable with mean reversion" if gamma_levels['gamma_regime'] == 'Positive' 
                    else "Market likely to be more volatile with trending behavior"
                ))

# Tab 3: Strike Intelligence
with tab3:
    st.markdown("### 💎 Strike-Level Intelligence")
    
    # Select expiry
    strike_expiry_label = st.selectbox(
        "Select Expiry",
        options=[label for label, _ in expiry_dates],
        index=0,
        key='strike_expiry'
    )
    
    selected_exp = [exp_date for label, exp_date in expiry_dates if label == strike_expiry_label][0]
    
    with st.spinner(f"Analyzing strikes for {strike_expiry_label}..."):
        chain = get_spx_options_chain(selected_exp, current_price, strike_range)
        
        if chain is not None and not chain.empty:
            # Calculate max pain
            max_pain = calculate_max_pain(chain)
            
            # Find call/put walls
            walls = find_call_put_walls(chain, current_price)
            
            # Display key levels
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if max_pain:
                    distance_from_max_pain = ((current_price - max_pain) / current_price) * 100
                    st.markdown(f"""
                    <div class="level-card">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Max Pain Level</div>
                        <div style="font-size: 28px; font-weight: 700; color: #3b82f6;">${max_pain:.0f}</div>
                        <div style="font-size: 13px; color: #666;">
                            {abs(distance_from_max_pain):.2f}% {'above' if distance_from_max_pain < 0 else 'below'} current
                        </div>
                        <div style="font-size: 11px; color: #666; margin-top: 8px;">
                            Price where most options expire worthless
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if walls['call_wall'] is not None:
                    call_strike = walls['call_wall']['strike']
                    call_oi = walls['call_wall']['openInterest']
                    distance = ((call_strike - current_price) / current_price) * 100
                    st.markdown(f"""
                    <div class="level-card alert-bearish">
                        <div style="font-size: 14px; margin-bottom: 5px;">Call Wall (Resistance)</div>
                        <div style="font-size: 28px; font-weight: 700;">${call_strike:.0f}</div>
                        <div style="font-size: 13px;">
                            {distance:+.2f}% from current
                        </div>
                        <div style="font-size: 11px; margin-top: 8px;">
                            OI: {call_oi:,.0f} contracts
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if walls['put_wall'] is not None:
                    put_strike = walls['put_wall']['strike']
                    put_oi = walls['put_wall']['openInterest']
                    distance = ((current_price - put_strike) / current_price) * 100
                    st.markdown(f"""
                    <div class="level-card alert-bullish">
                        <div style="font-size: 14px; margin-bottom: 5px;">Put Wall (Support)</div>
                        <div style="font-size: 28px; font-weight: 700;">${put_strike:.0f}</div>
                        <div style="font-size: 13px;">
                            {distance:.2f}% below current
                        </div>
                        <div style="font-size: 11px; margin-top: 8px;">
                            OI: {put_oi:,.0f} contracts
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Volume and OI profile
            st.markdown("#### Volume & Open Interest Profile")
            
            volume_by_strike = chain.groupby(['strike', 'type']).agg({
                'volume': 'sum',
                'openInterest': 'sum'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume profile
                fig_vol = px.bar(
                    volume_by_strike,
                    x='strike',
                    y='volume',
                    color='type',
                    title="Volume by Strike",
                    color_discrete_map={'CALL': '#22c55e', 'PUT': '#ef4444'},
                    barmode='group'
                )
                fig_vol.add_vline(x=current_price, line_dash="dash", annotation_text=f"${current_price:.0f}")
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col2:
                # OI profile
                fig_oi = px.bar(
                    volume_by_strike,
                    x='strike',
                    y='openInterest',
                    color='type',
                    title="Open Interest by Strike",
                    color_discrete_map={'CALL': '#22c55e', 'PUT': '#ef4444'},
                    barmode='group'
                )
                fig_oi.add_vline(x=current_price, line_dash="dash", annotation_text=f"${current_price:.0f}")
                st.plotly_chart(fig_oi, use_container_width=True)

# Tab 4: Flow Analysis
with tab4:
    st.markdown("### 📊 Options Flow Analysis")
    
    # Select expiry for flow
    flow_expiry_label = st.selectbox(
        "Select Expiry",
        options=[label for label, _ in expiry_dates],
        index=0,
        key='flow_expiry'
    )
    
    selected_exp = [exp_date for label, exp_date in expiry_dates if label == flow_expiry_label][0]
    
    with st.spinner(f"Analyzing flow for {flow_expiry_label}..."):
        chain = get_spx_options_chain(selected_exp, current_price, strike_range)
        
        if chain is not None and not chain.empty:
            # Filter by minimum volume
            active_chain = chain[chain['volume'] >= min_volume].copy()
            
            if not active_chain.empty:
                # Calculate metrics
                total_call_vol = active_chain[active_chain['type'] == 'CALL']['volume'].sum()
                total_put_vol = active_chain[active_chain['type'] == 'PUT']['volume'].sum()
                pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
                
                total_call_oi = active_chain[active_chain['type'] == 'CALL']['openInterest'].sum()
                total_put_oi = active_chain[active_chain['type'] == 'PUT']['openInterest'].sum()
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Call Volume", f"{total_call_vol:,.0f}")
                
                with col2:
                    st.metric("Total Put Volume", f"{total_put_vol:,.0f}")
                
                with col3:
                    sentiment = "🟢 Bullish" if pc_ratio < 0.8 else "🔴 Bearish" if pc_ratio > 1.2 else "⚪ Neutral"
                    st.metric("Put/Call Ratio", f"{pc_ratio:.2f}", sentiment)
                
                with col4:
                    st.metric("Total Open Interest", f"{(total_call_oi + total_put_oi):,.0f}")
                
                # Top unusual activity
                st.markdown("#### Top Unusual Activity")
                
                # Calculate vol/OI ratio
                active_chain['vol_oi_ratio'] = active_chain['volume'] / active_chain['openInterest'].replace(0, 1)
                
                # Sort by volume
                top_flow = active_chain.nlargest(20, 'volume')[
                    ['strike', 'type', 'volume', 'openInterest', 'vol_oi_ratio', 'impliedVolatility', 'delta']
                ].copy()
                
                top_flow.columns = ['Strike', 'Type', 'Volume', 'Open Interest', 'Vol/OI', 'IV %', 'Delta']
                
                st.dataframe(
                    top_flow,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Volume": st.column_config.NumberColumn(format="%d"),
                        "Open Interest": st.column_config.NumberColumn(format="%d"),
                        "Vol/OI": st.column_config.NumberColumn(format="%.2f"),
                        "IV %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Delta": st.column_config.NumberColumn(format="%.3f")
                    }
                )
                
                # Flow heatmap
                st.markdown("#### Flow Heatmap")
                
                flow_pivot = active_chain.pivot_table(
                    values='volume',
                    index='strike',
                    columns='type',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig = go.Figure()
                
                if 'CALL' in flow_pivot.columns:
                    fig.add_trace(go.Bar(
                        x=flow_pivot.index,
                        y=flow_pivot['CALL'],
                        name='Calls',
                        marker_color='#22c55e'
                    ))
                
                if 'PUT' in flow_pivot.columns:
                    fig.add_trace(go.Bar(
                        x=flow_pivot.index,
                        y=-flow_pivot['PUT'],  # Negative for visual separation
                        name='Puts',
                        marker_color='#ef4444'
                    ))
                
                fig.add_vline(x=current_price, line_dash="dash", annotation_text=f"SPX: ${current_price:.0f}")
                
                fig.update_layout(
                    xaxis_title="Strike",
                    yaxis_title="Volume (Calls positive, Puts negative)",
                    height=400,
                    barmode='relative',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No active flow above {min_volume} volume threshold")

st.markdown("---")

# Auto-refresh logic - continuously check if it's time to refresh
if st.session_state.auto_refresh_spx:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh_spx).seconds
    if time_since_refresh >= 60:
        st.cache_data.clear()
        st.session_state.last_refresh_spx = datetime.now()
        st.rerun()
    else:
        # Keep checking every 10 seconds to update countdown timer
        time.sleep(10)
        st.rerun()
    st.caption("🔄 Live streaming enabled (60s) | Professional market maker analysis for SPX options.")
else:
    st.caption("💡 Enable auto-refresh for live streaming updates | Professional market maker analysis for SPX options.")
