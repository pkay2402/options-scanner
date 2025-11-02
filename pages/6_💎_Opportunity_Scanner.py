#!/usr/bin/env python3
"""
Smart Options Opportunity Scanner
Automatically scans markets and identifies top trade setups with actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent; sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Configure Streamlit page
st.set_page_config(
    page_title="Trade Opportunity Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .opportunity-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 8px solid;
    }
    .setup-gamma {
        border-left-color: #9b59b6;
    }
    .setup-momentum {
        border-left-color: #3498db;
    }
    .setup-volatility {
        border-left-color: #e74c3c;
    }
    .setup-reversal {
        border-left-color: #f39c12;
    }
    .opportunity-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 2px solid #ecf0f1;
    }
    .setup-title {
        font-size: 1.8em;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence-badge {
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    .confidence-high {
        background-color: #2ecc71;
        color: white;
    }
    .confidence-medium {
        background-color: #f39c12;
        color: white;
    }
    .confidence-low {
        background-color: #e74c3c;
        color: white;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
    }
    .trade-suggestion {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .key-levels {
        background: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .level-item {
        display: flex;
        justify-content: space-between;
        padding: 8px;
        margin: 5px 0;
        background: white;
        border-radius: 5px;
    }
    .rationale-box {
        background: #fff9e6;
        border-left: 4px solid #f39c12;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .signal-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .signal-bullish {
        background-color: #2ecc71;
    }
    .signal-bearish {
        background-color: #e74c3c;
    }
    .signal-neutral {
        background-color: #f39c12;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .scanner-status {
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 10px 20px;
        background: #2ecc71;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        z-index: 1000;
        animation: slideIn 0.5s;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

# Default watchlist - most liquid stocks
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",  # Indices
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",  # Mega caps
    "AMD", "NFLX", "CRM", "AVGO", "ORCL",  # Tech
    "JPM", "BAC", "GS", "XLF",  # Financials
    "XLE", "XOM", "CVX",  # Energy
    "COIN", "XYZ", "PYPL"  # Fintech
    "CRWD", "PANW", "ZS"  # Cybersecurity
    "NNE", "GEV", "CCJ","BE"  # Nuclear
    "RDDT", "PINS", "SNAP"  # Social
    "AMD", "MU", "ALAB","QCOM"  # Semis

]

def estimate_underlying_from_strikes(options_data):
    """Estimate underlying price from ATM options strikes"""
    try:
        if not options_data or 'callExpDateMap' not in options_data:
            return None
        
        exp_dates = list(options_data['callExpDateMap'].keys())
        if not exp_dates:
            return None
        
        first_exp = options_data['callExpDateMap'][exp_dates[0]]
        strikes = [float(s) for s in first_exp.keys()]
        
        if strikes:
            strike_data = []
            for strike_str, contracts in first_exp.items():
                if contracts:
                    contract = contracts[0]
                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)
                    strike = float(strike_str)
                    activity = volume + open_interest * 0.1
                    strike_data.append((strike, activity))
            
            if strike_data:
                strike_data.sort(key=lambda x: x[1], reverse=True)
                most_active_strike = strike_data[0][0]
                
                if 50 < most_active_strike < 2000:
                    return most_active_strike
            
            strikes.sort()
            mid_index = len(strikes) // 2
            return strikes[mid_index]
        
        return None
    except:
        return None

@st.cache_data(ttl=120)
def get_options_data(symbol):
    """Fetch options data"""
    try:
        client = SchwabClient()
        
        quote_data = client.get_quote(symbol)
        if not quote_data or symbol not in quote_data:
            return None, None
        
        underlying_price = quote_data[symbol].get('lastPrice', 0)
        if underlying_price == 0:
            underlying_price = quote_data[symbol].get('mark', 0)
        
        options_data = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not options_data:
            return None, None
        
        if 'underlying' in options_data and options_data['underlying']:
            options_underlying_price = options_data['underlying'].get('last', 0)
            if options_underlying_price and options_underlying_price > 0:
                underlying_price = options_underlying_price
        
        if underlying_price == 0 or underlying_price == 100.0:
            estimated_price = estimate_underlying_from_strikes(options_data)
            if estimated_price:
                underlying_price = estimated_price
        
        return options_data, underlying_price
        
    except Exception as e:
        return None, None

def analyze_gamma_squeeze_setup(options_data, underlying_price, symbol):
    """Detect gamma squeeze opportunities"""
    
    if not options_data:
        return None
    
    # Get first 2 expiries
    exp_dates = list(options_data.get('callExpDateMap', {}).keys())[:2]
    if not exp_dates:
        return None
    
    total_call_oi_above = 0
    total_put_oi_below = 0
    max_call_strike = None
    max_call_oi = 0
    
    for exp_date in exp_dates:
        if exp_date not in options_data.get('callExpDateMap', {}):
            continue
        
        strikes_data = options_data['callExpDateMap'][exp_date]
        
        for strike_str, contracts in strikes_data.items():
            if not contracts:
                continue
            
            strike = float(strike_str)
            contract = contracts[0]
            oi = contract.get('openInterest', 0)
            
            # Count OI above current price
            if strike > underlying_price and strike < underlying_price * 1.1:
                total_call_oi_above += oi
                
                if oi > max_call_oi:
                    max_call_oi = oi
                    max_call_strike = strike
    
    # Check puts below
    for exp_date in exp_dates:
        if exp_date not in options_data.get('putExpDateMap', {}):
            continue
        
        strikes_data = options_data['putExpDateMap'][exp_date]
        
        for strike_str, contracts in strikes_data.items():
            if not contracts:
                continue
            
            strike = float(strike_str)
            contract = contracts[0]
            oi = contract.get('openInterest', 0)
            
            if strike < underlying_price and strike > underlying_price * 0.9:
                total_put_oi_below += oi
    
    # Gamma squeeze criteria
    if total_call_oi_above > total_put_oi_below * 1.5 and total_call_oi_above > 10000:
        
        confidence = "HIGH" if total_call_oi_above > 50000 else "MEDIUM"
        
        return {
            'type': 'GAMMA_SQUEEZE',
            'symbol': symbol,
            'current_price': underlying_price,
            'target_strike': max_call_strike,
            'call_oi_above': total_call_oi_above,
            'put_oi_below': total_put_oi_below,
            'ratio': total_call_oi_above / total_put_oi_below if total_put_oi_below > 0 else 999,
            'confidence': confidence,
            'upside_potential': ((max_call_strike - underlying_price) / underlying_price) * 100,
            'score': total_call_oi_above / 1000  # Simple scoring
        }
    
    return None

def analyze_momentum_flow(options_data, underlying_price, symbol):
    """Detect directional momentum with flow confirmation"""
    
    if not options_data:
        return None
    
    exp_dates = list(options_data.get('callExpDateMap', {}).keys())[:3]
    if not exp_dates:
        return None
    
    call_volume = 0
    call_premium = 0
    put_volume = 0
    put_premium = 0
    
    otm_call_volume = 0
    otm_put_volume = 0
    
    # Analyze flow
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
        
        is_call = 'call' in option_type
        
        for exp_date in exp_dates:
            if exp_date not in options_data[option_type]:
                continue
            
            strikes_data = options_data[option_type][exp_date]
            
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                contract = contracts[0]
                
                volume = contract.get('totalVolume', 0)
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else contract.get('last', 0)
                
                premium = volume * mid_price * 100
                
                if is_call:
                    call_volume += volume
                    call_premium += premium
                    
                    if strike > underlying_price:
                        otm_call_volume += volume
                else:
                    put_volume += volume
                    put_premium += premium
                    
                    if strike < underlying_price:
                        otm_put_volume += volume
    
    total_volume = call_volume + put_volume
    
    if total_volume < 1000:
        return None
    
    # Detect directional bias
    call_put_volume_ratio = call_volume / put_volume if put_volume > 0 else 999
    call_put_premium_ratio = call_premium / put_premium if put_premium > 0 else 999
    
    # Strong bullish flow
    if call_put_volume_ratio > 1.5 and otm_call_volume > otm_put_volume * 1.3 and call_premium > 500000:
        return {
            'type': 'MOMENTUM_BULLISH',
            'symbol': symbol,
            'current_price': underlying_price,
            'direction': 'BULLISH',
            'call_volume': call_volume,
            'put_volume': put_volume,
            'cp_ratio': call_put_volume_ratio,
            'total_premium': call_premium + put_premium,
            'dominant_premium': call_premium,
            'confidence': 'HIGH' if call_put_volume_ratio > 2.0 else 'MEDIUM',
            'score': call_premium / 10000
        }
    
    # Strong bearish flow
    elif call_put_volume_ratio < 0.7 and otm_put_volume > otm_call_volume * 1.3 and put_premium > 500000:
        return {
            'type': 'MOMENTUM_BEARISH',
            'symbol': symbol,
            'current_price': underlying_price,
            'direction': 'BEARISH',
            'call_volume': call_volume,
            'put_volume': put_volume,
            'cp_ratio': call_put_volume_ratio,
            'total_premium': call_premium + put_premium,
            'dominant_premium': put_premium,
            'confidence': 'HIGH' if call_put_volume_ratio < 0.5 else 'MEDIUM',
            'score': put_premium / 10000
        }
    
    return None

def analyze_volatility_play(options_data, underlying_price, symbol):
    """Detect volatility expansion opportunities"""
    
    if not options_data:
        return None
    
    exp_dates = list(options_data.get('callExpDateMap', {}).keys())[:2]
    if not exp_dates:
        return None
    
    # Get ATM options IV
    atm_ivs = []
    straddle_price = 0
    atm_strike = None
    
    for exp_date in exp_dates[:1]:  # Just first expiry
        # Find ATM strike
        if exp_date in options_data.get('callExpDateMap', {}):
            strikes = [float(s) for s in options_data['callExpDateMap'][exp_date].keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            
            # Get ATM call
            atm_call = options_data['callExpDateMap'][exp_date].get(str(atm_strike), [None])[0]
            if atm_call:
                call_iv = atm_call.get('volatility', 0) * 100
                call_price = atm_call.get('mark', 0)
                
                if call_iv > 0:
                    atm_ivs.append(call_iv)
                    straddle_price += call_price
        
        # Get ATM put
        if exp_date in options_data.get('putExpDateMap', {}) and atm_strike:
            atm_put = options_data['putExpDateMap'][exp_date].get(str(atm_strike), [None])[0]
            if atm_put:
                put_iv = atm_put.get('volatility', 0) * 100
                put_price = atm_put.get('mark', 0)
                
                if put_iv > 0:
                    atm_ivs.append(put_iv)
                    straddle_price += put_price
    
    if not atm_ivs:
        return None
    
    avg_iv = np.mean(atm_ivs)
    
    # Low IV criteria (potential expansion)
    if avg_iv < 35 and straddle_price > 0:  # Relatively low IV
        
        expected_move = (straddle_price / underlying_price) * 100
        
        return {
            'type': 'VOLATILITY_EXPANSION',
            'symbol': symbol,
            'current_price': underlying_price,
            'current_iv': avg_iv,
            'atm_strike': atm_strike,
            'straddle_price': straddle_price,
            'expected_move': expected_move,
            'confidence': 'MEDIUM',  # Would need historical IV for better confidence
            'score': (40 - avg_iv) * 2  # Lower IV = higher score
        }
    
    return None

def analyze_reversal_setup(options_data, underlying_price, symbol):
    """Detect institutional positioning reversal"""
    
    if not options_data:
        return None
    
    exp_dates = list(options_data.get('callExpDateMap', {}).keys())[:2]
    if not exp_dates:
        return None
    
    # Look for unusual OI vs Volume patterns
    unusual_strikes = []
    
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in options_data:
            continue
        
        is_call = 'call' in option_type
        
        for exp_date in exp_dates:
            if exp_date not in options_data[option_type]:
                continue
            
            strikes_data = options_data[option_type][exp_date]
            
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                
                strike = float(strike_str)
                contract = contracts[0]
                
                volume = contract.get('totalVolume', 0)
                oi = contract.get('openInterest', 0)
                
                # High volume relative to OI (new positioning)
                if oi > 0 and volume > oi * 0.8 and volume > 500:
                    
                    bid = contract.get('bid', 0)
                    ask = contract.get('ask', 0)
                    mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else contract.get('last', 0)
                    premium = volume * mid_price * 100
                    
                    if premium > 100000:
                        unusual_strikes.append({
                            'strike': strike,
                            'type': 'CALL' if is_call else 'PUT',
                            'volume': volume,
                            'oi': oi,
                            'vol_oi_ratio': volume / oi,
                            'premium': premium
                        })
    
    if not unusual_strikes:
        return None
    
    # Sort by premium
    unusual_strikes.sort(key=lambda x: x['premium'], reverse=True)
    top_strike = unusual_strikes[0]
    
    # Determine direction
    if top_strike['type'] == 'CALL' and top_strike['strike'] > underlying_price:
        direction = 'BULLISH'
    elif top_strike['type'] == 'PUT' and top_strike['strike'] < underlying_price:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'
    
    total_premium = sum([s['premium'] for s in unusual_strikes[:3]])
    
    if total_premium > 500000:
        return {
            'type': 'REVERSAL_SETUP',
            'symbol': symbol,
            'current_price': underlying_price,
            'direction': direction,
            'key_strike': top_strike['strike'],
            'option_type': top_strike['type'],
            'unusual_volume': top_strike['volume'],
            'premium_flow': total_premium,
            'num_unusual': len(unusual_strikes),
            'confidence': 'HIGH' if total_premium > 1000000 else 'MEDIUM',
            'score': total_premium / 10000
        }
    
    return None

def scan_opportunities(symbols, progress_callback=None):
    """Scan all symbols for opportunities"""
    
    opportunities = []
    
    for idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(idx + 1, len(symbols), symbol)
        
        try:
            options_data, underlying_price = get_options_data(symbol)
            
            if not options_data or not underlying_price:
                continue
            
            # Run all analyzers
            gamma_setup = analyze_gamma_squeeze_setup(options_data, underlying_price, symbol)
            if gamma_setup:
                opportunities.append(gamma_setup)
            
            momentum_setup = analyze_momentum_flow(options_data, underlying_price, symbol)
            if momentum_setup:
                opportunities.append(momentum_setup)
            
            volatility_setup = analyze_volatility_play(options_data, underlying_price, symbol)
            if volatility_setup:
                opportunities.append(volatility_setup)
            
            reversal_setup = analyze_reversal_setup(options_data, underlying_price, symbol)
            if reversal_setup:
                opportunities.append(reversal_setup)
        
        except Exception as e:
            continue
    
    # Sort by score
    opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return opportunities

def display_opportunity(opp, rank):
    """Display a single opportunity with full details"""
    
    setup_type = opp['type']
    symbol = opp['symbol']
    current_price = opp['current_price']
    confidence = opp['confidence']
    
    # Determine styling
    if 'GAMMA' in setup_type:
        card_class = 'setup-gamma'
        setup_name = '🚀 Gamma Squeeze Setup'
        color = '#9b59b6'
    elif 'MOMENTUM' in setup_type:
        card_class = 'setup-momentum'
        direction = opp.get('direction', 'BULLISH')
        setup_name = f'📈 {direction.title()} Momentum Flow'
        color = '#3498db'
    elif 'VOLATILITY' in setup_type:
        card_class = 'setup-volatility'
        setup_name = '⚡ Volatility Expansion Play'
        color = '#e74c3c'
    else:
        card_class = 'setup-reversal'
        direction = opp.get('direction', 'NEUTRAL')
        setup_name = f'🔄 {direction.title()} Reversal Setup'
        color = '#f39c12'
    
    # Confidence badge
    conf_class = f"confidence-{confidence.lower()}"
    
    # Start card HTML
    html = f"""
    <div class="opportunity-card {card_class}">
        <div class="opportunity-header">
            <div>
                <div class="setup-title">#{rank} - {symbol} @ ${current_price:.2f}</div>
                <div style="color: {color}; font-size: 1.2em; margin-top: 5px;">{setup_name}</div>
            </div>
            <div class="confidence-badge {conf_class}">{confidence} CONFIDENCE</div>
        </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Metrics based on setup type
    if 'GAMMA' in setup_type:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Target Strike", f"${opp['target_strike']:.2f}")
        with col2:
            st.metric("Upside Potential", f"+{opp['upside_potential']:.1f}%")
        with col3:
            st.metric("Call OI Above", f"{opp['call_oi_above']:,.0f}")
        with col4:
            st.metric("Call/Put Ratio", f"{opp['ratio']:.2f}x")
        
        # Rationale
        st.markdown(f"""
        <div class="rationale-box">
            <strong>📋 Setup Rationale:</strong><br>
            Heavy call open interest ({opp['call_oi_above']:,.0f} contracts) concentrated above current price at ${opp['target_strike']:.2f}. 
            As price approaches this level, dealers must buy underlying to hedge, creating upward pressure. 
            Call/Put ratio of {opp['ratio']:.2f}x indicates strong bullish positioning.
        </div>
        """, unsafe_allow_html=True)
        
        # Trade suggestion
        st.markdown(f"""
        <div class="trade-suggestion">
            <strong>💡 Suggested Trade:</strong><br>
            • Buy {symbol} ${opp['target_strike']:.0f}C (near-term expiry)<br>
            • Or buy debit call spread: Long ${opp['target_strike']:.0f}C / Short ${opp['target_strike'] * 1.05:.0f}C<br>
            • Target: ${opp['target_strike']:.2f} ({opp['upside_potential']:.1f}% gain)<br>
            • Risk: Price fails to reach gamma wall, time decay
        </div>
        """, unsafe_allow_html=True)
        
        # Key levels
        st.markdown(f"""
        <div class="key-levels">
            <strong>🎯 Key Levels:</strong>
            <div class="level-item">
                <span>Current Price:</span>
                <strong>${current_price:.2f}</strong>
            </div>
            <div class="level-item">
                <span>Gamma Wall:</span>
                <strong>${opp['target_strike']:.2f}</strong>
            </div>
            <div class="level-item">
                <span>Resistance:</span>
                <strong>${opp['target_strike'] * 1.02:.2f}</strong>
            </div>
            <div class="level-item">
                <span>Support:</span>
                <strong>${current_price * 0.98:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif 'MOMENTUM' in setup_type:
        col1, col2, col3, col4 = st.columns(4)
        
        direction = opp['direction']
        
        with col1:
            st.metric("Direction", direction)
        with col2:
            st.metric("Dominant Premium", f"${opp['dominant_premium']/1e6:.1f}M")
        with col3:
            st.metric("C/P Volume Ratio", f"{opp['cp_ratio']:.2f}")
        with col4:
            st.metric("Total Volume", f"{opp['call_volume'] + opp['put_volume']:,.0f}")
        
        # Rationale
        st.markdown(f"""
        <div class="rationale-box">
            <strong>📋 Setup Rationale:</strong><br>
            Strong {direction.lower()} flow detected with ${opp['dominant_premium']/1e6:.1f}M in premium. 
            Volume bias ({opp['call_volume']:,.0f} calls vs {opp['put_volume']:,.0f} puts) suggests directional conviction.
            Institutional money is positioning for continued {direction.lower()} move.
        </div>
        """, unsafe_allow_html=True)
        
        # Trade suggestion
        if direction == 'BULLISH':
            st.markdown(f"""
            <div class="trade-suggestion">
                <strong>💡 Suggested Trade:</strong><br>
                • Buy ATM/slightly OTM calls (1-2 weeks out)<br>
                • Target strike: ${current_price * 1.05:.0f} - ${current_price * 1.10:.0f}<br>
                • Or use bull call spread to reduce cost<br>
                • Stop loss: {current_price * 0.97:.2f} (3% below entry)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="trade-suggestion">
                <strong>💡 Suggested Trade:</strong><br>
                • Buy ATM/slightly OTM puts (1-2 weeks out)<br>
                • Target strike: ${current_price * 0.95:.0f} - ${current_price * 0.90:.0f}<br>
                • Or use bear put spread to reduce cost<br>
                • Stop loss: {current_price * 1.03:.2f} (3% above entry)
            </div>
            """, unsafe_allow_html=True)
    
    elif 'VOLATILITY' in setup_type:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current IV", f"{opp['current_iv']:.1f}%")
        with col2:
            st.metric("ATM Straddle", f"${opp['straddle_price']:.2f}")
        with col3:
            st.metric("Expected Move", f"±{opp['expected_move']:.1f}%")
        with col4:
            st.metric("ATM Strike", f"${opp['atm_strike']:.2f}")
        
        # Rationale
        st.markdown(f"""
        <div class="rationale-box">
            <strong>📋 Setup Rationale:</strong><br>
            Implied volatility at {opp['current_iv']:.1f}% is relatively low, suggesting options are underpriced. 
            Expected move of ±{opp['expected_move']:.1f}% priced into ATM straddle at ${opp['straddle_price']:.2f}.
            Potential for volatility expansion on upcoming catalyst or market event.
        </div>
        """, unsafe_allow_html=True)
        
        # Trade suggestion
        st.markdown(f"""
        <div class="trade-suggestion">
            <strong>💡 Suggested Trade:</strong><br>
            • Buy ATM straddle: Long ${opp['atm_strike']:.0f}C + Long ${opp['atm_strike']:.0f}P<br>
            • Cost: ${opp['straddle_price']:.2f} per share (${opp['straddle_price'] * 100:.0f} per contract)<br>
            • Breakevens: ${opp['atm_strike'] - opp['straddle_price']:.2f} / ${opp['atm_strike'] + opp['straddle_price']:.2f}<br>
            • Profit if price moves beyond ±{opp['expected_move']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    else:  # REVERSAL
        col1, col2, col3, col4 = st.columns(4)
        
        direction = opp['direction']
        
        with col1:
            st.metric("Direction", direction)
        with col2:
            st.metric("Premium Flow", f"${opp['premium_flow']/1e6:.1f}M")
        with col3:
            st.metric("Key Strike", f"${opp['key_strike']:.2f}")
        with col4:
            st.metric("Unusual Volume", f"{opp['unusual_volume']:,.0f}")
        
        # Rationale
        st.markdown(f"""
        <div class="rationale-box">
            <strong>📋 Setup Rationale:</strong><br>
            Detected unusual institutional activity with ${opp['premium_flow']/1e6:.1f}M flowing into {opp['option_type']}s.
            High volume relative to open interest ({opp['unusual_volume']:,.0f} contracts) suggests new positioning.
            {opp['num_unusual']} strikes show abnormal activity, indicating potential {direction.lower()} reversal.
        </div>
        """, unsafe_allow_html=True)
        
        # Trade suggestion
        st.markdown(f"""
        <div class="trade-suggestion">
            <strong>💡 Suggested Trade:</strong><br>
            • Follow the institutional flow: {opp['option_type']}s near ${opp['key_strike']:.0f}<br>
            • Direction: {direction}<br>
            • Watch for continued flow in same direction<br>
            • Risk management: Exit if flow reverses or price breaks key levels
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def main():
    st.title("🎯 Smart Options Opportunity Scanner")
    st.markdown("Automated detection of high-probability trade setups across the market")
    
    # Sidebar
    st.sidebar.header("Scanner Settings")
    
    # Watchlist selection
    use_default = st.sidebar.checkbox("Use Default Watchlist", value=True)
    
    if use_default:
        symbols = DEFAULT_WATCHLIST
        st.sidebar.info(f"Scanning {len(symbols)} symbols from default watchlist")
    else:
        custom_symbols = st.sidebar.text_area(
            "Custom Watchlist (one per line)",
            value="\n".join(DEFAULT_WATCHLIST[:10]),
            height=200
        )
        symbols = [s.strip().upper() for s in custom_symbols.split('\n') if s.strip()]
    
    # Filters
    st.sidebar.subheader("Setup Filters")
    
    setup_types = st.sidebar.multiselect(
        "Setup Types",
        ["Gamma Squeeze", "Momentum Flow", "Volatility Play", "Reversal"],
        default=["Gamma Squeeze", "Momentum Flow", "Volatility Play", "Reversal"]
    )
    
    min_confidence = st.sidebar.selectbox(
        "Minimum Confidence",
        ["ALL", "MEDIUM", "HIGH"],
        index=0
    )
    
    max_results = st.sidebar.slider(
        "Max Results to Display",
        1, 20, 10
    )
    
    # Scan button
    if st.sidebar.button("🔍 SCAN NOW", type="primary"):
        st.cache_data.clear()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress_bar.progress(current / total)
            status_text.text(f"Scanning {symbol}... ({current}/{total})")
        
        # Run scan
        with st.spinner("Scanning markets for opportunities..."):
            opportunities = scan_opportunities(symbols, update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Filter by setup type
        type_map = {
            'Gamma Squeeze': 'GAMMA_SQUEEZE',
            'Momentum Flow': 'MOMENTUM',
            'Volatility Play': 'VOLATILITY_EXPANSION',
            'Reversal': 'REVERSAL_SETUP'
        }
        
        selected_types = [type_map[t] for t in setup_types]
        opportunities = [o for o in opportunities if any(st in o['type'] for st in selected_types)]
        
        # Filter by confidence
        if min_confidence != "ALL":
            opportunities = [o for o in opportunities if o['confidence'] == min_confidence or 
                           (min_confidence == "MEDIUM" and o['confidence'] == "HIGH")]
        
        # Limit results
        opportunities = opportunities[:max_results]
        
        # Store in session state
        st.session_state['opportunities'] = opportunities
        st.session_state['scan_time'] = datetime.now()
        
        st.success(f"✅ Scan complete! Found {len(opportunities)} opportunities")
    
    # Display results
    if 'opportunities' in st.session_state and st.session_state['opportunities']:
        opportunities = st.session_state['opportunities']
        scan_time = st.session_state.get('scan_time', datetime.now())
        
        # Summary
        st.header(f"📊 Top {len(opportunities)} Trade Opportunities")
        st.caption(f"Last scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gamma_count = len([o for o in opportunities if 'GAMMA' in o['type']])
            st.metric("🚀 Gamma Setups", gamma_count)
        
        with col2:
            momentum_count = len([o for o in opportunities if 'MOMENTUM' in o['type']])
            st.metric("📈 Momentum Plays", momentum_count)
        
        with col3:
            vol_count = len([o for o in opportunities if 'VOLATILITY' in o['type']])
            st.metric("⚡ Vol Plays", vol_count)
        
        with col4:
            reversal_count = len([o for o in opportunities if 'REVERSAL' in o['type']])
            st.metric("🔄 Reversals", reversal_count)
        
        st.markdown("---")
        
        # Display each opportunity
        for idx, opp in enumerate(opportunities, 1):
            display_opportunity(opp, idx)
        
    else:
        # Initial state - show instructions
        st.info("👈 Click **SCAN NOW** in the sidebar to find trade opportunities")
        
        st.markdown("""
        ### What This Scanner Detects:
        
        **🚀 Gamma Squeeze Setups**
        - Heavy call OI above current price
        - Potential for explosive upside moves
        - Dealer hedging pressure
        
        **📈 Momentum Flow Plays**
        - Directional conviction in options flow
        - Large premium flows (>$500K)
        - Bullish or bearish bias
        
        **⚡ Volatility Expansion Plays**
        - Low IV relative to potential
        - Straddle/strangle opportunities
        - Pre-catalyst positioning
        
        **🔄 Reversal Setups**
        - Unusual institutional positioning
        - High volume relative to OI
        - Smart money divergence
        
        ### How to Use:
        1. Select your watchlist (default or custom)
        2. Choose which setup types to scan for
        3. Set minimum confidence level
        4. Click **SCAN NOW**
        5. Review ranked opportunities with trade suggestions
        """)

if __name__ == "__main__":
    main()
