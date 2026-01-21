#!/usr/bin/env python3
"""
Options-Based Price Analysis & Probability Zones
Uses Greeks, OI, Volume, and Flow to identify probability ranges and key levels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client


# Page config
st.set_page_config(
    page_title="Price Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .zone-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }
    
    .zone-bullish { border-left-color: #10b981; }
    .zone-bearish { border-left-color: #ef4444; }
    .zone-neutral { border-left-color: #6b7280; }
    
    .prob-high { color: #10b981; font-weight: bold; }
    .prob-medium { color: #f59e0b; }
    .prob-low { color: #94a3b8; }
    
    .metric-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .bias-meter {
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(90deg, #ef4444 0%, #6b7280 50%, #10b981 100%);
        position: relative;
        margin: 10px 0;
    }
    
    .level-support { color: #10b981; }
    .level-resistance { color: #ef4444; }
    .level-neutral { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_available_expiries(symbol: str) -> list:
    """Get available expiration dates for a symbol"""
    try:
        client = get_client()
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=5)
        if chain and chain.get('status') == 'SUCCESS':
            expiries = []
            for exp_date in chain.get('callExpDateMap', {}).keys():
                exp_key = exp_date.split(':')[0]
                dte = int(exp_date.split(':')[1]) if ':' in exp_date else 0
                expiries.append({'date': exp_key, 'dte': dte})
            expiries.sort(key=lambda x: x['date'])
            return expiries
    except:
        pass
    return []


class OptionsAnalyzer:
    """
    Analyzes options data to produce:
    1. Probability zones based on implied volatility
    2. Key support/resistance levels from OI
    3. Directional bias score from multiple signals
    4. Volatility regime from GEX
    """
    
    def __init__(self, symbol: str, expiry_filter: str = None):
        self.symbol = symbol.upper()
        self.expiry_filter = expiry_filter
        self.client = get_client()
        self.chain_data = None
        self.underlying_price = 0
        self.calls_df = None
        self.puts_df = None
        
    def fetch_data(self) -> bool:
        """Fetch options chain data"""
        try:
            self.chain_data = self.client.get_options_chain(
                symbol=self.symbol,
                contract_type='ALL',
                strike_count=50
            )
            
            if not self.chain_data or self.chain_data.get('status') != 'SUCCESS':
                return False
            
            self.underlying_price = self.chain_data.get('underlyingPrice', 0)
            self._parse_chain_data()
            return True
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False
    
    def _parse_chain_data(self):
        """Parse options chain into DataFrames"""
        calls_list = []
        puts_list = []
        
        for exp_date, strikes in self.chain_data.get('callExpDateMap', {}).items():
            exp_key = exp_date.split(':')[0]
            if self.expiry_filter and exp_key != self.expiry_filter:
                continue
                
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    calls_list.append({
                        'expiry': exp_key,
                        'dte': c.get('daysToExpiration', 0),
                        'strike': float(strike_str),
                        'bid': c.get('bid', 0),
                        'ask': c.get('ask', 0),
                        'mark': c.get('mark', 0),
                        'volume': c.get('totalVolume', 0),
                        'oi': c.get('openInterest', 0),
                        'delta': c.get('delta', 0),
                        'gamma': c.get('gamma', 0),
                        'iv': c.get('volatility', 0),
                        'type': 'CALL'
                    })
        
        for exp_date, strikes in self.chain_data.get('putExpDateMap', {}).items():
            exp_key = exp_date.split(':')[0]
            if self.expiry_filter and exp_key != self.expiry_filter:
                continue
                
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    puts_list.append({
                        'expiry': exp_key,
                        'dte': c.get('daysToExpiration', 0),
                        'strike': float(strike_str),
                        'bid': c.get('bid', 0),
                        'ask': c.get('ask', 0),
                        'mark': c.get('mark', 0),
                        'volume': c.get('totalVolume', 0),
                        'oi': c.get('openInterest', 0),
                        'delta': c.get('delta', 0),
                        'gamma': c.get('gamma', 0),
                        'iv': c.get('volatility', 0),
                        'type': 'PUT'
                    })
        
        self.calls_df = pd.DataFrame(calls_list)
        self.puts_df = pd.DataFrame(puts_list)
    
    def calculate_probability_zones(self) -> dict:
        """
        Calculate probability zones using implied volatility
        
        This is the MOST statistically valid prediction:
        - Uses ATM IV to calculate expected move
        - Returns 1σ (68%), 1.5σ (87%), 2σ (95%) probability zones
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Get nearest expiry
        nearest_exp = self.calls_df['expiry'].min()
        exp_calls = self.calls_df[self.calls_df['expiry'] == nearest_exp]
        exp_puts = self.puts_df[self.puts_df['expiry'] == nearest_exp]
        
        if exp_calls.empty:
            return {}
        
        dte = exp_calls['dte'].values[0]
        
        # Find ATM strike
        atm_idx = (exp_calls['strike'] - self.underlying_price).abs().argsort()[:1]
        atm_strike = exp_calls.iloc[atm_idx]['strike'].values[0]
        
        # Get ATM IV (average of call and put)
        atm_call_iv = exp_calls[exp_calls['strike'] == atm_strike]['iv'].values
        atm_put_iv = exp_puts[exp_puts['strike'] == atm_strike]['iv'].values
        
        if len(atm_call_iv) == 0 or len(atm_put_iv) == 0:
            return {}
        
        atm_iv = (atm_call_iv[0] + atm_put_iv[0]) / 2 / 100  # Convert to decimal
        
        # Calculate expected move using IV
        # IV is annualized, so we need to adjust for DTE
        # Expected Move = Price × IV × sqrt(DTE/365)
        time_factor = np.sqrt(max(dte, 1) / 365)
        expected_move = self.underlying_price * atm_iv * time_factor
        
        # Calculate straddle price for comparison
        atm_call_price = exp_calls[exp_calls['strike'] == atm_strike]['mark'].values[0]
        atm_put_price = exp_puts[exp_puts['strike'] == atm_strike]['mark'].values[0]
        straddle_price = atm_call_price + atm_put_price
        
        # Probability zones
        zones = {
            'current_price': self.underlying_price,
            'atm_strike': atm_strike,
            'atm_iv': atm_iv * 100,
            'dte': dte,
            'expiry': nearest_exp,
            'expected_move': expected_move,
            'expected_move_pct': (expected_move / self.underlying_price) * 100,
            'straddle_price': straddle_price,
            'straddle_implied_move': straddle_price,
            'zones': {
                '1_sigma': {  # 68.2% probability
                    'probability': 68.2,
                    'upper': self.underlying_price + expected_move,
                    'lower': self.underlying_price - expected_move,
                    'label': '68% Range (1σ)'
                },
                '1.5_sigma': {  # 86.6% probability
                    'probability': 86.6,
                    'upper': self.underlying_price + (expected_move * 1.5),
                    'lower': self.underlying_price - (expected_move * 1.5),
                    'label': '87% Range (1.5σ)'
                },
                '2_sigma': {  # 95.4% probability
                    'probability': 95.4,
                    'upper': self.underlying_price + (expected_move * 2),
                    'lower': self.underlying_price - (expected_move * 2),
                    'label': '95% Range (2σ)'
                }
            }
        }
        
        return zones
    
    def calculate_key_levels(self) -> dict:
        """
        Identify key support/resistance levels from OI
        
        These are NOT price targets, but levels where:
        - High call OI above price = potential resistance (MM hedging)
        - High put OI below price = potential support (MM hedging)
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Use nearest expiry
        nearest_exp = self.calls_df['expiry'].min()
        calls_near = self.calls_df[self.calls_df['expiry'] == nearest_exp]
        puts_near = self.puts_df[self.puts_df['expiry'] == nearest_exp]
        
        # Resistance levels (call OI above current price)
        calls_above = calls_near[calls_near['strike'] > self.underlying_price]
        calls_above = calls_above.nlargest(5, 'oi')
        
        resistance_levels = []
        for _, row in calls_above.iterrows():
            if row['oi'] > 0:
                resistance_levels.append({
                    'strike': row['strike'],
                    'oi': row['oi'],
                    'gamma': row['gamma'],
                    'distance_pct': ((row['strike'] - self.underlying_price) / self.underlying_price) * 100,
                    'strength': 'STRONG' if row['oi'] > calls_near['oi'].quantile(0.9) else 'MODERATE'
                })
        
        # Support levels (put OI below current price)
        puts_below = puts_near[puts_near['strike'] < self.underlying_price]
        puts_below = puts_below.nlargest(5, 'oi')
        
        support_levels = []
        for _, row in puts_below.iterrows():
            if row['oi'] > 0:
                support_levels.append({
                    'strike': row['strike'],
                    'oi': row['oi'],
                    'gamma': row['gamma'],
                    'distance_pct': ((self.underlying_price - row['strike']) / self.underlying_price) * 100,
                    'strength': 'STRONG' if row['oi'] > puts_near['oi'].quantile(0.9) else 'MODERATE'
                })
        
        # Calculate max pain
        max_pain = self._calculate_max_pain(calls_near, puts_near)
        
        return {
            'resistance': sorted(resistance_levels, key=lambda x: x['strike']),
            'support': sorted(support_levels, key=lambda x: x['strike'], reverse=True),
            'max_pain': max_pain,
            'nearest_resistance': resistance_levels[0]['strike'] if resistance_levels else None,
            'nearest_support': support_levels[0]['strike'] if support_levels else None
        }
    
    def _calculate_max_pain(self, calls_df, puts_df) -> dict:
        """Calculate max pain strike"""
        all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
        
        call_strikes = calls_df['strike'].values
        call_oi = calls_df['oi'].values
        put_strikes = puts_df['strike'].values
        put_oi = puts_df['oi'].values
        
        pain_by_strike = {}
        
        for test_price in all_strikes:
            call_mask = test_price > call_strikes
            call_pain = np.sum((test_price - call_strikes[call_mask]) * call_oi[call_mask] * 100)
            
            put_mask = test_price < put_strikes
            put_pain = np.sum((put_strikes[put_mask] - test_price) * put_oi[put_mask] * 100)
            
            pain_by_strike[test_price] = call_pain + put_pain
        
        max_pain_strike = min(pain_by_strike.keys(), key=lambda x: pain_by_strike[x])
        distance_pct = ((max_pain_strike - self.underlying_price) / self.underlying_price) * 100
        
        return {
            'strike': max_pain_strike,
            'distance_pct': distance_pct,
            'pain_by_strike': pain_by_strike
        }
    
    def calculate_directional_bias(self) -> dict:
        """
        Calculate directional bias score from multiple signals
        
        Returns a score from -100 (extremely bearish) to +100 (extremely bullish)
        This is NOT a price target, but indicates market positioning
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        signals = []
        
        # Signal 1: Put/Call OI Ratio
        total_call_oi = self.calls_df['oi'].sum()
        total_put_oi = self.puts_df['oi'].sum()
        pc_oi_ratio = total_put_oi / max(total_call_oi, 1)
        
        # P/C < 0.7 = bullish, > 1.0 = bearish
        if pc_oi_ratio < 0.7:
            oi_signal = min(50, (0.7 - pc_oi_ratio) * 100)
        elif pc_oi_ratio > 1.0:
            oi_signal = max(-50, (1.0 - pc_oi_ratio) * 50)
        else:
            oi_signal = 0
        
        signals.append({'name': 'P/C OI Ratio', 'value': oi_signal, 'raw': pc_oi_ratio})
        
        # Signal 2: Put/Call Volume Ratio (today's activity)
        total_call_vol = self.calls_df['volume'].sum()
        total_put_vol = self.puts_df['volume'].sum()
        pc_vol_ratio = total_put_vol / max(total_call_vol, 1)
        
        if pc_vol_ratio < 0.7:
            vol_signal = min(50, (0.7 - pc_vol_ratio) * 100)
        elif pc_vol_ratio > 1.0:
            vol_signal = max(-50, (1.0 - pc_vol_ratio) * 50)
        else:
            vol_signal = 0
        
        signals.append({'name': 'P/C Volume', 'value': vol_signal, 'raw': pc_vol_ratio})
        
        # Signal 3: Delta-weighted OI (net positioning)
        call_delta_oi = (self.calls_df['delta'] * self.calls_df['oi']).sum()
        put_delta_oi = (self.puts_df['delta'] * self.puts_df['oi']).sum()  # Already negative
        net_delta = call_delta_oi + put_delta_oi
        
        # Normalize to -50 to +50
        total_oi = total_call_oi + total_put_oi
        if total_oi > 0:
            delta_signal = (net_delta / total_oi) * 100
            delta_signal = max(-50, min(50, delta_signal))
        else:
            delta_signal = 0
        
        signals.append({'name': 'Net Delta', 'value': delta_signal, 'raw': net_delta})
        
        # Signal 4: Premium Flow (volume × mark)
        call_premium = (self.calls_df['volume'] * self.calls_df['mark']).sum()
        put_premium = (self.puts_df['volume'] * self.puts_df['mark']).sum()
        total_premium = call_premium + put_premium
        
        if total_premium > 0:
            premium_ratio = call_premium / total_premium
            premium_signal = (premium_ratio - 0.5) * 100  # 0.5 is neutral
            premium_signal = max(-50, min(50, premium_signal))
        else:
            premium_signal = 0
        
        signals.append({'name': 'Premium Flow', 'value': premium_signal, 'raw': call_premium / max(put_premium, 1)})
        
        # Calculate overall bias score
        total_score = sum(s['value'] for s in signals)
        # Normalize to -100 to +100
        bias_score = max(-100, min(100, total_score))
        
        # Determine bias label
        if bias_score > 30:
            bias_label = 'BULLISH'
            bias_strength = 'Strong' if bias_score > 60 else 'Moderate'
        elif bias_score < -30:
            bias_label = 'BEARISH'
            bias_strength = 'Strong' if bias_score < -60 else 'Moderate'
        else:
            bias_label = 'NEUTRAL'
            bias_strength = 'Weak bias'
        
        return {
            'score': bias_score,
            'label': bias_label,
            'strength': bias_strength,
            'signals': signals,
            'pc_oi_ratio': pc_oi_ratio,
            'pc_vol_ratio': pc_vol_ratio,
            'call_premium': call_premium,
            'put_premium': put_premium
        }
    
    def calculate_gex_regime(self) -> dict:
        """
        Calculate Gamma Exposure regime
        
        This tells us about VOLATILITY, not direction:
        - Positive GEX = Price stabilization (mean reversion likely)
        - Negative GEX = Price acceleration (momentum/trends likely)
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        nearest_exp = self.calls_df['expiry'].min()
        calls_near = self.calls_df[self.calls_df['expiry'] == nearest_exp]
        puts_near = self.puts_df[self.puts_df['expiry'] == nearest_exp]
        
        gex_by_strike = {}
        
        call_dict = dict(zip(calls_near['strike'].values, 
                             zip(calls_near['gamma'].fillna(0).values, calls_near['oi'].values)))
        put_dict = dict(zip(puts_near['strike'].values, 
                            zip(puts_near['gamma'].fillna(0).values, puts_near['oi'].values)))
        
        all_strikes = sorted(set(calls_near['strike'].tolist() + puts_near['strike'].tolist()))
        
        for strike in all_strikes:
            call_gex = 0
            put_gex = 0
            
            if strike in call_dict:
                call_gamma, call_oi = call_dict[strike]
                call_gex = -call_gamma * call_oi * 100 * self.underlying_price
            
            if strike in put_dict:
                put_gamma, put_oi = put_dict[strike]
                put_gex = put_gamma * put_oi * 100 * self.underlying_price
            
            gex_by_strike[strike] = call_gex + put_gex
        
        total_gex = sum(gex_by_strike.values())
        
        # Find GEX flip point (where cumulative GEX crosses zero)
        cumulative = 0
        flip_strike = self.underlying_price
        for strike in sorted(gex_by_strike.keys()):
            cumulative += gex_by_strike[strike]
            if cumulative >= 0 and strike >= self.underlying_price:
                flip_strike = strike
                break
        
        # Find highest GEX strikes (magnetic levels)
        sorted_by_abs_gex = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)
        magnetic_levels = [{'strike': s, 'gex': g} for s, g in sorted_by_abs_gex[:5]]
        
        regime = 'POSITIVE' if total_gex > 0 else 'NEGATIVE'
        
        return {
            'total_gex': total_gex,
            'regime': regime,
            'flip_strike': flip_strike,
            'magnetic_levels': magnetic_levels,
            'by_strike': gex_by_strike,
            'interpretation': 'Mean reversion likely (price stabilizes)' if regime == 'POSITIVE' 
                            else 'Momentum likely (price can accelerate)'
        }
    
    def generate_analysis(self) -> dict:
        """Generate complete analysis"""
        prob_zones = self.calculate_probability_zones()
        key_levels = self.calculate_key_levels()
        bias = self.calculate_directional_bias()
        gex = self.calculate_gex_regime()
        
        return {
            'symbol': self.symbol,
            'current_price': self.underlying_price,
            'probability_zones': prob_zones,
            'key_levels': key_levels,
            'directional_bias': bias,
            'gex_regime': gex,
            'timestamp': datetime.now().isoformat()
        }


def create_probability_chart(analysis):
    """Create probability zones visualization"""
    zones = analysis['probability_zones']
    if not zones:
        return None
    
    current = zones['current_price']
    z = zones['zones']
    
    fig = go.Figure()
    
    # Add probability zones as horizontal bands
    colors = ['rgba(16, 185, 129, 0.3)', 'rgba(59, 130, 246, 0.2)', 'rgba(148, 163, 184, 0.1)']
    zone_keys = ['1_sigma', '1.5_sigma', '2_sigma']
    
    for i, key in enumerate(zone_keys):
        zone = z[key]
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=zone['lower'], y1=zone['upper'],
            fillcolor=colors[i],
            line=dict(width=0),
            layer="below"
        )
        
        # Add zone labels
        fig.add_annotation(
            x=1.05,
            y=(zone['upper'] + zone['lower']) / 2,
            text=f"{zone['label']}<br>\${zone['lower']:.2f} - \${zone['upper']:.2f}",
            showarrow=False,
            font=dict(size=10),
            xanchor='left'
        )
    
    # Current price line
    fig.add_hline(
        y=current,
        line_color="#f59e0b",
        line_width=3,
        annotation_text=f"Current: \${current:.2f}",
        annotation_position="left"
    )
    
    # Key levels
    key_levels = analysis['key_levels']
    
    if key_levels.get('nearest_resistance'):
        fig.add_hline(
            y=key_levels['nearest_resistance'],
            line_color="#ef4444",
            line_dash="dash",
            annotation_text=f"Resistance: \${key_levels['nearest_resistance']:.2f}",
            annotation_position="left"
        )
    
    if key_levels.get('nearest_support'):
        fig.add_hline(
            y=key_levels['nearest_support'],
            line_color="#10b981",
            line_dash="dash",
            annotation_text=f"Support: \${key_levels['nearest_support']:.2f}",
            annotation_position="left"
        )
    
    if key_levels.get('max_pain', {}).get('strike'):
        fig.add_hline(
            y=key_levels['max_pain']['strike'],
            line_color="#8b5cf6",
            line_dash="dot",
            annotation_text=f"Max Pain: \${key_levels['max_pain']['strike']:.2f}",
            annotation_position="left"
        )
    
    fig.update_layout(
        title=f"Probability Zones by {zones['expiry']} ({zones['dte']} DTE)",
        yaxis_title="Price (\$)",
        xaxis=dict(showticklabels=False, showgrid=False, range=[-0.1, 1.3]),
        yaxis=dict(gridcolor='#374151'),
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117',
        margin=dict(r=150)
    )
    
    return fig


def create_gex_chart(gex_data, underlying_price):
    """Create GEX by strike chart"""
    if not gex_data or 'by_strike' not in gex_data:
        return None
    
    # Filter to ±10% of current price
    strikes = []
    gex_values = []
    
    for strike, gex in sorted(gex_data['by_strike'].items()):
        if abs(strike - underlying_price) / underlying_price < 0.10:
            strikes.append(strike)
            gex_values.append(gex)
    
    if not strikes:
        return None
    
    colors = ['#10b981' if g > 0 else '#ef4444' for g in gex_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=strikes,
        y=gex_values,
        marker_color=colors,
        name='Net GEX'
    ))
    
    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Current: \${underlying_price:.2f}"
    )
    
    # Mark flip point
    if gex_data.get('flip_strike'):
        fig.add_vline(
            x=gex_data['flip_strike'],
            line_dash="dot",
            line_color="#8b5cf6",
            annotation_text=f"GEX Flip: \${gex_data['flip_strike']:.2f}"
        )
    
    fig.update_layout(
        title=f"Gamma Exposure by Strike ({gex_data['regime']} Regime)",
        xaxis_title="Strike Price",
        yaxis_title="Net GEX (\$)",
        template='plotly_dark',
        height=350,
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117'
    )
    
    return fig


def create_bias_gauge(bias_score):
    """Create a bias gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bias_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [-100, -30], 'color': '#ef4444'},
                {'range': [-30, 30], 'color': '#6b7280'},
                {'range': [30, 100], 'color': '#10b981'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': bias_score
            }
        },
        title={'text': "Directional Bias Score"}
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=250,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    return fig


def main():
    """Main application"""
    
    st.title("🔮 Options-Based Price Analysis")
    st.caption("Probability zones, key levels, and directional bias from options data")
    
    # Settings
    with st.container():
        col_sym, col_exp, col_refresh = st.columns([2, 3, 1])
        
        with col_sym:
            symbol = st.text_input(
                "Symbol",
                value=st.session_state.get('predictor_symbol', 'SPY'),
                label_visibility="collapsed",
                placeholder="Enter Symbol (e.g., SPY)"
            ).upper().strip()
            st.session_state['predictor_symbol'] = symbol
        
        expiries = get_available_expiries(symbol) if symbol else []
        
        with col_exp:
            if expiries:
                expiry_options = [f"{e['date']} ({e['dte']} DTE)" for e in expiries]
                selected_idx = st.selectbox(
                    "Expiry",
                    range(len(expiry_options)),
                    format_func=lambda x: expiry_options[x],
                    label_visibility="collapsed"
                )
                selected_expiry = expiries[selected_idx]['date']
            else:
                st.selectbox("Expiry", ["Enter symbol first"], disabled=True, label_visibility="collapsed")
                selected_expiry = None
        
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
    
    st.divider()
    
    # Sidebar explanation
    with st.sidebar:
        st.header("📚 How to Use")
        
        st.markdown("""
        ### Probability Zones
        Based on implied volatility, these show where price is **statistically likely** to stay:
        - **68% Zone (1σ)**: Price stays here ~68% of the time
        - **87% Zone (1.5σ)**: ~87% probability
        - **95% Zone (2σ)**: ~95% probability
        
        ### Key Levels
        - **Support**: High put OI creates buying pressure
        - **Resistance**: High call OI creates selling pressure
        - **Max Pain**: Where most options expire worthless
        
        ### Directional Bias
        Score from -100 to +100 based on:
        - Put/Call ratios
        - Net delta positioning
        - Premium flow direction
        
        ### GEX Regime
        - **Positive**: Mean reversion (price stabilizes)
        - **Negative**: Momentum (price can run)
        """)
        
        st.divider()
        
        st.warning("""
        ⚠️ **Important**: These are probability ranges, not predictions. 
        Markets can and do move outside expected ranges.
        """)
    
    if not symbol:
        st.warning("Enter a symbol to begin analysis")
        return
    
    # Run analysis
    with st.spinner(f"Analyzing {symbol}..."):
        analyzer = OptionsAnalyzer(symbol, expiry_filter=selected_expiry)
        
        if not analyzer.fetch_data():
            st.error("Failed to fetch options data. Try again or try a different symbol.")
            return
        
        analysis = analyzer.generate_analysis()
    
    # === MAIN DISPLAY ===
    
    # Current Price & Expected Range
    zones = analysis['probability_zones']
    bias = analysis['directional_bias']
    
    if zones:
        st.header(f"📊 {symbol} @ \${zones['current_price']:.2f}")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "ATM IV",
                f"{zones['atm_iv']:.1f}%",
                help="At-the-money implied volatility"
            )
        
        with col2:
            st.metric(
                "Expected Move",
                f"±\${zones['expected_move']:.2f}",
                f"±{zones['expected_move_pct']:.1f}%",
                help="1 standard deviation move by expiry"
            )
        
        with col3:
            st.metric(
                "Straddle Price",
                f"\${zones['straddle_price']:.2f}",
                help="Cost to buy ATM straddle (market's implied move)"
            )
        
        with col4:
            bias_emoji = "🟢" if bias['label'] == 'BULLISH' else ("🔴" if bias['label'] == 'BEARISH' else "⚪")
            st.metric(
                "Bias",
                f"{bias_emoji} {bias['label']}",
                f"Score: {bias['score']:.0f}",
                help="Directional bias from positioning"
            )
        
        # 68% Range highlight
        z68 = zones['zones']['1_sigma']
        st.success(f"""
        **68% Probability Range by {zones['expiry']}:** 
        \${z68['lower']:.2f} - \${z68['upper']:.2f}
        """)
    
    # Charts row
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        prob_fig = create_probability_chart(analysis)
        if prob_fig:
            st.plotly_chart(prob_fig, use_container_width=True)
    
    with col_chart2:
        bias_fig = create_bias_gauge(bias['score'])
        st.plotly_chart(bias_fig, use_container_width=True)
        
        # Signal breakdown
        st.markdown("**Signal Breakdown:**")
        for signal in bias['signals']:
            direction = "🟢" if signal['value'] > 10 else ("🔴" if signal['value'] < -10 else "⚪")
            st.markdown(f"{direction} **{signal['name']}**: {signal['value']:+.0f} (raw: {signal['raw']:.2f})")
    
    # GEX Analysis
    st.subheader("⚡ Gamma Exposure (Volatility Regime)")
    
    gex = analysis['gex_regime']
    
    col_gex1, col_gex2 = st.columns([2, 1])
    
    with col_gex1:
        gex_fig = create_gex_chart(gex, analysis['current_price'])
        if gex_fig:
            st.plotly_chart(gex_fig, use_container_width=True)
    
    with col_gex2:
        regime_color = "#10b981" if gex['regime'] == 'POSITIVE' else "#ef4444"
        st.markdown(f"""
        <div class="zone-card">
            <h3 style="color: {regime_color}">{gex['regime']} GEX</h3>
            <p><strong>Total GEX:</strong> \${gex['total_gex']/1e9:.2f}B</p>
            <p><strong>GEX Flip:</strong> \${gex['flip_strike']:.2f}</p>
            <p><em>{gex['interpretation']}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Key GEX Levels:**")
        for level in gex['magnetic_levels'][:3]:
            emoji = "🟢" if level['gex'] > 0 else "🔴"
            st.markdown(f"{emoji} \${level['strike']:.2f}: \${level['gex']/1e6:.1f}M")
    
    # Key Levels Table
    st.subheader("🎯 Key Support & Resistance Levels")
    
    key_levels = analysis['key_levels']
    
    col_sup, col_res = st.columns(2)
    
    with col_sup:
        st.markdown("**🟢 Support Levels (Put Walls)**")
        if key_levels['support']:
            for level in key_levels['support'][:5]:
                strength_emoji = "💪" if level['strength'] == 'STRONG' else ""
                st.markdown(f"• **\${level['strike']:.2f}** (-{level['distance_pct']:.1f}%) - OI: {level['oi']:,} {strength_emoji}")
        else:
            st.markdown("_No significant support levels_")
    
    with col_res:
        st.markdown("**🔴 Resistance Levels (Call Walls)**")
        if key_levels['resistance']:
            for level in key_levels['resistance'][:5]:
                strength_emoji = "💪" if level['strength'] == 'STRONG' else ""
                st.markdown(f"• **\${level['strike']:.2f}** (+{level['distance_pct']:.1f}%) - OI: {level['oi']:,} {strength_emoji}")
        else:
            st.markdown("_No significant resistance levels_")
    
    # Max Pain
    if key_levels.get('max_pain'):
        mp = key_levels['max_pain']
        mp_direction = "above" if mp['distance_pct'] > 0 else "below"
        st.info(f"""
        **Max Pain:** \${mp['strike']:.2f} ({abs(mp['distance_pct']):.1f}% {mp_direction} current price)
        
        _Max pain is where most options expire worthless. Price often gravitates toward this level near expiration, 
        but it's more reliable for index options (SPY, QQQ) than individual stocks._
        """)
    
    # Summary
    st.divider()
    st.subheader("📝 Analysis Summary")
    
    # Build summary text
    summary_points = []
    
    # Probability zone summary
    if zones:
        z68 = zones['zones']['1_sigma']
        summary_points.append(f"• **Expected Range:** \${z68['lower']:.2f} - \${z68['upper']:.2f} (68% probability by {zones['expiry']})")
        summary_points.append(f"• **Implied Volatility:** {zones['atm_iv']:.1f}% ATM IV implies ±{zones['expected_move_pct']:.1f}% move")
    
    # Bias summary
    if bias['label'] != 'NEUTRAL':
        summary_points.append(f"• **Directional Bias:** {bias['strength']} {bias['label'].lower()} positioning (score: {bias['score']:.0f})")
    else:
        summary_points.append(f"• **Directional Bias:** Neutral/mixed positioning")
    
    # GEX summary
    summary_points.append(f"• **Volatility Regime:** {gex['regime']} GEX - {gex['interpretation'].lower()}")
    
    # Key levels summary
    if key_levels.get('nearest_support') and key_levels.get('nearest_resistance'):
        summary_points.append(f"• **Trading Range:** Support at \${key_levels['nearest_support']:.2f}, Resistance at \${key_levels['nearest_resistance']:.2f}")
    
    st.markdown("\n".join(summary_points))
    
    # Disclaimer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This analysis uses current options market data which changes constantly. 
    Probability zones represent statistical likelihood, not guaranteed outcomes. 
    Options positioning can shift rapidly. This is for educational purposes only.
    """)


if __name__ == "__main__":
    main()
