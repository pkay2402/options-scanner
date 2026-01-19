#!/usr/bin/env python3
"""
Options-Based Price Predictor
Uses Greeks, OI, Volume, and Flow to predict price targets by expiration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cached_client import get_client


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
            # Sort by date
            expiries.sort(key=lambda x: x['date'])
            return expiries
    except:
        pass
    return []

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
    .prediction-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #3b82f6;
        text-align: center;
    }
    
    .prediction-label {
        color: #93c5fd;
        font-size: 0.9em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-value {
        color: #ffffff;
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .prediction-range {
        color: #60a5fa;
        font-size: 1.1em;
    }
    
    .signal-bullish {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border-color: #10b981;
    }
    
    .signal-bearish {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border-color: #ef4444;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border-color: #6b7280;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #374151;
    }
    
    .confidence-high { color: #10b981; }
    .confidence-medium { color: #f59e0b; }
    .confidence-low { color: #ef4444; }
    
    .gex-positive { background-color: rgba(16, 185, 129, 0.2); }
    .gex-negative { background-color: rgba(239, 68, 68, 0.2); }
</style>
""", unsafe_allow_html=True)


class OptionsPricePredictor:
    """
    Predicts stock price using multiple options-based signals:
    1. Gamma Exposure (GEX) - Price magnetism/volatility zones
    2. Max Pain - Expiration price target
    3. Implied Move - Expected range from ATM straddle
    4. Delta-Weighted OI - Directional pressure
    5. Volume Flow Analysis - Smart money direction
    6. Put/Call Skew - Sentiment indicator
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
            # Get options chain - use smaller strike count to reduce timeout risk
            self.chain_data = self.client.get_options_chain(
                symbol=self.symbol,
                contract_type='ALL',
                strike_count=50  # Reduced to improve speed
            )
            
            if not self.chain_data:
                st.error(f"Could not fetch options data for {self.symbol}. API may be slow - try again.")
                return False
            
            if self.chain_data.get('status') != 'SUCCESS':
                st.error(f"Options chain error: {self.chain_data.get('status', 'Unknown')}")
                return False
            
            self.underlying_price = self.chain_data.get('underlyingPrice', 0)
            
            # Parse into DataFrames
            self._parse_chain_data()
            
            return True
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False
    
    def _parse_chain_data(self):
        """Parse options chain into structured DataFrames"""
        calls_list = []
        puts_list = []
        
        # Parse calls
        for exp_date, strikes in self.chain_data.get('callExpDateMap', {}).items():
            exp_key = exp_date.split(':')[0]
            
            # Filter by expiry if specified
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
                        'theta': c.get('theta', 0),
                        'vega': c.get('vega', 0),
                        'iv': c.get('volatility', 0),
                        'itm': c.get('inTheMoney', False),
                        'intrinsic': c.get('intrinsicValue', 0),
                        'extrinsic': c.get('extrinsicValue', 0),
                        'type': 'CALL'
                    })
        
        # Parse puts
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
                        'theta': c.get('theta', 0),
                        'vega': c.get('vega', 0),
                        'iv': c.get('volatility', 0),
                        'itm': c.get('inTheMoney', False),
                        'intrinsic': c.get('intrinsicValue', 0),
                        'extrinsic': c.get('extrinsicValue', 0),
                        'type': 'PUT'
                    })
        
        self.calls_df = pd.DataFrame(calls_list)
        self.puts_df = pd.DataFrame(puts_list)
    
    def calculate_gex(self) -> dict:
        """
        Calculate Gamma Exposure (GEX) by strike
        
        GEX = Gamma × OI × 100 × Spot
        
        Market makers are typically:
        - Short calls (retail buys calls) → MM has negative gamma on calls
        - Short puts (retail buys puts) → MM has positive gamma on puts
        
        Net GEX = Put GEX - Call GEX
        
        Positive GEX = Price stabilization (mean reversion)
        Negative GEX = Price acceleration (momentum)
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Use nearest expiry for primary GEX calculation
        nearest_exp = self.calls_df['expiry'].min()
        calls_near = self.calls_df[self.calls_df['expiry'] == nearest_exp]
        puts_near = self.puts_df[self.puts_df['expiry'] == nearest_exp]
        
        gex_data = {}
        
        # Get unique strikes
        all_strikes = sorted(set(calls_near['strike'].tolist() + puts_near['strike'].tolist()))
        
        # Build strike lookup dictionaries for faster access
        call_dict = dict(zip(calls_near['strike'].values, 
                             zip(calls_near['gamma'].fillna(0).values, calls_near['oi'].values)))
        put_dict = dict(zip(puts_near['strike'].values, 
                            zip(puts_near['gamma'].fillna(0).values, puts_near['oi'].values)))
        
        for strike in all_strikes:
            call_gex = 0
            put_gex = 0
            call_oi = 0
            put_oi = 0
            
            if strike in call_dict:
                call_gamma, call_oi = call_dict[strike]
                # MM is typically short calls → negative gamma exposure
                call_gex = -call_gamma * call_oi * 100 * self.underlying_price
            
            if strike in put_dict:
                put_gamma, put_oi = put_dict[strike]
                # MM is typically short puts → positive gamma effect
                put_gex = put_gamma * put_oi * 100 * self.underlying_price
            
            net_gex = call_gex + put_gex
            
            gex_data[strike] = {
                'call_gex': call_gex,
                'put_gex': put_gex,
                'net_gex': net_gex,
                'call_oi': call_oi,
                'put_oi': put_oi
            }
        
        # Find key GEX levels
        sorted_by_gex = sorted(gex_data.items(), key=lambda x: abs(x[1]['net_gex']), reverse=True)
        
        # Total GEX
        total_gex = sum(d['net_gex'] for d in gex_data.values())
        
        # Find zero gamma level (flip point)
        cumulative_gex = 0
        zero_gamma_strike = self.underlying_price
        for strike in sorted(gex_data.keys()):
            cumulative_gex += gex_data[strike]['net_gex']
            if cumulative_gex >= 0:
                zero_gamma_strike = strike
                break
        
        return {
            'by_strike': gex_data,
            'total_gex': total_gex,
            'gex_regime': 'POSITIVE' if total_gex > 0 else 'NEGATIVE',
            'top_strikes': sorted_by_gex[:10],
            'zero_gamma_strike': zero_gamma_strike
        }
    
    def calculate_max_pain(self) -> dict:
        """
        Calculate Max Pain - the strike where option buyers lose the most money
        (where most options expire worthless)
        
        This is often where price gravitates toward expiration
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Use nearest expiry for max pain (most relevant)
        nearest_exp = self.calls_df['expiry'].min()
        calls_near = self.calls_df[self.calls_df['expiry'] == nearest_exp]
        puts_near = self.puts_df[self.puts_df['expiry'] == nearest_exp]
        
        # Get unique strikes
        all_strikes = sorted(set(calls_near['strike'].tolist() + puts_near['strike'].tolist()))
        
        # Vectorized calculation using numpy arrays
        call_strikes = calls_near['strike'].values
        call_oi = calls_near['oi'].values
        put_strikes = puts_near['strike'].values
        put_oi = puts_near['oi'].values
        
        pain_by_strike = {}
        
        for test_price in all_strikes:
            # Call pain: sum where test_price > strike (calls ITM)
            call_mask = test_price > call_strikes
            call_pain = np.sum((test_price - call_strikes[call_mask]) * call_oi[call_mask] * 100)
            
            # Put pain: sum where test_price < strike (puts ITM)
            put_mask = test_price < put_strikes
            put_pain = np.sum((put_strikes[put_mask] - test_price) * put_oi[put_mask] * 100)
            
            pain_by_strike[test_price] = call_pain + put_pain
        
        # Find minimum pain strike
        max_pain_strike = min(pain_by_strike.keys(), key=lambda x: pain_by_strike[x])
        
        return {
            'max_pain_strike': max_pain_strike,
            'pain_by_strike': pain_by_strike,
            'distance_from_current': ((max_pain_strike - self.underlying_price) / self.underlying_price) * 100
        }
    
    def calculate_implied_move(self) -> dict:
        """
        Calculate expected move using ATM straddle
        
        Implied Move = ATM Straddle Price / Underlying Price
        
        This gives the market's expected range by expiration
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        results = {}
        
        # Group by expiry
        for expiry in self.calls_df['expiry'].unique():
            exp_calls = self.calls_df[self.calls_df['expiry'] == expiry]
            exp_puts = self.puts_df[self.puts_df['expiry'] == expiry]
            
            if exp_calls.empty or exp_puts.empty:
                continue
            
            dte = exp_calls['dte'].values[0]
            
            # Find ATM strike (closest to current price)
            atm_strike = exp_calls.iloc[(exp_calls['strike'] - self.underlying_price).abs().argsort()[:1]]['strike'].values[0]
            
            # Get ATM call and put prices
            atm_call = exp_calls[exp_calls['strike'] == atm_strike]['mark'].values
            atm_put = exp_puts[exp_puts['strike'] == atm_strike]['mark'].values
            
            if len(atm_call) > 0 and len(atm_put) > 0:
                straddle_price = atm_call[0] + atm_put[0]
                implied_move_pct = (straddle_price / self.underlying_price) * 100
                
                # Calculate expected range
                upper_bound = self.underlying_price * (1 + straddle_price / self.underlying_price)
                lower_bound = self.underlying_price * (1 - straddle_price / self.underlying_price)
                
                # 1 standard deviation range (68% probability)
                std_move = straddle_price * 0.8  # Approximate
                
                results[expiry] = {
                    'dte': dte,
                    'atm_strike': atm_strike,
                    'straddle_price': straddle_price,
                    'implied_move_pct': implied_move_pct,
                    'upper_bound': upper_bound,
                    'lower_bound': lower_bound,
                    'std_move': std_move,
                    'upper_1std': self.underlying_price + std_move,
                    'lower_1std': self.underlying_price - std_move
                }
        
        return results
    
    def calculate_delta_weighted_oi(self) -> dict:
        """
        Calculate delta-weighted open interest
        
        This shows the net directional exposure:
        - Positive = Bullish bias (more call delta)
        - Negative = Bearish bias (more put delta)
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Call delta is positive, Put delta is negative
        call_delta_oi = (self.calls_df['delta'] * self.calls_df['oi'] * 100).sum()
        put_delta_oi = (self.puts_df['delta'] * self.puts_df['oi'] * 100).sum()  # Already negative
        
        net_delta = call_delta_oi + put_delta_oi
        
        # Normalize by underlying price for comparison
        net_delta_shares = net_delta  # This is equivalent shares
        net_delta_value = net_delta * self.underlying_price
        
        return {
            'call_delta_oi': call_delta_oi,
            'put_delta_oi': put_delta_oi,
            'net_delta': net_delta,
            'net_delta_value': net_delta_value,
            'bias': 'BULLISH' if net_delta > 0 else 'BEARISH',
            'strength': abs(net_delta) / 10000  # Normalized strength
        }
    
    def calculate_volume_flow(self) -> dict:
        """
        Analyze today's volume flow direction
        
        Uses volume * delta to estimate directional flow
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Volume-weighted delta flow
        call_flow = (self.calls_df['volume'] * self.calls_df['delta'] * self.calls_df['mark'] * 100).sum()
        put_flow = (self.puts_df['volume'] * self.puts_df['delta'] * self.puts_df['mark'] * 100).sum()
        
        # Total premium
        call_premium = (self.calls_df['volume'] * self.calls_df['mark'] * 100).sum()
        put_premium = (self.puts_df['volume'] * self.puts_df['mark'] * 100).sum()
        
        net_flow = call_flow + put_flow
        total_premium = call_premium + put_premium
        
        # P/C ratios
        pc_volume = self.puts_df['volume'].sum() / max(self.calls_df['volume'].sum(), 1)
        pc_premium = put_premium / max(call_premium, 1)
        
        return {
            'call_flow': call_flow,
            'put_flow': put_flow,
            'net_flow': net_flow,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'pc_volume': pc_volume,
            'pc_premium': pc_premium,
            'flow_bias': 'BULLISH' if net_flow > 0 else 'BEARISH'
        }
    
    def calculate_oi_walls(self) -> dict:
        """
        Find support/resistance levels based on high OI strikes
        
        High call OI above price = Resistance (MMs sell delta → selling pressure)
        High put OI below price = Support (MMs buy delta → buying pressure)
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Find resistance (call OI above current price)
        calls_above = self.calls_df[self.calls_df['strike'] > self.underlying_price].nlargest(5, 'oi')
        
        resistance_levels = []
        for strike, oi, gamma in zip(calls_above['strike'].values, calls_above['oi'].values, calls_above['gamma'].values):
            if oi > 0:
                resistance_levels.append({
                    'strike': strike,
                    'oi': oi,
                    'gamma': gamma,
                    'distance_pct': ((strike - self.underlying_price) / self.underlying_price) * 100
                })
        
        # Find support (put OI below current price)
        puts_below = self.puts_df[self.puts_df['strike'] < self.underlying_price].nlargest(5, 'oi')
        
        support_levels = []
        for strike, oi, gamma in zip(puts_below['strike'].values, puts_below['oi'].values, puts_below['gamma'].values):
            if oi > 0:
                support_levels.append({
                    'strike': strike,
                    'oi': oi,
                    'gamma': gamma,
                    'distance_pct': ((self.underlying_price - strike) / self.underlying_price) * 100
                })
        
        return {
            'resistance': resistance_levels,
            'support': support_levels,
            'nearest_resistance': resistance_levels[0]['strike'] if resistance_levels else None,
            'nearest_support': support_levels[0]['strike'] if support_levels else None
        }
    
    def calculate_put_call_skew(self) -> dict:
        """
        Analyze put/call IV skew
        
        Higher put IV = Fear/hedging demand
        Higher call IV = Speculation/FOMO
        """
        if self.calls_df.empty or self.puts_df.empty:
            return {}
        
        # Get OTM options for skew analysis
        otm_calls = self.calls_df[self.calls_df['strike'] > self.underlying_price]
        otm_puts = self.puts_df[self.puts_df['strike'] < self.underlying_price]
        
        # Average IV for OTM options (weighted by OI)
        if not otm_calls.empty and otm_calls['oi'].sum() > 0:
            call_iv = np.average(otm_calls['iv'], weights=otm_calls['oi'] + 1)
        else:
            call_iv = self.calls_df['iv'].mean()
        
        if not otm_puts.empty and otm_puts['oi'].sum() > 0:
            put_iv = np.average(otm_puts['iv'], weights=otm_puts['oi'] + 1)
        else:
            put_iv = self.puts_df['iv'].mean()
        
        skew = put_iv - call_iv
        skew_pct = (put_iv / max(call_iv, 0.01) - 1) * 100
        
        return {
            'call_iv': call_iv,
            'put_iv': put_iv,
            'skew': skew,
            'skew_pct': skew_pct,
            'interpretation': 'FEARFUL' if skew_pct > 10 else ('GREEDY' if skew_pct < -10 else 'NEUTRAL')
        }
    
    def generate_prediction(self) -> dict:
        """
        Combine all signals into a comprehensive price prediction
        """
        # Calculate all signals
        gex = self.calculate_gex()
        max_pain = self.calculate_max_pain()
        implied_move = self.calculate_implied_move()
        delta_oi = self.calculate_delta_weighted_oi()
        volume_flow = self.calculate_volume_flow()
        oi_walls = self.calculate_oi_walls()
        skew = self.calculate_put_call_skew()
        
        # Get nearest expiry for primary prediction
        if implied_move:
            nearest_exp = min(implied_move.keys())
            exp_data = implied_move[nearest_exp]
        else:
            exp_data = {}
        
        # Weight signals for prediction
        signals = []
        
        # Signal 1: Max Pain (weight: 30%) - Most reliable for expiration
        if max_pain and max_pain.get('max_pain_strike'):
            mp_strike = max_pain['max_pain_strike']
            distance = abs(max_pain.get('distance_from_current', 0))
            signals.append({
                'name': 'Max Pain',
                'target': mp_strike,
                'weight': 0.30,
                'confidence': 0.8 if distance < 3 else (0.6 if distance < 5 else 0.4),
                'direction': 'BULLISH' if mp_strike > self.underlying_price else 'BEARISH'
            })
        
        # Signal 2: OI Walls (weight: 25%) - Strong support/resistance
        if oi_walls:
            nearest_res = oi_walls.get('nearest_resistance')
            nearest_sup = oi_walls.get('nearest_support')
            
            if nearest_res and nearest_sup:
                # Price likely to stay between walls, bias toward center
                midpoint = (nearest_res + nearest_sup) / 2
                # Determine direction based on where current price is relative to midpoint
                direction = 'BULLISH' if midpoint > self.underlying_price else 'BEARISH'
                signals.append({
                    'name': 'OI Walls',
                    'target': midpoint,
                    'upper_wall': nearest_res,
                    'lower_wall': nearest_sup,
                    'weight': 0.25,
                    'confidence': 0.7,
                    'direction': direction
                })
        
        # Signal 3: Volume Flow (weight: 25%) - Today's activity
        if volume_flow and volume_flow.get('pc_premium'):
            pc_prem = volume_flow['pc_premium']
            # If P/C < 0.7, bullish; > 1.3, bearish
            if pc_prem < 0.7:
                flow_dir = 'BULLISH'
                flow_target = self.underlying_price * 1.01  # 1% up bias
                conf = 0.7
            elif pc_prem > 1.3:
                flow_dir = 'BEARISH'
                flow_target = self.underlying_price * 0.99  # 1% down bias
                conf = 0.7
            else:
                flow_dir = 'NEUTRAL'
                flow_target = self.underlying_price
                conf = 0.3
            
            signals.append({
                'name': 'Volume Flow',
                'target': flow_target,
                'weight': 0.25,
                'confidence': conf,
                'direction': flow_dir
            })
        
        # Signal 4: GEX Regime (weight: 20%) - Volatility indicator
        if gex and gex.get('zero_gamma_strike'):
            gex_strike = gex['zero_gamma_strike']
            signals.append({
                'name': 'GEX Flip',
                'target': gex_strike,
                'weight': 0.20,
                'confidence': 0.5,  # GEX is more about volatility than direction
                'direction': 'BULLISH' if gex_strike > self.underlying_price else 'BEARISH',
                'regime': gex.get('gex_regime', 'UNKNOWN')
            })
        
        # Calculate weighted prediction - simpler approach
        if signals:
            # Simple weighted average of targets
            total_weight = sum(s['weight'] for s in signals)
            predicted_price = sum(s['target'] * s['weight'] for s in signals) / total_weight
            
            # Sanity check - predicted price should be within 10% of current
            max_move = self.underlying_price * 0.10
            predicted_price = max(self.underlying_price - max_move, 
                                  min(self.underlying_price + max_move, predicted_price))
        else:
            predicted_price = self.underlying_price
        
        # Calculate confidence score
        bullish_signals = sum(1 for s in signals if s.get('direction') == 'BULLISH')
        bearish_signals = sum(1 for s in signals if s.get('direction') == 'BEARISH')
        signal_agreement = abs(bullish_signals - bearish_signals) / max(len(signals), 1)
        
        avg_confidence = sum(s['confidence'] for s in signals) / max(len(signals), 1)
        overall_confidence = (signal_agreement * 0.5 + avg_confidence * 0.5)
        
        # Determine overall bias
        if bullish_signals > bearish_signals:
            overall_bias = 'BULLISH'
        elif bearish_signals > bullish_signals:
            overall_bias = 'BEARISH'
        else:
            overall_bias = 'NEUTRAL'
        
        return {
            'current_price': self.underlying_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': ((predicted_price - self.underlying_price) / self.underlying_price) * 100,
            'overall_bias': overall_bias,
            'confidence': overall_confidence,
            'confidence_label': 'HIGH' if overall_confidence > 0.6 else ('MEDIUM' if overall_confidence > 0.4 else 'LOW'),
            'signals': signals,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'expected_range': {
                'upper': exp_data.get('upper_bound', self.underlying_price * 1.05),
                'lower': exp_data.get('lower_bound', self.underlying_price * 0.95),
                'implied_move_pct': exp_data.get('implied_move_pct', 5)
            },
            'key_levels': {
                'max_pain': max_pain.get('max_pain_strike'),
                'gex_flip': gex.get('zero_gamma_strike'),
                'resistance': oi_walls.get('nearest_resistance'),
                'support': oi_walls.get('nearest_support')
            },
            'raw_data': {
                'gex': gex,
                'max_pain': max_pain,
                'implied_move': implied_move,
                'delta_oi': delta_oi,
                'volume_flow': volume_flow,
                'oi_walls': oi_walls,
                'skew': skew
            }
        }


def create_prediction_chart(predictor, prediction):
    """Create visualization of prediction with key levels"""
    
    current = prediction['current_price']
    predicted = prediction['predicted_price']
    key_levels = prediction['key_levels']
    exp_range = prediction['expected_range']
    
    fig = go.Figure()
    
    # Add current price marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current],
        mode='markers+text',
        marker=dict(size=20, color='#3b82f6', symbol='diamond'),
        text=[f"Current: ${current:.2f}"],
        textposition='middle right',
        name='Current Price'
    ))
    
    # Add predicted price
    fig.add_trace(go.Scatter(
        x=[1],
        y=[predicted],
        mode='markers+text',
        marker=dict(size=25, color='#10b981' if predicted > current else '#ef4444', symbol='star'),
        text=[f"Predicted: ${predicted:.2f}"],
        textposition='middle right',
        name='Predicted Price'
    ))
    
    # Add expected range box
    fig.add_shape(
        type="rect",
        x0=-0.2, x1=1.2,
        y0=exp_range['lower'], y1=exp_range['upper'],
        fillcolor="rgba(59, 130, 246, 0.1)",
        line=dict(color="rgba(59, 130, 246, 0.5)", width=2, dash="dash"),
    )
    
    # Add key levels
    level_colors = {
        'max_pain': '#f59e0b',
        'gex_flip': '#8b5cf6',
        'resistance': '#ef4444',
        'support': '#10b981'
    }
    
    level_names = {
        'max_pain': 'Max Pain',
        'gex_flip': 'GEX Flip',
        'resistance': 'Resistance',
        'support': 'Support'
    }
    
    for level_key, level_value in key_levels.items():
        if level_value:
            fig.add_hline(
                y=level_value,
                line_dash="dot",
                line_color=level_colors.get(level_key, '#6b7280'),
                annotation_text=f"{level_names.get(level_key, level_key)}: ${level_value:.2f}",
                annotation_position="right",
                opacity=0.7
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Price Prediction with Key Levels',
            font=dict(size=20, color='white')
        ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, 1.5]
        ),
        yaxis=dict(
            title='Price ($)',
            gridcolor='#374151'
        ),
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117'
    )
    
    return fig


def create_gex_chart(gex_data, underlying_price):
    """Create GEX by strike visualization"""
    
    if not gex_data or 'by_strike' not in gex_data:
        return None
    
    strikes = []
    net_gex = []
    call_gex = []
    put_gex = []
    
    # Filter to strikes near current price (within 10%)
    for strike, data in gex_data['by_strike'].items():
        if abs(strike - underlying_price) / underlying_price < 0.10:
            strikes.append(strike)
            net_gex.append(data['net_gex'])
            call_gex.append(data['call_gex'])
            put_gex.append(data['put_gex'])
    
    if not strikes:
        return None
    
    # Sort by strike
    sorted_data = sorted(zip(strikes, net_gex, call_gex, put_gex))
    strikes, net_gex, call_gex, put_gex = zip(*sorted_data)
    
    fig = go.Figure()
    
    # Color bars by positive/negative
    colors = ['#10b981' if g > 0 else '#ef4444' for g in net_gex]
    
    fig.add_trace(go.Bar(
        x=strikes,
        y=net_gex,
        marker_color=colors,
        name='Net GEX',
        text=[f"${abs(g)/1000000:.1f}M" for g in net_gex],
        textposition='outside'
    ))
    
    # Add current price line
    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Current: ${underlying_price:.2f}"
    )
    
    fig.update_layout(
        title='Gamma Exposure (GEX) by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Net GEX ($)',
        template='plotly_dark',
        height=350,
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117'
    )
    
    return fig


def create_oi_profile_chart(predictor, underlying_price):
    """Create open interest profile chart"""
    
    calls_df = predictor.calls_df
    puts_df = predictor.puts_df
    
    if calls_df.empty or puts_df.empty:
        return None
    
    # Filter to strikes within 10% of current price
    calls_filtered = calls_df[abs(calls_df['strike'] - underlying_price) / underlying_price < 0.10]
    puts_filtered = puts_df[abs(puts_df['strike'] - underlying_price) / underlying_price < 0.10]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Call Open Interest', 'Put Open Interest'))
    
    # Call OI
    fig.add_trace(go.Bar(
        y=calls_filtered['strike'],
        x=calls_filtered['oi'],
        orientation='h',
        marker_color='#10b981',
        name='Call OI'
    ), row=1, col=1)
    
    # Put OI
    fig.add_trace(go.Bar(
        y=puts_filtered['strike'],
        x=puts_filtered['oi'],
        orientation='h',
        marker_color='#ef4444',
        name='Put OI'
    ), row=1, col=2)
    
    # Add current price lines
    fig.add_hline(y=underlying_price, line_dash="dash", line_color="yellow", row=1, col=1)
    fig.add_hline(y=underlying_price, line_dash="dash", line_color="yellow", row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False,
        plot_bgcolor='#1e2329',
        paper_bgcolor='#0e1117'
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.title("🔮 Options-Based Price Predictor")
    st.caption("Predicting stock price targets using options Greeks, volume, and positioning data")
    
    # === SETTINGS SECTION AT TOP ===
    with st.container():
        st.markdown("### ⚙️ Settings")
        col_sym, col_exp, col_refresh = st.columns([2, 3, 1])
        
        with col_sym:
            symbol = st.text_input(
                "Symbol",
                value=st.session_state.get('predictor_symbol', 'SPY'),
                help="Enter stock symbol",
                label_visibility="collapsed",
                placeholder="Enter Symbol (e.g., SPY)"
            ).upper().strip()
            st.session_state['predictor_symbol'] = symbol
        
        # Get available expiries for the symbol
        expiries = []
        selected_expiry = None
        
        if symbol:
            expiries = get_available_expiries(symbol)
        
        with col_exp:
            if expiries:
                expiry_options = [f"{e['date']} ({e['dte']} DTE)" for e in expiries]
                selected_idx = st.selectbox(
                    "Expiry",
                    range(len(expiry_options)),
                    format_func=lambda x: expiry_options[x],
                    help="Select expiration date for analysis",
                    label_visibility="collapsed"
                )
                selected_expiry = expiries[selected_idx]['date'] if expiries else None
            else:
                st.selectbox("Expiry", ["Enter symbol first"], disabled=True, label_visibility="collapsed")
        
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True, help="Clear cache and refresh data"):
                st.cache_data.clear()
                st.rerun()
    
    st.divider()
    
    # Sidebar for info
    with st.sidebar:
        st.header("📊 Signals Used")
        
        st.markdown("""
        1. **Max Pain** - Strike where most options expire worthless
        2. **GEX** - Gamma exposure & volatility regime
        3. **OI Walls** - Support/resistance from open interest
        4. **Volume Flow** - Directional premium flow
        5. **Delta OI** - Net positioning bias
        6. **IV Skew** - Fear/greed indicator
        """)
        
        st.divider()
        
        st.info("""
        ⚠️ **Note:** Options data is dynamic and changes constantly. 
        Predictions are based on current positioning.
        """)
    
    if not symbol:
        st.warning("Enter a symbol to generate prediction")
        return
    
    # Initialize predictor with selected expiry
    with st.spinner(f"Analyzing {symbol} options data{f' for {selected_expiry}' if selected_expiry else ''}..."):
        predictor = OptionsPricePredictor(symbol, expiry_filter=selected_expiry)
        
        if not predictor.fetch_data():
            col_err1, col_err2 = st.columns([3, 1])
            with col_err1:
                st.error("Failed to fetch options data. The API may be slow or timing out.")
            with col_err2:
                if st.button("🔄 Retry", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            return
        
        prediction = predictor.generate_prediction()
    
    # Display main prediction with expiry info
    expiry_display = f" (Exp: {selected_expiry})" if selected_expiry else ""
    st.header(f"📈 {symbol} Price Prediction{expiry_display}")
    
    # Current price and prediction cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Current Price</div>
            <div class="prediction-value">${prediction['current_price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        bias_class = prediction['overall_bias'].lower()
        change_pct = prediction['predicted_change_pct']
        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
        
        st.markdown(f"""
        <div class="prediction-card signal-{bias_class}">
            <div class="prediction-label">Predicted Price</div>
            <div class="prediction-value">${prediction['predicted_price']:.2f}</div>
            <div class="prediction-range">{arrow} {change_pct:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conf_class = prediction['confidence_label'].lower()
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Confidence</div>
            <div class="prediction-value confidence-{conf_class}">{prediction['confidence']:.0%}</div>
            <div class="prediction-range">{prediction['confidence_label']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Expected range
    exp_range = prediction['expected_range']
    expiry_label = selected_expiry if selected_expiry else "Nearest Expiry"
    st.info(f"""
    **Expected Range by {expiry_label}:** ${exp_range['lower']:.2f} - ${exp_range['upper']:.2f} 
    (±{exp_range['implied_move_pct']:.1f}% implied move)
    """)
    
    # Signal breakdown
    st.subheader("🎯 Signal Breakdown")
    
    signals = prediction['signals']
    num_cols = min(len(signals), 4)  # Max 4 columns
    cols = st.columns(num_cols)
    
    for i, signal in enumerate(signals):
        with cols[i % num_cols]:
            direction = signal.get('direction', 'NEUTRAL')
            color = '#10b981' if direction == 'BULLISH' else ('#ef4444' if direction == 'BEARISH' else '#6b7280')
            
            st.markdown(f"""
            <div style="background: #1e2329; padding: 15px; border-radius: 10px; border-left: 4px solid {color}; margin-bottom: 10px;">
                <strong>{signal['name']}</strong><br>
                <span style="font-size: 1.3em; color: {color};">${signal['target']:.2f}</span><br>
                <small>Weight: {signal['weight']:.0%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📊 Price Levels")
        fig = create_prediction_chart(predictor, prediction)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("⚡ Gamma Exposure")
        gex_fig = create_gex_chart(prediction['raw_data']['gex'], prediction['current_price'])
        if gex_fig:
            st.plotly_chart(gex_fig, use_container_width=True)
        else:
            st.warning("Insufficient GEX data")
    
    # OI Profile
    st.subheader("📈 Open Interest Profile")
    oi_fig = create_oi_profile_chart(predictor, prediction['current_price'])
    if oi_fig:
        st.plotly_chart(oi_fig, use_container_width=True)
    
    # Key Levels Table
    st.subheader("🎯 Key Price Levels")
    
    key_levels = prediction['key_levels']
    raw = prediction['raw_data']
    
    levels_data = []
    
    if key_levels.get('support'):
        levels_data.append({
            'Level': 'Support',
            'Price': f"${key_levels['support']:.2f}",
            'Distance': f"{((prediction['current_price'] - key_levels['support']) / prediction['current_price'] * 100):.1f}%",
            'Type': '🟢 Put Wall'
        })
    
    if key_levels.get('max_pain'):
        levels_data.append({
            'Level': 'Max Pain',
            'Price': f"${key_levels['max_pain']:.2f}",
            'Distance': f"{((key_levels['max_pain'] - prediction['current_price']) / prediction['current_price'] * 100):+.1f}%",
            'Type': '🟡 Expiry Target'
        })
    
    if key_levels.get('gex_flip'):
        levels_data.append({
            'Level': 'GEX Flip',
            'Price': f"${key_levels['gex_flip']:.2f}",
            'Distance': f"{((key_levels['gex_flip'] - prediction['current_price']) / prediction['current_price'] * 100):+.1f}%",
            'Type': '🟣 Volatility Zone'
        })
    
    if key_levels.get('resistance'):
        levels_data.append({
            'Level': 'Resistance',
            'Price': f"${key_levels['resistance']:.2f}",
            'Distance': f"{((key_levels['resistance'] - prediction['current_price']) / prediction['current_price'] * 100):+.1f}%",
            'Type': '🔴 Call Wall'
        })
    
    if levels_data:
        st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)
    
    # Detailed Analysis Expander
    with st.expander("📊 Detailed Analysis"):
        tab1, tab2, tab3, tab4 = st.tabs(["GEX Analysis", "Volume Flow", "Skew", "Raw Data"])
        
        with tab1:
            gex = raw['gex']
            st.markdown(f"""
            ### Gamma Exposure Analysis
            
            - **GEX Regime:** {gex.get('gex_regime', 'N/A')}
            - **Total GEX:** ${gex.get('total_gex', 0)/1000000:.2f}M
            - **Zero Gamma Strike:** ${gex.get('zero_gamma_strike', 0):.2f}
            
            **Interpretation:**
            - {'✅ Positive GEX = Mean reversion likely (price stabilization)' if gex.get('gex_regime') == 'POSITIVE' else '⚠️ Negative GEX = Momentum likely (price acceleration)'}
            """)
        
        with tab2:
            flow = raw['volume_flow']
            st.markdown(f"""
            ### Volume Flow Analysis
            
            - **Call Premium:** ${flow.get('call_premium', 0)/1000000:.2f}M
            - **Put Premium:** ${flow.get('put_premium', 0)/1000000:.2f}M
            - **P/C Volume Ratio:** {flow.get('pc_volume', 0):.2f}
            - **P/C Premium Ratio:** {flow.get('pc_premium', 0):.2f}
            - **Flow Bias:** {flow.get('flow_bias', 'N/A')}
            """)
        
        with tab3:
            skew = raw['skew']
            st.markdown(f"""
            ### IV Skew Analysis
            
            - **Call IV:** {skew.get('call_iv', 0):.1f}%
            - **Put IV:** {skew.get('put_iv', 0):.1f}%
            - **Skew:** {skew.get('skew_pct', 0):+.1f}%
            - **Interpretation:** {skew.get('interpretation', 'N/A')}
            
            **Meaning:**
            - Put IV > Call IV = Fear/hedging demand
            - Call IV > Put IV = Speculation/FOMO
            """)
        
        with tab4:
            st.json(prediction)
    
    # Disclaimer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This prediction model uses options market data which is constantly changing. 
    Predictions are based on current positioning and may shift rapidly with market conditions. 
    This is for educational/informational purposes only and should not be used as sole basis for trading decisions.
    Past options patterns do not guarantee future price movements.
    """)


if __name__ == "__main__":
    main()
