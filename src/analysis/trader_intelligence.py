"""
Professional Trader Intelligence Module
Combines multiple data sources to provide actionable trading insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..api.schwab_client import SchwabClient
from ..data.database import DatabaseManager
from .market_dynamics import MarketDynamicsAnalyzer
from .big_trades import BigTradesDetector
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off" 
    ROTATION = "rotation"
    CONSOLIDATION = "consolidation"
    VOLATILITY_EXPANSION = "vol_expansion"
    VOLATILITY_CONTRACTION = "vol_contraction"

class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class SmartMoneyFlow:
    """Tracks institutional money flows"""
    symbol: str
    net_flow: float  # Net dollar flow (positive = inflow)
    call_flow: float
    put_flow: float
    unusual_volume: float
    avg_trade_size: float
    institutional_trades: int
    retail_trades: int
    confidence: float
    timestamp: datetime

@dataclass
class PositioningAnalysis:
    """Dealer and institutional positioning"""
    symbol: str
    dealer_gamma_exposure: float
    dealer_delta_exposure: float
    put_call_skew: float
    term_structure: Dict[str, float]  # IV term structure
    max_pain: float  # Max pain level
    gamma_wall: float  # Largest gamma concentration
    support_levels: List[float]
    resistance_levels: List[float]
    next_expiration_oi: Dict[str, int]

@dataclass
class TradingOpportunity:
    """Specific trading opportunity"""
    symbol: str
    signal: TradingSignal
    strategy: str  # "long_calls", "put_spread", "iron_condor", etc.
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    probability: float
    time_horizon: str  # "intraday", "1-3 days", "1-2 weeks"
    catalyst: str
    position_size: float  # Suggested % of portfolio
    confidence: float

@dataclass
class MarketIntelligence:
    """Complete market intelligence package"""
    timestamp: datetime
    market_regime: MarketRegime
    vix_regime: str
    smart_money_flows: List[SmartMoneyFlow]
    positioning_analysis: List[PositioningAnalysis]
    trading_opportunities: List[TradingOpportunity]
    sector_rotation: Dict[str, float]
    key_events: List[str]
    risk_factors: List[str]
    overnight_setup: str

class TraderIntelligenceEngine:
    """
    Professional trader's intelligence engine
    Combines options flow, positioning, and market data for actionable insights
    """
    
    def __init__(self, schwab_client: SchwabClient = None):
        self.settings = get_settings()
        self.schwab_client = schwab_client or SchwabClient()
        self.db_manager = DatabaseManager()
        self.market_analyzer = MarketDynamicsAnalyzer(self.schwab_client)
        self.big_trades_detector = BigTradesDetector(self.schwab_client)
        
        # Professional watchlists
        self.core_indices = ['SPY', 'QQQ', 'IWM', 'VIX']
        self.mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
        self.sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLRE', 'XLB', 'XLY']
        self.momentum_plays = ['SMCI', 'ARM', 'PLTR', 'SHOP', 'SQ', 'ROKU', 'ZM']
        
    def generate_market_intelligence(self) -> MarketIntelligence:
        """
        Generate comprehensive market intelligence for professional trading
        """
        logger.info("🧠 Generating professional market intelligence...")
        
        # Determine market regime
        market_regime = self._determine_market_regime()
        vix_regime = self._analyze_vix_regime()
        
        # Analyze smart money flows
        smart_money_flows = self._analyze_smart_money_flows()
        
        # Positioning analysis
        positioning_analysis = self._analyze_positioning()
        
        # Generate trading opportunities
        trading_opportunities = self._identify_trading_opportunities(
            market_regime, smart_money_flows, positioning_analysis
        )
        
        # Sector rotation analysis
        sector_rotation = self._analyze_sector_rotation()
        
        # Key events and catalysts
        key_events = self._identify_key_events()
        
        # Risk factors
        risk_factors = self._assess_risk_factors()
        
        # Overnight positioning setup
        overnight_setup = self._generate_overnight_setup()
        
        return MarketIntelligence(
            timestamp=datetime.now(),
            market_regime=market_regime,
            vix_regime=vix_regime,
            smart_money_flows=smart_money_flows,
            positioning_analysis=positioning_analysis,
            trading_opportunities=trading_opportunities,
            sector_rotation=sector_rotation,
            key_events=key_events,
            risk_factors=risk_factors,
            overnight_setup=overnight_setup
        )
    
    def _determine_market_regime(self) -> MarketRegime:
        """Determine current market regime"""
        try:
            # Get VIX data
            vix_data = self.schwab_client.get_quote('VIX')
            vix_level = vix_data.get('VIX', {}).get('lastPrice', 20)
            
            # Get major indices
            spy_data = self.schwab_client.get_quote('SPY')
            spy_price = spy_data.get('SPY', {}).get('lastPrice', 0)
            spy_change = spy_data.get('SPY', {}).get('netChange', 0)
            
            qqq_data = self.schwab_client.get_quote('QQQ')
            qqq_change = qqq_data.get('QQQ', {}).get('netChange', 0)
            
            iwm_data = self.schwab_client.get_quote('IWM')
            iwm_change = iwm_data.get('IWM', {}).get('netChange', 0)
            
            # Regime logic
            if vix_level > 30:
                return MarketRegime.VOLATILITY_EXPANSION
            elif vix_level < 15:
                return MarketRegime.VOLATILITY_CONTRACTION
            elif abs(spy_change) > spy_price * 0.02:  # 2% move
                return MarketRegime.RISK_OFF if spy_change < 0 else MarketRegime.RISK_ON
            elif (qqq_change > 0) != (iwm_change > 0):  # Growth vs Value divergence
                return MarketRegime.ROTATION
            else:
                return MarketRegime.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return MarketRegime.CONSOLIDATION
    
    def _analyze_vix_regime(self) -> str:
        """Analyze VIX regime and implications"""
        try:
            vix_data = self.schwab_client.get_quote('VIX')
            vix_level = vix_data.get('VIX', {}).get('lastPrice', 20)
            
            if vix_level > 35:
                return "CRISIS MODE: Sell premium, hedge long positions"
            elif vix_level > 25:
                return "ELEVATED: Tactical trades, short vol strategies"
            elif vix_level > 20:
                return "NORMAL: Balanced approach, momentum trades"
            elif vix_level > 15:
                return "COMPLACENT: Buy protection, momentum plays"
            else:
                return "EXTREMELY LOW: Buy cheap hedges, expect expansion"
                
        except Exception as e:
            logger.error(f"Error analyzing VIX regime: {e}")
            return "UNKNOWN: Unable to determine VIX regime"
    
    def _analyze_smart_money_flows(self) -> List[SmartMoneyFlow]:
        """Analyze institutional vs retail money flows"""
        smart_flows = []
        
        # Combine all watchlist symbols
        all_symbols = (self.core_indices + self.mega_caps + 
                      self.sector_etfs + self.momentum_plays)
        
        for symbol in all_symbols[:10]:  # Limit for performance
            try:
                # Get big trades
                big_trades = self.big_trades_detector.scan_for_big_trades(
                    [symbol], min_premium=100000  # $100k minimum
                )
                
                if not big_trades:
                    continue
                
                # Calculate flows
                call_flow = sum(t.notional_value for t in big_trades 
                              if 'call' in t.contract_type.lower())
                put_flow = sum(t.notional_value for t in big_trades 
                             if 'put' in t.contract_type.lower())
                net_flow = call_flow - put_flow
                
                # Estimate institutional vs retail
                large_trades = [t for t in big_trades if t.notional_value > 500000]
                institutional_trades = len(large_trades)
                retail_trades = len(big_trades) - institutional_trades
                
                avg_trade_size = np.mean([t.notional_value for t in big_trades])
                unusual_volume = sum(t.volume for t in big_trades)
                
                confidence = min(len(big_trades) / 10.0, 1.0)  # More trades = higher confidence
                
                smart_flows.append(SmartMoneyFlow(
                    symbol=symbol,
                    net_flow=net_flow,
                    call_flow=call_flow,
                    put_flow=put_flow,
                    unusual_volume=unusual_volume,
                    avg_trade_size=avg_trade_size,
                    institutional_trades=institutional_trades,
                    retail_trades=retail_trades,
                    confidence=confidence,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.warning(f"Error analyzing flows for {symbol}: {e}")
                continue
        
        # Sort by net flow magnitude
        smart_flows.sort(key=lambda x: abs(x.net_flow), reverse=True)
        return smart_flows[:10]  # Top 10
    
    def _analyze_positioning(self) -> List[PositioningAnalysis]:
        """Analyze dealer and institutional positioning"""
        positioning = []
        
        for symbol in (self.core_indices + self.mega_caps)[:8]:
            try:
                # Get options chain
                chain_data = self.schwab_client.get_options_chain(
                    symbol=symbol, contract_type="ALL", strike_count=20
                )
                
                # Get current price
                quote = self.schwab_client.get_quote(symbol)
                current_price = quote.get(symbol, {}).get('lastPrice', 0)
                
                if not current_price or not chain_data:
                    continue
                
                # Calculate positioning metrics
                dealer_gamma = self._calculate_dealer_gamma_exposure(chain_data, current_price)
                dealer_delta = self._calculate_dealer_delta_exposure(chain_data, current_price)
                put_call_skew = self._calculate_put_call_skew(chain_data)
                max_pain = self._calculate_max_pain(chain_data)
                gamma_wall = self._find_gamma_wall(chain_data, current_price)
                
                # Support/Resistance from option strikes
                support_levels, resistance_levels = self._identify_key_levels(
                    chain_data, current_price
                )
                
                positioning.append(PositioningAnalysis(
                    symbol=symbol,
                    dealer_gamma_exposure=dealer_gamma,
                    dealer_delta_exposure=dealer_delta,
                    put_call_skew=put_call_skew,
                    term_structure={},  # TODO: Implement
                    max_pain=max_pain,
                    gamma_wall=gamma_wall,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    next_expiration_oi={}  # TODO: Implement
                ))
                
            except Exception as e:
                logger.warning(f"Error analyzing positioning for {symbol}: {e}")
                continue
        
        return positioning
    
    def _identify_trading_opportunities(self, market_regime: MarketRegime, 
                                     smart_flows: List[SmartMoneyFlow],
                                     positioning: List[PositioningAnalysis]) -> List[TradingOpportunity]:
        """Identify specific trading opportunities"""
        opportunities = []
        
        # Create a map for quick lookups
        flow_map = {flow.symbol: flow for flow in smart_flows}
        position_map = {pos.symbol: pos for pos in positioning}
        
        for symbol in (self.core_indices + self.mega_caps)[:10]:
            try:
                # Get current quote
                quote = self.schwab_client.get_quote(symbol)
                current_price = quote.get(symbol, {}).get('lastPrice', 0)
                
                if not current_price:
                    continue
                
                flow = flow_map.get(symbol)
                pos = position_map.get(symbol)
                
                # Generate opportunities based on multiple factors
                if flow and pos:
                    opps = self._generate_symbol_opportunities(
                        symbol, current_price, flow, pos, market_regime
                    )
                    opportunities.extend(opps)
                    
            except Exception as e:
                logger.warning(f"Error generating opportunities for {symbol}: {e}")
                continue
        
        # Sort by confidence * risk_reward
        opportunities.sort(key=lambda x: x.confidence * x.risk_reward, reverse=True)
        return opportunities[:15]  # Top 15 opportunities
    
    def _generate_symbol_opportunities(self, symbol: str, current_price: float,
                                     flow: SmartMoneyFlow, pos: PositioningAnalysis,
                                     regime: MarketRegime) -> List[TradingOpportunity]:
        """Generate trading opportunities for a specific symbol"""
        opportunities = []
        
        # Bullish flow + positive gamma wall = Call spread opportunity
        if (flow.net_flow > 0 and flow.confidence > 0.6 and 
            pos.gamma_wall > current_price):
            
            target = min(pos.gamma_wall, current_price * 1.05)  # 5% max target
            stop = current_price * 0.98  # 2% stop
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                signal=TradingSignal.BUY,
                strategy="CALL_SPREAD",
                entry_price=current_price,
                target_price=target,
                stop_loss=stop,
                risk_reward=(target - current_price) / (current_price - stop),
                probability=0.65,
                time_horizon="1-3 days",
                catalyst=f"Smart money call flow ${flow.call_flow/1e6:.1f}M",
                position_size=0.02,  # 2% of portfolio
                confidence=flow.confidence
            ))
        
        # Bearish flow + negative gamma exposure = Put spread
        if (flow.net_flow < -1000000 and pos.dealer_gamma_exposure < 0):  # $1M+ bearish flow
            
            target = max(pos.max_pain, current_price * 0.95)  # 5% max drop to max pain
            stop = current_price * 1.02  # 2% stop
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                signal=TradingSignal.SELL,
                strategy="PUT_SPREAD",
                entry_price=current_price,
                target_price=target,
                stop_loss=stop,
                risk_reward=(current_price - target) / (stop - current_price),
                probability=0.6,
                time_horizon="1-2 weeks",
                catalyst=f"Large put flow ${abs(flow.put_flow)/1e6:.1f}M",
                position_size=0.015,  # 1.5% of portfolio
                confidence=flow.confidence * 0.9
            ))
        
        # Low VIX + High gamma exposure = Volatility expansion play
        if regime == MarketRegime.VOLATILITY_CONTRACTION and abs(pos.dealer_gamma_exposure) > 1e9:
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                signal=TradingSignal.BUY,
                strategy="LONG_STRADDLE",
                entry_price=current_price,
                target_price=current_price * 1.08,  # 8% move expected
                stop_loss=current_price * 0.92,    # 8% move expected
                risk_reward=1.0,  # Symmetric
                probability=0.4,  # Lower probability but high payoff
                time_horizon="2-4 weeks",
                catalyst="VIX expansion play - low vol unsustainable",
                position_size=0.01,  # 1% of portfolio
                confidence=0.7
            ))
        
        return opportunities
    
    def _analyze_sector_rotation(self) -> Dict[str, float]:
        """Analyze sector rotation patterns"""
        sector_performance = {}
        
        for etf in self.sector_etfs:
            try:
                quote = self.schwab_client.get_quote(etf)
                data = quote.get(etf, {})
                daily_change = data.get('netChange', 0)
                last_price = data.get('lastPrice', 1)
                
                if last_price > 0:
                    pct_change = (daily_change / last_price) * 100
                    sector_performance[etf] = pct_change
                    
            except Exception as e:
                logger.warning(f"Error getting sector data for {etf}: {e}")
                continue
        
        return dict(sorted(sector_performance.items(), key=lambda x: x[1], reverse=True))
    
    def _identify_key_events(self) -> List[str]:
        """Identify key market events and catalysts"""
        events = []
        
        # Check for unusual VIX moves
        try:
            vix_data = self.schwab_client.get_quote('VIX')
            vix_change = vix_data.get('VIX', {}).get('netChange', 0)
            
            if abs(vix_change) > 2:
                direction = "spike" if vix_change > 0 else "crush"
                events.append(f"🔥 VIX {direction}: {vix_change:+.1f} points")
        
        except Exception:
            pass
        
        # Check for major index moves
        try:
            spy_data = self.schwab_client.get_quote('SPY')
            spy_change_pct = (spy_data.get('SPY', {}).get('netChange', 0) / 
                            spy_data.get('SPY', {}).get('lastPrice', 1)) * 100
            
            if abs(spy_change_pct) > 1.5:
                direction = "rally" if spy_change_pct > 0 else "selloff"
                events.append(f"📈 SPY {direction}: {spy_change_pct:+.1f}%")
                
        except Exception:
            pass
        
        return events
    
    def _assess_risk_factors(self) -> List[str]:
        """Assess current risk factors"""
        risks = []
        
        # VIX risk
        try:
            vix_data = self.schwab_client.get_quote('VIX')
            vix_level = vix_data.get('VIX', {}).get('lastPrice', 20)
            
            if vix_level > 30:
                risks.append("⚠️ Elevated volatility - reduce position sizes")
            elif vix_level < 12:
                risks.append("⚠️ Complacency risk - vol expansion likely")
                
        except Exception:
            pass
        
        return risks
    
    def _generate_overnight_setup(self) -> str:
        """Generate overnight positioning guidance"""
        try:
            spy_data = self.schwab_client.get_quote('SPY')
            spy_change = spy_data.get('SPY', {}).get('netChange', 0)
            
            if spy_change > 0:
                return "🌙 OVERNIGHT: Bullish close - watch for gap continuation or fade"
            elif spy_change < 0:
                return "🌙 OVERNIGHT: Bearish close - watch for oversold bounce or breakdown"
            else:
                return "🌙 OVERNIGHT: Flat close - expect low vol continuation"
                
        except Exception:
            return "🌙 OVERNIGHT: Monitor futures for direction"
    
    # Helper methods for positioning calculations
    def _calculate_dealer_gamma_exposure(self, chain_data: Dict, current_price: float) -> float:
        """Calculate dealer gamma exposure"""
        total_gamma_exposure = 0
        
        # Process calls (dealers are short)
        if 'callExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['callExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    for option in options:
                        gamma = option.get('gamma', 0)
                        oi = option.get('openInterest', 0)
                        # Official Professional Net GEX Formula: Γ × 100 × OI × S² × 0.01
                        # Calls: positive contribution to Net GEX
                        total_gamma_exposure += gamma * 100 * oi * current_price * current_price * 0.01
        
        # Process puts 
        if 'putExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['putExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    for option in options:
                        gamma = option.get('gamma', 0)
                        oi = option.get('openInterest', 0)
                        # Official Professional Net GEX Formula: Γ × 100 × OI × S² × 0.01
                        # Puts: negative contribution to Net GEX
                        total_gamma_exposure -= gamma * 100 * oi * current_price * current_price * 0.01
        
        return total_gamma_exposure
    
    def _calculate_dealer_delta_exposure(self, chain_data: Dict, current_price: float) -> float:
        """Calculate dealer delta exposure"""
        total_delta_exposure = 0
        
        # Process calls (dealers are short)
        if 'callExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['callExpDateMap'].items():
                for strike_price, options in strikes.items():
                    for option in options:
                        delta = option.get('delta', 0)
                        oi = option.get('openInterest', 0)
                        total_delta_exposure -= delta * oi * 100
        
        # Process puts (dealers are long)
        if 'putExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['putExpDateMap'].items():
                for strike_price, options in strikes.items():
                    for option in options:
                        delta = option.get('delta', 0)
                        oi = option.get('openInterest', 0)
                        total_delta_exposure += delta * oi * 100
        
        return total_delta_exposure
    
    def _calculate_put_call_skew(self, chain_data: Dict) -> float:
        """Calculate put/call volatility skew"""
        # Simplified - would need ATM options for accurate skew
        return 0.0
    
    def _calculate_max_pain(self, chain_data: Dict) -> float:
        """Calculate max pain level"""
        pain_levels = {}
        
        # Get all strikes and their total OI
        all_strikes = set()
        
        if 'callExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['callExpDateMap'].items():
                all_strikes.update(float(s) for s in strikes.keys())
        
        if 'putExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['putExpDateMap'].items():
                all_strikes.update(float(s) for s in strikes.keys())
        
        # Calculate pain for each strike
        for test_strike in all_strikes:
            total_pain = 0
            
            # Call pain
            if 'callExpDateMap' in chain_data:
                for exp_date, strikes in chain_data['callExpDateMap'].items():
                    for strike_price, options in strikes.items():
                        strike = float(strike_price)
                        if test_strike > strike:  # ITM calls
                            for option in options:
                                oi = option.get('openInterest', 0)
                                total_pain += (test_strike - strike) * oi * 100
            
            # Put pain  
            if 'putExpDateMap' in chain_data:
                for exp_date, strikes in chain_data['putExpDateMap'].items():
                    for strike_price, options in strikes.items():
                        strike = float(strike_price)
                        if test_strike < strike:  # ITM puts
                            for option in options:
                                oi = option.get('openInterest', 0)
                                total_pain += (strike - test_strike) * oi * 100
            
            pain_levels[test_strike] = total_pain
        
        # Return strike with minimum pain
        if pain_levels:
            return min(pain_levels, key=pain_levels.get)
        return 0.0
    
    def _find_gamma_wall(self, chain_data: Dict, current_price: float) -> float:
        """Find the largest gamma concentration (gamma wall)"""
        gamma_by_strike = {}
        
        # Aggregate gamma by strike
        if 'callExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['callExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    
                    for option in options:
                        gamma = option.get('gamma', 0)
                        oi = option.get('openInterest', 0)
                        gamma_by_strike[strike] += gamma * oi
        
        if 'putExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['putExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    
                    for option in options:
                        gamma = option.get('gamma', 0)
                        oi = option.get('openInterest', 0)
                        gamma_by_strike[strike] += gamma * oi
        
        # Find strike with highest gamma
        if gamma_by_strike:
            max_gamma_strike = max(gamma_by_strike, key=gamma_by_strike.get)
            return max_gamma_strike
        
        return current_price
    
    def _identify_key_levels(self, chain_data: Dict, current_price: float) -> Tuple[List[float], List[float]]:
        """Identify key support and resistance levels from options data"""
        oi_by_strike = {}
        
        # Aggregate OI by strike
        if 'callExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['callExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    
                    for option in options:
                        oi = option.get('openInterest', 0)
                        oi_by_strike[strike] += oi
        
        if 'putExpDateMap' in chain_data:
            for exp_date, strikes in chain_data['putExpDateMap'].items():
                for strike_price, options in strikes.items():
                    strike = float(strike_price)
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    
                    for option in options:
                        oi = option.get('openInterest', 0)
                        oi_by_strike[strike] += oi
        
        # Find high OI strikes
        sorted_strikes = sorted(oi_by_strike.items(), key=lambda x: x[1], reverse=True)
        
        support_levels = [strike for strike, oi in sorted_strikes[:3] if strike < current_price]
        resistance_levels = [strike for strike, oi in sorted_strikes[:3] if strike > current_price]
        
        return support_levels[:2], resistance_levels[:2]  # Top 2 each