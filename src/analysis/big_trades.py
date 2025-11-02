"""
Big Options Trades Detection Module
Identifies and analyzes significant options trades and unusual activity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from ..api.schwab_client import SchwabClient
from ..utils.config import get_settings, BIG_TRADE_THRESHOLDS, VOLUME_THRESHOLDS

logger = logging.getLogger(__name__)

@dataclass
class BigTrade:
    """Big options trade data structure"""
    symbol: str
    timestamp: datetime
    contract_type: str  # 'call' or 'put'
    expiration: str
    strike: float
    volume: int
    open_interest: int
    premium: float
    notional_value: float
    trade_type: str  # 'opening', 'closing', 'unknown'
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    size_score: float  # How unusual the size is (1-10)
    urgency_score: float  # How urgent/unusual the timing is (1-10)
    confidence_level: float
    
    # Greeks at time of trade
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    implied_volatility: float = 0.0
    
    # Market context
    underlying_price: float = 0.0
    time_to_expiration: int = 0  # days
    moneyness: float = 0.0  # strike/underlying ratio
    
    # Analysis metadata
    analysis_notes: List[str] = field(default_factory=list)
    related_trades: List[str] = field(default_factory=list)

@dataclass
class UnusualActivity:
    """Unusual options activity summary"""
    symbol: str
    timestamp: datetime
    activity_type: str  # 'volume_spike', 'oi_increase', 'iv_surge', 'sweep'
    description: str
    metrics: Dict[str, float]
    big_trades: List[BigTrade]
    severity: float  # 1-10 scale
    market_impact: str  # 'low', 'medium', 'high'

class BigTradesDetector:
    """
    Detects and analyzes big options trades and unusual activity
    """
    
    def __init__(self, schwab_client: SchwabClient = None):
        self.settings = get_settings()
        self.schwab_client = schwab_client or SchwabClient()
        self.thresholds = BIG_TRADE_THRESHOLDS
        
    def scan_for_big_trades(self, symbols: List[str] = None,
                           min_premium: float = None) -> List[BigTrade]:
        """
        Scan for big options trades across specified symbols
        """
        if symbols is None:
            # Default to high-volume stocks and indices
            from ..utils.config import HIGH_VOLUME_STOCKS, MAJOR_INDICES
            symbols = MAJOR_INDICES + HIGH_VOLUME_STOCKS[:50]
        
        if min_premium is None:
            min_premium = self.thresholds['min_premium']
        
        logger.info(f"Scanning for big trades in {len(symbols)} symbols with min premium ${min_premium:,.0f}")
        
        big_trades = []
        
        for symbol in symbols:
            try:
                symbol_trades = self._analyze_symbol_for_big_trades(symbol, min_premium)
                big_trades.extend(symbol_trades)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for big trades: {str(e)}")
                continue
        
        # Sort by notional value (largest first)
        big_trades.sort(key=lambda x: x.notional_value, reverse=True)
        
        logger.info(f"Found {len(big_trades)} big trades")
        return big_trades
    
    def _analyze_symbol_for_big_trades(self, symbol: str, min_premium: float) -> List[BigTrade]:
        """
        Analyze a single symbol for big trades
        """
        big_trades = []
        
        try:
            # Get options chain
            chain_data = self.schwab_client.get_options_chain(
                symbol=symbol,
                contract_type="ALL",
                strike_count=50,
                include_quotes=True
            )
            
            # Get underlying quote for context
            quotes = self.schwab_client.get_quotes([symbol])
            underlying_price = quotes.get(symbol, {}).get('lastPrice', 0)
            
            # Analyze calls
            if 'callExpDateMap' in chain_data:
                call_trades = self._analyze_options_for_big_trades(
                    symbol, chain_data['callExpDateMap'], 'call', 
                    underlying_price, min_premium
                )
                big_trades.extend(call_trades)
            
            # Analyze puts
            if 'putExpDateMap' in chain_data:
                put_trades = self._analyze_options_for_big_trades(
                    symbol, chain_data['putExpDateMap'], 'put',
                    underlying_price, min_premium
                )
                big_trades.extend(put_trades)
                
        except Exception as e:
            logger.error(f"Error getting options data for {symbol}: {str(e)}")
        
        return big_trades
    
    def _analyze_options_for_big_trades(self, symbol: str, exp_date_map: Dict,
                                       contract_type: str, underlying_price: float,
                                       min_premium: float) -> List[BigTrade]:
        """
        Analyze options data for big trades
        """
        big_trades = []
        
        for exp_date, strikes_data in exp_date_map.items():
            exp_datetime = self._parse_expiration_date(exp_date)
            days_to_exp = (exp_datetime - datetime.now()).days if exp_datetime else 0
            
            for strike_str, contracts in strikes_data.items():
                strike = float(strike_str)
                
                for contract in contracts:
                    # Check if this qualifies as a big trade
                    trade = self._evaluate_big_trade(
                        symbol, contract, contract_type, exp_date,
                        strike, underlying_price, days_to_exp, min_premium
                    )
                    
                    if trade:
                        big_trades.append(trade)
        
        return big_trades
    
    def _evaluate_big_trade(self, symbol: str, contract: Dict, contract_type: str,
                           exp_date: str, strike: float, underlying_price: float,
                           days_to_exp: int, min_premium: float) -> Optional[BigTrade]:
        """
        Evaluate if a contract represents a big trade
        """
        # Extract key metrics
        volume = contract.get('totalVolume', 0)
        open_interest = contract.get('openInterest', 0)
        mark_price = contract.get('mark', 0)
        bid = contract.get('bid', 0)
        ask = contract.get('ask', 0)
        
        # Skip if no volume or invalid pricing
        if volume == 0 or mark_price <= 0:
            return None
        
        # Calculate trade value
        premium_per_contract = mark_price
        total_premium = premium_per_contract * volume * 100  # Options are in lots of 100
        notional_value = strike * volume * 100
        
        # Check thresholds
        meets_premium_threshold = total_premium >= min_premium
        meets_volume_threshold = volume >= self.thresholds['min_volume']
        meets_notional_threshold = notional_value >= self.thresholds['min_notional']
        
        # Must meet at least one major threshold
        if not (meets_premium_threshold or meets_volume_threshold or meets_notional_threshold):
            return None
        
        # Calculate additional metrics
        moneyness = strike / underlying_price if underlying_price > 0 else 1.0
        
        # Determine trade sentiment
        sentiment = self._determine_trade_sentiment(
            contract_type, moneyness, days_to_exp, volume, open_interest
        )
        
        # Calculate size score (how unusual the size is)
        size_score = self._calculate_size_score(volume, total_premium, notional_value)
        
        # Calculate urgency score (how urgent/unusual the timing is)
        urgency_score = self._calculate_urgency_score(
            days_to_exp, moneyness, volume, open_interest
        )
        
        # Determine trade type (opening vs closing)
        trade_type = self._determine_trade_type(volume, open_interest)
        
        # Calculate confidence level
        confidence = self._calculate_trade_confidence(contract, volume, mark_price)
        
        # Create analysis notes
        analysis_notes = self._generate_trade_analysis(
            symbol, contract_type, strike, underlying_price, 
            volume, total_premium, moneyness, days_to_exp
        )
        
        return BigTrade(
            symbol=symbol,
            timestamp=datetime.now(),
            contract_type=contract_type,
            expiration=exp_date,
            strike=strike,
            volume=volume,
            open_interest=open_interest,
            premium=premium_per_contract,
            notional_value=notional_value,
            trade_type=trade_type,
            sentiment=sentiment,
            size_score=size_score,
            urgency_score=urgency_score,
            confidence_level=confidence,
            delta=contract.get('delta', 0),
            gamma=contract.get('gamma', 0),
            theta=contract.get('theta', 0),
            vega=contract.get('vega', 0),
            implied_volatility=contract.get('volatility', 0),
            underlying_price=underlying_price,
            time_to_expiration=days_to_exp,
            moneyness=moneyness,
            analysis_notes=analysis_notes
        )
    
    def _determine_trade_sentiment(self, contract_type: str, moneyness: float,
                                 days_to_exp: int, volume: int, 
                                 open_interest: int) -> str:
        """
        Determine the sentiment of the trade
        """
        if contract_type == 'call':
            if moneyness < 0.95:  # OTM calls
                return 'bullish'
            elif moneyness > 1.05:  # ITM calls
                return 'neutral'  # Could be delta hedging
            else:  # ATM calls
                return 'bullish'
        else:  # puts
            if moneyness > 1.05:  # OTM puts
                return 'bearish'
            elif moneyness < 0.95:  # ITM puts
                return 'neutral'  # Could be delta hedging
            else:  # ATM puts
                return 'bearish'
    
    def _calculate_size_score(self, volume: int, total_premium: float,
                            notional_value: float) -> float:
        """
        Calculate how unusual the trade size is (1-10 scale)
        """
        score = 1.0
        
        # Volume score
        if volume >= 10000:
            score += 3.0
        elif volume >= 5000:
            score += 2.0
        elif volume >= 1000:
            score += 1.0
        
        # Premium score
        if total_premium >= 10000000:  # $10M+
            score += 3.0
        elif total_premium >= 5000000:  # $5M+
            score += 2.0
        elif total_premium >= 1000000:  # $1M+
            score += 1.0
        
        # Notional score
        if notional_value >= 100000000:  # $100M+
            score += 2.0
        elif notional_value >= 50000000:  # $50M+
            score += 1.0
        
        return min(10.0, score)
    
    def _calculate_urgency_score(self, days_to_exp: int, moneyness: float,
                               volume: int, open_interest: int) -> float:
        """
        Calculate how urgent/unusual the timing is (1-10 scale)
        """
        score = 1.0
        
        # Time decay urgency
        if days_to_exp <= 1:
            score += 4.0  # Same day expiration
        elif days_to_exp <= 3:
            score += 3.0  # This week expiration
        elif days_to_exp <= 7:
            score += 2.0  # Next week
        elif days_to_exp <= 30:
            score += 1.0  # This month
        
        # Moneyness urgency (very OTM options are more speculative)
        if moneyness < 0.8 or moneyness > 1.2:
            score += 2.0
        elif moneyness < 0.9 or moneyness > 1.1:
            score += 1.0
        
        # Volume vs OI ratio (new positions are more urgent)
        if open_interest > 0:
            vol_oi_ratio = volume / open_interest
            if vol_oi_ratio > 1.0:
                score += 2.0
            elif vol_oi_ratio > 0.5:
                score += 1.0
        
        return min(10.0, score)
    
    def _determine_trade_type(self, volume: int, open_interest: int) -> str:
        """
        Determine if trade is opening or closing position
        """
        if open_interest == 0:
            return 'opening'
        
        vol_oi_ratio = volume / open_interest if open_interest > 0 else 0
        
        if vol_oi_ratio > 0.5:
            return 'opening'
        elif vol_oi_ratio < 0.2:
            return 'closing'
        else:
            return 'unknown'
    
    def _calculate_trade_confidence(self, contract: Dict, volume: int,
                                  mark_price: float) -> float:
        """
        Calculate confidence in the trade analysis
        """
        confidence = 0.5  # Base confidence
        
        # Bid-ask spread quality
        bid = contract.get('bid', 0)
        ask = contract.get('ask', 0)
        
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / mark_price if mark_price > 0 else 1
            if spread_pct < 0.05:  # Tight spread
                confidence += 0.3
            elif spread_pct < 0.10:
                confidence += 0.2
            elif spread_pct < 0.20:
                confidence += 0.1
        
        # Volume confidence
        if volume >= 1000:
            confidence += 0.2
        elif volume >= 500:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _generate_trade_analysis(self, symbol: str, contract_type: str,
                               strike: float, underlying_price: float,
                               volume: int, total_premium: float,
                               moneyness: float, days_to_exp: int) -> List[str]:
        """
        Generate analysis notes for the trade
        """
        notes = []
        
        # Size analysis
        notes.append(f"${total_premium:,.0f} premium on {volume:,} contracts")
        
        # Moneyness analysis
        if moneyness < 0.9:
            notes.append(f"Buying OTM {contract_type}s - bullish bet" if contract_type == 'call' else f"Buying OTM {contract_type}s - bearish bet")
        elif moneyness > 1.1:
            notes.append(f"Buying ITM {contract_type}s - potential hedging or exercising strategy")
        else:
            notes.append(f"Buying ATM {contract_type}s - directional play")
        
        # Time analysis
        if days_to_exp <= 3:
            notes.append("Very short-term trade - high gamma exposure")
        elif days_to_exp <= 30:
            notes.append("Short-term trade - earnings or event play possible")
        elif days_to_exp <= 90:
            notes.append("Medium-term trade - trend following strategy")
        else:
            notes.append("Long-term trade - structural position")
        
        # Market impact
        if total_premium >= 10000000:
            notes.append("Institutional-size trade - significant market impact expected")
        elif total_premium >= 1000000:
            notes.append("Large trade - monitor for follow-through")
        
        return notes
    
    def _parse_expiration_date(self, exp_date_str: str) -> Optional[datetime]:
        """
        Parse expiration date string to datetime
        """
        try:
            # Handle different date formats from Schwab API
            if ':' in exp_date_str:
                # Format: "2024-01-19:45"
                date_part = exp_date_str.split(':')[0]
                return datetime.strptime(date_part, '%Y-%m-%d')
            else:
                # Direct date format
                return datetime.strptime(exp_date_str, '%Y-%m-%d')
        except Exception:
            return None
    
    def detect_unusual_activity(self, symbols: List[str] = None,
                              lookback_hours: int = 4) -> List[UnusualActivity]:
        """
        Detect various types of unusual options activity
        """
        if symbols is None:
            from ..utils.config import HIGH_VOLUME_STOCKS, MAJOR_INDICES
            symbols = MAJOR_INDICES + HIGH_VOLUME_STOCKS[:30]
        
        logger.info(f"Detecting unusual activity in {len(symbols)} symbols")
        
        unusual_activities = []
        
        for symbol in symbols:
            try:
                activities = self._analyze_symbol_unusual_activity(symbol, lookback_hours)
                unusual_activities.extend(activities)
            except Exception as e:
                logger.error(f"Error analyzing unusual activity for {symbol}: {str(e)}")
                continue
        
        # Sort by severity
        unusual_activities.sort(key=lambda x: x.severity, reverse=True)
        
        return unusual_activities
    
    def _analyze_symbol_unusual_activity(self, symbol: str,
                                       lookback_hours: int) -> List[UnusualActivity]:
        """
        Analyze unusual activity for a single symbol
        """
        activities = []
        
        try:
            # Get current options data
            chain_data = self.schwab_client.get_options_chain(
                symbol=symbol,
                contract_type="ALL",
                strike_count=30,
                include_quotes=True
            )
            
            # Get underlying quote
            quotes = self.schwab_client.get_quotes([symbol])
            underlying_price = quotes.get(symbol, {}).get('lastPrice', 0)
            
            # Check for volume spikes
            volume_activity = self._detect_volume_spikes(symbol, chain_data)
            if volume_activity:
                activities.append(volume_activity)
            
            # Check for OI increases
            oi_activity = self._detect_oi_increases(symbol, chain_data)
            if oi_activity:
                activities.append(oi_activity)
            
            # Check for IV surges
            iv_activity = self._detect_iv_surges(symbol, chain_data)
            if iv_activity:
                activities.append(iv_activity)
            
            # Check for option sweeps
            sweep_activity = self._detect_option_sweeps(symbol, chain_data)
            if sweep_activity:
                activities.append(sweep_activity)
                
        except Exception as e:
            logger.error(f"Error analyzing unusual activity for {symbol}: {str(e)}")
        
        return activities
    
    def _detect_volume_spikes(self, symbol: str, chain_data: Dict) -> Optional[UnusualActivity]:
        """
        Detect unusual volume spikes
        """
        all_volumes = []
        
        # Collect all volumes
        for exp_map in [chain_data.get('callExpDateMap', {}), chain_data.get('putExpDateMap', {})]:
            for exp_date, strikes in exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        volume = contract.get('totalVolume', 0)
                        if volume > 0:
                            all_volumes.append(volume)
        
        if len(all_volumes) < 10:  # Need enough data
            return None
        
        # Calculate thresholds
        avg_volume = np.mean(all_volumes)
        volume_threshold = avg_volume * VOLUME_THRESHOLDS['unusual_volume_ratio']
        
        # Find contracts exceeding threshold
        unusual_contracts = []
        total_unusual_volume = 0
        
        for exp_map in [chain_data.get('callExpDateMap', {}), chain_data.get('putExpDateMap', {})]:
            for exp_date, strikes in exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        volume = contract.get('totalVolume', 0)
                        if volume > volume_threshold and volume > 100:
                            unusual_contracts.append({
                                'strike': float(strike),
                                'volume': volume,
                                'expiration': exp_date,
                                'contract': contract
                            })
                            total_unusual_volume += volume
        
        if not unusual_contracts:
            return None
        
        # Calculate severity
        severity = min(10.0, len(unusual_contracts) / 5 + total_unusual_volume / 10000)
        
        return UnusualActivity(
            symbol=symbol,
            timestamp=datetime.now(),
            activity_type='volume_spike',
            description=f"Unusual volume in {len(unusual_contracts)} contracts",
            metrics={
                'unusual_contracts': len(unusual_contracts),
                'total_unusual_volume': total_unusual_volume,
                'avg_volume': avg_volume,
                'threshold': volume_threshold
            },
            big_trades=[],  # Would populate with related big trades
            severity=severity,
            market_impact='medium' if severity > 5 else 'low'
        )
    
    def _detect_oi_increases(self, symbol: str, chain_data: Dict) -> Optional[UnusualActivity]:
        """
        Detect significant open interest increases
        """
        # This would require historical OI data for comparison
        # For now, identify high absolute OI levels
        high_oi_contracts = []
        
        oi_threshold = 5000  # Contracts
        
        for exp_map in [chain_data.get('callExpDateMap', {}), chain_data.get('putExpDateMap', {})]:
            for exp_date, strikes in exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        oi = contract.get('openInterest', 0)
                        if oi > oi_threshold:
                            high_oi_contracts.append({
                                'strike': float(strike),
                                'oi': oi,
                                'expiration': exp_date,
                                'contract': contract
                            })
        
        if len(high_oi_contracts) < 3:
            return None
        
        total_oi = sum(c['oi'] for c in high_oi_contracts)
        severity = min(10.0, len(high_oi_contracts) / 10 + total_oi / 100000)
        
        return UnusualActivity(
            symbol=symbol,
            timestamp=datetime.now(),
            activity_type='oi_increase',
            description=f"High open interest in {len(high_oi_contracts)} contracts",
            metrics={
                'high_oi_contracts': len(high_oi_contracts),
                'total_oi': total_oi,
                'threshold': oi_threshold
            },
            big_trades=[],
            severity=severity,
            market_impact='high' if severity > 7 else 'medium'
        )
    
    def _detect_iv_surges(self, symbol: str, chain_data: Dict) -> Optional[UnusualActivity]:
        """
        Detect implied volatility surges
        """
        all_ivs = []
        
        # Collect all IVs
        for exp_map in [chain_data.get('callExpDateMap', {}), chain_data.get('putExpDateMap', {})]:
            for exp_date, strikes in exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        iv = contract.get('volatility', 0)
                        if iv > 0:
                            all_ivs.append(iv)
        
        if len(all_ivs) < 10:
            return None
        
        avg_iv = np.mean(all_ivs)
        max_iv = max(all_ivs)
        
        # Check if IV is unusually high
        if avg_iv > 0.5 or max_iv > 0.8:  # 50% average or 80% max
            severity = min(10.0, avg_iv * 10 + max_iv * 5)
            
            return UnusualActivity(
                symbol=symbol,
                timestamp=datetime.now(),
                activity_type='iv_surge',
                description=f"High implied volatility: avg {avg_iv:.1%}, max {max_iv:.1%}",
                metrics={
                    'avg_iv': avg_iv,
                    'max_iv': max_iv,
                    'iv_contracts': len(all_ivs)
                },
                big_trades=[],
                severity=severity,
                market_impact='high' if severity > 8 else 'medium'
            )
        
        return None
    
    def _detect_option_sweeps(self, symbol: str, chain_data: Dict) -> Optional[UnusualActivity]:
        """
        Detect option sweeps (large orders that sweep multiple strikes)
        """
        # This would require real-time tick data to detect sweeps
        # For now, identify potential sweep patterns based on volume distribution
        
        strikes_with_volume = []
        
        for contract_type in ['call', 'put']:
            exp_map = chain_data.get(f'{contract_type}ExpDateMap', {})
            
            for exp_date, strikes in exp_map.items():
                exp_volumes = []
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        volume = contract.get('totalVolume', 0)
                        if volume > 100:  # Minimum volume threshold
                            exp_volumes.append({
                                'strike': float(strike),
                                'volume': volume,
                                'type': contract_type,
                                'expiration': exp_date
                            })
                
                # Look for patterns indicating sweeps
                if len(exp_volumes) >= 3:
                    total_volume = sum(v['volume'] for v in exp_volumes)
                    if total_volume > 2000:  # Significant total volume
                        strikes_with_volume.extend(exp_volumes)
        
        if len(strikes_with_volume) < 5:
            return None
        
        total_sweep_volume = sum(s['volume'] for s in strikes_with_volume)
        severity = min(10.0, len(strikes_with_volume) / 5 + total_sweep_volume / 5000)
        
        return UnusualActivity(
            symbol=symbol,
            timestamp=datetime.now(),
            activity_type='sweep',
            description=f"Potential sweep across {len(strikes_with_volume)} strikes",
            metrics={
                'strikes_hit': len(strikes_with_volume),
                'total_volume': total_sweep_volume,
                'avg_volume_per_strike': total_sweep_volume / len(strikes_with_volume)
            },
            big_trades=[],
            severity=severity,
            market_impact='high' if severity > 6 else 'medium'
        )