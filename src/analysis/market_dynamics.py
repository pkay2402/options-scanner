"""
Market Dynamics Analysis Module
Analyzes short-term and mid-term market dynamics using options data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from ..api.schwab_client import SchwabClient
from ..utils.config import get_settings, MAJOR_INDICES, HIGH_VOLUME_STOCKS
from ..data.database import DatabaseManager
from .big_trades import BigTradesDetector, BigTrade

logger = logging.getLogger(__name__)

@dataclass
class MarketSentiment:
    """Market sentiment indicators"""
    put_call_ratio: float
    vix_level: float
    gamma_exposure: float
    dealer_positioning: str
    sentiment_score: float
    confidence_level: float

@dataclass
class OptionsFlow:
    """Options flow data structure"""
    symbol: str
    timestamp: datetime
    contract_type: str  # 'call' or 'put'
    expiration: str
    strike: float
    volume: int
    open_interest: int
    premium: float
    bid: float
    ask: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float

@dataclass
class MarketAnalysisResult:
    """Market analysis result structure"""
    timestamp: datetime
    analysis_type: str  # 'short_term', 'mid_term'
    sentiment: MarketSentiment
    key_levels: Dict[str, float]
    unusual_activity: List[OptionsFlow]
    recommendations: List[str]
    risk_factors: List[str]
    confidence_score: float

class MarketDynamicsAnalyzer:
    """
    Comprehensive market dynamics analyzer using options data
    """
    
    def __init__(self, schwab_client: SchwabClient = None):
        self.settings = get_settings()
        self.schwab_client = schwab_client or SchwabClient()
        self.db_manager = DatabaseManager()
        self.big_trades_detector = BigTradesDetector(self.schwab_client)
        
    def analyze_short_term_dynamics(self, symbols: List[str] = None) -> MarketAnalysisResult:
        """
        Analyze short-term market dynamics (intraday to 1 week)
        """
        if symbols is None:
            symbols = MAJOR_INDICES + HIGH_VOLUME_STOCKS[:10]
        
        logger.info(f"Starting short-term analysis for {len(symbols)} symbols")
        
        # Collect options data
        options_data = self._collect_options_data(symbols)
        
        # Calculate market sentiment
        sentiment = self._calculate_market_sentiment(options_data)
        
        # Identify key support/resistance levels
        key_levels = self._identify_key_levels(options_data)
        
        # Detect unusual options activity
        unusual_activity = self._detect_unusual_activity(options_data)
        
        # Generate recommendations
        recommendations = self._generate_short_term_recommendations(
            sentiment, key_levels, unusual_activity
        )
        
        # Assess risk factors
        risk_factors = self._assess_short_term_risks(sentiment, unusual_activity)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            len(options_data), len(unusual_activity)
        )
        
        return MarketAnalysisResult(
            timestamp=datetime.now(),
            analysis_type="short_term",
            sentiment=sentiment,
            key_levels=key_levels,
            unusual_activity=unusual_activity,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence_score=confidence_score
        )
    
    def analyze_mid_term_dynamics(self, symbols: List[str] = None, 
                                 lookback_days: int = 30) -> MarketAnalysisResult:
        """
        Analyze mid-term market dynamics (1 week to 3 months)
        """
        if symbols is None:
            symbols = MAJOR_INDICES + HIGH_VOLUME_STOCKS[:20]
        
        logger.info(f"Starting mid-term analysis for {len(symbols)} symbols over {lookback_days} days")
        
        # Collect historical options data
        historical_data = self._collect_historical_options_data(symbols, lookback_days)
        
        # Analyze trends and patterns
        sentiment = self._calculate_historical_sentiment(historical_data)
        
        # Identify structural levels
        key_levels = self._identify_structural_levels(historical_data)
        
        # Find persistent unusual activity
        unusual_activity = self._detect_persistent_unusual_activity(historical_data)
        
        # Generate mid-term recommendations
        recommendations = self._generate_mid_term_recommendations(
            sentiment, key_levels, unusual_activity
        )
        
        # Assess mid-term risks
        risk_factors = self._assess_mid_term_risks(sentiment, historical_data)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            len(historical_data), len(unusual_activity)
        )
        
        return MarketAnalysisResult(
            timestamp=datetime.now(),
            analysis_type="mid_term",
            sentiment=sentiment,
            key_levels=key_levels,
            unusual_activity=unusual_activity,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence_score=confidence_score
        )
    
    def _collect_options_data(self, symbols: List[str]) -> List[OptionsFlow]:
        """
        Collect current options data for given symbols
        """
        options_data = []
        
        for symbol in symbols:
            try:
                # Get options chain
                chain_data = self.schwab_client.get_options_chain(
                    symbol=symbol,
                    contract_type="ALL",
                    strike_count=20,
                    include_quotes=True
                )
                
                # Process calls
                if 'callExpDateMap' in chain_data:
                    for exp_date, strikes in chain_data['callExpDateMap'].items():
                        for strike_price, contracts in strikes.items():
                            for contract in contracts:
                                options_data.append(self._create_options_flow(
                                    symbol, contract, 'call', exp_date, strike_price
                                ))
                
                # Process puts
                if 'putExpDateMap' in chain_data:
                    for exp_date, strikes in chain_data['putExpDateMap'].items():
                        for strike_price, contracts in strikes.items():
                            for contract in contracts:
                                options_data.append(self._create_options_flow(
                                    symbol, contract, 'put', exp_date, strike_price
                                ))
                
            except Exception as e:
                logger.error(f"Error collecting options data for {symbol}: {str(e)}")
                continue
        
        return options_data
    
    def _create_options_flow(self, symbol: str, contract: Dict, 
                           contract_type: str, exp_date: str, strike_price: str) -> OptionsFlow:
        """
        Create OptionsFlow object from contract data
        """
        return OptionsFlow(
            symbol=symbol,
            timestamp=datetime.now(),
            contract_type=contract_type,
            expiration=exp_date,
            strike=float(strike_price),
            volume=contract.get('totalVolume', 0),
            open_interest=contract.get('openInterest', 0),
            premium=contract.get('mark', 0.0),
            bid=contract.get('bid', 0.0),
            ask=contract.get('ask', 0.0),
            delta=contract.get('delta', 0.0),
            gamma=contract.get('gamma', 0.0),
            theta=contract.get('theta', 0.0),
            vega=contract.get('vega', 0.0),
            implied_volatility=contract.get('volatility', 0.0)
        )
    
    def _calculate_market_sentiment(self, options_data: List[OptionsFlow]) -> MarketSentiment:
        """
        Calculate overall market sentiment from options data
        """
        if not options_data:
            return MarketSentiment(0.0, 0.0, 0.0, "neutral", 0.0, 0.0)
        
        # Calculate put/call ratio by volume
        call_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'call')
        put_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'put')
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0.0
        
        # Get real VIX data or estimate
        try:
            vix_data = self.schwab_client.get_quote('VIX')
            vix_level = vix_data['VIX']['lastPrice'] if 'VIX' in vix_data else None
        except Exception as e:
            logger.warning(f"Could not fetch VIX data: {e}")
            vix_level = None
            
        # Fallback to estimated VIX from options IV
        if vix_level is None:
            avg_iv = np.mean([opt.implied_volatility for opt in options_data if opt.implied_volatility > 0])
            # Convert decimal IV to VIX-like percentage (typical range 10-80)
            vix_level = min(max(avg_iv * 100, 10), 80) if avg_iv > 0 else 20.0
        
        # Calculate gamma exposure
        total_gamma = sum(abs(opt.gamma * opt.open_interest * 100) for opt in options_data)
        gamma_exposure = total_gamma / 1e9  # Normalize to billions
        
        # Determine dealer positioning
        if put_call_ratio > 1.2:
            dealer_positioning = "short_gamma"
        elif put_call_ratio < 0.8:
            dealer_positioning = "long_gamma"
        else:
            dealer_positioning = "neutral"
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (1 - put_call_ratio) if put_call_ratio <= 2 else -1
        sentiment_score = max(-1, min(1, sentiment_score))
        
        # Calculate confidence level
        data_quality = min(1.0, len(options_data) / 1000)  # More data = higher confidence
        confidence_level = data_quality * 0.8 + 0.2  # Base confidence of 20%
        
        return MarketSentiment(
            put_call_ratio=put_call_ratio,
            vix_level=vix_level,
            gamma_exposure=gamma_exposure,
            dealer_positioning=dealer_positioning,
            sentiment_score=sentiment_score,
            confidence_level=confidence_level
        )
    
    def _identify_key_levels(self, options_data: List[OptionsFlow]) -> Dict[str, float]:
        """
        Identify key support and resistance levels from options data
        """
        key_levels = {}
        
        # Group by symbol
        symbol_data = {}
        for opt in options_data:
            if opt.symbol not in symbol_data:
                symbol_data[opt.symbol] = []
            symbol_data[opt.symbol].append(opt)
        
        # Analyze each symbol
        for symbol, opts in symbol_data.items():
            if not opts:
                continue
            
            # Find strikes with high open interest
            oi_by_strike = {}
            for opt in opts:
                strike = opt.strike
                if strike not in oi_by_strike:
                    oi_by_strike[strike] = 0
                oi_by_strike[strike] += opt.open_interest
            
            # Sort by open interest
            sorted_strikes = sorted(oi_by_strike.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_strikes:
                key_levels[f"{symbol}_max_pain"] = sorted_strikes[0][0]
                
                # Find put wall (high put OI below current price)
                put_strikes = [(strike, oi) for strike, oi in sorted_strikes 
                              if any(opt.contract_type == 'put' and opt.strike == strike 
                                   for opt in opts)]
                if put_strikes:
                    key_levels[f"{symbol}_put_wall"] = put_strikes[0][0]
                
                # Find call wall (high call OI above current price)
                call_strikes = [(strike, oi) for strike, oi in sorted_strikes 
                               if any(opt.contract_type == 'call' and opt.strike == strike 
                                    for opt in opts)]
                if call_strikes:
                    key_levels[f"{symbol}_call_wall"] = call_strikes[0][0]
        
        return key_levels
    
    def _detect_unusual_activity(self, options_data: List[OptionsFlow]) -> List[OptionsFlow]:
        """
        Detect unusual options activity
        """
        unusual_activity = []
        
        # Calculate volume percentiles for each symbol
        symbol_volumes = {}
        for opt in options_data:
            if opt.symbol not in symbol_volumes:
                symbol_volumes[opt.symbol] = []
            symbol_volumes[opt.symbol].append(opt.volume)
        
        # Calculate thresholds
        for symbol, volumes in symbol_volumes.items():
            if len(volumes) < 10:  # Need enough data
                continue
            
            threshold = np.percentile(volumes, 95)  # Top 5% volume
            
            # Find options exceeding threshold
            for opt in options_data:
                if opt.symbol == symbol and opt.volume > threshold and opt.volume > 100:
                    unusual_activity.append(opt)
        
        # Sort by volume (highest first)
        unusual_activity.sort(key=lambda x: x.volume, reverse=True)
        
        return unusual_activity[:50]  # Top 50 unusual trades
    
    def _generate_short_term_recommendations(self, sentiment: MarketSentiment,
                                           key_levels: Dict[str, float],
                                           unusual_activity: List[OptionsFlow]) -> List[str]:
        """
        Generate specific, actionable trading recommendations
        """
        recommendations = []
        
        # Get current market price for context
        current_prices = {}
        for symbol in ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA']:
            try:
                quote = self.schwab_client.get_quote(symbol)
                if quote and symbol in quote:
                    current_prices[symbol] = quote[symbol]['quote']['lastPrice']
            except:
                pass
        
        # Sentiment-based specific trade recommendations
        if sentiment.sentiment_score > 0.3:
            # BULLISH RECOMMENDATIONS
            if 'SPY' in current_prices:
                spy_price = current_prices['SPY']
                target_upside = spy_price * 1.03  # 3% upside target
                stop_loss = spy_price * 0.98     # 2% downside stop
                
                recommendations.extend([
                    f"🔥 BULLISH SETUP: SPY @ ${spy_price:.2f}",
                    f"   • CALL SPREAD: Buy {spy_price:.0f}C, Sell {target_upside:.0f}C (1-2 weeks)",
                    f"   • TARGET: ${target_upside:.2f} (+3.0%)",
                    f"   • STOP LOSS: ${stop_loss:.2f} (-2.0%)",
                    f"   • RISK/REWARD: 1:1.5"
                ])
        
        elif sentiment.sentiment_score < -0.3:
            # BEARISH RECOMMENDATIONS  
            if 'SPY' in current_prices:
                spy_price = current_prices['SPY']
                target_downside = spy_price * 0.97  # 3% downside target
                stop_loss = spy_price * 1.02       # 2% upside stop
                
                recommendations.extend([
                    f"🔻 BEARISH SETUP: SPY @ ${spy_price:.2f}",
                    f"   • PUT SPREAD: Buy {spy_price:.0f}P, Sell {target_downside:.0f}P (1-2 weeks)",
                    f"   • TARGET: ${target_downside:.2f} (-3.0%)",
                    f"   • STOP LOSS: ${stop_loss:.2f} (+2.0%)",
                    f"   • RISK/REWARD: 1:1.5"
                ])
        
        else:
            # NEUTRAL RECOMMENDATIONS
            if 'SPY' in current_prices:
                spy_price = current_prices['SPY']
                upper_range = spy_price * 1.02
                lower_range = spy_price * 0.98
                
                recommendations.extend([
                    f"⚖️  NEUTRAL SETUP: SPY @ ${spy_price:.2f}",
                    f"   • IRON CONDOR: Sell {lower_range:.0f}P/{spy_price:.0f}C, Buy {lower_range*0.98:.0f}P/{upper_range:.0f}C",
                    f"   • RANGE: ${lower_range:.2f} - ${upper_range:.2f}",
                    f"   • MAX PROFIT: If SPY stays in range",
                    f"   • DAYS TO EXPIRY: 7-14 days"
                ])
        
        # VIX-based specific strategies
        if sentiment.vix_level > 30:
            recommendations.extend([
                f"📈 HIGH VIX ({sentiment.vix_level:.1f}): VOLATILITY PLAYS",
                f"   • SELL premium: Short strangles 2 weeks out",
                f"   • IRON CONDORS: Target 15-20 delta strikes",
                f"   • AVOID: Long options (high premiums)"
            ])
        elif sentiment.vix_level < 15:
            recommendations.extend([
                f"📉 LOW VIX ({sentiment.vix_level:.1f}): VOLATILITY EXPANSION",
                f"   • BUY cheap options: Long straddles on earnings",
                f"   • BUTTERFLY spreads: Profit from low movement",
                f"   • CONSIDER: Calendar spreads"
            ])
        
        # Key levels with specific trade zones
        for symbol, level in key_levels.items():
            if any(sym in symbol for sym in current_prices.keys()):
                base_symbol = next(sym for sym in current_prices.keys() if sym in symbol)
                current = current_prices[base_symbol]
                distance = abs(current - level) / current * 100
                
                if 'support' in symbol and distance < 2:
                    recommendations.append(f"🎯 {base_symbol} NEAR SUPPORT: ${level:.2f} (${current:.2f} current) - BOUNCE PLAY")
                elif 'resistance' in symbol and distance < 2:
                    recommendations.append(f"🎯 {base_symbol} NEAR RESISTANCE: ${level:.2f} (${current:.2f} current) - BREAKDOWN PLAY")
        
        # Unusual activity specific alerts
        high_activity_symbols = {}
        for activity in unusual_activity:
            if activity.symbol not in high_activity_symbols:
                high_activity_symbols[activity.symbol] = []
            high_activity_symbols[activity.symbol].append(activity)
        
        # Detect big trades for high-activity symbols
        try:
            big_trades = self.big_trades_detector.scan_for_big_trades(
                list(high_activity_symbols.keys())[:5],  # Top 5 active symbols
                min_premium=50000  # $50k minimum
            )
            
            # Add big trades to recommendations
            for trade in big_trades[:3]:  # Top 3 big trades
                notional = trade.notional_value / 1000000  # Convert to millions
                recommendations.append(
                    f"💰 BIG TRADE ALERT: {trade.symbol} {trade.strike}{trade.contract_type[0].upper()} "
                    f"${notional:.1f}M - {trade.sentiment.upper()}"
                )
        except Exception as e:
            logger.warning(f"Could not detect big trades: {e}")
        
        for symbol, activities in high_activity_symbols.items():
            if len(activities) >= 5 and symbol in current_prices:  # High activity threshold
                call_flow = sum(1 for a in activities if 'call' in a.contract_type.lower())
                put_flow = sum(1 for a in activities if 'put' in a.contract_type.lower())
                
                if call_flow > put_flow * 2:
                    recommendations.append(f"🚀 {symbol} UNUSUAL CALL ACTIVITY: {call_flow} vs {put_flow} puts - BULLISH FLOW")
                elif put_flow > call_flow * 2:
                    recommendations.append(f"🔽 {symbol} UNUSUAL PUT ACTIVITY: {put_flow} vs {call_flow} calls - BEARISH FLOW")
        
        return recommendations
    
    def _assess_short_term_risks(self, sentiment: MarketSentiment,
                               unusual_activity: List[OptionsFlow]) -> List[str]:
        """
        Assess specific, actionable market risks with mitigation strategies
        """
        risks = []
        
        # Volatility risks with specific thresholds
        if sentiment.vix_level > 35:
            risks.extend([
                f"⚠️  EXTREME VIX ({sentiment.vix_level:.1f}): Expect 3-5% daily moves",
                f"   • REDUCE position sizes by 50%",
                f"   • AVOID overnight holds",
                f"   • USE wider stops (+/- 4%)"
            ])
        elif sentiment.vix_level > 25:
            risks.extend([
                f"⚠️  ELEVATED VIX ({sentiment.vix_level:.1f}): Expect 2-3% daily moves",
                f"   • REDUCE position sizes by 25%",
                f"   • USE protective stops (+/- 2.5%)"
            ])
        
        # Gamma risks with specific price levels
        if sentiment.gamma_exposure > 8:
            risks.extend([
                f"⚠️  HIGH GAMMA EXPOSURE ({sentiment.gamma_exposure:.1f}B): Pin risk near whole numbers",
                f"   • AVOID trades near round strikes (670, 675, 680)",
                f"   • EXPECT acceleration through levels"
            ])
        
        # Sentiment extremes with reversal probability
        if sentiment.sentiment_score > 0.7:
            risks.extend([
                f"⚠️  EXTREME BULLISH SENTIMENT ({sentiment.sentiment_score:.2f}): 70% reversal probability",
                f"   • BOOK profits on long calls",
                f"   • CONSIDER protective puts",
                f"   • REDUCE new bullish positions"
            ])
        elif sentiment.sentiment_score < -0.7:
            risks.extend([
                f"⚠️  EXTREME BEARISH SENTIMENT ({sentiment.sentiment_score:.2f}): 70% reversal probability",
                f"   • BOOK profits on puts", 
                f"   • CONSIDER protective calls",
                f"   • REDUCE new bearish positions"
            ])
        
        # Time decay risks
        risks.append("⏰ THETA DECAY: Options lose ~3% value per day (1 week to expiry)")
        
        # Liquidity risks for specific symbols
        low_liquidity_symbols = []
        for activity in unusual_activity:
            if hasattr(activity, 'bid_ask_spread') and activity.bid_ask_spread > 0.05:
                low_liquidity_symbols.append(activity.symbol)
        
        if low_liquidity_symbols:
            unique_symbols = list(set(low_liquidity_symbols))
            risks.append(f"⚠️  WIDE SPREADS: {', '.join(unique_symbols[:3])} - Use limit orders only")
        
        return risks
    
    def _calculate_confidence_score(self, data_points: int, unusual_count: int) -> float:
        """
        Calculate confidence score for the analysis
        """
        # Base confidence from data quantity
        data_confidence = min(1.0, data_points / 1000)
        
        # Adjustment for unusual activity
        activity_factor = min(1.0, unusual_count / 20)
        
        # Combined confidence (weighted average)
        confidence = 0.7 * data_confidence + 0.3 * activity_factor
        
        return max(0.1, min(0.95, confidence))  # Bound between 10% and 95%
    
    def _collect_historical_options_data(self, symbols: List[str], 
                                       lookback_days: int) -> List[OptionsFlow]:
        """
        Collect historical options data (placeholder - would need historical data source)
        """
        # This would integrate with historical data storage
        # For now, return current data as placeholder
        return self._collect_options_data(symbols)
    
    def _calculate_historical_sentiment(self, historical_data: List[OptionsFlow]) -> MarketSentiment:
        """
        Calculate sentiment trends from historical data
        """
        # For now, use current calculation
        # In full implementation, this would analyze trends over time
        return self._calculate_market_sentiment(historical_data)
    
    def _identify_structural_levels(self, historical_data: List[OptionsFlow]) -> Dict[str, float]:
        """
        Identify structural support/resistance levels from historical data
        """
        # This would analyze historical price action and options data
        # For now, use current level identification
        return self._identify_key_levels(historical_data)
    
    def _detect_persistent_unusual_activity(self, historical_data: List[OptionsFlow]) -> List[OptionsFlow]:
        """
        Detect persistent unusual activity patterns
        """
        # This would look for patterns that persist over time
        # For now, use current unusual activity detection
        return self._detect_unusual_activity(historical_data)
    
    def _generate_mid_term_recommendations(self, sentiment: MarketSentiment,
                                         key_levels: Dict[str, float],
                                         unusual_activity: List[OptionsFlow]) -> List[str]:
        """
        Generate mid-term recommendations
        """
        recommendations = self._generate_short_term_recommendations(
            sentiment, key_levels, unusual_activity
        )
        
        # Add mid-term specific recommendations
        recommendations.append("Monitor monthly expiration positioning")
        recommendations.append("Consider longer-dated strategies for trend plays")
        
        return recommendations
    
    def _assess_mid_term_risks(self, sentiment: MarketSentiment,
                             historical_data: List[OptionsFlow]) -> List[str]:
        """
        Assess mid-term risks
        """
        risks = self._assess_short_term_risks(sentiment, [])
        
        # Add mid-term specific risks
        risks.append("Monitor for quarterly earnings impacts")
        risks.append("Watch for changes in monetary policy")
        
        return risks

class IndividualStockAnalyzer:
    """
    Analyze individual stock dynamics using options data
    """
    
    def __init__(self, schwab_client: SchwabClient = None):
        self.schwab_client = schwab_client or SchwabClient()
        self.market_analyzer = MarketDynamicsAnalyzer(schwab_client)
    
    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of individual stock
        """
        logger.info(f"Analyzing individual stock: {symbol}")
        
        try:
            # Get options chain
            options_data = self.market_analyzer._collect_options_data([symbol])
            
            # Calculate metrics
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'options_metrics': self._calculate_options_metrics(options_data),
                'unusual_activity': self._find_stock_unusual_activity(symbol, options_data),
                'key_levels': self._identify_stock_levels(symbol, options_data),
                'sentiment': self._calculate_stock_sentiment(options_data),
                'recommendations': []
            }
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_stock_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _calculate_options_metrics(self, options_data: List[OptionsFlow]) -> Dict[str, float]:
        """
        Calculate key options metrics for the stock
        """
        if not options_data:
            return {}
        
        calls = [opt for opt in options_data if opt.contract_type == 'call']
        puts = [opt for opt in options_data if opt.contract_type == 'put']
        
        metrics = {
            'total_volume': sum(opt.volume for opt in options_data),
            'total_oi': sum(opt.open_interest for opt in options_data),
            'call_volume': sum(opt.volume for opt in calls),
            'put_volume': sum(opt.volume for opt in puts),
            'call_oi': sum(opt.open_interest for opt in calls),
            'put_oi': sum(opt.open_interest for opt in puts),
            'avg_iv': np.mean([opt.implied_volatility for opt in options_data if opt.implied_volatility > 0]),
            'max_pain': self._calculate_max_pain(options_data)
        }
        
        # Calculate ratios
        if metrics['call_volume'] > 0:
            metrics['put_call_volume_ratio'] = metrics['put_volume'] / metrics['call_volume']
        if metrics['call_oi'] > 0:
            metrics['put_call_oi_ratio'] = metrics['put_oi'] / metrics['call_oi']
        
        return metrics
    
    def _calculate_max_pain(self, options_data: List[OptionsFlow]) -> float:
        """
        Calculate max pain point (strike with maximum option seller pain)
        """
        strikes = {}
        
        for opt in options_data:
            strike = opt.strike
            if strike not in strikes:
                strikes[strike] = {'call_oi': 0, 'put_oi': 0}
            
            if opt.contract_type == 'call':
                strikes[strike]['call_oi'] += opt.open_interest
            else:
                strikes[strike]['put_oi'] += opt.open_interest
        
        # Calculate pain for each strike
        pain_by_strike = {}
        for strike, oi in strikes.items():
            call_pain = sum(max(0, strike - s) * strikes[s]['call_oi'] 
                           for s in strikes.keys() if s < strike)
            put_pain = sum(max(0, s - strike) * strikes[s]['put_oi'] 
                          for s in strikes.keys() if s > strike)
            pain_by_strike[strike] = call_pain + put_pain
        
        # Return strike with maximum pain
        if pain_by_strike:
            return max(pain_by_strike.items(), key=lambda x: x[1])[0]
        return 0.0
    
    def _find_stock_unusual_activity(self, symbol: str, 
                                   options_data: List[OptionsFlow]) -> List[OptionsFlow]:
        """
        Find unusual activity specific to this stock
        """
        if not options_data:
            return []
        
        # Calculate volume thresholds
        volumes = [opt.volume for opt in options_data if opt.volume > 0]
        if not volumes:
            return []
        
        threshold = np.percentile(volumes, 90)
        
        # Find unusual trades
        unusual = [opt for opt in options_data 
                  if opt.volume > threshold and opt.volume > 50]
        
        return sorted(unusual, key=lambda x: x.volume, reverse=True)[:20]
    
    def _identify_stock_levels(self, symbol: str, 
                             options_data: List[OptionsFlow]) -> Dict[str, float]:
        """
        Identify key levels for the stock
        """
        levels = {}
        
        if not options_data:
            return levels
        
        # Group by strike and sum open interest
        strike_oi = {}
        for opt in options_data:
            strike = opt.strike
            if strike not in strike_oi:
                strike_oi[strike] = 0
            strike_oi[strike] += opt.open_interest
        
        # Find top OI strikes
        sorted_strikes = sorted(strike_oi.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_strikes) >= 1:
            levels['max_oi_strike'] = sorted_strikes[0][0]
        if len(sorted_strikes) >= 2:
            levels['second_max_oi'] = sorted_strikes[1][0]
        if len(sorted_strikes) >= 3:
            levels['third_max_oi'] = sorted_strikes[2][0]
        
        return levels
    
    def _calculate_stock_sentiment(self, options_data: List[OptionsFlow]) -> Dict[str, Any]:
        """
        Calculate sentiment specific to the stock
        """
        if not options_data:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        call_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'call')
        put_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'put')
        
        if call_volume + put_volume == 0:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        call_ratio = call_volume / (call_volume + put_volume)
        
        if call_ratio > 0.6:
            sentiment = 'bullish'
        elif call_ratio < 0.4:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Confidence based on volume
        total_volume = call_volume + put_volume
        confidence = min(1.0, total_volume / 1000)  # Higher volume = higher confidence
        
        return {
            'sentiment': sentiment,
            'call_ratio': call_ratio,
            'confidence': confidence,
            'total_volume': total_volume
        }
    
    def _generate_stock_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for the individual stock
        """
        recommendations = []
        
        sentiment = analysis.get('sentiment', {})
        metrics = analysis.get('options_metrics', {})
        
        # Sentiment-based recommendations
        if sentiment.get('sentiment') == 'bullish' and sentiment.get('confidence', 0) > 0.5:
            recommendations.append("Bullish options activity - consider calls or call spreads")
        elif sentiment.get('sentiment') == 'bearish' and sentiment.get('confidence', 0) > 0.5:
            recommendations.append("Bearish options activity - consider puts or put spreads")
        
        # Volume-based recommendations
        if metrics.get('total_volume', 0) > 1000:
            recommendations.append("High options volume - increased liquidity for trading")
        
        # IV recommendations
        avg_iv = metrics.get('avg_iv', 0)
        if avg_iv > 0.4:
            recommendations.append("High implied volatility - consider selling premium")
        elif avg_iv < 0.2:
            recommendations.append("Low implied volatility - consider buying options")
        
        return recommendations