"""
Real-time Options Flow Monitoring Module
Monitors and alerts on real-time options flow and unusual activity
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from dataclasses import dataclass, field
import json

from ..api.schwab_client import SchwabClient, AsyncSchwabClient
from ..analysis.market_dynamics import MarketDynamicsAnalyzer, IndividualStockAnalyzer
from ..analysis.big_trades import BigTradesDetector, BigTrade, UnusualActivity
from ..data.database import DatabaseManager
from ..utils.config import get_settings, HIGH_VOLUME_STOCKS, MAJOR_INDICES

logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition_type: str  # 'volume_spike', 'big_trade', 'iv_surge', 'unusual_flow'
    parameters: Dict[str, Any]
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 30

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    rule_name: str
    symbol: str
    alert_type: str
    message: str
    data: Dict[str, Any]
    severity: str  # 'low', 'medium', 'high', 'critical'
    actions_taken: List[str] = field(default_factory=list)

class OptionsFlowMonitor:
    """
    Real-time options flow monitoring system
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.schwab_client = SchwabClient()
        self.async_client = AsyncSchwabClient(self.schwab_client)
        self.market_analyzer = MarketDynamicsAnalyzer(self.schwab_client)
        self.stock_analyzer = IndividualStockAnalyzer(self.schwab_client)
        self.big_trades_detector = BigTradesDetector(self.schwab_client)
        self.db_manager = DatabaseManager()
        
        # Monitoring configuration
        self.is_monitoring = False
        self.monitor_thread = None
        self.update_interval = self.settings.MARKET_DATA_UPDATE_INTERVAL
        
        # Symbols to monitor
        self.monitored_symbols = MAJOR_INDICES + HIGH_VOLUME_STOCKS[:30]
        
        # Alert configuration
        self.alert_rules = self._initialize_default_alert_rules()
        self.alert_handlers = []
        
        # Data cache
        self.data_cache = {}
        self.last_update = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_alerts': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'avg_update_time': 0.0
        }
    
    def start_monitoring(self, symbols: List[str] = None):
        """
        Start real-time monitoring
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        if symbols:
            self.monitored_symbols = symbols
        
        logger.info(f"Starting options flow monitoring for {len(self.monitored_symbols)} symbols")
        logger.info(f"Update interval: {self.update_interval} seconds")
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Options flow monitoring started")
    
    def stop_monitoring(self):
        """
        Stop real-time monitoring
        """
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return
        
        logger.info("Stopping options flow monitoring")
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        logger.info("Options flow monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop
        """
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Update options data for all monitored symbols
                self._update_options_data()
                
                # Analyze for big trades and unusual activity
                self._analyze_for_alerts()
                
                # Update performance metrics
                update_time = time.time() - start_time
                self._update_performance_metrics(update_time, success=True)
                
                # Sleep until next update
                sleep_time = max(0, self.update_interval - update_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self._update_performance_metrics(0, success=False)
                time.sleep(self.update_interval)
    
    def _update_options_data(self):
        """
        Update options data for monitored symbols
        """
        updated_symbols = []
        
        for symbol in self.monitored_symbols:
            try:
                # Check if we need to update this symbol
                if self._should_update_symbol(symbol):
                    options_data = self.market_analyzer._collect_options_data([symbol])
                    
                    if options_data:
                        self.data_cache[symbol] = {
                            'options_data': options_data,
                            'timestamp': datetime.now()
                        }
                        self.last_update[symbol] = datetime.now()
                        updated_symbols.append(symbol)
                        
                        # Store in database
                        self._store_options_data(symbol, options_data)
                
            except Exception as e:
                logger.error(f"Error updating options data for {symbol}: {str(e)}")
                continue
        
        if updated_symbols:
            logger.debug(f"Updated options data for {len(updated_symbols)} symbols")
    
    def _should_update_symbol(self, symbol: str) -> bool:
        """
        Determine if a symbol needs data update
        """
        # Always update indices frequently
        if symbol in MAJOR_INDICES:
            return True
        
        # Check last update time
        last_update = self.last_update.get(symbol)
        if not last_update:
            return True
        
        # Update if data is older than interval
        time_since_update = (datetime.now() - last_update).total_seconds()
        return time_since_update >= self.update_interval
    
    def _store_options_data(self, symbol: str, options_data: List[Any]):
        """
        Store options data in database
        """
        try:
            # Convert to database format
            db_data = []
            for opt in options_data:
                db_data.append({
                    'symbol': symbol,
                    'timestamp': opt.timestamp,
                    'contract_type': opt.contract_type,
                    'expiration': opt.expiration,
                    'strike': opt.strike,
                    'volume': opt.volume,
                    'open_interest': opt.open_interest,
                    'premium': opt.premium,
                    'bid': opt.bid,
                    'ask': opt.ask,
                    'delta': opt.delta,
                    'gamma': opt.gamma,
                    'theta': opt.theta,
                    'vega': opt.vega,
                    'implied_volatility': opt.implied_volatility,
                    'underlying_price': getattr(opt, 'underlying_price', 0.0)
                })
            
            self.db_manager.store_options_data(db_data)
            
        except Exception as e:
            logger.error(f"Error storing options data for {symbol}: {str(e)}")
    
    def _analyze_for_alerts(self):
        """
        Analyze data for alert conditions
        """
        for symbol, data in self.data_cache.items():
            if not data or 'options_data' not in data:
                continue
            
            try:
                options_data = data['options_data']
                
                # Check each alert rule
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    # Check cooldown
                    if self._is_rule_in_cooldown(rule):
                        continue
                    
                    # Check rule condition
                    alert = self._check_alert_rule(rule, symbol, options_data)
                    if alert:
                        self._trigger_alert(alert)
                        rule.last_triggered = datetime.now()
                
            except Exception as e:
                logger.error(f"Error analyzing alerts for {symbol}: {str(e)}")
    
    def _check_alert_rule(self, rule: AlertRule, symbol: str, 
                         options_data: List[Any]) -> Optional[Alert]:
        """
        Check if an alert rule condition is met
        """
        try:
            if rule.condition_type == 'volume_spike':
                return self._check_volume_spike_rule(rule, symbol, options_data)
            elif rule.condition_type == 'big_trade':
                return self._check_big_trade_rule(rule, symbol, options_data)
            elif rule.condition_type == 'iv_surge':
                return self._check_iv_surge_rule(rule, symbol, options_data)
            elif rule.condition_type == 'unusual_flow':
                return self._check_unusual_flow_rule(rule, symbol, options_data)
            
        except Exception as e:
            logger.error(f"Error checking alert rule {rule.name}: {str(e)}")
        
        return None
    
    def _check_volume_spike_rule(self, rule: AlertRule, symbol: str,
                               options_data: List[Any]) -> Optional[Alert]:
        """
        Check for volume spike alerts
        """
        params = rule.parameters
        min_volume = params.get('min_volume', 1000)
        spike_multiplier = params.get('spike_multiplier', 3.0)
        
        # Calculate current vs average volume
        total_volume = sum(opt.volume for opt in options_data)
        
        # Get historical average (simplified - would use database in production)
        avg_volume = params.get('avg_volume', total_volume / 2)  # Placeholder
        
        if total_volume > avg_volume * spike_multiplier and total_volume > min_volume:
            return Alert(
                id=f"volume_spike_{symbol}_{int(time.time())}",
                timestamp=datetime.now(),
                rule_name=rule.name,
                symbol=symbol,
                alert_type='volume_spike',
                message=f"Volume spike in {symbol}: {total_volume:,} vs avg {avg_volume:,}",
                data={
                    'current_volume': total_volume,
                    'average_volume': avg_volume,
                    'spike_ratio': total_volume / avg_volume,
                    'options_count': len(options_data)
                },
                severity='medium' if total_volume > avg_volume * 5 else 'low'
            )
        
        return None
    
    def _check_big_trade_rule(self, rule: AlertRule, symbol: str,
                            options_data: List[Any]) -> Optional[Alert]:
        """
        Check for big trade alerts
        """
        params = rule.parameters
        min_premium = params.get('min_premium', 100000)
        min_notional = params.get('min_notional', 1000000)
        
        # Find big trades
        big_trades = []
        for opt in options_data:
            premium = opt.premium * opt.volume * 100
            notional = opt.strike * opt.volume * 100
            
            if premium >= min_premium or notional >= min_notional:
                big_trades.append({
                    'contract_type': opt.contract_type,
                    'strike': opt.strike,
                    'expiration': opt.expiration,
                    'volume': opt.volume,
                    'premium': premium,
                    'notional': notional
                })
        
        if big_trades:
            total_premium = sum(t['premium'] for t in big_trades)
            total_notional = sum(t['notional'] for t in big_trades)
            
            return Alert(
                id=f"big_trade_{symbol}_{int(time.time())}",
                timestamp=datetime.now(),
                rule_name=rule.name,
                symbol=symbol,
                alert_type='big_trade',
                message=f"Big trades in {symbol}: {len(big_trades)} trades, ${total_premium:,.0f} premium",
                data={
                    'trades_count': len(big_trades),
                    'total_premium': total_premium,
                    'total_notional': total_notional,
                    'biggest_trade': max(big_trades, key=lambda x: x['premium']),
                    'all_trades': big_trades
                },
                severity='high' if total_premium > 1000000 else 'medium'
            )
        
        return None
    
    def _check_iv_surge_rule(self, rule: AlertRule, symbol: str,
                           options_data: List[Any]) -> Optional[Alert]:
        """
        Check for implied volatility surge alerts
        """
        params = rule.parameters
        iv_threshold = params.get('iv_threshold', 0.5)  # 50%
        min_contracts = params.get('min_contracts', 5)
        
        # Calculate average IV
        ivs = [opt.implied_volatility for opt in options_data if opt.implied_volatility > 0]
        
        if len(ivs) < min_contracts:
            return None
        
        avg_iv = sum(ivs) / len(ivs)
        max_iv = max(ivs)
        
        if avg_iv > iv_threshold:
            return Alert(
                id=f"iv_surge_{symbol}_{int(time.time())}",
                timestamp=datetime.now(),
                rule_name=rule.name,
                symbol=symbol,
                alert_type='iv_surge',
                message=f"IV surge in {symbol}: avg {avg_iv:.1%}, max {max_iv:.1%}",
                data={
                    'avg_iv': avg_iv,
                    'max_iv': max_iv,
                    'contracts_analyzed': len(ivs),
                    'threshold': iv_threshold
                },
                severity='high' if avg_iv > 0.8 else 'medium'
            )
        
        return None
    
    def _check_unusual_flow_rule(self, rule: AlertRule, symbol: str,
                               options_data: List[Any]) -> Optional[Alert]:
        """
        Check for unusual flow patterns
        """
        params = rule.parameters
        
        # Calculate put/call ratios
        call_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'call')
        put_volume = sum(opt.volume for opt in options_data if opt.contract_type == 'put')
        
        if call_volume + put_volume == 0:
            return None
        
        put_call_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
        
        # Check for extreme ratios
        extreme_threshold = params.get('extreme_ratio', 3.0)
        min_total_volume = params.get('min_total_volume', 500)
        
        total_volume = call_volume + put_volume
        
        if total_volume > min_total_volume and (put_call_ratio > extreme_threshold or put_call_ratio < 1/extreme_threshold):
            sentiment = 'bearish' if put_call_ratio > extreme_threshold else 'bullish'
            
            return Alert(
                id=f"unusual_flow_{symbol}_{int(time.time())}",
                timestamp=datetime.now(),
                rule_name=rule.name,
                symbol=symbol,
                alert_type='unusual_flow',
                message=f"Unusual {sentiment} flow in {symbol}: P/C ratio {put_call_ratio:.2f}",
                data={
                    'put_call_ratio': put_call_ratio,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'total_volume': total_volume,
                    'sentiment': sentiment
                },
                severity='medium'
            )
        
        return None
    
    def _is_rule_in_cooldown(self, rule: AlertRule) -> bool:
        """
        Check if rule is in cooldown period
        """
        if not rule.last_triggered:
            return False
        
        time_since_trigger = (datetime.now() - rule.last_triggered).total_seconds() / 60
        return time_since_trigger < rule.cooldown_minutes
    
    def _trigger_alert(self, alert: Alert):
        """
        Trigger an alert
        """
        logger.info(f"ALERT: {alert.message}")
        
        # Store alert in database
        self._store_alert(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
        
        # Update metrics
        self.performance_metrics['total_alerts'] += 1
    
    def _store_alert(self, alert: Alert):
        """
        Store alert in database
        """
        try:
            # Store as unusual activity
            activity_data = {
                'symbol': alert.symbol,
                'timestamp': alert.timestamp,
                'activity_type': alert.alert_type,
                'description': alert.message,
                'metrics': alert.data,
                'severity': self._convert_severity_to_numeric(alert.severity),
                'market_impact': alert.severity
            }
            
            self.db_manager.store_unusual_activity(activity_data)
            
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
    
    def _convert_severity_to_numeric(self, severity: str) -> float:
        """
        Convert severity string to numeric value
        """
        severity_map = {
            'low': 3.0,
            'medium': 6.0,
            'high': 8.0,
            'critical': 10.0
        }
        return severity_map.get(severity, 5.0)
    
    def _update_performance_metrics(self, update_time: float, success: bool):
        """
        Update performance tracking metrics
        """
        if success:
            self.performance_metrics['successful_updates'] += 1
            # Update average update time (exponential moving average)
            alpha = 0.1
            current_avg = self.performance_metrics['avg_update_time']
            self.performance_metrics['avg_update_time'] = (
                alpha * update_time + (1 - alpha) * current_avg
            )
        else:
            self.performance_metrics['failed_updates'] += 1
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        Add a custom alert handler function
        """
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add a custom alert rule
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """
        Remove an alert rule by name
        """
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def enable_alert_rule(self, rule_name: str):
        """
        Enable an alert rule
        """
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled alert rule: {rule_name}")
                return
        logger.warning(f"Alert rule not found: {rule_name}")
    
    def disable_alert_rule(self, rule_name: str):
        """
        Disable an alert rule
        """
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled alert rule: {rule_name}")
                return
        logger.warning(f"Alert rule not found: {rule_name}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get monitoring performance metrics
        """
        total_updates = self.performance_metrics['successful_updates'] + self.performance_metrics['failed_updates']
        success_rate = (self.performance_metrics['successful_updates'] / total_updates * 100) if total_updates > 0 else 0
        
        return {
            **self.performance_metrics,
            'total_updates': total_updates,
            'success_rate': success_rate,
            'monitored_symbols': len(self.monitored_symbols),
            'active_rules': len([r for r in self.alert_rules if r.enabled]),
            'cache_size': len(self.data_cache)
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts from database
        """
        start_date = datetime.now() - timedelta(hours=hours)
        df = self.db_manager.get_unusual_activity(start_date=start_date)
        
        return df.to_dict('records') if not df.empty else []
    
    def _initialize_default_alert_rules(self) -> List[AlertRule]:
        """
        Initialize default alert rules
        """
        return [
            AlertRule(
                name="Volume Spike - Major Indices",
                condition_type="volume_spike",
                parameters={
                    'min_volume': 5000,
                    'spike_multiplier': 2.5,
                    'symbols': MAJOR_INDICES
                },
                cooldown_minutes=15
            ),
            AlertRule(
                name="Big Trade - All Symbols",
                condition_type="big_trade",
                parameters={
                    'min_premium': 250000,  # $250k
                    'min_notional': 2000000  # $2M
                },
                cooldown_minutes=10
            ),
            AlertRule(
                name="IV Surge - High Volume Stocks",
                condition_type="iv_surge",
                parameters={
                    'iv_threshold': 0.6,  # 60%
                    'min_contracts': 10
                },
                cooldown_minutes=30
            ),
            AlertRule(
                name="Unusual Flow - All Symbols",
                condition_type="unusual_flow",
                parameters={
                    'extreme_ratio': 4.0,
                    'min_total_volume': 1000
                },
                cooldown_minutes=20
            )
        ]

# Alert handler functions
def console_alert_handler(alert: Alert):
    """
    Print alerts to console
    """
    print(f"\n{'='*60}")
    print(f"🚨 {alert.severity.upper()} ALERT: {alert.message}")
    print(f"Symbol: {alert.symbol}")
    print(f"Time: {alert.timestamp}")
    print(f"Type: {alert.alert_type}")
    print(f"Data: {json.dumps(alert.data, indent=2, default=str)}")
    print(f"{'='*60}\n")

def log_alert_handler(alert: Alert):
    """
    Log alerts to file
    """
    logger.warning(f"ALERT [{alert.severity}]: {alert.message} | {alert.symbol} | {alert.data}")

# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = OptionsFlowMonitor()
    
    # Add alert handlers
    monitor.add_alert_handler(console_alert_handler)
    monitor.add_alert_handler(log_alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Monitor for 1 hour
        time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping monitoring...")
    finally:
        monitor.stop_monitoring()