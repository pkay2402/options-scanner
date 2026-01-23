"""
Multi-Channel Automated Alert Service for Discord Bot
Separate channels for: Whale Flows, 0DTE Levels, Market Intelligence (SPY/QQQ)
"""

import asyncio
import discord
import logging
from datetime import datetime, time, timedelta
from typing import Set, Optional, Dict, List
from pathlib import Path
import sys
import pytz
import json

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


class MultiChannelAlertService:
    """
    Enhanced alert service with separate channels for different alert types
    - Whale Flows Channel: Individual stock whale flows (score > threshold)
    - 0DTE Channel: SPY/QQQ/SPX wall levels and positioning
    - Market Intelligence Channel: Aggregate SPY/QQQ market signals
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.alert_task = None
        
        # Track sent alerts with timestamps (for time-based deduplication)
        self.sent_whale_alerts: Dict[str, datetime] = {}
        self.sent_market_intel: Dict[str, datetime] = {}
        
        # Channel IDs (to be set via commands)
        self.whale_channel_id: Optional[int] = None
        self.dte_channel_id: Optional[int] = None
        self.market_intel_channel_id: Optional[int] = None
        
        # Configuration
        self.whale_score_threshold = 50
        self.whale_scan_interval_minutes = 5  # More time-sensitive for whale flows
        self.scan_interval_minutes = 15  # For 0DTE and market intelligence
        
        # Market hours (Eastern Time - 9:30 AM to 4:00 PM)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Config file for persistence
        self.config_file = project_root / "discord-bot" / "alerts_config.json"
        self._load_config()
    
    def _load_config(self):
        """Load saved channel IDs and running state"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.whale_channel_id = config.get('whale_channel_id')
                    self.dte_channel_id = config.get('dte_channel_id')
                    self.market_intel_channel_id = config.get('market_intel_channel_id')
                    was_running = config.get('is_running', False)
                    
                    if was_running:
                        logger.info(f"Loaded config: whale={self.whale_channel_id}, dte={self.dte_channel_id}, intel={self.market_intel_channel_id}, will auto-start")
        except Exception as e:
            logger.error(f"Error loading alerts config: {e}")
    
    def _save_config(self):
        """Save current channel IDs and running state"""
        try:
            config = {
                'whale_channel_id': self.whale_channel_id,
                'dte_channel_id': self.dte_channel_id,
                'market_intel_channel_id': self.market_intel_channel_id,
                'is_running': self.is_running
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Alerts config saved")
        except Exception as e:
            logger.error(f"Error saving alerts config: {e}")
    
    async def auto_start_if_configured(self):
        """Auto-start alerts if they were running before restart"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if config.get('is_running', False):
                        # Wait a bit for bot to be fully ready
                        await asyncio.sleep(2)
                        await self.start()
                        logger.info("✅ Auto-started multi-channel alerts")
        except Exception as e:
            logger.error(f"Error auto-starting alerts: {e}")
    
    def set_whale_channel(self, channel_id: int):
        """Set the Discord channel ID for whale flow alerts"""
        self.whale_channel_id = channel_id
        self._save_config()
        logger.info(f"Whale flow alert channel set to: {channel_id}")
    
    def set_dte_channel(self, channel_id: int):
        """Set the Discord channel ID for 0DTE alerts"""
        self.dte_channel_id = channel_id
        self._save_config()
        logger.info(f"0DTE alert channel set to: {channel_id}")
    
    def set_market_intel_channel(self, channel_id: int):
        """Set the Discord channel ID for market intelligence alerts"""
        self.market_intel_channel_id = channel_id
        self._save_config()
        logger.info(f"Market intelligence channel set to: {channel_id}")
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (ET)"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        current_time = now_et.time()
        
        # Skip weekends
        weekday = now_et.weekday()
        if weekday >= 5:
            return False
        
        return self.market_open <= current_time < self.market_close
    
    async def start(self):
        """Start the automated alert service"""
        if self.is_running:
            logger.warning("Alert service already running")
            return
        
        # Verify at least one channel is set
        if not any([self.whale_channel_id, self.dte_channel_id, self.market_intel_channel_id]):
            logger.error("No channels configured. Set at least one channel first")
            return
        
        self.is_running = True
        self.alert_task = asyncio.create_task(self._alert_loop())
        self._save_config()
        logger.info("Multi-channel alert service started")
    
    async def stop(self):
        """Stop the automated alert service"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        
        self._save_config()
        logger.info("Multi-channel alert service stopped")
    
    async def _alert_loop(self):
        """Main alert loop - whale scans every 5 min, others every 15 min"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        whale_scan_counter = 0  # Track whale scan cycles
        
        while self.is_running:
            try:
                if self.is_market_hours():
                    logger.info("Running scheduled scans...")
                    
                    # Run scans with individual error handling
                    tasks = []
                    
                    # Whale flows every 5 minutes (more time-sensitive)
                    if self.whale_channel_id:
                        tasks.append(self._scan_whale_flows())
                    
                    # 0DTE and market intel every 15 minutes (every 3rd whale scan)
                    if whale_scan_counter % 3 == 0:
                        if self.dte_channel_id:
                            tasks.append(self._scan_0dte_levels())
                        
                        if self.market_intel_channel_id:
                            tasks.append(self._scan_market_intelligence())
                    
                    # Execute all enabled scans
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    logger.info("Scheduled scans completed")
                    whale_scan_counter += 1
                    consecutive_errors = 0
                else:
                    # Clear caches when market is closed
                    if self.sent_whale_alerts:
                        self.sent_whale_alerts.clear()
                    if self.sent_market_intel:
                        self.sent_market_intel.clear()
                    whale_scan_counter = 0
                    logger.info("Cleared alert caches (market closed)")
                
                # Wait for next whale scan interval (5 minutes)
                await asyncio.sleep(self.whale_scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Alert loop cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in alert loop (count: {consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors, stopping alert service")
                    self.is_running = False
                    break
                
                await asyncio.sleep(60)
    
    async def _scan_whale_flows(self):
        """Scan for individual stock whale flows > threshold (read from database)"""
        # DISABLED: Using whale_score.py scanner instead (richer alerts with expiry concentration & directional bias)
        # The WhaleScoreCommands cog handles whale flow alerts via /start_whale_scanner
        logger.debug("Whale flow scanning disabled in multi_channel_alert_service - using whale_score.py instead")
        return
        
        try:
            from src.data.market_cache import MarketCache
            
            channel = self.bot.get_channel(self.whale_channel_id)
            if not channel:
                logger.error(f"Whale channel {self.whale_channel_id} not found")
                return
            
            # Get whale flows from database (populated by market_data_worker)
            # Query recent flows only (last 15 minutes) to catch new detections
            cache = MarketCache()
            db_flows = cache.get_whale_flows(sort_by='time', limit=50, hours_lookback=0.25)
            
            if not db_flows:
                logger.info("No whale flows in database")
                return
            
            # Validate database schema
            if db_flows and 'type' not in db_flows[0]:
                logger.error("Database schema mismatch - 'type' column not found in whale_flows")
                logger.error(f"Available columns: {list(db_flows[0].keys())}")
                return
            
            # Filter and format whale flows
            whale_alerts = []
            for flow in db_flows:
                whale_score = flow.get('whale_score', 0)
                
                # Filter by threshold
                if whale_score < self.whale_score_threshold:
                    continue
                
                # Create unique alert key
                alert_key = f"{flow.get('symbol')}_{flow.get('strike')}_{flow.get('type')}_{flow.get('expiry')}"
                
                # Check if we've alerted on this recently (within last hour)
                now = datetime.now()
                if alert_key in self.sent_whale_alerts:
                    last_sent = self.sent_whale_alerts[alert_key]
                    if (now - last_sent).total_seconds() < 3600:  # 1 hour cooldown
                        continue
                
                # New alert or cooldown expired
                # Get underlying price for distance calculation
                underlying_price = flow.get('underlying_price', flow.get('strike', 0))
                
                whale_alerts.append({
                    'symbol': flow.get('symbol'),
                    'strike': flow.get('strike'),
                    'type': flow.get('type', 'CALL'),
                    'expiry': flow.get('expiry'),
                    'whale_score': whale_score,
                    'volume': flow.get('volume', 0),
                    'oi': flow.get('open_interest', 0),
                    'iv': flow.get('iv', 0) * 100,  # Convert to percentage
                    'underlying_price': underlying_price
                })
                self.sent_whale_alerts[alert_key] = now
            
            if whale_alerts:
                whale_alerts.sort(key=lambda x: x['whale_score'], reverse=True)
                
                # Get unique expiry dates from flows
                unique_expiries = sorted(set([flow['expiry'] for flow in whale_alerts if flow.get('expiry')]))
                expiry_display = ", ".join([datetime.strptime(exp, '%Y-%m-%d').strftime('%b %d') for exp in unique_expiries[:3]])
                
                embed = discord.Embed(
                    title="🐋 Whale Flow Alert",
                    description=f"Found {len(whale_alerts)} new whale flows (Score > {self.whale_score_threshold})",
                    color=0x00ff00,
                    timestamp=datetime.utcnow()
                )
                
                for flow in whale_alerts[:10]:
                    distance = ((flow['strike'] - flow['underlying_price']) / flow['underlying_price'] * 100) if flow['underlying_price'] else 0
                    exp_date = datetime.strptime(flow['expiry'], '%Y-%m-%d').strftime('%m/%d') if flow.get('expiry') else 'N/A'
                    
                    field_name = f"{flow['symbol']} ${flow['strike']:.2f} {flow['type']} [{exp_date}]"
                    field_value = (
                        f"**Score:** {int(flow['whale_score']):,}\n"
                        f"**Vol:** {int(flow['volume']):,} | **OI:** {int(flow['oi']):,}\n"
                        f"**Distance:** {distance:+.1f}% | **IV:** {flow['iv']:.1f}%"
                    )
                    
                    embed.add_field(name=field_name, value=field_value, inline=False)
                
                embed.set_footer(text=f"Expiries: {expiry_display} | Auto-scan every 5min" if expiry_display else "Auto-scan every 5min")
                
                await channel.send(embed=embed)
                logger.info(f"Sent whale flow alert: {len(whale_alerts)} flows")
            else:
                logger.info("No new whale flows detected")
                
        except Exception as e:
            logger.error(f"Error scanning whale flows: {e}", exc_info=True)
    
    async def _scan_0dte_levels(self):
        """Scan 0DTE levels for SPY, QQQ, SPX"""
        try:
            from bot.commands.dte_commands import get_next_expiry, calculate_option_metrics
            
            channel = self.bot.get_channel(self.dte_channel_id)
            if not channel:
                logger.error(f"0DTE channel {self.dte_channel_id} not found")
                return
            
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            symbols = ['SPY', 'QQQ', '$SPX']
            
            embed = discord.Embed(
                title="📊 0DTE Levels Update",
                description="Current price vs Call/Put Walls",
                color=0x3498db,
                timestamp=datetime.utcnow()
            )
            
            for symbol in symbols:
                try:
                    quote = client.get_quote(symbol)
                    if not quote or symbol not in quote:
                        continue
                    
                    underlying_price = quote[symbol]['quote']['lastPrice']
                    expiry_date = get_next_expiry(symbol)
                    expiry_str = expiry_date.strftime("%Y-%m-%d")
                    
                    options_chain = client.get_options_chain(
                        symbol=symbol,
                        contract_type='ALL',
                        from_date=expiry_str,
                        to_date=expiry_str
                    )
                    
                    if not options_chain:
                        continue
                    
                    metrics = calculate_option_metrics(options_chain, underlying_price, expiry_date)
                    
                    if not metrics:
                        continue
                    
                    call_walls = sorted(metrics.get('call_walls', []), key=lambda x: x[1], reverse=True)
                    put_walls = sorted(metrics.get('put_walls', []), key=lambda x: x[1], reverse=True)
                    
                    call_wall_strike = call_walls[0][0] if call_walls else None
                    call_wall_volume = call_walls[0][1] if call_walls else 0
                    
                    put_wall_strike = put_walls[0][0] if put_walls else None
                    put_wall_volume = put_walls[0][1] if put_walls else 0
                    
                    flip_level = metrics.get('flip_level')
                    
                    position = ""
                    if call_wall_strike and put_wall_strike:
                        if underlying_price > call_wall_strike:
                            position = "🟢 Above Call Wall"
                        elif underlying_price < put_wall_strike:
                            position = "🔴 Below Put Wall"
                        else:
                            position = "🟡 Between Walls"
                    
                    # Build field value with flip level
                    flip_level_str = f"${flip_level:.2f}" if flip_level else "N/A"
                    field_value = (
                        f"**Current:** ${underlying_price:.2f} {position}\n"
                        f"**Call Wall:** ${call_wall_strike:.2f} ({call_wall_volume:,.0f} vol)\n"
                        f"**Put Wall:** ${put_wall_strike:.2f} ({put_wall_volume:,.0f} vol)\n"
                        f"**Flip Level:** {flip_level_str}\n"
                        f"**Call/Put Vol:** {metrics['total_call_volume']:,.0f} / {metrics['total_put_volume']:,.0f}"
                    )
                    
                    embed.add_field(name=f"**{symbol}**", value=field_value, inline=False)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            if len(embed.fields) > 0:
                embed.set_footer(text="Auto-update every 15min during market hours")
                await channel.send(embed=embed)
                logger.info("Sent 0DTE levels update")
                
        except Exception as e:
            logger.error(f"Error scanning 0DTE levels: {e}", exc_info=True)
    
    async def _scan_market_intelligence(self):
        """
        Compact SPY/QQQ market intelligence combining both indices
        Key innovations:
        - Dealer flow pressure (intraday hedging)
        - Dark pool positioning (institutional intent)
        - Term structure bias (near vs far positioning)
        - Skew analysis (fear/greed gauge)
        """
        try:
            channel = self.bot.get_channel(self.market_intel_channel_id)
            if not channel:
                logger.error(f"Market intel channel {self.market_intel_channel_id} not found")
                return
            
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            
            # Analyze both SPY and QQQ
            spy_intel = await self._analyze_market_structure(client, 'SPY')
            qqq_intel = await self._analyze_market_structure(client, 'QQQ')
            
            if spy_intel and qqq_intel:
                # Create combined alert (hourly)
                alert_key = f"MARKET_{datetime.now().strftime('%H')}"
                
                if alert_key not in self.sent_market_intel:
                    await self._send_combined_market_intel(channel, spy_intel, qqq_intel)
                    self.sent_market_intel.add(alert_key)
                        
        except Exception as e:
            logger.error(f"Error in market intelligence scan: {e}", exc_info=True)
    
    async def _analyze_market_structure(self, client, symbol: str) -> Optional[Dict]:
        """
        Innovative options analysis for directional prediction
        Key metrics:
        1. Dealer flow pressure (gamma/delta hedging creates buying/selling pressure)
        2. Term structure (0-7DTE vs 7-30DTE positioning differences)
        3. Skew ratio (OTM put premium / OTM call premium - fear gauge)
        4. Dark pool delta (net institutional delta positioning)
        """
        try:
            # Get quote
            quote = client.get_quote(symbol)
            if not quote or symbol not in quote:
                return None
            
            current_price = quote[symbol]['quote']['lastPrice']
            
            # Get options chain across all expiries
            options_chain = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                strike_count=50
            )
            
            if not options_chain:
                return None
            
            # Separate near-term (0-7 DTE) and mid-term (7-30 DTE) analysis
            near_term_data = {'call_vol': 0, 'put_vol': 0, 'call_oi': 0, 'put_oi': 0, 
                             'call_delta': 0, 'put_delta': 0, 'net_gex': 0}
            mid_term_data = {'call_vol': 0, 'put_vol': 0, 'call_oi': 0, 'put_oi': 0,
                            'call_delta': 0, 'put_delta': 0, 'net_gex': 0}
            
            # Track flow velocity (volume acceleration)
            total_fresh_calls = 0  # Vol/OI >= 3.0x
            total_fresh_puts = 0
            
            # Skew calculation (OTM premium differential)
            otm_call_premium = 0
            otm_put_premium = 0
            otm_strikes_analyzed = 0
            
            today = datetime.now().date()
            
            # Process calls
            if 'callExpDateMap' in options_chain:
                for exp_date_str, strikes in options_chain['callExpDateMap'].items():
                    # Parse expiry date
                    expiry_str = exp_date_str.split(':')[0]
                    try:
                        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                        dte = (expiry_date - today).days
                    except:
                        continue
                    
                    if dte > 30:
                        continue
                    
                    bucket = near_term_data if dte <= 7 else mid_term_data
                    
                    for strike_str, contracts in strikes.items():
                        if not contracts:
                            continue
                        
                        strike = float(strike_str)
                        contract = contracts[0]
                        
                        volume = contract.get('totalVolume', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        delta = contract.get('delta', 0) or 0
                        gamma = contract.get('gamma', 0) or 0
                        mark = contract.get('mark', 0) or 0
                        
                        bucket['call_vol'] += volume
                        bucket['call_oi'] += oi
                        bucket['call_delta'] += delta * oi * 100  # Net dealer short delta
                        
                        # GEX calculation (dealer perspective)
                        if gamma and oi:
                            gex = gamma * oi * 100 * current_price * current_price * 0.01
                            bucket['net_gex'] += gex
                        
                        # Fresh institutional flow
                        if oi > 0 and volume / oi >= 3.0:
                            total_fresh_calls += 1
                        
                        # Skew: 5-10% OTM calls
                        if strike > current_price * 1.05 and strike < current_price * 1.10:
                            otm_call_premium += mark
                            otm_strikes_analyzed += 1
            
            # Process puts
            if 'putExpDateMap' in options_chain:
                for exp_date_str, strikes in options_chain['putExpDateMap'].items():
                    expiry_str = exp_date_str.split(':')[0]
                    try:
                        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                        dte = (expiry_date - today).days
                    except:
                        continue
                    
                    if dte > 30:
                        continue
                    
                    bucket = near_term_data if dte <= 7 else mid_term_data
                    
                    for strike_str, contracts in strikes.items():
                        if not contracts:
                            continue
                        
                        strike = float(strike_str)
                        contract = contracts[0]
                        
                        volume = contract.get('totalVolume', 0) or 0
                        oi = contract.get('openInterest', 0) or 0
                        delta = contract.get('delta', 0) or 0
                        gamma = contract.get('gamma', 0) or 0
                        mark = contract.get('mark', 0) or 0
                        
                        bucket['put_vol'] += volume
                        bucket['put_oi'] += oi
                        bucket['put_delta'] += abs(delta) * oi * 100  # Net dealer long delta
                        
                        # GEX (negative for puts)
                        if gamma and oi:
                            gex = gamma * oi * 100 * current_price * current_price * 0.01
                            bucket['net_gex'] -= gex
                        
                        # Fresh institutional flow
                        if oi > 0 and volume / oi >= 3.0:
                            total_fresh_puts += 1
                        
                        # Skew: 5-10% OTM puts
                        if strike < current_price * 0.95 and strike > current_price * 0.90:
                            otm_put_premium += mark
            
            # Calculate derived metrics
            
            # 1. Dealer Flow Pressure (net delta that dealers must hedge)
            # Positive = dealers long, will sell on rallies (resistance)
            # Negative = dealers short, will buy on dips (support)
            near_dealer_delta = near_term_data['put_delta'] - near_term_data['call_delta']
            mid_dealer_delta = mid_term_data['put_delta'] - mid_term_data['call_delta']
            
            # 2. Term Structure Divergence (positioning mismatch = directional signal)
            near_pc_ratio = near_term_data['put_vol'] / max(near_term_data['call_vol'], 1)
            mid_pc_ratio = mid_term_data['put_vol'] / max(mid_term_data['call_vol'], 1)
            term_divergence = mid_pc_ratio - near_pc_ratio
            
            # 3. Skew Ratio (fear gauge)
            skew_ratio = (otm_put_premium / max(otm_call_premium, 0.01)) if otm_strikes_analyzed > 0 else 1.0
            
            # 4. Flow Velocity (institutional urgency)
            flow_velocity = total_fresh_calls / max(total_fresh_puts, 1)
            
            # 5. Net GEX (volatility suppression/amplification)
            total_net_gex = near_term_data['net_gex'] + mid_term_data['net_gex']
            
            # Signal generation with strength
            signal_strength = 0
            intraday_bias = ""
            multi_day_bias = ""
            
            # Near-term dealer delta (intraday pressure)
            if near_dealer_delta < -50000:  # Dealers short = must buy dips
                intraday_bias = "BULLISH"
                signal_strength += 2
            elif near_dealer_delta > 50000:  # Dealers long = will sell rallies
                intraday_bias = "BEARISH"
                signal_strength -= 2
            else:
                intraday_bias = "NEUTRAL"
            
            # Term structure (multi-day positioning)
            if term_divergence < -0.3:  # Near-term puts low, mid-term calls high = bullish
                multi_day_bias = "BULLISH"
                signal_strength += 2
            elif term_divergence > 0.3:  # Near-term puts high, mid-term protection = bearish
                multi_day_bias = "BEARISH"
                signal_strength -= 2
            else:
                multi_day_bias = "NEUTRAL"
            
            # Skew (fear/greed confirmation)
            if skew_ratio > 1.5:  # High put premium = fear (contrarian bullish if overdone)
                signal_strength -= 1
            elif skew_ratio < 1.0:  # Low put premium = complacency (potential risk)
                signal_strength += 1
            
            # Flow velocity (institutional direction)
            if flow_velocity > 1.5:
                signal_strength += 1
            elif flow_velocity < 0.67:
                signal_strength -= 1
            
            # Overall signal
            if signal_strength >= 4:
                signal = "STRONG_BULLISH"
            elif signal_strength >= 2:
                signal = "BULLISH"
            elif signal_strength <= -4:
                signal = "STRONG_BEARISH"
            elif signal_strength <= -2:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            return {
                'symbol': symbol,
                'signal': signal,
                'strength': signal_strength,
                'price': current_price,
                'intraday_bias': intraday_bias,
                'multiday_bias': multi_day_bias,
                'near_dealer_delta': near_dealer_delta,
                'mid_dealer_delta': mid_dealer_delta,
                'term_divergence': term_divergence,
                'skew_ratio': skew_ratio,
                'flow_velocity': flow_velocity,
                'net_gex': total_net_gex,
                'near_pc': near_pc_ratio,
                'mid_pc': mid_pc_ratio,
                'fresh_calls': total_fresh_calls,
                'fresh_puts': total_fresh_puts
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure for {symbol}: {e}", exc_info=True)
            return None
    
    async def _send_combined_market_intel(self, channel, spy_intel: Dict, qqq_intel: Dict):
        """Send compact combined SPY/QQQ intelligence in single card"""
        try:
            # Determine overall market signal (weighted: SPY 70%, QQQ 30%)
            market_strength = int(spy_intel['strength'] * 0.7 + qqq_intel['strength'] * 0.3)
            
            if market_strength >= 4:
                market_signal = "STRONG_BULLISH"
                color = 0x00ff00
                emoji = "🚀"
            elif market_strength >= 2:
                market_signal = "BULLISH"
                color = 0x90ee90
                emoji = "🟢"
            elif market_strength <= -4:
                market_signal = "STRONG_BEARISH"
                color = 0xff0000
                emoji = "💀"
            elif market_strength <= -2:
                market_signal = "BEARISH"
                color = 0xff6347
                emoji = "🔴"
            else:
                market_signal = "NEUTRAL"
                color = 0xffff00
                emoji = "🟡"
            
            embed = discord.Embed(
                title=f"{emoji} Market Intelligence",
                description=f"**Overall Signal: {market_signal}** (Strength: {market_strength})",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            # --- SPY SECTION ---
            spy_intraday_emoji = "🟢" if spy_intel['intraday_bias'] == "BULLISH" else "🔴" if spy_intel['intraday_bias'] == "BEARISH" else "🟡"
            spy_multiday_emoji = "🟢" if spy_intel['multiday_bias'] == "BULLISH" else "🔴" if spy_intel['multiday_bias'] == "BEARISH" else "🟡"
            
            embed.add_field(
                name="🔵 SPY",
                value=(
                    f"**Price:** ${spy_intel['price']:.2f} | **Signal:** {spy_intel['signal']}\n"
                    f"{spy_intraday_emoji} **Intraday:** {spy_intel['intraday_bias']} | "
                    f"{spy_multiday_emoji} **Multi-Day:** {spy_intel['multiday_bias']}\n\n"
                    f"**📊 Dealer Position**\n"
                    f"Near-Term Δ: {spy_intel['near_dealer_delta']/1000:.0f}K (intraday pressure)\n"
                    f"Mid-Term Δ: {spy_intel['mid_dealer_delta']/1000:.0f}K (multi-day trend)\n\n"
                    f"**📈 Flow Metrics**\n"
                    f"Skew: {spy_intel['skew_ratio']:.2f} (fear={spy_intel['skew_ratio'] > 1.3})\n"
                    f"Flow Velocity: {spy_intel['flow_velocity']:.2f}x\n"
                    f"Term Divergence: {spy_intel['term_divergence']:.2f}\n"
                    f"Net GEX: ${spy_intel['net_gex']/1e9:.2f}B\n"
                ),
                inline=False
            )
            
            # --- QQQ SECTION ---
            qqq_intraday_emoji = "🟢" if qqq_intel['intraday_bias'] == "BULLISH" else "🔴" if qqq_intel['intraday_bias'] == "BEARISH" else "🟡"
            qqq_multiday_emoji = "🟢" if qqq_intel['multiday_bias'] == "BULLISH" else "🔴" if qqq_intel['multiday_bias'] == "BEARISH" else "🟡"
            
            embed.add_field(
                name="🟢 QQQ",
                value=(
                    f"**Price:** ${qqq_intel['price']:.2f} | **Signal:** {qqq_intel['signal']}\n"
                    f"{qqq_intraday_emoji} **Intraday:** {qqq_intel['intraday_bias']} | "
                    f"{qqq_multiday_emoji} **Multi-Day:** {qqq_intel['multiday_bias']}\n\n"
                    f"**📊 Dealer Position**\n"
                    f"Near-Term Δ: {qqq_intel['near_dealer_delta']/1000:.0f}K (intraday pressure)\n"
                    f"Mid-Term Δ: {qqq_intel['mid_dealer_delta']/1000:.0f}K (multi-day trend)\n\n"
                    f"**📈 Flow Metrics**\n"
                    f"Skew: {qqq_intel['skew_ratio']:.2f} (fear={qqq_intel['skew_ratio'] > 1.3})\n"
                    f"Flow Velocity: {qqq_intel['flow_velocity']:.2f}x\n"
                    f"Term Divergence: {qqq_intel['term_divergence']:.2f}\n"
                    f"Net GEX: ${qqq_intel['net_gex']/1e9:.2f}B\n"
                ),
                inline=False
            )
            
            # --- TRADING IMPLICATION ---
            # Determine actionable bias
            if spy_intel['intraday_bias'] == qqq_intel['intraday_bias']:
                intraday_action = f"**Intraday:** {spy_intel['intraday_bias']} (aligned)"
            else:
                intraday_action = f"**Intraday:** MIXED (SPY {spy_intel['intraday_bias']}, QQQ {qqq_intel['intraday_bias']})"
            
            if spy_intel['multiday_bias'] == qqq_intel['multiday_bias']:
                multiday_action = f"**Multi-Day:** {spy_intel['multiday_bias']} (aligned)"
            else:
                multiday_action = f"**Multi-Day:** MIXED (SPY {spy_intel['multiday_bias']}, QQQ {qqq_intel['multiday_bias']})"
            
            # Dealer pressure interpretation
            avg_near_delta = (spy_intel['near_dealer_delta'] + qqq_intel['near_dealer_delta']) / 2
            if avg_near_delta < -50000:
                dealer_action = "Dealers SHORT → Will BUY dips (support)"
            elif avg_near_delta > 50000:
                dealer_action = "Dealers LONG → Will SELL rallies (resistance)"
            else:
                dealer_action = "Dealers NEUTRAL → Low hedging pressure"
            
            embed.add_field(
                name="💡 Trading Implication",
                value=(
                    f"{intraday_action}\n"
                    f"{multiday_action}\n\n"
                    f"**Dealer Pressure:** {dealer_action}\n\n"
                    f"⚠️ **Note:** Dealer delta shows intraday hedging pressure. "
                    f"Term divergence shows institutional positioning mismatch."
                ),
                inline=False
            )
            
            embed.set_footer(text=f"Market Intel • Updated every 15min • 0-30 DTE analyzed • {datetime.now().strftime('%H:%M')}")
            
            await channel.send(embed=embed)
            logger.info(f"Sent combined market intelligence: {market_signal}")
            
        except Exception as e:
            logger.error(f"Error sending combined market intel: {e}", exc_info=True)
