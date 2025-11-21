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
        
        # Track sent alerts to avoid duplicates
        self.sent_whale_alerts: Set[str] = set()
        self.sent_market_intel: Set[str] = set()
        
        # Channel IDs (to be set via commands)
        self.whale_channel_id: Optional[int] = None
        self.dte_channel_id: Optional[int] = None
        self.market_intel_channel_id: Optional[int] = None
        
        # Configuration
        self.whale_score_threshold = 300
        self.scan_interval_minutes = 15
        
        # Market hours (Eastern Time - 9:30 AM to 4:00 PM)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
    
    def set_whale_channel(self, channel_id: int):
        """Set the Discord channel ID for whale flow alerts"""
        self.whale_channel_id = channel_id
        logger.info(f"Whale flow alert channel set to: {channel_id}")
    
    def set_dte_channel(self, channel_id: int):
        """Set the Discord channel ID for 0DTE alerts"""
        self.dte_channel_id = channel_id
        logger.info(f"0DTE alert channel set to: {channel_id}")
    
    def set_market_intel_channel(self, channel_id: int):
        """Set the Discord channel ID for market intelligence alerts"""
        self.market_intel_channel_id = channel_id
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
        
        logger.info("Multi-channel alert service stopped")
    
    async def _alert_loop(self):
        """Main alert loop - runs every 15 minutes during market hours"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                if self.is_market_hours():
                    logger.info("Running scheduled scans...")
                    
                    # Run scans with individual error handling
                    tasks = []
                    
                    if self.whale_channel_id:
                        tasks.append(self._scan_whale_flows())
                    
                    if self.dte_channel_id:
                        tasks.append(self._scan_0dte_levels())
                    
                    if self.market_intel_channel_id:
                        tasks.append(self._scan_market_intelligence())
                    
                    # Execute all enabled scans
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    logger.info("Scheduled scans completed")
                    consecutive_errors = 0
                else:
                    # Clear caches when market is closed
                    if self.sent_whale_alerts:
                        self.sent_whale_alerts.clear()
                    if self.sent_market_intel:
                        self.sent_market_intel.clear()
                    logger.info("Cleared alert caches (market closed)")
                
                # Wait for next scan interval
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
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
        """Scan for individual stock whale flows > threshold"""
        try:
            from bot.commands.whale_score import scan_stock_whale_flows, get_next_friday, TOP_TECH_STOCKS
            
            channel = self.bot.get_channel(self.whale_channel_id)
            if not channel:
                logger.error(f"Whale channel {self.whale_channel_id} not found")
                return
            
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            expiry_date = get_next_friday()
            
            # Scan stocks
            whale_alerts = []
            for symbol in TOP_TECH_STOCKS:
                flows = scan_stock_whale_flows(client, symbol, expiry_date, min_whale_score=self.whale_score_threshold)
                
                if flows:
                    for flow in flows:
                        alert_key = f"{flow['symbol']}_{flow['strike']}_{flow['type']}"
                        
                        if alert_key not in self.sent_whale_alerts:
                            whale_alerts.append(flow)
                            self.sent_whale_alerts.add(alert_key)
            
            if whale_alerts:
                whale_alerts.sort(key=lambda x: x['whale_score'], reverse=True)
                
                embed = discord.Embed(
                    title="🐋 Whale Flow Alert",
                    description=f"Found {len(whale_alerts)} new whale flows (Score > {self.whale_score_threshold})",
                    color=0x00ff00,
                    timestamp=datetime.utcnow()
                )
                
                for flow in whale_alerts[:10]:
                    distance = ((flow['strike'] - flow['underlying_price']) / flow['underlying_price'] * 100)
                    
                    field_name = f"{flow['symbol']} ${flow['strike']:.2f} {flow['type']}"
                    field_value = (
                        f"**Score:** {int(flow['whale_score']):,}\n"
                        f"**Vol:** {int(flow['volume']):,} | **OI:** {int(flow['oi']):,}\n"
                        f"**Distance:** {distance:+.1f}% | **IV:** {flow['iv']:.1f}%"
                    )
                    
                    embed.add_field(name=field_name, value=field_value, inline=False)
                
                embed.set_footer(text=f"Expiry: {expiry_date.strftime('%b %d, %Y')} | Auto-scan every 15min")
                
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
                    
                    max_pain = metrics.get('max_pain', underlying_price)
                    
                    position = ""
                    if call_wall_strike and put_wall_strike:
                        if underlying_price > call_wall_strike:
                            position = "🟢 Above Call Wall"
                        elif underlying_price < put_wall_strike:
                            position = "🔴 Below Put Wall"
                        else:
                            position = "🟡 Between Walls"
                    
                    field_value = (
                        f"**Current:** ${underlying_price:.2f} {position}\n"
                        f"**Call Wall:** ${call_wall_strike:.2f} ({call_wall_volume:,.0f} vol)\n"
                        f"**Put Wall:** ${put_wall_strike:.2f} ({put_wall_volume:,.0f} vol)\n"
                        f"**Max Pain:** ${max_pain:.2f}\n"
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
        Advanced market intelligence for SPY/QQQ
        Analyzes next 10 expiries to detect:
        - Directional bias (call vs put flow momentum)
        - Institutional positioning (OI buildups)
        - Gamma flip levels and net GEX
        - Volume acceleration patterns
        - Put/Call ratio trends
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
            
            for symbol in ['SPY', 'QQQ']:
                intel = await self._analyze_market_structure(client, symbol)
                
                if intel:
                    # Create unique key to avoid duplicate alerts
                    alert_key = f"{symbol}_{intel['signal']}_{datetime.now().strftime('%H')}"
                    
                    if alert_key not in self.sent_market_intel:
                        await self._send_market_intel_alert(channel, symbol, intel)
                        self.sent_market_intel.add(alert_key)
                        
        except Exception as e:
            logger.error(f"Error in market intelligence scan: {e}", exc_info=True)
    
    async def _analyze_market_structure(self, client, symbol: str) -> Optional[Dict]:
        """
        Analyze market structure across next 10 expiries
        Returns actionable market intelligence
        """
        try:
            # Get quote
            quote = client.get_quote(symbol)
            if not quote or symbol not in quote:
                return None
            
            current_price = quote[symbol]['quote']['lastPrice']
            
            # Get next 10 expiries
            options_chain = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                strike_count=50
            )
            
            if not options_chain:
                return None
            
            # Analyze call vs put flow and OI across all expiries
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            atm_call_volume = 0
            atm_put_volume = 0
            
            otm_call_volume = 0  # Above current price
            otm_put_volume = 0   # Below current price
            
            net_gamma_exposure = 0
            gamma_flip_estimate = current_price
            
            # Track unusual activity
            high_vol_oi_calls = []
            high_vol_oi_puts = []
            
            expiry_count = 0
            
            # Process calls
            if 'callExpDateMap' in options_chain:
                for exp_date, strikes in options_chain['callExpDateMap'].items():
                    expiry_count += 1
                    if expiry_count > 10:
                        break
                    
                    for strike_str, contracts in strikes.items():
                        if not contracts:
                            continue
                        
                        strike = float(strike_str)
                        contract = contracts[0]
                        
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        total_call_volume += volume
                        total_call_oi += oi
                        
                        # ATM = within 2% of current price
                        if abs(strike - current_price) / current_price <= 0.02:
                            atm_call_volume += volume
                        
                        # OTM calls (bullish)
                        if strike > current_price * 1.02:
                            otm_call_volume += volume
                        
                        # Calculate net GEX (dealer perspective)
                        if gamma and oi:
                            gex = gamma * oi * 100 * current_price * current_price * 0.01
                            net_gamma_exposure += gex
                        
                        # Track high Vol/OI ratios (fresh flows)
                        if oi > 0 and volume / oi >= 3.0:
                            high_vol_oi_calls.append({
                                'strike': strike,
                                'ratio': volume / oi,
                                'volume': volume,
                                'oi': oi
                            })
            
            # Process puts
            if 'putExpDateMap' in options_chain:
                expiry_count = 0
                for exp_date, strikes in options_chain['putExpDateMap'].items():
                    expiry_count += 1
                    if expiry_count > 10:
                        break
                    
                    for strike_str, contracts in strikes.items():
                        if not contracts:
                            continue
                        
                        strike = float(strike_str)
                        contract = contracts[0]
                        
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        total_put_volume += volume
                        total_put_oi += oi
                        
                        # ATM
                        if abs(strike - current_price) / current_price <= 0.02:
                            atm_put_volume += volume
                        
                        # OTM puts (bearish protection)
                        if strike < current_price * 0.98:
                            otm_put_volume += volume
                        
                        # Net GEX (puts are negative from dealer perspective)
                        if gamma and oi:
                            gex = gamma * oi * 100 * current_price * current_price * 0.01
                            net_gamma_exposure -= gex
                        
                        # Track high Vol/OI
                        if oi > 0 and volume / oi >= 3.0:
                            high_vol_oi_puts.append({
                                'strike': strike,
                                'ratio': volume / oi,
                                'volume': volume,
                                'oi': oi
                            })
            
            # Calculate metrics
            pc_ratio = total_put_volume / max(total_call_volume, 1)
            oi_pc_ratio = total_put_oi / max(total_call_oi, 1)
            
            atm_flow_ratio = atm_call_volume / max(atm_put_volume, 1)
            otm_flow_ratio = otm_call_volume / max(otm_put_volume, 1)
            
            # Determine signal
            signal = "NEUTRAL"
            signal_strength = 0
            bias_reasons = []
            
            # Signal 1: Net GEX (positive = support, negative = volatility)
            if net_gamma_exposure > 0:
                bias_reasons.append(f"✅ Positive GEX ({net_gamma_exposure/1e9:.2f}B) - dealers buy dips")
                signal_strength += 2
            else:
                bias_reasons.append(f"⚠️ Negative GEX ({net_gamma_exposure/1e9:.2f}B) - expect volatility")
                signal_strength -= 2
            
            # Signal 2: P/C Ratio
            if pc_ratio < 0.7:
                bias_reasons.append(f"🟢 Bullish P/C: {pc_ratio:.2f} (heavy call buying)")
                signal_strength += 3
            elif pc_ratio > 1.3:
                bias_reasons.append(f"🔴 Bearish P/C: {pc_ratio:.2f} (heavy put buying)")
                signal_strength -= 3
            else:
                bias_reasons.append(f"🟡 Neutral P/C: {pc_ratio:.2f}")
            
            # Signal 3: ATM Flow (immediate directional intent)
            if atm_flow_ratio > 1.5:
                bias_reasons.append(f"🟢 ATM Flow: {atm_flow_ratio:.2f}x more calls (bullish)")
                signal_strength += 2
            elif atm_flow_ratio < 0.67:
                bias_reasons.append(f"🔴 ATM Flow: {1/atm_flow_ratio:.2f}x more puts (bearish)")
                signal_strength -= 2
            
            # Signal 4: OTM Flow (positioning for moves)
            if otm_flow_ratio > 1.5:
                bias_reasons.append(f"🚀 OTM Calls: {otm_flow_ratio:.2f}x > OTM Puts (bullish positioning)")
                signal_strength += 2
            elif otm_flow_ratio < 0.67:
                bias_reasons.append(f"🛡️ OTM Puts: {1/otm_flow_ratio:.2f}x > OTM Calls (hedging/bearish)")
                signal_strength -= 2
            
            # Signal 5: Fresh flow direction (Vol/OI >= 3.0)
            if len(high_vol_oi_calls) > len(high_vol_oi_puts) * 1.5:
                bias_reasons.append(f"💰 Fresh Call Flows: {len(high_vol_oi_calls)} vs {len(high_vol_oi_puts)} puts")
                signal_strength += 2
            elif len(high_vol_oi_puts) > len(high_vol_oi_calls) * 1.5:
                bias_reasons.append(f"⚡ Fresh Put Flows: {len(high_vol_oi_puts)} vs {len(high_vol_oi_calls)} calls")
                signal_strength -= 2
            
            # Determine overall signal
            if signal_strength >= 5:
                signal = "STRONG_BULLISH"
            elif signal_strength >= 2:
                signal = "BULLISH"
            elif signal_strength <= -5:
                signal = "STRONG_BEARISH"
            elif signal_strength <= -2:
                signal = "BEARISH"
            
            return {
                'signal': signal,
                'strength': signal_strength,
                'current_price': current_price,
                'pc_ratio': pc_ratio,
                'oi_pc_ratio': oi_pc_ratio,
                'atm_flow_ratio': atm_flow_ratio,
                'otm_flow_ratio': otm_flow_ratio,
                'net_gex': net_gamma_exposure,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'fresh_call_flows': len(high_vol_oi_calls),
                'fresh_put_flows': len(high_vol_oi_puts),
                'bias_reasons': bias_reasons,
                'top_fresh_calls': sorted(high_vol_oi_calls, key=lambda x: x['volume'], reverse=True)[:3],
                'top_fresh_puts': sorted(high_vol_oi_puts, key=lambda x: x['volume'], reverse=True)[:3]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure for {symbol}: {e}", exc_info=True)
            return None
    
    async def _send_market_intel_alert(self, channel, symbol: str, intel: Dict):
        """Send formatted market intelligence alert"""
        try:
            # Color based on signal
            color_map = {
                'STRONG_BULLISH': 0x00ff00,  # Bright green
                'BULLISH': 0x90ee90,          # Light green
                'NEUTRAL': 0xffff00,          # Yellow
                'BEARISH': 0xff6347,          # Tomato
                'STRONG_BEARISH': 0xff0000   # Red
            }
            
            emoji_map = {
                'STRONG_BULLISH': '🚀',
                'BULLISH': '🟢',
                'NEUTRAL': '🟡',
                'BEARISH': '🔴',
                'STRONG_BEARISH': '💀'
            }
            
            signal = intel['signal']
            embed = discord.Embed(
                title=f"{emoji_map[signal]} {symbol} Market Intelligence",
                description=f"**Signal: {signal}** (Strength: {intel['strength']})\nAnalyzing next 10 expiries",
                color=color_map[signal],
                timestamp=datetime.utcnow()
            )
            
            # Current state
            embed.add_field(
                name="📊 Current State",
                value=(
                    f"**Price:** ${intel['current_price']:.2f}\n"
                    f"**Net GEX:** ${intel['net_gex']/1e9:.2f}B\n"
                    f"**P/C Ratio:** {intel['pc_ratio']:.2f}"
                ),
                inline=True
            )
            
            # Flow metrics
            embed.add_field(
                name="💹 Flow Analysis",
                value=(
                    f"**Total Vol:** {intel['total_call_volume']:,.0f}C / {intel['total_put_volume']:,.0f}P\n"
                    f"**ATM Flow:** {intel['atm_flow_ratio']:.2f}x\n"
                    f"**OTM Flow:** {intel['otm_flow_ratio']:.2f}x"
                ),
                inline=True
            )
            
            # Fresh flows
            embed.add_field(
                name="⚡ Fresh Institutional Flows",
                value=(
                    f"**Call Flows:** {intel['fresh_call_flows']} strikes\n"
                    f"**Put Flows:** {intel['fresh_put_flows']} strikes\n"
                    f"**Total OI:** {intel['total_call_oi']:,.0f}C / {intel['total_put_oi']:,.0f}P"
                ),
                inline=True
            )
            
            # Key reasons
            reasons_text = "\n".join(intel['bias_reasons'])
            embed.add_field(
                name="🎯 Key Signals",
                value=reasons_text,
                inline=False
            )
            
            # Top fresh call flows
            if intel['top_fresh_calls']:
                calls_text = "\n".join([
                    f"${f['strike']:.2f}: {f['volume']:,.0f} vol ({f['ratio']:.1f}x Vol/OI)"
                    for f in intel['top_fresh_calls']
                ])
                embed.add_field(
                    name="🟢 Top Fresh Call Strikes",
                    value=calls_text,
                    inline=True
                )
            
            # Top fresh put flows
            if intel['top_fresh_puts']:
                puts_text = "\n".join([
                    f"${f['strike']:.2f}: {f['volume']:,.0f} vol ({f['ratio']:.1f}x Vol/OI)"
                    for f in intel['top_fresh_puts']
                ])
                embed.add_field(
                    name="🔴 Top Fresh Put Strikes",
                    value=puts_text,
                    inline=True
                )
            
            # Trading suggestion
            if signal in ['STRONG_BULLISH', 'BULLISH']:
                suggestion = "✅ **Bias:** BULLISH - Consider call positions or long exposure"
            elif signal in ['STRONG_BEARISH', 'BEARISH']:
                suggestion = "⚠️ **Bias:** BEARISH - Consider put hedges or reduced exposure"
            else:
                suggestion = "⚪ **Bias:** NEUTRAL - Wait for clearer signals"
            
            embed.add_field(
                name="💡 Trading Implication",
                value=suggestion,
                inline=False
            )
            
            embed.set_footer(text="Market Intel • Updated every 15min • Next 10 expiries analyzed")
            
            await channel.send(embed=embed)
            logger.info(f"Sent market intelligence alert for {symbol}: {signal}")
            
        except Exception as e:
            logger.error(f"Error sending market intel alert: {e}", exc_info=True)
