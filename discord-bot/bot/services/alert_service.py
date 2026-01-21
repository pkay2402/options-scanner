"""
Automated Alert Service for Discord Bot
Sends scheduled alerts for whale flows and 0DTE analysis
"""

import asyncio
import discord
import logging
from datetime import datetime, time
from typing import Set, Optional
from pathlib import Path
import sys

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


class AutomatedAlertService:
    """
    Automated alert service that runs scheduled scans and sends notifications
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.alert_task = None
        
        # Track sent alerts to avoid duplicates
        self.sent_whale_alerts: Set[str] = set()  # symbol_strike_type
        
        # Configuration
        self.whale_score_threshold = 300
        self.scan_interval_minutes = 15
        
        # Market hours (Eastern Time - 9:30 AM to 4:00 PM)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
    
    def set_channel_id(self, channel_id: int):
        """Set the Discord channel ID for alerts"""
        self.channel_id = channel_id
        logger.info(f"Alert channel set to: {channel_id}")
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (ET)"""
        now = datetime.now().time()
        
        # Skip weekends
        weekday = datetime.now().weekday()
        if weekday >= 5:  # Saturday=5, Sunday=6
            return False
        
        return self.market_open <= now <= self.market_close
    
    async def start(self):
        """Start the automated alert service"""
        if self.is_running:
            logger.warning("Alert service already running")
            return
        
        if not hasattr(self, 'channel_id'):
            logger.error("Channel ID not set. Use set_channel_id() first")
            return
        
        self.is_running = True
        self.alert_task = asyncio.create_task(self._alert_loop())
        logger.info("Automated alert service started")
    
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
        
        logger.info("Automated alert service stopped")
    
    async def _alert_loop(self):
        """Main alert loop - runs every 15 minutes during market hours"""
        while self.is_running:
            try:
                if self.is_market_hours():
                    logger.info("Running scheduled scans...")
                    
                    # Run both scans
                    await self._scan_whale_flows()
                    await self._scan_0dte_levels()
                    
                    logger.info("Scheduled scans completed")
                else:
                    # Clear sent alerts cache when market is closed
                    if self.sent_whale_alerts:
                        self.sent_whale_alerts.clear()
                        logger.info("Cleared whale alerts cache (market closed)")
                
                # Wait for next scan interval
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _scan_whale_flows(self):
        """Scan for whale flows > 300 score"""
        # DISABLED: Using whale_score.py scanner instead (richer alerts with expiry concentration & directional bias)
        # The WhaleScoreCommands cog handles whale flow alerts via /start_whale_scanner
        logger.debug("Whale flow scanning disabled in alert_service - using whale_score.py instead")
        return
        
        try:
            from bot.commands.whale_score import scan_stock_whale_flows, get_next_friday, TOP_TECH_STOCKS
            
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            expiry_date = get_next_friday()
            
            # Scan stocks for whale flows
            whale_alerts = []
            for symbol in TOP_TECH_STOCKS:
                flows = scan_stock_whale_flows(client, symbol, expiry_date, min_whale_score=self.whale_score_threshold)
                
                if flows:
                    for flow in flows:
                        # Create unique key to avoid duplicates
                        alert_key = f"{flow['symbol']}_{flow['strike']}_{flow['type']}"
                        
                        # Only send if not already sent in this session
                        if alert_key not in self.sent_whale_alerts:
                            whale_alerts.append(flow)
                            self.sent_whale_alerts.add(alert_key)
            
            # Send alerts if found
            if whale_alerts:
                # Sort by whale score descending
                whale_alerts.sort(key=lambda x: x['whale_score'], reverse=True)
                
                # Create embed
                embed = discord.Embed(
                    title="🐋 Whale Flow Alert",
                    description=f"Found {len(whale_alerts)} new whale flows (Score > {self.whale_score_threshold})",
                    color=0x00ff00,  # Green
                    timestamp=datetime.utcnow()
                )
                
                # Add top 10 flows
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
            
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            
            symbols = ['SPY', 'QQQ', '$SPX']
            
            # Create embed
            embed = discord.Embed(
                title="📊 0DTE Levels Update",
                description="Current price vs Call/Put Walls",
                color=0x3498db,  # Blue
                timestamp=datetime.utcnow()
            )
            
            for symbol in symbols:
                try:
                    # Get quote
                    quote = client.get_quote(symbol)
                    if not quote or symbol not in quote:
                        continue
                    
                    underlying_price = quote[symbol]['quote']['lastPrice']
                    
                    # Get expiry for this symbol
                    expiry_date = get_next_expiry(symbol)
                    expiry_str = expiry_date.strftime("%Y-%m-%d")
                    
                    # Get options chain
                    options_chain = client.get_options_chain(
                        symbol=symbol,
                        contract_type='ALL',
                        from_date=expiry_str,
                        to_date=expiry_str
                    )
                    
                    if not options_chain:
                        continue
                    
                    # Calculate metrics
                    metrics = calculate_option_metrics(options_chain, underlying_price, expiry_date)
                    
                    if not metrics:
                        continue
                    
                    # Get call wall and put wall (already max volume from calculate_option_metrics)
                    call_walls = metrics.get('call_walls', [(None, 0)])
                    put_walls = metrics.get('put_walls', [(None, 0)])
                    
                    call_wall_strike = call_walls[0][0] if call_walls else None
                    call_wall_volume = call_walls[0][1] if call_walls else 0
                    
                    put_wall_strike = put_walls[0][0] if put_walls else None
                    put_wall_volume = put_walls[0][1] if put_walls else 0
                    
                    # Get flip level (where net volume changes sign)
                    flip_level = metrics.get('flip_level', None)
                    
                    # Determine position relative to walls
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
                        f"**Flip Level:** ${flip_level:.2f}\n" if flip_level else "**Flip Level:** N/A\n"
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
            else:
                logger.warning("No 0DTE data to send")
                
        except Exception as e:
            logger.error(f"Error scanning 0DTE levels: {e}", exc_info=True)
