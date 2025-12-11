"""
Watchlist Flip Level Scanner Command for Discord Bot
Monitors watchlist stocks for flip level crossings every 30 minutes
"""

import asyncio
import discord
import logging
from datetime import datetime, timedelta
from typing import Set, Optional, Dict
from pathlib import Path
import sys
import json
import pytz
import aiohttp

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup watchlist scanner commands"""
    await bot.add_cog(WatchlistScannerCommands(bot))


class WatchlistScannerCommands(discord.ext.commands.Cog):
    """Commands for monitoring watchlist flip level crossings"""
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        self.scan_interval_minutes = 30
        
        # Track previous flip level positions to detect crossings
        self.previous_positions: Dict[str, str] = {}  # symbol -> "above" or "below"
        
        # Load watchlist
        self.watchlist = self._load_watchlist()
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
    
    def _load_watchlist(self):
        """Load watchlist from user_preferences.json"""
        try:
            prefs_path = project_root / "user_preferences.json"
            if prefs_path.exists():
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
                    return prefs.get('watchlist', [])
            return []
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return []
    
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
    
    @discord.app_commands.command(
        name="setup_watchlist_scanner",
        description="Set this channel for watchlist flip level crossing alerts (scans every 30min)"
    )
    async def setup_watchlist_scanner(self, interaction: discord.Interaction):
        """Set the current channel for watchlist scanner alerts"""
        try:
            channel_id = interaction.channel_id
            self.channel_id = channel_id
            
            watchlist_str = ", ".join(self.watchlist) if self.watchlist else "No stocks in watchlist"
            
            embed = discord.Embed(
                title="📊 Watchlist Flip Level Scanner Configured",
                description=f"This channel will receive alerts when watchlist stocks cross flip levels",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="Watchlist",
                value=watchlist_str,
                inline=False
            )
            embed.add_field(
                name="Scan Frequency",
                value="Every 30 minutes during market hours",
                inline=True
            )
            embed.add_field(
                name="Alert Triggers",
                value="• Price crosses above flip level\n• Price crosses below flip level",
                inline=False
            )
            embed.add_field(
                name="Data Included",
                value="Stock, Current Price, Flip Level, Call Wall, Put Wall, Max GEX",
                inline=False
            )
            embed.set_footer(text="Use /start_watchlist_scanner to begin monitoring")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Watchlist scanner channel set to: {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting watchlist scanner channel: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="start_watchlist_scanner",
        description="Start automated watchlist flip level crossing monitoring"
    )
    async def start_watchlist_scanner(self, interaction: discord.Interaction):
        """Start the watchlist scanner service"""
        try:
            if not self.channel_id:
                await interaction.response.send_message(
                    "❌ Please setup the scanner channel first using `/setup_watchlist_scanner`",
                    ephemeral=True
                )
                return
            
            if self.is_running:
                await interaction.response.send_message(
                    "⚠️ Watchlist scanner is already running!",
                    ephemeral=True
                )
                return
            
            if not self.watchlist:
                await interaction.response.send_message(
                    "❌ No stocks in watchlist. Add stocks to user_preferences.json",
                    ephemeral=True
                )
                return
            
            self.is_running = True
            self.scanner_task = asyncio.create_task(self._scanner_loop())
            
            embed = discord.Embed(
                title="✅ Watchlist Scanner Started",
                description=f"Monitoring {len(self.watchlist)} stocks for flip level crossings",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="Status",
                value="🟢 Active",
                inline=True
            )
            embed.add_field(
                name="Next Scan",
                value=f"In {self.scan_interval_minutes} minutes",
                inline=True
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Watchlist scanner started")
            
        except Exception as e:
            logger.error(f"Error starting watchlist scanner: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="stop_watchlist_scanner",
        description="Stop automated watchlist monitoring"
    )
    async def stop_watchlist_scanner(self, interaction: discord.Interaction):
        """Stop the watchlist scanner service"""
        try:
            if not self.is_running:
                await interaction.response.send_message(
                    "⚠️ Watchlist scanner is not running",
                    ephemeral=True
                )
                return
            
            self.is_running = False
            if self.scanner_task:
                self.scanner_task.cancel()
                try:
                    await self.scanner_task
                except asyncio.CancelledError:
                    pass
            
            embed = discord.Embed(
                title="🛑 Watchlist Scanner Stopped",
                description="Flip level monitoring has been disabled",
                color=discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Watchlist scanner stopped")
            
        except Exception as e:
            logger.error(f"Error stopping watchlist scanner: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="watchlist_status",
        description="Check watchlist scanner status"
    )
    async def watchlist_status(self, interaction: discord.Interaction):
        """Show current watchlist scanner status"""
        try:
            status = "🟢 Running" if self.is_running else "🔴 Stopped"
            channel_info = f"<#{self.channel_id}>" if self.channel_id else "Not configured"
            watchlist_str = ", ".join(self.watchlist) if self.watchlist else "No stocks"
            
            embed = discord.Embed(
                title="📊 Watchlist Scanner Status",
                color=discord.Color.blue() if self.is_running else discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="Status", value=status, inline=True)
            embed.add_field(name="Channel", value=channel_info, inline=True)
            embed.add_field(name="Scan Interval", value=f"{self.scan_interval_minutes} min", inline=True)
            embed.add_field(name="Watchlist", value=watchlist_str, inline=False)
            
            if self.previous_positions:
                positions_str = "\n".join([
                    f"{sym}: {'📈 Above' if pos == 'above' else '📉 Below'} flip"
                    for sym, pos in list(self.previous_positions.items())[:10]
                ])
                embed.add_field(name="Current Positions", value=positions_str or "None", inline=False)
            
            await interaction.response.send_message(embed=embed)
            
        except Exception as e:
            logger.error(f"Error getting watchlist status: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    async def _scanner_loop(self):
        """Main scanner loop that runs every 30 minutes"""
        logger.info("Watchlist scanner loop started")
        
        while self.is_running:
            try:
                if self.is_market_hours():
                    await self._scan_watchlist()
                else:
                    logger.info("Outside market hours, skipping scan")
                
                # Wait for next scan interval
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Watchlist scanner loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in watchlist scanner loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute before retrying on error
    
    async def _scan_watchlist(self):
        """Scan all watchlist stocks for flip level crossings"""
        logger.info(f"Scanning {len(self.watchlist)} stocks for flip level crossings")
        
        try:
            client = SchwabClient()
            crossings = []
            
            for symbol in self.watchlist:
                try:
                    crossing = await self._check_flip_crossing(symbol, client)
                    if crossing:
                        crossings.append(crossing)
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error checking {symbol}: {e}")
                    continue
            
            # Send alerts for any crossings detected
            if crossings:
                await self._send_crossing_alerts(crossings)
                logger.info(f"Found {len(crossings)} flip level crossings")
            else:
                logger.info("No flip level crossings detected")
                
        except Exception as e:
            logger.error(f"Error scanning watchlist: {e}", exc_info=True)
    
    async def _check_flip_crossing(self, symbol: str, client: SchwabClient) -> Optional[Dict]:
        """Check if a stock has crossed its flip level"""
        try:
            from datetime import date
            
            # Get today's date for daily expiry
            today = date.today()
            exp_date_str = today.strftime('%Y-%m-%d')
            
            # Call the API server to get key levels (including flip level)
            api_url = f"http://localhost:8000/api/key_levels?symbol={symbol}&expiry={exp_date_str}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.warning(f"API returned status {response.status} for {symbol}")
                        return None
                    
                    data = await response.json()
                    
                    if not data.get('success') or not data.get('flip_level'):
                        return None
            
            current_price = data['underlying_price']
            flip_level = data['flip_level']
            call_wall = data.get('call_wall', {}).get('strike') if data.get('call_wall') else None
            put_wall = data.get('put_wall', {}).get('strike') if data.get('put_wall') else None
            max_gex = data.get('max_gex', {})
            max_gex_strike = max_gex.get('strike') if max_gex else None
            max_gex_value = max_gex.get('gex') if max_gex else None
            
            # Determine current position
            current_position = "above" if current_price >= flip_level else "below"
            previous_position = self.previous_positions.get(symbol)
            
            # Detect crossing
            crossing_detected = False
            crossing_direction = None
            
            if previous_position and previous_position != current_position:
                crossing_detected = True
                crossing_direction = "bullish" if current_position == "above" else "bearish"
            
            # Update position tracking
            self.previous_positions[symbol] = current_position
            
            # Return crossing data if detected
            if crossing_detected:
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'flip_level': flip_level,
                    'call_wall': call_wall,
                    'put_wall': put_wall,
                    'max_gex_strike': max_gex_strike,
                    'max_gex_value': max_gex_value,
                    'direction': crossing_direction,
                    'previous_position': previous_position,
                    'current_position': current_position
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking flip crossing for {symbol}: {e}")
            return None
    
    async def _send_crossing_alerts(self, crossings: list):
        """Send Discord alerts for flip level crossings"""
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            for crossing in crossings:
                embed = self._create_crossing_embed(crossing)
                await channel.send(embed=embed)
                await asyncio.sleep(1)  # Rate limit protection
                
        except Exception as e:
            logger.error(f"Error sending crossing alerts: {e}", exc_info=True)
    
    def _create_crossing_embed(self, crossing: Dict) -> discord.Embed:
        """Create a Discord embed for a flip level crossing"""
        symbol = crossing['symbol']
        direction = crossing['direction']
        
        # Set color and emoji based on direction
        if direction == "bullish":
            color = discord.Color.green()
            emoji = "🚀"
            title = f"{emoji} {symbol} Crossed ABOVE Flip Level"
        else:
            color = discord.Color.red()
            emoji = "📉"
            title = f"{emoji} {symbol} Crossed BELOW Flip Level"
        
        embed = discord.Embed(
            title=title,
            description=f"Price moved from **{crossing['previous_position']}** to **{crossing['current_position']}** flip level",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Current Price
        embed.add_field(
            name="💰 Current Price",
            value=f"${crossing['current_price']:.2f}",
            inline=True
        )
        
        # Flip Level
        flip_dist = crossing['current_price'] - crossing['flip_level']
        embed.add_field(
            name="🎯 Flip Level",
            value=f"${crossing['flip_level']:.2f}\n({flip_dist:+.2f})",
            inline=True
        )
        
        # Distance %
        flip_pct = (flip_dist / crossing['flip_level']) * 100
        embed.add_field(
            name="📊 Distance",
            value=f"{flip_pct:+.2f}%",
            inline=True
        )
        
        # Call Wall
        if crossing['call_wall']:
            cw_dist = crossing['call_wall'] - crossing['current_price']
            embed.add_field(
                name="📞 Call Wall",
                value=f"${crossing['call_wall']:.2f}\n({cw_dist:+.2f})",
                inline=True
            )
        
        # Put Wall
        if crossing['put_wall']:
            pw_dist = crossing['put_wall'] - crossing['current_price']
            embed.add_field(
                name="📱 Put Wall",
                value=f"${crossing['put_wall']:.2f}\n({pw_dist:+.2f})",
                inline=True
            )
        
        # Max GEX
        if crossing['max_gex_strike']:
            embed.add_field(
                name="⚡ Max GEX",
                value=f"${crossing['max_gex_strike']:.2f}\n{crossing['max_gex_value']/1e6:.1f}M",
                inline=True
            )
        
        embed.set_footer(text=f"Watchlist Scanner • Daily Timeframe")
        
        return embed


def setup(bot):
    """Register the cog"""
    return bot.add_cog(WatchlistScannerCommands(bot))
