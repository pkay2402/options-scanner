"""
ETF Momentum Scanner Command
Sends top 10 leveraged ETFs every 15 minutes during first hour of trading
Then repeats at market close
Uses Month/Week/Day performance criteria with volume filter
"""

import asyncio
import discord
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import sys
import json
import pytz
import yfinance as yf
import pandas as pd

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup ETF momentum scanner command"""
    await bot.add_cog(ETFMomentumCommands(bot))


class ETFMomentumCommands(discord.ext.commands.Cog):
    """Commands for ETF momentum scanning"""
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        
        # Load ETF list from extracted_symbols.csv
        self.etf_list = self._load_etf_list()
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Scan schedule (first hour: 9:30, 9:45, 10:00, 10:15, then at close)
        self.first_hour_scans = [
            datetime.strptime("09:30", "%H:%M").time(),
            datetime.strptime("09:45", "%H:%M").time(),
            datetime.strptime("10:00", "%H:%M").time(),
            datetime.strptime("10:15", "%H:%M").time(),
        ]
        self.close_scan_time = datetime.strptime("15:45", "%H:%M").time()  # 15 min before close
        
        # Config file for persistence
        self.config_file = project_root / "discord-bot" / "etf_momentum_config.json"
        self._load_config()
        
        # Cache
        self.last_scan_time = None
        self.last_results = None
    
    def _load_config(self):
        """Load saved channel and scanner state"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.channel_id = config.get('channel_id')
                    was_running = config.get('is_running', False)
                    
                    if self.channel_id and was_running:
                        logger.info(f"Loaded ETF momentum config: channel_id={self.channel_id}, auto-start enabled")
                    elif self.channel_id:
                        logger.info(f"Loaded ETF momentum config: channel_id={self.channel_id}, was stopped")
        except Exception as e:
            logger.error(f"Error loading ETF momentum config: {e}")
    
    def _save_config(self):
        """Save current channel and scanner state"""
        try:
            config = {
                'channel_id': self.channel_id,
                'is_running': self.is_running,
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"ETF momentum config saved: {config}")
        except Exception as e:
            logger.error(f"Error saving ETF momentum config: {e}")
    
    @discord.ext.commands.Cog.listener()
    async def on_ready(self):
        """Auto-start scanner if it was previously running"""
        if self.channel_id and not self.is_running:
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                        if config.get('is_running', False):
                            channel = self.bot.get_channel(self.channel_id)
                            if channel:
                                self.is_running = True
                                self.scanner_task = asyncio.create_task(self._scanner_loop())
                                logger.info(f"✅ Auto-started ETF momentum scanner in channel: {channel.name}")
                            else:
                                logger.warning(f"Could not find channel {self.channel_id} for ETF momentum auto-start")
            except Exception as e:
                logger.error(f"Error auto-starting ETF momentum scanner: {e}")
    
    def _load_etf_list(self):
        """Load ETF symbols from extracted_symbols.csv"""
        try:
            csv_path = project_root / "extracted_symbols.csv"
            if csv_path.exists():
                etfs = []
                with open(csv_path, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            symbol = line.split(',')[0].strip()
                            etfs.append(symbol)
                logger.info(f"Loaded {len(etfs)} ETF symbols from extracted_symbols.csv")
                return etfs
            else:
                logger.warning("extracted_symbols.csv not found, using default list")
                return ['SPY', 'QQQ', 'IWM', 'TQQQ', 'SQQQ', 'SPXL', 'SOXL', 'TNA', 'UVXY']
        except Exception as e:
            logger.error(f"Error loading ETF list: {e}")
            return ['SPY', 'QQQ', 'IWM', 'TQQQ', 'SQQQ']
    
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
    
    def should_scan_now(self) -> bool:
        """Check if we should scan at this time"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        current_time = now_et.time()
        
        # Check if market is open
        if not self.is_market_hours():
            return False
        
        # Check if it's one of our scan times (within 1 minute tolerance)
        for scan_time in self.first_hour_scans:
            if abs((datetime.combine(datetime.today(), current_time) - 
                   datetime.combine(datetime.today(), scan_time)).total_seconds()) < 60:
                return True
        
        # Check if it's close scan time
        if abs((datetime.combine(datetime.today(), current_time) - 
               datetime.combine(datetime.today(), self.close_scan_time)).total_seconds()) < 60:
            return True
        
        return False
    
    def calculate_etf_momentum(self, symbol: str) -> Optional[Dict]:
        """
        Calculate momentum metrics for an ETF
        Returns dict with scores or None if criteria not met
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data (need 30 days for monthly)
            hist = ticker.history(period="2mo", interval="1d")
            if hist.empty or len(hist) < 22:
                return None
            
            # Get latest price data
            latest_close = hist['Close'].iloc[-1]
            
            # Calculate percentage changes
            month_close = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]  # 21 trading days ago
            week_close = hist['Close'].iloc[-6] if len(hist) >= 6 else hist['Close'].iloc[0]  # 5 trading days ago
            day_close = hist['Close'].iloc[-2] if len(hist) >= 2 else hist['Close'].iloc[0]  # 1 trading day ago
            
            month_change = ((latest_close - month_close) / month_close) * 100
            week_change = ((latest_close - week_close) / week_close) * 100
            day_change = ((latest_close - day_close) / day_close) * 100
            
            # Volume filter (20-day average)
            avg_volume = hist['Volume'].tail(20).mean()
            
            # Apply scan criteria
            if (month_change > 5 and 
                week_change > 2 and 
                day_change > 0.5 and 
                avg_volume > 500000):
                
                # Calculate momentum score (weighted)
                momentum_score = (month_change * 0.4) + (week_change * 0.35) + (day_change * 0.25)
                
                return {
                    'symbol': symbol,
                    'price': latest_close,
                    'month_change': month_change,
                    'week_change': week_change,
                    'day_change': day_change,
                    'avg_volume': avg_volume,
                    'momentum_score': momentum_score
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {e}")
            return None
    
    async def _scan_etfs(self) -> List[Dict]:
        """Scan all ETFs and return top 10 by momentum"""
        try:
            logger.info(f"Scanning {len(self.etf_list)} ETFs for momentum...")
            results = []
            
            # Scan in batches to avoid overwhelming yfinance
            batch_size = 10
            for i in range(0, len(self.etf_list), batch_size):
                batch = self.etf_list[i:i+batch_size]
                
                for symbol in batch:
                    try:
                        momentum_data = self.calculate_etf_momentum(symbol)
                        if momentum_data:
                            results.append(momentum_data)
                    except Exception as e:
                        logger.error(f"Error scanning {symbol}: {e}")
                        continue
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            # Sort by momentum score and return top 10
            results.sort(key=lambda x: x['momentum_score'], reverse=True)
            top_10 = results[:10]
            
            logger.info(f"Found {len(results)} qualifying ETFs, returning top 10")
            return top_10
            
        except Exception as e:
            logger.error(f"Error in ETF scan: {e}")
            return []
    
    def _create_momentum_embed(self, etfs: List[Dict], scan_time: datetime) -> discord.Embed:
        """Create Discord embed for ETF momentum results"""
        eastern = pytz.timezone('US/Eastern')
        scan_time_et = scan_time.astimezone(eastern)
        
        embed = discord.Embed(
            title="🚀 ETF Momentum Scanner",
            description=f"Top 10 Leveraged ETFs • {scan_time_et.strftime('%I:%M %p ET')}",
            color=discord.Color.green(),
            timestamp=scan_time
        )
        
        if not etfs:
            embed.add_field(
                name="No Results",
                value="No ETFs currently meet the momentum criteria:\n• Month: >5%\n• Week: >2%\n• Day: >0.5%\n• Avg Volume: >500K",
                inline=False
            )
            return embed
        
        # Add criteria info
        embed.add_field(
            name="📋 Scan Criteria",
            value="Month >5% • Week >2% • Day >0.5% • Vol >500K",
            inline=False
        )
        
        # Add top 10 ETFs
        for i, etf in enumerate(etfs, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"**{i}.**"
            
            value_lines = [
                f"**Price:** ${etf['price']:.2f}",
                f"**Month:** {etf['month_change']:+.1f}% | **Week:** {etf['week_change']:+.1f}% | **Day:** {etf['day_change']:+.1f}%",
                f"**Score:** {etf['momentum_score']:.1f} | **Avg Vol:** {etf['avg_volume']:,.0f}"
            ]
            
            embed.add_field(
                name=f"{medal} {etf['symbol']}",
                value="\n".join(value_lines),
                inline=False
            )
        
        # Add footer with next scan time
        current_time_et = scan_time_et.time()
        next_scan = None
        
        for scan_time in self.first_hour_scans:
            if current_time_et < scan_time:
                next_scan = scan_time
                break
        
        if next_scan:
            embed.set_footer(text=f"Next scan at {next_scan.strftime('%I:%M %p ET')}")
        else:
            embed.set_footer(text=f"Next scan at {self.close_scan_time.strftime('%I:%M %p ET')} (Market Close)")
        
        return embed
    
    async def _scanner_loop(self):
        """Main scanner loop - runs at specific times during market hours"""
        logger.info("ETF Momentum scanner started")
        
        while self.is_running:
            try:
                # Check if we should scan now
                if not self.should_scan_now():
                    await asyncio.sleep(30)  # Check every 30 seconds
                    continue
                
                # Check if we already scanned in the last minute
                if self.last_scan_time:
                    time_since_last = (datetime.now(pytz.UTC) - self.last_scan_time).total_seconds()
                    if time_since_last < 60:
                        await asyncio.sleep(30)
                        continue
                
                # Check if channel is set
                if not self.channel_id:
                    logger.warning("No channel set for ETF Momentum alerts")
                    await asyncio.sleep(30)
                    continue
                
                # Get channel
                channel = self.bot.get_channel(self.channel_id)
                if not channel:
                    logger.error(f"Channel {self.channel_id} not found")
                    await asyncio.sleep(30)
                    continue
                
                # Run scan
                logger.info("Running ETF Momentum scan...")
                scan_time = datetime.now(pytz.UTC)
                etf_results = await self._scan_etfs()
                
                # Create and send embed
                embed = self._create_momentum_embed(etf_results, scan_time)
                await channel.send(embed=embed)
                
                logger.info(f"Sent ETF Momentum alert with {len(etf_results)} ETFs")
                
                # Cache results
                self.last_scan_time = scan_time
                self.last_results = etf_results
                
                # Wait a minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in ETF momentum scanner loop: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    @discord.app_commands.command(
        name="setup_etf_momentum",
        description="Setup channel for ETF momentum alerts (scans first hour + close)"
    )
    async def setup_etf_momentum(self, interaction: discord.Interaction):
        """Setup the channel for ETF momentum alerts"""
        try:
            self.channel_id = interaction.channel_id
            self._save_config()
            
            embed = discord.Embed(
                title="✅ ETF Momentum Scanner Setup Complete",
                description=f"This channel will receive ETF momentum alerts",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="📅 Scan Schedule",
                value="• **First Hour:** 9:30, 9:45, 10:00, 10:15 AM ET\n• **Market Close:** 3:45 PM ET",
                inline=False
            )
            
            embed.add_field(
                name="📊 Criteria",
                value="• Monthly Change: >5%\n• Weekly Change: >2%\n• Daily Change: >0.5%\n• Average Volume: >500,000",
                inline=False
            )
            
            embed.add_field(
                name="📝 ETF Universe",
                value=f"{len(self.etf_list)} leveraged ETFs from extracted_symbols.csv",
                inline=False
            )
            
            embed.add_field(
                name="🎮 Commands",
                value="• `/start_etf_momentum` - Start scanner\n• `/stop_etf_momentum` - Stop scanner\n• `/etf_momentum_now` - Run scan immediately",
                inline=False
            )
            
            embed.set_footer(text="Use /start_etf_momentum to begin monitoring")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"ETF momentum scanner setup in channel: {interaction.channel.name}")
            
        except Exception as e:
            logger.error(f"Error setting up ETF momentum: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error setting up ETF Momentum scanner: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="start_etf_momentum",
        description="Start the ETF momentum scanner (must setup channel first)"
    )
    async def start_etf_momentum(self, interaction: discord.Interaction):
        """Start the ETF momentum scanner"""
        try:
            if self.is_running:
                await interaction.response.send_message(
                    "⚠️ ETF Momentum scanner is already running!",
                    ephemeral=True
                )
                return
            
            if not self.channel_id:
                await interaction.response.send_message(
                    "❌ Please setup a channel first using `/setup_etf_momentum`",
                    ephemeral=True
                )
                return
            
            self.is_running = True
            self._save_config()
            self.scanner_task = asyncio.create_task(self._scanner_loop())
            
            await interaction.response.send_message(
                "✅ ETF Momentum scanner started! Will scan at 9:30, 9:45, 10:00, 10:15 AM and 3:45 PM ET during market hours.",
                ephemeral=True
            )
            logger.info("ETF Momentum scanner started by user")
            
        except Exception as e:
            logger.error(f"Error starting ETF momentum scanner: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error starting scanner: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="stop_etf_momentum",
        description="Stop the ETF momentum scanner"
    )
    async def stop_etf_momentum(self, interaction: discord.Interaction):
        """Stop the ETF momentum scanner"""
        try:
            if not self.is_running:
                await interaction.response.send_message(
                    "⚠️ ETF Momentum scanner is not running!",
                    ephemeral=True
                )
                return
            
            self.is_running = False
            self._save_config()
            if self.scanner_task:
                self.scanner_task.cancel()
                self.scanner_task = None
            
            await interaction.response.send_message(
                "✅ ETF Momentum scanner stopped.",
                ephemeral=True
            )
            logger.info("ETF Momentum scanner stopped by user")
            
        except Exception as e:
            logger.error(f"Error stopping ETF momentum scanner: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error stopping scanner: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="etf_momentum_now",
        description="Run ETF momentum scan immediately (manual test)"
    )
    async def etf_momentum_now(self, interaction: discord.Interaction):
        """Run ETF momentum scan immediately"""
        try:
            await interaction.response.defer(thinking=True)
            
            logger.info("Running manual ETF Momentum scan...")
            scan_time = datetime.now(pytz.UTC)
            etf_results = await self._scan_etfs()
            
            # Create and send embed
            embed = self._create_momentum_embed(etf_results, scan_time)
            
            await interaction.followup.send(embed=embed)
            logger.info(f"Manual ETF Momentum scan completed with {len(etf_results)} ETFs")
            
        except Exception as e:
            logger.error(f"Error in manual ETF momentum scan: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ Error running scan: {str(e)}"
            )
