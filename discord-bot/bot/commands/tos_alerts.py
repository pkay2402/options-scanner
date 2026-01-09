"""
TOS Scan Alert Command
Monitors ThinkorSwim email alerts and sends to Discord automatically
Runs every 2-5 minutes during market hours
"""

import asyncio
import discord
from discord import app_commands
from discord.ext import commands
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Set
from pathlib import Path
import sys
import imaplib
import email
import re
import json
from dateutil import parser
from bs4 import BeautifulSoup
import pytz

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup TOS alerts command"""
    await bot.add_cog(TOSAlertsCommands(bot))


class TOSAlertsCommands(commands.Cog):
    """Commands for automated TOS scan alerts"""
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        self.check_interval_minutes = 3  # Check every 3 minutes
        
        # Email configuration (from bot config or env)
        self.email_address = None
        self.email_password = None
        self.sender_email = "alerts@thinkorswim.com"
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Track sent alerts to avoid duplicates (ticker_signal_date: timestamp)
        self.sent_today: Set[str] = set()
        self.processed_email_ids: Set[bytes] = set()
        
        # TOS scan keywords to monitor
        self.keywords = ["HG_30mins_L", "HG_30mins_S"]
        
        # Config file for persistence
        self.config_file = project_root / "discord-bot" / "tos_alerts_config.json"
        self._load_config()
        
        # Load email credentials
        self._load_credentials()
    
    def _load_config(self):
        """Load saved channel and scanner state"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.channel_id = config.get('channel_id')
                    was_running = config.get('is_running', False)
                    
                    if self.channel_id and was_running:
                        logger.info(f"Loaded TOS config: channel_id={self.channel_id}, auto-start enabled")
                    elif self.channel_id:
                        logger.info(f"Loaded TOS config: channel_id={self.channel_id}, was stopped")
        except Exception as e:
            logger.error(f"Error loading TOS config: {e}")
    
    def _save_config(self):
        """Save current channel and scanner state"""
        try:
            config = {
                'channel_id': self.channel_id,
                'is_running': self.is_running,
                'check_interval_minutes': self.check_interval_minutes
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"TOS config saved: {config}")
        except Exception as e:
            logger.error(f"Error saving TOS config: {e}")
    
    def _load_credentials(self):
        """Load email credentials from config"""
        try:
            import os
            self.email_address = os.getenv('TOS_EMAIL_ADDRESS')
            self.email_password = os.getenv('TOS_EMAIL_PASSWORD')
            
            if not self.email_address or not self.email_password:
                logger.warning("TOS email credentials not found in environment variables")
                logger.warning("Set TOS_EMAIL_ADDRESS and TOS_EMAIL_PASSWORD")
        except Exception as e:
            logger.error(f"Error loading TOS email credentials: {e}")
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Auto-start scanner if it was previously running"""
        if self.channel_id and not self.is_running:
            try:
                # Check if scanner was running before restart
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                        was_running = config.get('is_running', False)
                        
                        if was_running:
                            logger.info(f"Auto-starting TOS alerts monitor for channel {self.channel_id}")
                            self.is_running = True
                            self.scanner_task = asyncio.create_task(self._scanner_loop())
                            
                            # Send notification to channel
                            channel = self.bot.get_channel(self.channel_id)
                            if channel:
                                embed = discord.Embed(
                                    title="📧 TOS Alerts Auto-Resumed",
                                    description="Monitoring resumed after bot restart",
                                    color=discord.Color.green()
                                )
                                await channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Error auto-starting TOS alerts: {e}")
    
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
    
    def _connect_to_email(self, retries=3):
        """Establish email connection with retry logic"""
        for attempt in range(retries):
            try:
                mail = imaplib.IMAP4_SSL('imap.gmail.com')
                mail.login(self.email_address, self.email_password)
                return mail
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Email connection attempt {attempt + 1} failed: {e}")
                asyncio.sleep(2)
        return None
    
    def _parse_email_body(self, msg):
        """Parse email body with HTML handling"""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() in ["text/plain", "text/html"]:
                        body = part.get_payload(decode=True).decode()
                        if part.get_content_type() == "text/html":
                            soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                            return soup.get_text(separator=' ', strip=True)
                        return body
            else:
                body = msg.get_payload(decode=True).decode()
                if msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                    return soup.get_text(separator=' ', strip=True)
                return body
        except Exception as e:
            logger.error(f"Error parsing email body: {e}")
            return ""
    
    async def _check_tos_alerts(self):
        """Check email for new TOS alerts and send to Discord"""
        if not self.email_address or not self.email_password:
            logger.error("Email credentials not configured")
            return
        
        eastern = pytz.timezone('US/Eastern')
        today_et = datetime.now(eastern).date()
        
        try:
            mail = self._connect_to_email()
            if not mail:
                return
            
            mail.select('inbox')
            
            # Search for today's emails from TOS
            date_since = today_et.strftime("%d-%b-%Y")
            
            for keyword in self.keywords:
                search_criteria = f'(FROM "{self.sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
                _, data = mail.search(None, search_criteria)
                
                for num in data[0].split():
                    if num in self.processed_email_ids:
                        continue
                    
                    _, email_data = mail.fetch(num, '(RFC822)')
                    msg = email.message_from_bytes(email_data[0][1])
                    
                    # Parse date
                    email_datetime = parser.parse(msg['Date'])
                    email_date = email_datetime.date()
                    
                    # Skip if not today or weekend
                    if email_date != today_et or email_datetime.weekday() >= 5:
                        continue
                    
                    # Parse body for symbols
                    body = self._parse_email_body(msg)
                    symbols = re.findall(
                        r'New symbols:\s*([A-Z,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', 
                        body
                    )
                    
                    if symbols:
                        for symbol_group in symbols:
                            extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                            signal_type = symbol_group[1]
                            
                            for ticker in extracted_symbols:
                                if ticker.isalpha():
                                    # Create deduplication key
                                    dedup_key = f"{ticker}_{signal_type}_{today_et}"
                                    
                                    if dedup_key not in self.sent_today:
                                        # Send to Discord
                                        await self._send_alert_to_discord(
                                            ticker, signal_type, email_datetime
                                        )
                                        self.sent_today.add(dedup_key)
                                        await asyncio.sleep(0.5)  # Rate limiting
                    
                    self.processed_email_ids.add(num)
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            logger.error(f"Error checking TOS alerts: {e}")
    
    async def _send_alert_to_discord(self, ticker: str, signal: str, alert_time: datetime):
        """Send TOS alert to Discord channel"""
        if not self.channel_id:
            return
        
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            # Determine signal type and color
            is_long = "_L" in signal
            signal_type = "LONG" if is_long else "SHORT"
            color = discord.Color.green() if is_long else discord.Color.red()
            emoji = "🟢" if is_long else "🔴"
            
            # Create embed
            embed = discord.Embed(
                title=f"{emoji} TOS {signal_type} Alert: {ticker}",
                description=f"**ThinkorSwim High Grade 30-Min Signal**\n`{signal}`",
                color=color,
                timestamp=alert_time
            )
            
            embed.add_field(name="Ticker", value=ticker, inline=True)
            embed.add_field(name="Signal Type", value=signal_type, inline=True)
            embed.add_field(name="Scan", value=signal, inline=True)
            embed.set_footer(text="TOS Scan Alert • 30-Min Timeframe")
            
            await channel.send(embed=embed)
            logger.info(f"Sent TOS alert: {ticker} {signal}")
            
        except Exception as e:
            logger.error(f"Error sending Discord alert for {ticker}: {e}")
    
    async def _scanner_loop(self):
        """Main loop - checks for alerts during market hours"""
        logger.info("TOS alerts scanner loop started")
        
        while self.is_running:
            try:
                # Check if market hours
                if self.is_market_hours():
                    logger.info("Checking for new TOS alerts...")
                    await self._check_tos_alerts()
                else:
                    # Clear sent cache at start of new day
                    eastern = pytz.timezone('US/Eastern')
                    now_et = datetime.now(eastern)
                    
                    # If it's a new day, clear the sent cache
                    if now_et.hour == 0 and now_et.minute < self.check_interval_minutes:
                        self.sent_today.clear()
                        self.processed_email_ids.clear()
                        logger.info("New day - cleared sent alerts cache")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in TOS alerts scanner loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    @discord.app_commands.command(name="setup_tos_alerts", description="Setup TOS scan alerts for this channel")
    async def setup_tos_alerts(self, interaction: discord.Interaction):
        """Configure TOS alerts for current channel"""
        self.channel_id = interaction.channel_id
        self._save_config()
        await interaction.response.send_message(
            f"✅ TOS scan alerts configured for this channel!\n"
            f"Use `/start_tos_alerts` to begin monitoring.",
            ephemeral=True
        )
        logger.info(f"TOS alerts configured for channel {self.channel_id}")
    
    @discord.app_commands.command(name="start_tos_alerts", description="Start automated TOS alert monitoring")
    async def start_tos_alerts(self, interaction: discord.Interaction):
        """Start the automated TOS alert scanner"""
        if not self.channel_id:
            await interaction.response.send_message(
                "❌ Please run `/setup_tos_alerts` first to configure the channel.",
                ephemeral=True
            )
            return
        
        if not self.email_address or not self.email_password:
            await interaction.response.send_message(
                "❌ Email credentials not configured. Please set TOS_EMAIL_ADDRESS and TOS_EMAIL_PASSWORD environment variables.",
                ephemeral=True
            )
            return
        
        if self.is_running:
            await interaction.response.send_message(
                "⚠️ TOS alerts monitoring is already running!",
                ephemeral=True
            )
            return
        
        self.is_running = True
        self._save_config()
        self.scanner_task = asyncio.create_task(self._scanner_loop())
        
        embed = discord.Embed(
            title="📧 TOS Alerts Monitor Started",
            description="Monitoring ThinkorSwim email alerts during market hours",
            color=discord.Color.green()
        )
        embed.add_field(name="Check Interval", value=f"{self.check_interval_minutes} minutes", inline=True)
        embed.add_field(name="Market Hours", value="9:30 AM - 4:00 PM ET", inline=True)
        embed.add_field(name="Scans", value="HG_30mins_L, HG_30mins_S", inline=False)
        embed.set_footer(text="Alerts will be sent automatically when detected")
        
        await interaction.response.send_message(embed=embed)
        logger.info("TOS alerts monitoring started")
    
    @discord.app_commands.command(name="stop_tos_alerts", description="Stop automated TOS alert monitoring")
    async def stop_tos_alerts(self, interaction: discord.Interaction):
        """Stop the automated TOS alert scanner"""
        if not self.is_running:
            await interaction.response.send_message(
                "⚠️ TOS alerts monitoring is not running.",
                ephemeral=True
            )
            return
        
        self.is_running = False
        self._save_config()
        if self.scanner_task:
            self.scanner_task.cancel()
            self.scanner_task = None
        
        await interaction.response.send_message(
            "✅ TOS alerts monitoring stopped.",
            ephemeral=True
        )
        logger.info("TOS alerts monitoring stopped")
    
    @discord.app_commands.command(name="tos_alerts_status", description="Check TOS alerts monitor status")
    async def tos_alerts_status(self, interaction: discord.Interaction):
        """Show current status of TOS alerts monitor"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        embed = discord.Embed(
            title="📊 TOS Alerts Monitor Status",
            color=discord.Color.blue()
        )
        
        # Running status
        status_emoji = "🟢" if self.is_running else "🔴"
        status_text = "Running" if self.is_running else "Stopped"
        embed.add_field(name="Status", value=f"{status_emoji} {status_text}", inline=True)
        
        # Market hours status
        market_status = "🟢 OPEN" if self.is_market_hours() else "🔴 CLOSED"
        embed.add_field(name="Market", value=market_status, inline=True)
        
        # Credentials status
        creds_status = "✅" if self.email_address and self.email_password else "❌"
        embed.add_field(name="Email Config", value=creds_status, inline=True)
        
        # Channel
        if self.channel_id:
            embed.add_field(name="Alert Channel", value=f"<#{self.channel_id}>", inline=True)
        
        # Stats
        embed.add_field(name="Alerts Sent Today", value=str(len(self.sent_today)), inline=True)
        embed.add_field(name="Check Interval", value=f"{self.check_interval_minutes} min", inline=True)
        
        embed.set_footer(text=f"Current time: {now_et.strftime('%I:%M %p ET')}")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
