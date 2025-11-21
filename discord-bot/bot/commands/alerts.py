"""
Enhanced Multi-Channel Alert Commands for Discord Bot
Separate channels for: Whale Flows, 0DTE Levels, Market Intelligence
"""

import asyncio
import discord
import logging
from datetime import datetime, time
from typing import Set, Optional
from pathlib import Path
import sys
import pytz

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient
from bot.services.multi_channel_alert_service import MultiChannelAlertService

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup alert commands"""
    await bot.add_cog(AlertCommands(bot))


class AlertCommands(discord.ext.commands.Cog):
    """Commands for managing automated alerts across multiple channels"""
    
    def __init__(self, bot):
        self.bot = bot
        # Initialize multi-channel alert service
        if not hasattr(bot, 'multi_alert_service'):
            bot.multi_alert_service = MultiChannelAlertService(bot)
        self.alert_service = bot.multi_alert_service
    
    @discord.app_commands.command(
        name="setup_whale_alerts",
        description="Set this channel for whale flow alerts (score > 300)"
    )
    async def setup_whale_alerts(self, interaction: discord.Interaction):
        """Set the current channel for whale flow alerts"""
        try:
            channel_id = interaction.channel_id
            self.alert_service.set_whale_channel(channel_id)
            
            embed = discord.Embed(
                title="🐋 Whale Flow Alerts Configured",
                description=(
                    f"This channel will receive alerts for:\n"
                    f"• Individual stock whale flows (score > 300)\n"
                    f"• High conviction institutional options activity\n"
                    f"• Auto-updates every 15 minutes during market hours"
                ),
                color=0x00ff00
            )
            embed.add_field(
                name="Monitored Stocks",
                value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, CRM, PLTR, COIN, SNOW, CRWD, APP",
                inline=False
            )
            embed.set_footer(text=f"Channel ID: {channel_id}")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Whale alerts configured for channel {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting up whale alerts: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error setting up whale alerts: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="setup_0dte_alerts",
        description="Set this channel for 0DTE level alerts (SPY/QQQ/SPX)"
    )
    async def setup_dte_alerts(self, interaction: discord.Interaction):
        """Set the current channel for 0DTE alerts"""
        try:
            channel_id = interaction.channel_id
            self.alert_service.set_dte_channel(channel_id)
            
            embed = discord.Embed(
                title="📊 0DTE Alerts Configured",
                description=(
                    f"This channel will receive alerts for:\n"
                    f"• SPY, QQQ, SPX 0DTE levels\n"
                    f"• Call/Put walls and max pain\n"
                    f"• Price positioning relative to key strikes\n"
                    f"• Auto-updates every 15 minutes during market hours"
                ),
                color=0x3498db
            )
            embed.add_field(
                name="Symbols Tracked",
                value="SPY, QQQ, $SPX",
                inline=False
            )
            embed.set_footer(text=f"Channel ID: {channel_id}")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"0DTE alerts configured for channel {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting up 0DTE alerts: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error setting up 0DTE alerts: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="setup_market_intel",
        description="Set this channel for SPY/QQQ market intelligence alerts"
    )
    async def setup_market_intel(self, interaction: discord.Interaction):
        """Set the current channel for market intelligence alerts"""
        try:
            channel_id = interaction.channel_id
            self.alert_service.set_market_intel_channel(channel_id)
            
            embed = discord.Embed(
                title="🧠 Market Intelligence Configured",
                description=(
                    f"This channel will receive advanced market analysis:\n\n"
                    f"**SPY & QQQ Intelligence (Next 10 Expiries):**\n"
                    f"• Directional bias detection (bullish/bearish signals)\n"
                    f"• Net Gamma Exposure (GEX) analysis\n"
                    f"• Put/Call ratio trends\n"
                    f"• ATM vs OTM flow analysis\n"
                    f"• Fresh institutional positioning (high Vol/OI)\n"
                    f"• Volume acceleration patterns\n"
                    f"• Actionable trading implications\n\n"
                    f"Auto-updates every 15 minutes during market hours"
                ),
                color=0x9b59b6
            )
            embed.add_field(
                name="📈 Analysis Scope",
                value="Aggregates data across next 10 expiries for comprehensive market view",
                inline=False
            )
            embed.add_field(
                name="🎯 Signal Types",
                value="STRONG_BULLISH 🚀 | BULLISH 🟢 | NEUTRAL 🟡 | BEARISH 🔴 | STRONG_BEARISH 💀",
                inline=False
            )
            embed.set_footer(text=f"Channel ID: {channel_id}")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Market intelligence alerts configured for channel {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting up market intel: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error setting up market intelligence: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="start_alerts",
        description="Start automated alert service (all configured channels)"
    )
    @discord.app_commands.checks.has_permissions(administrator=True)
    async def start_alerts(self, interaction: discord.Interaction):
        """Start the automated alert service"""
        try:
            await interaction.response.defer()
            
            # Check if any channels are configured
            configured = []
            if self.alert_service.whale_channel_id:
                configured.append(f"🐋 Whale Flows: <#{self.alert_service.whale_channel_id}>")
            if self.alert_service.dte_channel_id:
                configured.append(f"📊 0DTE Levels: <#{self.alert_service.dte_channel_id}>")
            if self.alert_service.market_intel_channel_id:
                configured.append(f"🧠 Market Intel: <#{self.alert_service.market_intel_channel_id}>")
            
            if not configured:
                await interaction.followup.send(
                    "❌ No alert channels configured! Use `/setup_whale_alerts`, `/setup_0dte_alerts`, or `/setup_market_intel` first.",
                    ephemeral=True
                )
                return
            
            await self.alert_service.start()
            
            embed = discord.Embed(
                title="✅ Alert Service Started",
                description="Automated alerts are now running during market hours (9:30 AM - 4:00 PM ET)",
                color=0x00ff00
            )
            
            embed.add_field(
                name="📡 Active Channels",
                value="\n".join(configured),
                inline=False
            )
            
            embed.add_field(
                name="⏱️ Scan Interval",
                value="Every 15 minutes",
                inline=True
            )
            
            embed.add_field(
                name="📅 Active Days",
                value="Monday - Friday",
                inline=True
            )
            
            embed.set_footer(text="Use /stop_alerts to disable")
            
            await interaction.followup.send(embed=embed)
            logger.info("Alert service started via command")
            
        except Exception as e:
            logger.error(f"Error starting alerts: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ Error starting alerts: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="stop_alerts",
        description="Stop automated alert service"
    )
    @discord.app_commands.checks.has_permissions(administrator=True)
    async def stop_alerts(self, interaction: discord.Interaction):
        """Stop the automated alert service"""
        try:
            await self.alert_service.stop()
            
            embed = discord.Embed(
                title="🛑 Alert Service Stopped",
                description="Automated alerts have been disabled",
                color=0xff0000
            )
            embed.set_footer(text="Use /start_alerts to re-enable")
            
            await interaction.response.send_message(embed=embed)
            logger.info("Alert service stopped via command")
            
        except Exception as e:
            logger.error(f"Error stopping alerts: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error stopping alerts: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="alert_status",
        description="Check alert service status and configuration"
    )
    async def alert_status(self, interaction: discord.Interaction):
        """Check the status of the alert service"""
        try:
            embed = discord.Embed(
                title="📊 Alert Service Status",
                color=0x3498db if self.alert_service.is_running else 0x95a5a6
            )
            
            # Status
            status = "🟢 Running" if self.alert_service.is_running else "🔴 Stopped"
            embed.add_field(name="Status", value=status, inline=True)
            
            # Market hours check
            is_market_hours = self.alert_service.is_market_hours()
            market_status = "🟢 Open" if is_market_hours else "🔴 Closed"
            embed.add_field(name="Market", value=market_status, inline=True)
            
            # Scan interval
            embed.add_field(
                name="Scan Interval",
                value=f"{self.alert_service.scan_interval_minutes} minutes",
                inline=True
            )
            
            # Configured channels
            channels_info = []
            if self.alert_service.whale_channel_id:
                channels_info.append(f"🐋 Whale Flows: <#{self.alert_service.whale_channel_id}>")
            if self.alert_service.dte_channel_id:
                channels_info.append(f"📊 0DTE Levels: <#{self.alert_service.dte_channel_id}>")
            if self.alert_service.market_intel_channel_id:
                channels_info.append(f"🧠 Market Intel: <#{self.alert_service.market_intel_channel_id}>")
            
            if channels_info:
                embed.add_field(
                    name="📡 Configured Channels",
                    value="\n".join(channels_info),
                    inline=False
                )
            else:
                embed.add_field(
                    name="📡 Configured Channels",
                    value="❌ No channels configured",
                    inline=False
                )
            
            # Cache stats
            embed.add_field(
                name="📊 Alert Cache",
                value=f"Whale: {len(self.alert_service.sent_whale_alerts)} | Market Intel: {len(self.alert_service.sent_market_intel)}",
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
            
        except Exception as e:
            logger.error(f"Error checking alert status: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error checking status: {str(e)}",
                ephemeral=True
            )
