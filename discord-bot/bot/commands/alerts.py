"""
Alert Configuration Commands
Commands to setup and control automated alerts
"""

import discord
from discord import app_commands
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)


class AlertCommands(commands.Cog):
    """Alert configuration and control commands"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="alerts_setup", description="Setup automated alerts in this channel")
    @app_commands.checks.has_permissions(administrator=True)
    async def setup_alerts(self, interaction: discord.Interaction):
        """Setup automated alerts to send to this channel"""
        try:
            # Set this channel as the alert channel
            channel_id = interaction.channel_id
            
            if hasattr(self.bot, 'alert_service'):
                self.bot.alert_service.set_channel_id(channel_id)
                
                embed = discord.Embed(
                    title="✅ Automated Alerts Configured",
                    description=f"Alerts will be sent to this channel: <#{channel_id}>",
                    color=0x00ff00
                )
                
                embed.add_field(
                    name="🐋 Whale Flow Alerts",
                    value="• Whale Score > 300\n• Scans every 15 minutes\n• No duplicate alerts per session\n• Top 10 results",
                    inline=False
                )
                
                embed.add_field(
                    name="📊 0DTE Levels",
                    value="• SPY, QQQ, $SPX\n• Current price vs walls\n• Call/Put volume\n• Max pain levels\n• Every 15 minutes",
                    inline=False
                )
                
                embed.add_field(
                    name="⏰ Schedule",
                    value="Active during market hours: 9:30 AM - 4:00 PM ET",
                    inline=False
                )
                
                embed.set_footer(text="Use /alerts_start to begin monitoring")
                
                await interaction.response.send_message(embed=embed)
                logger.info(f"Alerts configured for channel {channel_id}")
            else:
                await interaction.response.send_message("❌ Alert service not available", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error setting up alerts: {e}")
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="alerts_start", description="Start automated alert scanning")
    @app_commands.checks.has_permissions(administrator=True)
    async def start_alerts(self, interaction: discord.Interaction):
        """Start the automated alert service"""
        try:
            if not hasattr(self.bot, 'alert_service'):
                await interaction.response.send_message("❌ Alert service not available", ephemeral=True)
                return
            
            if not hasattr(self.bot.alert_service, 'channel_id'):
                await interaction.response.send_message(
                    "❌ Please run `/alerts_setup` first to configure the alert channel",
                    ephemeral=True
                )
                return
            
            await self.bot.alert_service.start()
            
            embed = discord.Embed(
                title="🚀 Automated Alerts Started",
                description="Alert service is now running",
                color=0x00ff00
            )
            
            embed.add_field(
                name="Status",
                value="✅ Active during market hours\n🔄 Scanning every 15 minutes",
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Alert service started via command")
            
        except Exception as e:
            logger.error(f"Error starting alerts: {e}")
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="alerts_stop", description="Stop automated alert scanning")
    @app_commands.checks.has_permissions(administrator=True)
    async def stop_alerts(self, interaction: discord.Interaction):
        """Stop the automated alert service"""
        try:
            if not hasattr(self.bot, 'alert_service'):
                await interaction.response.send_message("❌ Alert service not available", ephemeral=True)
                return
            
            await self.bot.alert_service.stop()
            
            embed = discord.Embed(
                title="⏹️ Automated Alerts Stopped",
                description="Alert service has been stopped",
                color=0xff0000
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Alert service stopped via command")
            
        except Exception as e:
            logger.error(f"Error stopping alerts: {e}")
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="alerts_status", description="Check automated alerts status")
    async def alerts_status(self, interaction: discord.Interaction):
        """Check the status of automated alerts"""
        try:
            if not hasattr(self.bot, 'alert_service'):
                await interaction.response.send_message("❌ Alert service not available", ephemeral=True)
                return
            
            service = self.bot.alert_service
            
            status_color = 0x00ff00 if service.is_running else 0xff0000
            status_text = "🟢 Running" if service.is_running else "🔴 Stopped"
            
            market_status = "🟢 Open" if service.is_market_hours() else "🔴 Closed"
            
            embed = discord.Embed(
                title="📊 Alert Service Status",
                color=status_color
            )
            
            embed.add_field(name="Service Status", value=status_text, inline=True)
            embed.add_field(name="Market Status", value=market_status, inline=True)
            embed.add_field(name="Scan Interval", value=f"{service.scan_interval_minutes} minutes", inline=True)
            
            if hasattr(service, 'channel_id'):
                embed.add_field(name="Alert Channel", value=f"<#{service.channel_id}>", inline=False)
            
            embed.add_field(
                name="Whale Alerts Cached",
                value=f"{len(service.sent_whale_alerts)} (cleared at market close)",
                inline=True
            )
            
            embed.add_field(
                name="Whale Score Threshold",
                value=f"{service.whale_score_threshold}",
                inline=True
            )
            
            await interaction.response.send_message(embed=embed)
            
        except Exception as e:
            logger.error(f"Error checking alert status: {e}")
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="alerts_test", description="Send a test alert to verify configuration")
    @app_commands.checks.has_permissions(administrator=True)
    async def test_alerts(self, interaction: discord.Interaction):
        """Send a test alert"""
        try:
            if not hasattr(self.bot, 'alert_service'):
                await interaction.response.send_message("❌ Alert service not available", ephemeral=True)
                return
            
            if not hasattr(self.bot.alert_service, 'channel_id'):
                await interaction.response.send_message(
                    "❌ Please run `/alerts_setup` first",
                    ephemeral=True
                )
                return
            
            channel = self.bot.get_channel(self.bot.alert_service.channel_id)
            if not channel:
                await interaction.response.send_message("❌ Alert channel not found", ephemeral=True)
                return
            
            # Send test message
            embed = discord.Embed(
                title="🧪 Test Alert",
                description="This is a test alert to verify the configuration is working correctly.",
                color=0xffff00
            )
            
            embed.add_field(name="Status", value="✅ Alerts are configured correctly", inline=False)
            embed.set_footer(text="Automated alerts will appear in this format")
            
            await channel.send(embed=embed)
            await interaction.response.send_message("✅ Test alert sent!", ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error sending test alert: {e}")
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)


async def setup(bot):
    """Load the cog"""
    await bot.add_cog(AlertCommands(bot))
