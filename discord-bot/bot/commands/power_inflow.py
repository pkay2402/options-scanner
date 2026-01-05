"""
Discord bot command for power inflow scanning
Uses slash commands for better UX
"""

import discord
from discord import app_commands
from discord.ext import commands, tasks
import sys
import os
import logging
from datetime import datetime, time as dt_time

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from power_inflow_scanner import scan_for_power_inflows, get_summary_stats
    logger.info("✅ Successfully imported power_inflow_scanner module")
except Exception as e:
    logger.error(f"❌ Failed to import power_inflow_scanner: {e}", exc_info=True)
    raise


class PowerInflowCog(commands.Cog):
    """Power Inflow Scanner Commands"""
    
    def __init__(self, bot):
        self.bot = bot
        self.channel_id = None  # Set this to your Discord channel ID
        logger.info("Initializing PowerInflowCog...")
        self.auto_scan.start()
        logger.info("✅ PowerInflowCog initialized, auto_scan started")
    
    def cog_unload(self):
        self.auto_scan.cancel()
    
    @tasks.loop(minutes=3)
    async def auto_scan(self):
        """Auto-scan every 3 minutes during market hours"""
        # Check if market hours (9:30 AM - 4:00 PM ET)
        now = datetime.now()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        # Skip weekends
        if now.weekday() >= 5:
            return
        
        # Check market hours
        current_time = now.time()
        if not (market_open <= current_time <= market_close):
            return
        
        # Run scan
        try:
            messages = scan_for_power_inflows()
            
            if messages and self.channel_id:
                channel = self.bot.get_channel(self.channel_id)
                if channel:
                    for msg in messages:
                        await channel.send(msg)
        except Exception as e:
            print(f"Error in auto_scan: {e}")
    
    @auto_scan.before_loop
    async def before_auto_scan(self):
        """Wait until bot is ready"""
        await self.bot.wait_until_ready()
    
    @app_commands.command(name="flows", description="Scan for significant power inflows")
    async def manual_scan(self, interaction: discord.Interaction):
        """Manually trigger a power inflow scan"""
        await interaction.response.defer()
        
        try:
            messages = scan_for_power_inflows()
            
            if messages:
                await interaction.followup.send(f"🔍 Found {len(messages)} power inflow alerts:")
                for msg in messages:
                    await interaction.followup.send(msg)
            else:
                await interaction.followup.send("✅ No new significant flows detected.")
        except Exception as e:
            logger.error(f"Error in manual_scan: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error scanning: {str(e)}")
    
    @app_commands.command(name="flowstats", description="Show power inflow scanner statistics")
    async def show_stats(self, interaction: discord.Interaction):
        """Show scanner statistics"""
        await interaction.response.defer()
        
        try:
            stats = get_summary_stats()
            
            embed = discord.Embed(
                title="📊 Power Inflow Scanner Stats",
                color=discord.Color.blue()
            )
            embed.add_field(name="Last Run", value=stats['last_run'], inline=True)
            embed.add_field(name="Flows Today", value=stats['flows_today'], inline=True)
            embed.add_field(name="Watching", value=f"{stats['symbols_watching']} symbols", inline=True)
            
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Error in show_stats: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error getting stats: {str(e)}")
    
    @app_commands.command(name="setchannel", description="Set this channel for auto power inflow alerts")
    @app_commands.checks.has_permissions(administrator=True)
    async def set_channel(self, interaction: discord.Interaction):
        """Set this channel for auto-alerts (Admin only)"""
        self.channel_id = interaction.channel_id
        await interaction.response.send_message(
            f"✅ Auto-alerts will be sent to this channel (<#{interaction.channel_id}>)"
        )
    
    @app_commands.command(name="startscan", description="Start auto-scanning for power inflows")
    @app_commands.checks.has_permissions(administrator=True)
    async def start_scanning(self, interaction: discord.Interaction):
        """Start auto-scanning (Admin only)"""
        if not self.auto_scan.is_running():
            self.auto_scan.start()
            await interaction.response.send_message(
                "✅ Auto-scanning started (every 3 minutes during market hours)"
            )
        else:
            await interaction.response.send_message(
                "⚠️ Auto-scanning is already running"
            )
    
    @app_commands.command(name="stopscan", description="Stop auto-scanning for power inflows")
    @app_commands.checks.has_permissions(administrator=True)
    async def stop_scanning(self, interaction: discord.Interaction):
        """Stop auto-scanning (Admin only)"""
        if self.auto_scan.is_running():
            self.auto_scan.cancel()
            await interaction.response.send_message("✅ Auto-scanning stopped")
        else:
            await interaction.response.send_message("⚠️ Auto-scanning is not running")

async def setup(bot):
    """Setup function for loading the cog"""
    logger.info("Setting up PowerInflowCog...")
    await bot.add_cog(PowerInflowCog(bot))
    logger.info("✅ PowerInflowCog added to bot")
