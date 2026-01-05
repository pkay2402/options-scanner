"""
Discord bot command for power inflow scanning
Add this to your Discord bot's command handler
"""

import discord
from discord.ext import commands, tasks
import sys
import os
from datetime import datetime, time as dt_time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from discord_bot.power_inflow_scanner import scan_for_power_inflows, get_summary_stats


class PowerInflowCog(commands.Cog):
    """Power Inflow Scanner Commands"""
    
    def __init__(self, bot):
        self.bot = bot
        self.channel_id = None  # Set this to your Discord channel ID
        self.auto_scan.start()
    
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
    
    @commands.command(name='flows', aliases=['inflows', 'powerflows'])
    async def manual_scan(self, ctx):
        """Manually trigger a power inflow scan"""
        await ctx.send("🔍 Scanning for power inflows...")
        
        try:
            messages = scan_for_power_inflows()
            
            if messages:
                for msg in messages:
                    await ctx.send(msg)
            else:
                await ctx.send("✅ No new significant flows detected.")
        except Exception as e:
            await ctx.send(f"❌ Error scanning: {str(e)}")
    
    @commands.command(name='flowstats', aliases=['scanstats'])
    async def show_stats(self, ctx):
        """Show scanner statistics"""
        try:
            stats = get_summary_stats()
            
            embed = discord.Embed(
                title="📊 Power Inflow Scanner Stats",
                color=discord.Color.blue()
            )
            embed.add_field(name="Last Run", value=stats['last_run'], inline=True)
            embed.add_field(name="Flows Today", value=stats['flows_today'], inline=True)
            embed.add_field(name="Watching", value=f"{stats['symbols_watching']} symbols", inline=True)
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error getting stats: {str(e)}")
    
    @commands.command(name='setchannel')
    @commands.has_permissions(administrator=True)
    async def set_channel(self, ctx):
        """Set this channel for auto-alerts (Admin only)"""
        self.channel_id = ctx.channel.id
        await ctx.send(f"✅ Auto-alerts will be sent to this channel ({ctx.channel.mention})")
    
    @commands.command(name='startscan')
    @commands.has_permissions(administrator=True)
    async def start_scanning(self, ctx):
        """Start auto-scanning (Admin only)"""
        if not self.auto_scan.is_running():
            self.auto_scan.start()
            await ctx.send("✅ Auto-scanning started (every 3 minutes during market hours)")
        else:
            await ctx.send("⚠️ Auto-scanning is already running")
    
    @commands.command(name='stopscan')
    @commands.has_permissions(administrator=True)
    async def stop_scanning(self, ctx):
        """Stop auto-scanning (Admin only)"""
        if self.auto_scan.is_running():
            self.auto_scan.cancel()
            await ctx.send("✅ Auto-scanning stopped")
        else:
            await ctx.send("⚠️ Auto-scanning is not running")


async def setup(bot):
    """Setup function for loading the cog"""
    await bot.add_cog(PowerInflowCog(bot))
