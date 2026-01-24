"""
Market Commentary Commands
Discord slash commands to control the AI market commentary service
"""

import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional

from ..services.market_commentary import MarketCommentaryService

logger = logging.getLogger(__name__)


class MarketCommentaryCommands(commands.Cog):
    """Commands for AI-powered market commentary"""
    
    def __init__(self, bot):
        self.bot = bot
        self.service = MarketCommentaryService(bot)
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Auto-start if was previously running"""
        try:
            if self.service.config_file.exists():
                import json
                with open(self.service.config_file, 'r') as f:
                    config = json.load(f)
                    was_running = config.get('is_running', False)
                    channel_id = config.get('channel_id')
                    
                    if was_running and channel_id:
                        logger.info(f"Auto-starting market commentary for channel {channel_id}")
                        await self.service.start(channel_id)
                        
                        channel = self.bot.get_channel(channel_id)
                        if channel:
                            embed = discord.Embed(
                                title="🎙️ Market Commentary Resumed",
                                description="AI commentary service auto-resumed after bot restart",
                                color=discord.Color.green()
                            )
                            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Error auto-starting market commentary: {e}")
    
    @app_commands.command(
        name="setup_commentary",
        description="Setup AI market commentary for this channel"
    )
    @app_commands.describe(
        interval="Commentary interval in minutes (default: 15)"
    )
    async def setup_commentary(
        self,
        interaction: discord.Interaction,
        interval: int = 15
    ):
        """Configure AI market commentary for this channel"""
        self.service.channel_id = interaction.channel_id
        self.service.commentary_interval_minutes = interval
        self.service._save_config()
        
        status = self.service.get_status()
        
        embed = discord.Embed(
            title="🎙️ Market Commentary Configured",
            description="AI-powered market commentary will be posted to this channel",
            color=discord.Color.blue()
        )
        embed.add_field(name="Channel", value=f"<#{interaction.channel_id}>", inline=True)
        embed.add_field(name="Interval", value=f"{interval} minutes", inline=True)
        embed.add_field(name="Groq AI", value="✅ Ready" if status['groq_available'] else "❌ Not configured", inline=True)
        embed.set_footer(text="Use /start_commentary to begin")
        
        await interaction.response.send_message(embed=embed)
        logger.info(f"Market commentary configured for channel {interaction.channel_id}")
    
    @app_commands.command(
        name="start_commentary",
        description="Start AI market commentary updates"
    )
    async def start_commentary(self, interaction: discord.Interaction):
        """Start the market commentary service"""
        if not self.service.channel_id:
            await interaction.response.send_message(
                "❌ Please run `/setup_commentary` first to configure the channel.",
                ephemeral=True
            )
            return
        
        if self.service.is_running:
            await interaction.response.send_message(
                "⚠️ Market commentary is already running!",
                ephemeral=True
            )
            return
        
        await self.service.start()
        
        status = self.service.get_status()
        embed = discord.Embed(
            title="🎙️ Market Commentary Started",
            description="AI-powered commentary will be posted during market hours",
            color=discord.Color.green()
        )
        embed.add_field(name="Interval", value=f"Every {self.service.commentary_interval_minutes} minutes", inline=True)
        embed.add_field(name="Market Status", value="🟢 Open" if status['market_open'] else "🔴 Closed", inline=True)
        embed.add_field(name="Groq AI", value="✅ Enabled" if status['groq_available'] else "⚠️ Basic mode", inline=True)
        embed.set_footer(text="Commentary includes: Whale Flows, TOS Alerts, Z-Score, ETF Momentum")
        
        await interaction.response.send_message(embed=embed)
        logger.info("Market commentary started")
    
    @app_commands.command(
        name="stop_commentary",
        description="Stop AI market commentary updates"
    )
    async def stop_commentary(self, interaction: discord.Interaction):
        """Stop the market commentary service"""
        if not self.service.is_running:
            await interaction.response.send_message(
                "⚠️ Market commentary is not running.",
                ephemeral=True
            )
            return
        
        await self.service.stop()
        
        await interaction.response.send_message(
            "✅ Market commentary stopped.",
            ephemeral=True
        )
        logger.info("Market commentary stopped")
    
    @app_commands.command(
        name="commentary_status",
        description="Check market commentary service status"
    )
    async def commentary_status(self, interaction: discord.Interaction):
        """Show current status of market commentary"""
        status = self.service.get_status()
        
        embed = discord.Embed(
            title="📊 Market Commentary Status",
            color=discord.Color.blue()
        )
        
        # Service status
        status_emoji = "🟢" if status['is_running'] else "🔴"
        status_text = "Running" if status['is_running'] else "Stopped"
        embed.add_field(name="Status", value=f"{status_emoji} {status_text}", inline=True)
        
        # Channel
        if status['channel_id']:
            embed.add_field(name="Channel", value=f"<#{status['channel_id']}>", inline=True)
        else:
            embed.add_field(name="Channel", value="Not configured", inline=True)
        
        # Market status
        market_status = "🟢 Open" if status['market_open'] else "🔴 Closed"
        embed.add_field(name="Market", value=market_status, inline=True)
        
        # Config
        embed.add_field(name="Interval", value=f"{status['interval_minutes']} min", inline=True)
        embed.add_field(name="Groq AI", value="✅ Ready" if status['groq_available'] else "❌ Not available", inline=True)
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(
        name="commentary_now",
        description="Generate and post market commentary immediately"
    )
    async def commentary_now(self, interaction: discord.Interaction):
        """Generate commentary immediately (on-demand)"""
        await interaction.response.defer()
        
        try:
            # Temporarily set channel if not set
            original_channel = self.service.channel_id
            self.service.channel_id = interaction.channel_id
            
            # Collect data and generate
            data = await self.service.collect_scanner_data(lookback_minutes=30)
            commentary = self.service.generate_ai_commentary(data)
            
            # Restore original channel
            self.service.channel_id = original_channel
            
            # Create embed
            session = data.get('session', 'Market Update')
            
            # Determine embed color
            color = discord.Color.blue()
            spy_change = data.get('market_levels', {}).get('SPY', {}).get('change_pct') or 0
            if spy_change > 0.5:
                color = discord.Color.green()
            elif spy_change < -0.5:
                color = discord.Color.red()
            
            embed = discord.Embed(
                title=f"🎙️ Market Commentary - {session}",
                description=commentary,
                color=color,
                timestamp=interaction.created_at
            )
            
            # Add market levels
            if data.get('market_levels'):
                levels_str = " | ".join([
                    f"{sym}: ${info.get('price') or 0:.2f} ({(info.get('change_pct') or 0):+.2f}%)"
                    for sym, info in data['market_levels'].items()
                    if info.get('price')
                ])
                embed.set_footer(text=levels_str)
            
            # Signal counts
            total_signals = (
                len(data.get('whale_flows', [])) +
                len(data.get('tos_alerts', [])) +
                len(data.get('zscore_signals', [])) +
                len(data.get('etf_momentum', []))
            )
            
            if total_signals > 0:
                signal_summary = []
                if data.get('whale_flows'):
                    signal_summary.append(f"🐋 {len(data['whale_flows'])}")
                if data.get('tos_alerts'):
                    signal_summary.append(f"⚡ {len(data['tos_alerts'])}")
                if data.get('zscore_signals'):
                    signal_summary.append(f"📊 {len(data['zscore_signals'])}")
                if data.get('etf_momentum'):
                    signal_summary.append(f"📈 {len(data['etf_momentum'])}")
                
                embed.add_field(
                    name="Signals (Last 30min)",
                    value=" | ".join(signal_summary),
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in commentary_now: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error generating commentary: {str(e)}")


async def setup(bot):
    """Setup function called by Discord.py"""
    await bot.add_cog(MarketCommentaryCommands(bot))
