"""
Volume Walls Commands
Commands for call/put walls and gamma flip levels analysis
"""

import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.utils.chart_utils import plotly_to_discord_file, create_embed, format_large_number

logger = logging.getLogger(__name__)


class WallsCommands(commands.Cog):
    """Volume walls analysis Discord commands"""
    
    def __init__(self, bot):
        self.bot = bot
        
    def _analyze_volume_walls(self, options_data, underlying_price):
        """Analyze call/put walls and flip level - adapted from Option Volume Walls page"""
        try:
            call_strikes = {}
            put_strikes = {}
            
            # Process calls
            if 'callExpDateMap' in options_data:
                for exp_date, strikes in options_data['callExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            strike = float(strike_str)
                            contract = contracts[0]
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 0)
                            gamma = contract.get('gamma', 0)
                            
                            if strike not in call_strikes:
                                call_strikes[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                            
                            call_strikes[strike]['volume'] += volume
                            call_strikes[strike]['oi'] += oi
                            call_strikes[strike]['gamma'] += gamma * oi * 100
            
            # Process puts
            if 'putExpDateMap' in options_data:
                for exp_date, strikes in options_data['putExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            strike = float(strike_str)
                            contract = contracts[0]
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 0)
                            gamma = contract.get('gamma', 0)
                            
                            if strike not in put_strikes:
                                put_strikes[strike] = {'volume': 0, 'oi': 0, 'gamma': 0}
                            
                            put_strikes[strike]['volume'] += volume
                            put_strikes[strike]['oi'] += oi
                            put_strikes[strike]['gamma'] += gamma * oi * 100
            
            # Find walls (highest OI strikes)
            call_wall = None
            if call_strikes:
                max_call_strike = max(call_strikes.items(), key=lambda x: x[1]['oi'])
                call_wall = {
                    'strike': max_call_strike[0],
                    'volume': max_call_strike[1]['volume'],
                    'oi': max_call_strike[1]['oi'],
                    'gamma': max_call_strike[1]['gamma']
                }
            
            put_wall = None
            if put_strikes:
                max_put_strike = max(put_strikes.items(), key=lambda x: x[1]['oi'])
                put_wall = {
                    'strike': max_put_strike[0],
                    'volume': max_put_strike[1]['volume'],
                    'oi': max_put_strike[1]['oi'],
                    'gamma': max_put_strike[1]['gamma']
                }
            
            # Calculate flip level (zero gamma point)
            # Simplified: midpoint between call and put gamma concentrations
            flip_level = None
            if call_wall and put_wall:
                # Weight by gamma exposure
                total_gamma = abs(call_wall['gamma']) + abs(put_wall['gamma'])
                if total_gamma > 0:
                    flip_level = (
                        call_wall['strike'] * abs(call_wall['gamma']) +
                        put_wall['strike'] * abs(put_wall['gamma'])
                    ) / total_gamma
            
            return {
                'call_wall': call_wall,
                'put_wall': put_wall,
                'flip_level': flip_level,
                'call_strikes': call_strikes,
                'put_strikes': put_strikes
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume walls: {e}", exc_info=True)
            return None
    
    def _create_walls_chart(self, analysis, underlying_price, symbol):
        """Create volume walls visualization"""
        try:
            if not analysis:
                return None
            
            fig = go.Figure()
            
            # Create bar chart for call strikes
            if analysis['call_strikes']:
                strikes = list(analysis['call_strikes'].keys())
                ois = [analysis['call_strikes'][s]['oi'] for s in strikes]
                
                fig.add_trace(go.Bar(
                    x=strikes,
                    y=ois,
                    name='Call OI',
                    marker_color='rgba(34, 197, 94, 0.7)',
                    hovertemplate='<b>Call Strike: $%{x:.2f}</b><br>OI: %{y:,}<extra></extra>'
                ))
            
            # Create bar chart for put strikes
            if analysis['put_strikes']:
                strikes = list(analysis['put_strikes'].keys())
                ois = [-analysis['put_strikes'][s]['oi'] for s in strikes]  # Negative for visual separation
                
                fig.add_trace(go.Bar(
                    x=strikes,
                    y=ois,
                    name='Put OI',
                    marker_color='rgba(239, 68, 68, 0.7)',
                    hovertemplate='<b>Put Strike: $%{x:.2f}</b><br>OI: %{y:,}<extra></extra>'
                ))
            
            # Add current price line
            fig.add_vline(
                x=underlying_price,
                line=dict(color='gold', width=3, dash='dash'),
                annotation_text=f"Current: ${underlying_price:.2f}",
                annotation_position="top"
            )
            
            # Add call wall line
            if analysis['call_wall']:
                fig.add_vline(
                    x=analysis['call_wall']['strike'],
                    line=dict(color='green', width=2, dash='dot'),
                    annotation_text=f"Call Wall: ${analysis['call_wall']['strike']:.2f}",
                    annotation_position="top right"
                )
            
            # Add put wall line
            if analysis['put_wall']:
                fig.add_vline(
                    x=analysis['put_wall']['strike'],
                    line=dict(color='red', width=2, dash='dot'),
                    annotation_text=f"Put Wall: ${analysis['put_wall']['strike']:.2f}",
                    annotation_position="bottom left"
                )
            
            # Add flip level
            if analysis['flip_level']:
                fig.add_vline(
                    x=analysis['flip_level'],
                    line=dict(color='purple', width=2, dash='solid'),
                    annotation_text=f"Flip: ${analysis['flip_level']:.2f}",
                    annotation_position="top left"
                )
            
            fig.update_layout(
                title=f"{symbol} Options Volume Walls",
                xaxis_title="Strike Price",
                yaxis_title="Open Interest",
                height=600,
                template='plotly_white',
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating walls chart: {e}", exc_info=True)
            return None

    @app_commands.command(name="walls", description="🧱 Call and put volume walls analysis")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def walls(self, interaction: discord.Interaction, symbol: str):
        """Show call and put walls with flip level"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            logger.info(f"Fetching volume walls for {symbol} (user: {interaction.user})")
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Fetch options data
            options_data = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                strike_count=50
            )
            
            if not options_data:
                await interaction.followup.send(f"❌ No options data available for **{symbol}**")
                return
            
            # Get underlying price
            underlying_price = options_data.get('underlyingPrice', 0)
            if not underlying_price and 'underlying' in options_data:
                underlying_price = options_data['underlying'].get('last', 0)
            
            if not underlying_price:
                await interaction.followup.send(f"❌ Could not determine price for **{symbol}**")
                return
            
            # Analyze walls
            analysis = self._analyze_volume_walls(options_data, underlying_price)
            
            if not analysis or (not analysis['call_wall'] and not analysis['put_wall']):
                await interaction.followup.send(f"❌ No volume wall data available for **{symbol}**")
                return
            
            # Create embed
            embed = discord.Embed(
                title=f"🧱 {symbol} Volume Walls Analysis",
                description=f"**Current Price:** ${underlying_price:.2f}",
                color=discord.Color.purple()
            )
            
            # Add call wall info
            if analysis['call_wall']:
                call_info = (
                    f"**Strike:** ${analysis['call_wall']['strike']:.2f}\n"
                    f"**Open Interest:** {int(analysis['call_wall']['oi']):,}\n"
                    f"**Volume:** {int(analysis['call_wall']['volume']):,}\n"
                    f"**Distance:** {((analysis['call_wall']['strike'] - underlying_price) / underlying_price * 100):.2f}%"
                )
                embed.add_field(name="🟢 Call Wall (Resistance)", value=call_info, inline=True)
            
            # Add put wall info
            if analysis['put_wall']:
                put_info = (
                    f"**Strike:** ${analysis['put_wall']['strike']:.2f}\n"
                    f"**Open Interest:** {int(analysis['put_wall']['oi']):,}\n"
                    f"**Volume:** {int(analysis['put_wall']['volume']):,}\n"
                    f"**Distance:** {((underlying_price - analysis['put_wall']['strike']) / underlying_price * 100):.2f}%"
                )
                embed.add_field(name="🔴 Put Wall (Support)", value=put_info, inline=True)
            
            # Add flip level
            if analysis['flip_level']:
                flip_info = (
                    f"**Level:** ${analysis['flip_level']:.2f}\n"
                    f"**Status:** {'Above ⬆️' if underlying_price > analysis['flip_level'] else 'Below ⬇️'}\n"
                    f"**Distance:** {abs((underlying_price - analysis['flip_level']) / underlying_price * 100):.2f}%"
                )
                embed.add_field(name="🔄 Gamma Flip Level", value=flip_info, inline=False)
            
            # Create chart
            fig = self._create_walls_chart(analysis, underlying_price, symbol)
            
            if fig:
                filename = f"{symbol}_walls.png"
                file = plotly_to_discord_file(fig, filename, width=1400, height=700)
                
                if file:
                    embed.set_image(url=f"attachment://{filename}")
                    embed.set_footer(text="Data from Schwab API • Walls = highest OI strikes")
                    await interaction.followup.send(embed=embed, file=file)
                else:
                    embed.set_footer(text="Data from Schwab API • Chart generation failed")
                    await interaction.followup.send(embed=embed)
            else:
                embed.set_footer(text="Data from Schwab API")
                await interaction.followup.send(embed=embed)
            
            logger.info(f"Successfully sent volume walls for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in walls command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="call-wall", description="🟢 Call wall (resistance) analysis")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def call_wall(self, interaction: discord.Interaction, symbol: str):
        """Show call wall details"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Fetch options data
            options_data = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
            
            if not options_data:
                await interaction.followup.send(f"❌ No options data available for **{symbol}**")
                return
            
            # Get underlying price
            underlying_price = options_data.get('underlyingPrice', 0)
            if not underlying_price and 'underlying' in options_data:
                underlying_price = options_data['underlying'].get('last', 0)
            
            # Analyze walls
            analysis = self._analyze_volume_walls(options_data, underlying_price)
            
            if not analysis or not analysis['call_wall']:
                await interaction.followup.send(f"❌ No call wall data available for **{symbol}**")
                return
            
            call_wall = analysis['call_wall']
            
            embed = discord.Embed(
                title=f"🟢 {symbol} Call Wall (Resistance)",
                description=f"**Current Price:** ${underlying_price:.2f}",
                color=discord.Color.green()
            )
            
            info = (
                f"**Strike:** ${call_wall['strike']:.2f}\n"
                f"**Open Interest:** {int(call_wall['oi']):,}\n"
                f"**Volume:** {int(call_wall['volume']):,}\n"
                f"**Distance:** {((call_wall['strike'] - underlying_price) / underlying_price * 100):.2f}% above current\n"
                f"**Gamma Exposure:** {format_large_number(call_wall['gamma'])}"
            )
            
            embed.add_field(name="Wall Details", value=info, inline=False)
            
            # Add interpretation
            distance_pct = (call_wall['strike'] - underlying_price) / underlying_price * 100
            
            if distance_pct < 2:
                interpretation = "⚠️ **Very close** - Strong resistance nearby"
            elif distance_pct < 5:
                interpretation = "📊 **Moderate distance** - Watch for rejection"
            else:
                interpretation = "✅ **Far away** - Room to move up"
            
            embed.add_field(name="Interpretation", value=interpretation, inline=False)
            embed.set_footer(text="Call walls act as resistance • Dealers hedge by selling stock")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in call-wall command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="put-wall", description="🔴 Put wall (support) analysis")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def put_wall(self, interaction: discord.Interaction, symbol: str):
        """Show put wall details"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Fetch options data
            options_data = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
            
            if not options_data:
                await interaction.followup.send(f"❌ No options data available for **{symbol}**")
                return
            
            # Get underlying price
            underlying_price = options_data.get('underlyingPrice', 0)
            if not underlying_price and 'underlying' in options_data:
                underlying_price = options_data['underlying'].get('last', 0)
            
            # Analyze walls
            analysis = self._analyze_volume_walls(options_data, underlying_price)
            
            if not analysis or not analysis['put_wall']:
                await interaction.followup.send(f"❌ No put wall data available for **{symbol}**")
                return
            
            put_wall = analysis['put_wall']
            
            embed = discord.Embed(
                title=f"🔴 {symbol} Put Wall (Support)",
                description=f"**Current Price:** ${underlying_price:.2f}",
                color=discord.Color.red()
            )
            
            info = (
                f"**Strike:** ${put_wall['strike']:.2f}\n"
                f"**Open Interest:** {int(put_wall['oi']):,}\n"
                f"**Volume:** {int(put_wall['volume']):,}\n"
                f"**Distance:** {((underlying_price - put_wall['strike']) / underlying_price * 100):.2f}% below current\n"
                f"**Gamma Exposure:** {format_large_number(put_wall['gamma'])}"
            )
            
            embed.add_field(name="Wall Details", value=info, inline=False)
            
            # Add interpretation
            distance_pct = (underlying_price - put_wall['strike']) / underlying_price * 100
            
            if distance_pct < 2:
                interpretation = "⚠️ **Very close** - Strong support nearby"
            elif distance_pct < 5:
                interpretation = "📊 **Moderate distance** - Watch for bounce"
            else:
                interpretation = "✅ **Far away** - Room to move down"
            
            embed.add_field(name="Interpretation", value=interpretation, inline=False)
            embed.set_footer(text="Put walls act as support • Dealers hedge by buying stock")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in put-wall command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Load the cog"""
    await bot.add_cog(WallsCommands(bot))
