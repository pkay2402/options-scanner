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
        """
        Analyze call/put walls and flip level - exact logic from Option Volume Walls page
        Uses volume, OI, and gamma exposure (GEX) calculations
        """
        try:
            import numpy as np
            
            # Collect data from ALL strikes across ALL expirations
            call_volumes = {}
            put_volumes = {}
            call_oi = {}
            put_oi = {}
            call_gamma = {}
            put_gamma = {}
            
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
                            
                            call_volumes[strike] = call_volumes.get(strike, 0) + volume
                            call_oi[strike] = call_oi.get(strike, 0) + oi
                            call_gamma[strike] = call_gamma.get(strike, 0) + gamma
            
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
                            
                            put_volumes[strike] = put_volumes.get(strike, 0) + volume
                            put_oi[strike] = put_oi.get(strike, 0) + oi
                            put_gamma[strike] = put_gamma.get(strike, 0) + gamma
            
            # Get all strikes with data
            all_strikes = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
            
            if not all_strikes:
                return None
            
            # Calculate net volumes (Put - Call, positive = bearish, negative = bullish)
            net_volumes = {}
            for strike in all_strikes:
                call_vol = call_volumes.get(strike, 0)
                put_vol = put_volumes.get(strike, 0)
                net_volumes[strike] = put_vol - call_vol
            
            # Calculate Gamma Exposure (GEX) for each strike
            # GEX = Gamma * Open Interest * 100 * Spot^2 * 0.01
            # Dealer is short gamma, so: Call GEX is positive, Put GEX is negative
            gex_by_strike = {}
            for strike in all_strikes:
                call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01
                put_gex = put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01 * -1
                gex_by_strike[strike] = call_gex + put_gex
            
            # Find walls (max volumes)
            call_wall_strike = max(call_volumes.items(), key=lambda x: x[1])[0] if call_volumes else None
            call_wall = None
            if call_wall_strike:
                call_wall = {
                    'strike': call_wall_strike,
                    'volume': call_volumes.get(call_wall_strike, 0),
                    'oi': call_oi.get(call_wall_strike, 0),
                    'gex': gex_by_strike.get(call_wall_strike, 0)
                }
            
            put_wall_strike = max(put_volumes.items(), key=lambda x: x[1])[0] if put_volumes else None
            put_wall = None
            if put_wall_strike:
                put_wall = {
                    'strike': put_wall_strike,
                    'volume': put_volumes.get(put_wall_strike, 0),
                    'oi': put_oi.get(put_wall_strike, 0),
                    'gex': gex_by_strike.get(put_wall_strike, 0)
                }
            
            # Find flip level (where net volume changes sign near current price)
            # Look within ±10% of current price
            price_range = underlying_price * 0.10
            strikes_near_price = sorted([s for s in all_strikes 
                                        if abs(s - underlying_price) < price_range])
            
            flip_level = None
            for i in range(len(strikes_near_price) - 1):
                s1, s2 = strikes_near_price[i], strikes_near_price[i + 1]
                net_vol_s1 = net_volumes.get(s1, 0)
                net_vol_s2 = net_volumes.get(s2, 0)
                
                # Check for sign change (bearish to bullish or vice versa)
                if net_vol_s1 * net_vol_s2 < 0:
                    # Pick the strike with smallest absolute net volume (closest to neutral)
                    flip_level = s1 if abs(net_vol_s1) < abs(net_vol_s2) else s2
                    break
            
            # Calculate totals
            total_call_vol = sum(call_volumes.values())
            total_put_vol = sum(put_volumes.values())
            total_gex = sum(gex_by_strike.values())
            
            return {
                'call_wall': call_wall,
                'put_wall': put_wall,
                'flip_level': flip_level,
                'all_strikes': all_strikes,
                'call_volumes': call_volumes,
                'put_volumes': put_volumes,
                'net_volumes': net_volumes,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'gex_by_strike': gex_by_strike,
                'totals': {
                    'call_vol': total_call_vol,
                    'put_vol': total_put_vol,
                    'net_vol': total_put_vol - total_call_vol,
                    'total_gex': total_gex
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume walls: {e}", exc_info=True)
            return None
    
    def _create_walls_chart(self, analysis, underlying_price, symbol):
        """Create volume walls visualization using GEX (Gamma Exposure)"""
        try:
            if not analysis:
                return None
            
            fig = go.Figure()
            
            # Get strikes and GEX values
            all_strikes = analysis.get('all_strikes', [])
            gex_by_strike = analysis.get('gex_by_strike', {})
            
            if not all_strikes or not gex_by_strike:
                return None
            
            # Separate positive (call) and negative (put) GEX
            strikes_list = sorted(all_strikes)
            gex_values = [gex_by_strike.get(s, 0) for s in strikes_list]
            
            # Create colors: green for positive GEX (calls), red for negative GEX (puts)
            colors = ['rgba(34, 197, 94, 0.7)' if g > 0 else 'rgba(239, 68, 68, 0.7)' for g in gex_values]
            
            # Create bar chart for GEX
            fig.add_trace(go.Bar(
                x=strikes_list,
                y=gex_values,
                name='Net GEX',
                marker_color=colors,
                hovertemplate='<b>Strike: $%{x:.2f}</b><br>Net GEX: %{y:,.0f}<extra></extra>'
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
                title=f"{symbol} Gamma Exposure (GEX) by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Net Gamma Exposure (GEX)",
                height=600,
                template='plotly_white',
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating walls chart: {e}", exc_info=True)
            return None

    async def _send_walls_embed(self, interaction: discord.Interaction, symbol: str, expiry_date: str = None):
        """Helper to send walls embed and chart for a symbol/expiry"""
        symbol = symbol.upper()
        logger.info(f"Fetching volume walls for {symbol} (user: {interaction.user})")
        client = self.bot.schwab_service.get_client()
        chain_params = {
            'symbol': symbol,
            'contract_type': 'ALL',
            'strike_count': 50
        }
        if expiry_date:
            chain_params['from_date'] = expiry_date
            chain_params['to_date'] = expiry_date
        options_data = client.get_options_chain(**chain_params)
        if not options_data:
            await interaction.followup.send(f"❌ No options data available for **{symbol}**")
            return
        underlying_price = options_data.get('underlyingPrice', 0)
        if not underlying_price and 'underlying' in options_data:
            underlying_obj = options_data['underlying']
            if underlying_obj and isinstance(underlying_obj, dict):
                underlying_price = underlying_obj.get('last', 0)
        if not underlying_price:
            await interaction.followup.send(f"❌ Could not determine price for **{symbol}** (market may be closed or expiry invalid)")
            return
        analysis = self._analyze_volume_walls(options_data, underlying_price)
        if not analysis or (not analysis['call_wall'] and not analysis['put_wall']):
            await interaction.followup.send(f"❌ No volume wall data available for **{symbol}**")
            return
        embed = discord.Embed(
            title=f"🧱 {symbol} Volume Walls Analysis",
            description=f"**Current Price:** ${underlying_price:.2f}",
            color=discord.Color.purple()
        )
        if analysis['call_wall']:
            call_info = (
                f"**Strike:** ${analysis['call_wall']['strike']:.2f}\n"
                f"**Open Interest:** {int(analysis['call_wall']['oi']):,}\n"
                f"**Volume:** {int(analysis['call_wall']['volume']):,}\n"
                f"**GEX:** {format_large_number(analysis['call_wall']['gex'])}\n"
                f"**Distance:** {((analysis['call_wall']['strike'] - underlying_price) / underlying_price * 100):.2f}%"
            )
            embed.add_field(name="🟢 Call Wall (Resistance)", value=call_info, inline=True)
        if analysis['put_wall']:
            put_info = (
                f"**Strike:** ${analysis['put_wall']['strike']:.2f}\n"
                f"**Open Interest:** {int(analysis['put_wall']['oi']):,}\n"
                f"**Volume:** {int(analysis['put_wall']['volume']):,}\n"
                f"**GEX:** {format_large_number(analysis['put_wall']['gex'])}\n"
                f"**Distance:** {((underlying_price - analysis['put_wall']['strike']) / underlying_price * 100):.2f}%"
            )
            embed.add_field(name="🔴 Put Wall (Support)", value=put_info, inline=True)
        if analysis['flip_level']:
            flip_info = (
                f"**Level:** ${analysis['flip_level']:.2f}\n"
                f"**Status:** {'Above ⬆️' if underlying_price > analysis['flip_level'] else 'Below ⬇️'}\n"
                f"**Distance:** {abs((underlying_price - analysis['flip_level']) / underlying_price * 100):.2f}%"
            )
            embed.add_field(name="🔄 Gamma Flip Level", value=flip_info, inline=False)
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
    @app_commands.command(name="walls", description="🧱 Call and put volume walls analysis")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        expiry_date="Optional: Specific expiry date (YYYY-MM-DD format, e.g., 2025-11-15)"
    )
    async def walls(self, interaction: discord.Interaction, symbol: str, expiry_date: str = None):
        await interaction.response.defer(thinking=True)
        await self._send_walls_embed(interaction, symbol, expiry_date)

    @app_commands.command(name="call_wall", description="🟢 Call wall (resistance) analysis")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        expiry_date="Optional: Specific expiry date (YYYY-MM-DD format, e.g., 2025-11-15)"
    )
    async def call_wall(self, interaction: discord.Interaction, symbol: str, expiry_date: str = None):
        """Show call wall details"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Prepare options chain parameters
            chain_params = {
                'symbol': symbol,
                'contract_type': 'ALL',
                'strike_count': 50
            }
            
            # Add expiry date filter if specified
            if expiry_date:
                chain_params['from_date'] = expiry_date
                chain_params['to_date'] = expiry_date
            
            # Fetch options data
            options_data = client.get_options_chain(**chain_params)
            
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
                f"**Gamma Exposure (GEX):** {format_large_number(call_wall['gex'])}"
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
            logger.error(f"Error in call_wall command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="put_wall", description="🔴 Put wall (support) analysis")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        expiry_date="Optional: Specific expiry date (YYYY-MM-DD format, e.g., 2025-11-15)"
    )
    async def put_wall(self, interaction: discord.Interaction, symbol: str, expiry_date: str = None):
        """Show put wall details"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Prepare options chain parameters
            chain_params = {
                'symbol': symbol,
                'contract_type': 'ALL',
                'strike_count': 50
            }
            
            # Add expiry date filter if specified
            if expiry_date:
                chain_params['from_date'] = expiry_date
                chain_params['to_date'] = expiry_date
            
            # Fetch options data
            options_data = client.get_options_chain(**chain_params)
            
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
                f"**Gamma Exposure (GEX):** {format_large_number(put_wall['gex'])}"
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
            logger.error(f"Error in put_wall command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="walls_chart", description="📊 Price chart with volume walls overlay")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        expiry_date="Optional: Specific expiry date (YYYY-MM-DD format, e.g., 2025-11-15)"
    )
    async def walls_chart(self, interaction: discord.Interaction, symbol: str, expiry_date: str = None):
        """Show price chart with volume walls overlay"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            logger.info(f"Fetching walls chart for {symbol} (user: {interaction.user})")
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Prepare options chain parameters
            chain_params = {
                'symbol': symbol,
                'contract_type': 'ALL',
                'strike_count': 50
            }
            
            # Add expiry date filter if specified
            if expiry_date:
                chain_params['from_date'] = expiry_date
                chain_params['to_date'] = expiry_date
            
            # Fetch options data
            options_data = client.get_options_chain(**chain_params)
            
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
            
            if not analysis:
                await interaction.followup.send(f"❌ Could not analyze volume walls for **{symbol}**")
                return
            
            # Get price history (2 days of 5-minute data)
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            price_history = client.get_price_history(
                symbol=symbol,
                period_type='day',
                period=2,
                frequency_type='minute',
                frequency=5
            )
            
            if not price_history or 'candles' not in price_history or not price_history['candles']:
                await interaction.followup.send(f"❌ No price history available for **{symbol}**")
                return
            
            # Create chart
            import pandas as pd
            df = pd.DataFrame(price_history['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            # Keep only recent data (last 100 candles for cleaner chart)
            df = df.tail(100).reset_index(drop=True)
            
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))
            
            # Add 21 EMA
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['ema21'],
                mode='lines',
                name='21 EMA',
                line=dict(color='#ff9800', width=2),
                hovertemplate='<b>21 EMA</b>: $%{y:.2f}<extra></extra>'
            ))
            
            # Add VWAP
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='#00bcd4', width=2),
                hovertemplate='<b>VWAP</b>: $%{y:.2f}<extra></extra>'
            ))
            
            # Add volume walls and flip level as horizontal lines (matching Option Volume Walls logic)
            x_range = [df['datetime'].iloc[0], df['datetime'].iloc[-1]]

            if analysis['call_wall']:
                call_strike = analysis['call_wall']['strike']
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[call_strike, call_strike],
                    mode='lines',
                    name=f"🟢 Call Wall (${call_strike:.2f})",
                    line=dict(color='#22c55e', width=3, dash='dot'),
                    hovertemplate=f'<b>Call Wall (Resistance)</b><br>Strike: ${call_strike:.2f}<extra></extra>'
                ))

            if analysis['put_wall']:
                put_strike = analysis['put_wall']['strike']
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[put_strike, put_strike],
                    mode='lines',
                    name=f"🔴 Put Wall (${put_strike:.2f})",
                    line=dict(color='#ef4444', width=3, dash='dot'),
                    hovertemplate=f'<b>Put Wall (Support)</b><br>Strike: ${put_strike:.2f}<extra></extra>'
                ))

            if analysis['flip_level']:
                flip_strike = analysis['flip_level']
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[flip_strike, flip_strike],
                    mode='lines',
                    name=f"🔄 Flip Level (${flip_strike:.2f})",
                    line=dict(color='#a855f7', width=3.5, dash='solid'),
                    hovertemplate=f'<b>Gamma Flip Level</b><br>Strike: ${flip_strike:.2f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Chart with Volume Walls',
                yaxis_title='Price ($)',
                xaxis_title='Time',
                template='plotly_dark',
                height=600,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )
            
            # Convert to Discord file
            filename = f"{symbol}_walls_chart.png"
            file = plotly_to_discord_file(fig, filename)
            
            if not file:
                await interaction.followup.send(f"❌ Could not generate chart for **{symbol}**")
                return
            
            # Create embed
            embed = create_embed(
                title=f"📊 {symbol} Price Chart with Volume Walls",
                description=f"**Current Price:** ${underlying_price:.2f}",
                color=discord.Color.blue(),
                fields=[
                    {
                        'name': '🟢 Call Wall (Resistance)',
                        'value': f"${analysis['call_wall']['strike']:.2f}" if analysis['call_wall'] else "None",
                        'inline': True
                    },
                    {
                        'name': '🔴 Put Wall (Support)',
                        'value': f"${analysis['put_wall']['strike']:.2f}" if analysis['put_wall'] else "None",
                        'inline': True
                    },
                    {
                        'name': '🔄 Flip Level',
                        'value': f"${analysis['flip_level']:.2f}" if analysis['flip_level'] else "None",
                        'inline': True
                    }
                ],
                footer_text="Green candlesticks = up • Red candlesticks = down",
                image_filename=filename
            )
            
            await interaction.followup.send(embed=embed, file=file)
            logger.info(f"Successfully sent walls chart for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in walls_chart command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


    # --- Shortcut commands for frequent use ---
    # SPY0DTE
    @app_commands.command(name="spy_0dte", description="SPY 0DTE: Walls for today's expiry")
    async def spy_0dte(self, interaction: discord.Interaction):
        await interaction.response.defer()
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        await self._send_walls_embed(interaction, symbol="SPY", expiry_date=today)

    @app_commands.command(name="qqq_0dte", description="QQQ 0DTE: Walls for today's expiry")
    async def qqq_0dte(self, interaction: discord.Interaction):
        await interaction.response.defer()
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        await self._send_walls_embed(interaction, symbol="QQQ", expiry_date=today)

    @app_commands.command(name="spx_0dte", description="$SPX 0DTE: Walls for today's expiry")
    async def spx_0dte(self, interaction: discord.Interaction):
        await interaction.response.defer()
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        await self._send_walls_embed(interaction, symbol="$SPX", expiry_date=today)

    @app_commands.command(name="stock_weekly", description="Stock weekly: Walls for current week's expiry")
    @app_commands.describe(symbol="Stock symbol (e.g., TSLA, NVDA, AAPL)")
    async def stock_weekly(self, interaction: discord.Interaction, symbol: str):
        await interaction.response.defer()
        from datetime import datetime, timedelta
        today = datetime.now()
        days_ahead = 4 - today.weekday()
        if days_ahead < 0:
            days_ahead += 7
        friday = today + timedelta(days=days_ahead)
        expiry = friday.strftime('%Y-%m-%d')
        await self._send_walls_embed(interaction, symbol=symbol.upper(), expiry_date=expiry)

async def setup(bot):
    """Load the cog"""
    await bot.add_cog(WallsCommands(bot))
