"""
Gamma Analysis Commands
Commands for gamma exposure heatmaps and top gamma strikes analysis
"""

import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.utils.chart_utils import plotly_to_discord_file, create_embed, format_large_number

logger = logging.getLogger(__name__)


class GammaCommands(commands.Cog):
    """Gamma analysis Discord commands"""
    
    def __init__(self, bot):
        self.bot = bot
        
    def _calculate_gamma_strikes(self, options_data, underlying_price, num_expiries=5):
        """Calculate gamma for all strikes - reusing logic from Stock Option Finder"""
        if not options_data:
            return pd.DataFrame()
        
        results = []
        
        # Process both calls and puts
        for option_type in ['callExpDateMap', 'putExpDateMap']:
            if option_type not in options_data:
                continue
                
            exp_dates = list(options_data[option_type].keys())[:num_expiries]
            
            for exp_date in exp_dates:
                # Extract just the date from format "2025-11-08:7"
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                
                strikes_data = options_data[option_type][exp_date]
                
                for strike_str, contracts in strikes_data.items():
                    if not contracts:
                        continue
                        
                    strike = float(strike_str)
                    contract = contracts[0]
                    
                    # Extract data
                    gamma = contract.get('gamma', 0)
                    delta = contract.get('delta', 0)
                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)
                    
                    # Calculate notional gamma
                    notional_gamma = gamma * open_interest * 100 * underlying_price
                    
                    # Signed gamma (dealer perspective)
                    # Positive for calls (dealers short), negative for puts (dealers long)
                    signed_gamma = notional_gamma if 'call' in option_type.lower() else -notional_gamma
                    
                    results.append({
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': 'Call' if 'call' in option_type.lower() else 'Put',
                        'gamma': gamma,
                        'delta': delta,
                        'volume': volume,
                        'open_interest': open_interest,
                        'notional_gamma': abs(notional_gamma),
                        'signed_notional_gamma': signed_gamma
                    })
        
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('notional_gamma', ascending=False)
        
        return df_sorted
        
    def _create_gamma_heatmap(self, options_data, underlying_price, num_expiries=4):
        """Create gamma heatmap - adapted from Option Volume Walls page"""
        try:
            gamma_data = []
            
            # Process calls
            if 'callExpDateMap' in options_data:
                for exp_date, strikes in options_data['callExpDateMap'].items():
                    expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                    
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            contract = contracts[0]
                            strike = float(strike_str)
                            gamma = contract.get('gamma', 0)
                            oi = contract.get('openInterest', 0)
                            
                            signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            gamma_data.append({
                                'strike': strike,
                                'expiry': expiry,
                                'signed_notional_gamma': signed_gamma,
                                'type': 'call'
                            })
            
            # Process puts
            if 'putExpDateMap' in options_data:
                for exp_date, strikes in options_data['putExpDateMap'].items():
                    expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                    
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            contract = contracts[0]
                            strike = float(strike_str)
                            gamma = contract.get('gamma', 0)
                            oi = contract.get('openInterest', 0)
                            
                            signed_gamma = -1 * gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            gamma_data.append({
                                'strike': strike,
                                'expiry': expiry,
                                'signed_notional_gamma': signed_gamma,
                                'type': 'put'
                            })
            
            if not gamma_data:
                return None
            
            df_gamma = pd.DataFrame(gamma_data)
            
            # Get unique expiries and strikes
            expiries = sorted(df_gamma['expiry'].unique())[:min(num_expiries, 4)]
            all_strikes = sorted(df_gamma['strike'].unique())
            
            # Filter strikes (±5% around current price)
            min_strike = underlying_price * 0.95
            max_strike = underlying_price * 1.05
            
            filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
            
            # Limit to 12 strikes max
            if not filtered_strikes or len(filtered_strikes) > 12:
                sorted_by_distance = sorted(all_strikes, key=lambda x: abs(x - underlying_price))
                filtered_strikes = sorted(sorted_by_distance[:12])
            
            # Create data matrix
            heat_data = []
            for strike in filtered_strikes:
                row = []
                for expiry in expiries:
                    mask = (df_gamma['strike'] == strike) & (df_gamma['expiry'] == expiry)
                    strike_exp_data = df_gamma[mask]
                    
                    if not strike_exp_data.empty:
                        net_gex = strike_exp_data['signed_notional_gamma'].sum()
                        row.append(net_gex)
                    else:
                        row.append(0)
                
                heat_data.append(row)
            
            # Create labels
            strike_labels = [f"${s:.2f}" for s in filtered_strikes]
            expiry_labels = [exp.split('-')[1] + '/' + exp.split('-')[2] if '-' in exp else exp for exp in expiries]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heat_data,
                x=expiry_labels,
                y=strike_labels,
                colorscale='RdYlGn',
                zmid=0,
                showscale=True,
                colorbar=dict(
                    title="Net GEX",
                    titleside="right",
                    tickformat='$,.0s'
                ),
                hovertemplate='<b>Strike: %{y}</b><br>Expiry: %{x}<br>Net GEX: $%{z:,.0f}<extra></extra>'
            ))
            
            # Add current price line
            closest_strike = min(filtered_strikes, key=lambda x: abs(x - underlying_price))
            try:
                current_price_idx = filtered_strikes.index(closest_strike)
                fig.add_hline(
                    y=current_price_idx,
                    line=dict(color="yellow", width=3, dash="dash"),
                    annotation_text=f"  ${underlying_price:.2f}",
                    annotation_position="right"
                )
            except (ValueError, IndexError):
                pass
            
            fig.update_layout(
                title=dict(
                    text=f"Net Gamma Exposure (GEX) Heatmap - ${underlying_price:.2f}",
                    font=dict(size=18)
                ),
                xaxis=dict(title="Expiration Date"),
                yaxis=dict(title="Strike Price"),
                height=650,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gamma heatmap: {e}", exc_info=True)
            return None

    @app_commands.command(name="gamma-heatmap", description="🔥 Get gamma exposure heatmap for a symbol")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        expiries="Number of expiration dates to show (1-6, default: 4)"
    )
    async def gamma_heatmap(
        self, 
        interaction: discord.Interaction, 
        symbol: str,
        expiries: int = 4
    ):
        """Generate gamma exposure heatmap"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            expiries = max(1, min(6, expiries))  # Clamp between 1-6
            
            logger.info(f"Fetching gamma heatmap for {symbol} (user: {interaction.user})")
            
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
            
            # Create heatmap
            fig = self._create_gamma_heatmap(options_data, underlying_price, num_expiries=expiries)
            
            if fig is None:
                await interaction.followup.send(f"❌ No gamma data available for **{symbol}**")
                return
            
            # Convert to Discord file
            filename = f"{symbol}_gamma_heatmap.png"
            file = plotly_to_discord_file(fig, filename, width=1200, height=800)
            
            if not file:
                await interaction.followup.send(f"❌ Error generating chart for **{symbol}**")
                return
            
            # Create embed
            embed = create_embed(
                title=f"🔥 {symbol} Gamma Exposure Heatmap",
                description=f"**Current Price:** ${underlying_price:.2f}\n**Expiries:** {expiries}",
                color=discord.Color.red(),
                image_filename=filename
            )
            
            await interaction.followup.send(embed=embed, file=file)
            logger.info(f"Successfully sent gamma heatmap for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in gamma-heatmap command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="gamma-top", description="📊 Top gamma strikes for a symbol")
    @app_commands.describe(
        symbol="Stock symbol (e.g., SPY, QQQ, AAPL)",
        count="Number of top strikes to show (1-10, default: 5)"
    )
    async def gamma_top(
        self, 
        interaction: discord.Interaction, 
        symbol: str,
        count: int = 5
    ):
        """Show top gamma strikes"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            count = max(1, min(10, count))  # Clamp between 1-10
            
            logger.info(f"Fetching top {count} gamma strikes for {symbol} (user: {interaction.user})")
            
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
            
            # Calculate gamma strikes
            df = self._calculate_gamma_strikes(options_data, underlying_price, num_expiries=5)
            
            if df.empty:
                await interaction.followup.send(f"❌ No gamma data available for **{symbol}**")
                return
            
            top_strikes = df.head(count)
            
            # Create embed
            embed = discord.Embed(
                title=f"📊 {symbol} Top {count} Gamma Strikes",
                description=f"**Current Price:** ${underlying_price:.2f}",
                color=discord.Color.green()
            )
            
            for i, (_, row) in enumerate(top_strikes.iterrows(), 1):
                strike_info = (
                    f"**Strike:** ${row['strike']:.2f}\n"
                    f"**Type:** {row['option_type']}\n"
                    f"**Gamma Exposure:** {format_large_number(row['signed_notional_gamma'])}\n"
                    f"**Open Interest:** {int(row['open_interest']):,}\n"
                    f"**Expiry:** {row['expiry']}"
                )
                
                # Add emoji based on type
                emoji = "🟢" if row['option_type'] == 'Call' else "🔴"
                
                embed.add_field(
                    name=f"{emoji} #{i} - ${row['strike']:.2f} {row['option_type']}", 
                    value=strike_info,
                    inline=False
                )
            
            embed.set_footer(text="Data from Schwab API • Sorted by absolute notional gamma")
            
            await interaction.followup.send(embed=embed)
            logger.info(f"Successfully sent top gamma strikes for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in gamma-top command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Load the cog"""
    await bot.add_cog(GammaCommands(bot))
