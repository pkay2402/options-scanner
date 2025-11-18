"""
Stock Gamma Map Command
Generates gamma heatmap similar to Stock Option Finder page
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
import io

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


def calculate_gamma_heatmap_data(options_chain, underlying_price, num_expiries=4):
    """Calculate gamma exposure by strike and expiry"""
    try:
        gamma_data = []
        
        # Process calls
        if 'callExpDateMap' in options_chain:
            expiries = list(options_chain['callExpDateMap'].keys())[:num_expiries]
            
            for exp_date in expiries:
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                strikes = options_chain['callExpDateMap'][exp_date]
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        oi = contract.get('openInterest', 0)
                        
                        # Calculate signed notional gamma (positive for calls from dealer perspective)
                        signed_gamma = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'strike': strike,
                            'expiry': expiry,
                            'gamma_exposure': signed_gamma / 1_000_000,  # Convert to millions
                            'type': 'CALL'
                        })
        
        # Process puts
        if 'putExpDateMap' in options_chain:
            expiries = list(options_chain['putExpDateMap'].keys())[:num_expiries]
            
            for exp_date in expiries:
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                strikes = options_chain['putExpDateMap'][exp_date]
                
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        gamma = contract.get('gamma', 0)
                        oi = contract.get('openInterest', 0)
                        
                        # Calculate signed notional gamma (negative for puts from dealer perspective)
                        signed_gamma = -1 * gamma * oi * 100 * underlying_price * underlying_price * 0.01
                        
                        gamma_data.append({
                            'strike': strike,
                            'expiry': expiry,
                            'gamma_exposure': signed_gamma / 1_000_000,  # Convert to millions
                            'type': 'PUT'
                        })
        
        if not gamma_data:
            return None, None
        
        df = pd.DataFrame(gamma_data)
        
        # Filter strikes within ±10% of current price
        min_strike = underlying_price * 0.90
        max_strike = underlying_price * 1.10
        df = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
        
        # Get top gamma strikes across all expiries
        gamma_by_strike = df.groupby('strike')['gamma_exposure'].sum().abs().sort_values(ascending=False).head(10)
        top_strikes = [(strike, gamma) for strike, gamma in gamma_by_strike.items()]
        
        return df, top_strikes
        
    except Exception as e:
        logger.error(f"Error calculating gamma heatmap: {e}")
        return None, None


def create_gamma_heatmap_chart(df, underlying_price, symbol):
    """Create gamma heatmap chart"""
    try:
        # Pivot data for heatmap
        expiries = sorted(df['expiry'].unique())[:4]
        strikes = sorted(df['strike'].unique())
        
        # Create pivot table
        pivot_data = []
        expiry_labels = []
        
        for expiry in expiries:
            expiry_data = df[df['expiry'] == expiry]
            gamma_by_strike = expiry_data.groupby('strike')['gamma_exposure'].sum()
            pivot_data.append([gamma_by_strike.get(strike, 0) for strike in strikes])
            
            # Format expiry label (e.g., "Nov 23" instead of "2025-11-23")
            try:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                expiry_labels.append(exp_date.strftime("%b %d"))
            except:
                expiry_labels.append(expiry.split('-')[1] + '/' + expiry.split('-')[2])
        
        # Format text overlays with better visibility
        text_data = []
        for row in pivot_data:
            text_row = []
            for val in row:
                if abs(val) < 0.1:
                    text_row.append('')  # Hide very small values
                elif abs(val) < 1:
                    text_row.append(f'{val:.2f}M')
                else:
                    text_row.append(f'{val:.1f}M')
            text_data.append(text_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data,
            x=[f'${s:.0f}' for s in strikes],  # Format strikes with $
            y=expiry_labels,
            colorscale=[
                [0, '#d32f2f'],      # Darker red for negative (put gamma)
                [0.45, '#ff6b6b'],   # Light red
                [0.5, '#f5f5f5'],    # Near white for neutral
                [0.55, '#81c784'],   # Light green
                [1, '#2e7d32']       # Darker green for positive (call gamma)
            ],
            zmid=0,
            text=text_data,
            texttemplate='%{text}',
            textfont={"size": 9, "color": "#000000"},
            hovertemplate='<b>Strike:</b> %{x}<br><b>Expiry:</b> %{y}<br><b>Gamma:</b> %{z:.2f}M<extra></extra>',
            colorbar=dict(
                title=dict(
                    text="Gamma<br>Exposure<br>(Millions)",
                    font=dict(size=11, color='#e8e8e8')
                ),
                tickfont=dict(size=10, color='#e8e8e8'),
                x=1.02
            )
        ))
        
        # Add current price line
        current_strike_label = f'${underlying_price:.0f}'
        fig.add_vline(
            x=current_strike_label,
            line=dict(dash="dash", color="cyan", width=2),
            annotation=dict(
                text=f"Current: ${underlying_price:.2f}",
                font=dict(size=12, color="cyan", family="Arial Black"),
                bgcolor="rgba(0,0,0,0.7)",
                borderpad=4
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol} - Gamma Exposure Heatmap</b>",
                font=dict(size=18, color='#e8e8e8'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Strike Price",
                titlefont=dict(size=13, color='#e8e8e8'),
                tickfont=dict(size=10, color='#e8e8e8'),
                gridcolor='#2d3561',
                showgrid=True
            ),
            yaxis=dict(
                title="Expiration Date",
                titlefont=dict(size=13, color='#e8e8e8'),
                tickfont=dict(size=11, color='#e8e8e8'),
                gridcolor='#2d3561'
            ),
            height=600,
            width=900,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#16213e',
            font=dict(color='#e8e8e8', size=11),
            margin=dict(l=80, r=100, t=80, b=60)
        )
        
        # Save to bytes
        img_bytes = fig.to_image(format="png", width=900, height=600, scale=2)
        return io.BytesIO(img_bytes)
        
    except Exception as e:
        logger.error(f"Error creating heatmap chart: {e}")
        return None


class GammaMapCommands(commands.Cog):
    """Gamma Heatmap Commands"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="gammamap", description="Get gamma exposure heatmap for a stock")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, NVDA)")
    async def gamma_map(self, interaction: discord.Interaction, symbol: str):
        """Generate gamma heatmap"""
        await interaction.response.defer()
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                await interaction.followup.send("❌ Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            
            # Get quote
            quote = client.get_quote(symbol)
            if not quote or symbol not in quote:
                await interaction.followup.send(f"❌ Could not fetch quote for {symbol}")
                return
            
            underlying_price = quote[symbol]['quote']['lastPrice']
            
            # Get options chain (all available expiries)
            options_chain = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL'
            )
            
            if not options_chain or 'callExpDateMap' not in options_chain:
                await interaction.followup.send(f"❌ No options data for {symbol}")
                return
            
            # Calculate gamma data
            df, top_strikes = calculate_gamma_heatmap_data(options_chain, underlying_price)
            
            if df is None or df.empty:
                await interaction.followup.send(f"❌ Could not calculate gamma data for {symbol}")
                return
            
            # Create embed with summary
            embed = discord.Embed(
                title=f"🔥 {symbol} - Gamma Exposure Heatmap",
                description=f"**Current Price:** ${underlying_price:,.2f}\n"
                           f"**Total Strikes Analyzed:** {len(df['strike'].unique())}\n"
                           f"**Expiries Shown:** {len(df['expiry'].unique())}",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            
            # Add top gamma strikes
            top_strikes_text = ""
            for i, (strike, gamma) in enumerate(top_strikes[:5], 1):
                distance = ((strike - underlying_price) / underlying_price * 100)
                emoji = "🟢" if gamma > 0 else "🔴"
                top_strikes_text += f"{i}. {emoji} ${strike:,.2f} ({distance:+.1f}%) - {abs(gamma):,.1f}M\n"
            
            embed.add_field(
                name="⚡ Top 5 Gamma Strikes",
                value=top_strikes_text,
                inline=False
            )
            
            # Calculate total gamma exposure by type
            total_call_gamma = df[df['type'] == 'CALL']['gamma_exposure'].sum()
            total_put_gamma = abs(df[df['type'] == 'PUT']['gamma_exposure'].sum())
            net_gamma = total_call_gamma - total_put_gamma
            
            # Determine market sentiment
            if net_gamma > 0:
                sentiment = "🟢 Bullish (Call Gamma Dominates)"
            elif net_gamma < 0:
                sentiment = "🔴 Bearish (Put Gamma Dominates)"
            else:
                sentiment = "⚪ Neutral"
            
            embed.add_field(
                name="📊 Gamma Summary",
                value=f"**Call Gamma:** +{total_call_gamma:,.0f}M\n"
                      f"**Put Gamma:** -{total_put_gamma:,.0f}M\n"
                      f"**Net Gamma:** {net_gamma:+,.0f}M\n"
                      f"**Sentiment:** {sentiment}",
                inline=False
            )
            
            embed.set_footer(text="Green = Call Gamma (bullish) | Red = Put Gamma (bearish)")
            
            # Try to create and send chart
            try:
                chart_bytes = create_gamma_heatmap_chart(df, underlying_price, symbol)
                if chart_bytes:
                    file = discord.File(chart_bytes, filename=f"{symbol}_gamma_heatmap.png")
                    embed.set_image(url=f"attachment://{symbol}_gamma_heatmap.png")
                    await interaction.followup.send(embed=embed, file=file)
                else:
                    await interaction.followup.send(embed=embed)
            except Exception as e:
                logger.error(f"Could not create chart: {e}")
                await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in gamma_map: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Setup function called by Discord.py"""
    await bot.add_cog(GammaMapCommands(bot))
