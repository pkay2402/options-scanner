"""
Market Analysis Commands
Commands for dark pool sentiment, EMA trends, and market analysis
"""

import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.dark_pool import get_7day_dark_pool_sentiment, format_dark_pool_display
from bot.utils.chart_utils import create_embed

logger = logging.getLogger(__name__)


class AnalysisCommands(commands.Cog):
    """Market analysis Discord commands"""
    
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="dark-pool", description="💰 7-day dark pool sentiment analysis")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def dark_pool(self, interaction: discord.Interaction, symbol: str):
        """Show 7-day dark pool sentiment from FINRA data"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            logger.info(f"Fetching dark pool sentiment for {symbol} (user: {interaction.user})")
            
            # Get dark pool data (uses existing utility)
            dark_pool_data = get_7day_dark_pool_sentiment(symbol)
            
            if not dark_pool_data or dark_pool_data['days_available'] == 0:
                await interaction.followup.send(f"❌ No dark pool data available for **{symbol}**")
                return
            
            # Format display
            display_text, color, icon = format_dark_pool_display(dark_pool_data)
            
            # Convert hex color to Discord color
            color_int = int(color.replace('#', ''), 16)
            discord_color = discord.Color(color_int)
            
            # Create embed
            embed = discord.Embed(
                title=f"{icon} {symbol} Dark Pool Sentiment (7-Day)",
                description=display_text,
                color=discord_color
            )
            
            ratio = dark_pool_data['ratio']
            bought = dark_pool_data['total_bought']
            sold = dark_pool_data['total_sold']
            days = dark_pool_data['days_available']
            
            # Add detailed metrics
            metrics = (
                f"**Buy/Sell Ratio:** {ratio:.3f}\n"
                f"**Total Bought:** {bought:,} shares\n"
                f"**Total Sold:** {sold:,} shares\n"
                f"**Data Period:** {days} days"
            )
            embed.add_field(name="Metrics", value=metrics, inline=False)
            
            # Add interpretation
            if ratio > 1.2:
                interpretation = "🟢 **Strong Bullish** - Institutions aggressively buying"
            elif ratio > 1.0:
                interpretation = "🟡 **Neutral-Bullish** - Slight buying pressure"
            elif ratio > 0.8:
                interpretation = "⚪ **Neutral** - Balanced institutional flow"
            else:
                interpretation = "🔴 **Bearish** - Institutional selling pressure"
            
            embed.add_field(name="Interpretation", value=interpretation, inline=False)
            
            embed.set_footer(text="Data from FINRA • Dark pool = off-exchange trading")
            
            await interaction.followup.send(embed=embed)
            logger.info(f"Successfully sent dark pool sentiment for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in dark-pool command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="ema-trend", description="📈 EMA trend analysis (8/21/50/200)")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def ema_trend(self, interaction: discord.Interaction, symbol: str):
        """Show EMA positioning and trend analysis"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            logger.info(f"Fetching EMA trend for {symbol} (user: {interaction.user})")
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Get quote for current price
            quote_data = client.get_quote(symbol)
            if not quote_data or 'quote' not in quote_data:
                await interaction.followup.send(f"❌ No quote data available for **{symbol}**")
                return
            
            underlying_price = quote_data['quote'].get('lastPrice', 0)
            
            if not underlying_price:
                await interaction.followup.send(f"❌ Could not determine price for **{symbol}**")
                return
            
            # Get price history for EMA calculation
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                
                if hist.empty:
                    await interaction.followup.send(f"❌ No historical data available for **{symbol}**")
                    return
                
                # Calculate EMAs
                ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
                ema_21 = hist['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
                ema_50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                
            except Exception as e:
                logger.error(f"Error calculating EMAs: {e}")
                await interaction.followup.send(f"❌ Could not calculate EMAs for **{symbol}**")
                return
            
            # Determine trend
            if underlying_price > ema_8 > ema_21 > ema_50:
                trend = "📈 **Strong Uptrend**"
                color = discord.Color.green()
                emoji = "🟢"
            elif underlying_price < ema_8 < ema_21 < ema_50:
                trend = "📉 **Strong Downtrend**"
                color = discord.Color.red()
                emoji = "🔴"
            elif underlying_price > ema_50 > ema_200:
                trend = "🟢 **Bullish Trend**"
                color = discord.Color.green()
                emoji = "🟢"
            elif underlying_price < ema_50 < ema_200:
                trend = "🔴 **Bearish Trend**"
                color = discord.Color.red()
                emoji = "🔴"
            else:
                trend = "📊 **Mixed/Ranging**"
                color = discord.Color.gold()
                emoji = "⚪"
            
            # Create embed
            embed = discord.Embed(
                title=f"{emoji} {symbol} EMA Trend Analysis",
                description=f"**Current Price:** ${underlying_price:.2f}\n**Trend:** {trend}",
                color=color
            )
            
            # Add EMA values
            ema_values = (
                f"**EMA-8:** ${ema_8:.2f} ({((underlying_price - ema_8) / ema_8 * 100):+.2f}%)\n"
                f"**EMA-21:** ${ema_21:.2f} ({((underlying_price - ema_21) / ema_21 * 100):+.2f}%)\n"
                f"**EMA-50:** ${ema_50:.2f} ({((underlying_price - ema_50) / ema_50 * 100):+.2f}%)\n"
                f"**EMA-200:** ${ema_200:.2f} ({((underlying_price - ema_200) / ema_200 * 100):+.2f}%)"
            )
            embed.add_field(name="EMA Levels", value=ema_values, inline=False)
            
            # Add support/resistance
            if underlying_price > ema_8:
                support = f"Nearest support: ${ema_8:.2f} (EMA-8)"
            elif underlying_price > ema_21:
                support = f"Nearest support: ${ema_21:.2f} (EMA-21)"
            elif underlying_price > ema_50:
                support = f"Nearest support: ${ema_50:.2f} (EMA-50)"
            else:
                support = f"Nearest support: ${ema_200:.2f} (EMA-200)"
            
            if underlying_price < ema_8:
                resistance = f"Nearest resistance: ${ema_8:.2f} (EMA-8)"
            elif underlying_price < ema_21:
                resistance = f"Nearest resistance: ${ema_21:.2f} (EMA-21)"
            elif underlying_price < ema_50:
                resistance = f"Nearest resistance: ${ema_50:.2f} (EMA-50)"
            else:
                resistance = f"Nearest resistance: Above all EMAs"
            
            levels = f"{support}\n{resistance}"
            embed.add_field(name="Key Levels", value=levels, inline=False)
            
            embed.set_footer(text="EMAs = Exponential Moving Averages • Trend following indicators")
            
            await interaction.followup.send(embed=embed)
            logger.info(f"Successfully sent EMA trend for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in ema-trend command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="quote", description="💵 Quick quote for a symbol")
    @app_commands.describe(symbol="Stock symbol (e.g., SPY, QQQ, AAPL)")
    async def quote(self, interaction: discord.Interaction, symbol: str):
        """Get quick quote"""
        await interaction.response.defer(thinking=True)
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            client = self.bot.schwab_service.get_client()
            
            # Get quote
            quote_data = client.get_quote(symbol)
            
            if not quote_data or 'quote' not in quote_data:
                await interaction.followup.send(f"❌ No quote data available for **{symbol}**")
                return
            
            quote = quote_data['quote']
            
            # Extract quote data
            last_price = quote.get('lastPrice', 0)
            change = quote.get('netChange', 0)
            change_pct = quote.get('netPercentChange', 0)
            volume = quote.get('totalVolume', 0)
            bid = quote.get('bidPrice', 0)
            ask = quote.get('askPrice', 0)
            
            # Determine color based on change
            if change > 0:
                color = discord.Color.green()
                emoji = "📈"
            elif change < 0:
                color = discord.Color.red()
                emoji = "📉"
            else:
                color = discord.Color.gold()
                emoji = "➡️"
            
            # Create embed
            embed = discord.Embed(
                title=f"{emoji} {symbol} Quote",
                description=f"**${last_price:.2f}** {change:+.2f} ({change_pct:+.2f}%)",
                color=color
            )
            
            # Add details
            details = (
                f"**Bid:** ${bid:.2f}\n"
                f"**Ask:** ${ask:.2f}\n"
                f"**Volume:** {volume:,}"
            )
            embed.add_field(name="Details", value=details, inline=False)
            
            embed.set_footer(text="Data from Schwab API • Real-time quote")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in quote command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Load the cog"""
    await bot.add_cog(AnalysisCommands(bot))
