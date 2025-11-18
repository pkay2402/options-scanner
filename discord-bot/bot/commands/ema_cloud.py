"""
EMA Cloud Signals Command
Shows EMA cloud alignment and crossover signals on multiple timeframes (1H, 4H, Daily)
Based on EMA Cloud Scanner page logic
"""

import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Predefined stock lists
STOCK_LISTS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'ADBE', 'CRM', 'ORCL'],
    'mega': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'ai': ['NVDA', 'PLTR', 'SNOW', 'NET', 'DDOG', 'PANW', 'CRWD', 'ZS', 'COIN', 'APP'],
    'indices': ['SPY', 'QQQ', 'IWM', 'DIA']
}

ALL_PREDEFINED = list(set([s for stocks in STOCK_LISTS.values() for s in stocks]))


def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()


def get_ema_signals(ticker, timeframe='1d'):
    """
    Get EMA cloud signals for a ticker
    Returns: dict with signal data or None
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch appropriate data based on timeframe
        if timeframe == '1h':
            data = stock.history(period='60d', interval='1h')
        elif timeframe == '4h':
            data = stock.history(period='60d', interval='1h')
            # Resample to 4H
            if not data.empty:
                data = data.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
        else:  # daily
            data = stock.history(period='100d', interval='1d')
        
        if data.empty or len(data) < 50:
            return None
        
        # Calculate EMAs (5, 12, 34, 50)
        data['EMA_5'] = calculate_ema(data, 5)
        data['EMA_12'] = calculate_ema(data, 12)
        data['EMA_34'] = calculate_ema(data, 34)
        data['EMA_50'] = calculate_ema(data, 50)
        
        # Drop NaN values
        data = data.dropna()
        
        if len(data) < 2:
            return None
        
        # Get current and previous values
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        current_price = current['Close']
        
        # Check EMA alignment (bullish = 5 > 12 > 34 > 50)
        bullish_alignment = (
            current['EMA_5'] > current['EMA_12'] and
            current['EMA_12'] > current['EMA_34'] and
            current['EMA_34'] > current['EMA_50']
        )
        
        bearish_alignment = (
            current['EMA_5'] < current['EMA_12'] and
            current['EMA_12'] < current['EMA_34'] and
            current['EMA_34'] < current['EMA_50']
        )
        
        # Detect crossovers
        bullish_cross = (
            previous['EMA_5'] <= previous['EMA_12'] and
            current['EMA_5'] > current['EMA_12']
        )
        
        bearish_cross = (
            previous['EMA_5'] >= previous['EMA_12'] and
            current['EMA_5'] < current['EMA_12']
        )
        
        # Check price position relative to EMAs
        above_all_emas = (
            current_price > current['EMA_5'] and
            current_price > current['EMA_12'] and
            current_price > current['EMA_34']
        )
        
        below_all_emas = (
            current_price < current['EMA_5'] and
            current_price < current['EMA_12'] and
            current_price < current['EMA_34']
        )
        
        # Calculate distances from EMAs (as percentage)
        ema_5_dist = ((current_price - current['EMA_5']) / current_price * 100)
        ema_12_dist = ((current_price - current['EMA_12']) / current_price * 100)
        ema_34_dist = ((current_price - current['EMA_34']) / current_price * 100)
        
        # Determine signal strength
        if bullish_alignment and above_all_emas:
            signal = "STRONG_BULL"
            emoji = "🟢🟢"
        elif bullish_alignment:
            signal = "BULLISH"
            emoji = "🟢"
        elif bearish_alignment and below_all_emas:
            signal = "STRONG_BEAR"
            emoji = "🔴🔴"
        elif bearish_alignment:
            signal = "BEARISH"
            emoji = "🔴"
        else:
            signal = "NEUTRAL"
            emoji = "⚪"
        
        if bullish_cross:
            signal = "BULL_CROSS"
            emoji = "🚀"
        elif bearish_cross:
            signal = "BEAR_CROSS"
            emoji = "💥"
        
        return {
            'ticker': ticker,
            'price': current_price,
            'signal': signal,
            'emoji': emoji,
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross,
            'above_all_emas': above_all_emas,
            'ema_5': current['EMA_5'],
            'ema_12': current['EMA_12'],
            'ema_34': current['EMA_34'],
            'ema_50': current['EMA_50'],
            'ema_5_dist': ema_5_dist,
            'ema_12_dist': ema_12_dist,
            'ema_34_dist': ema_34_dist
        }
        
    except Exception as e:
        logger.error(f"Error getting EMA signals for {ticker}: {e}")
        return None


class EMACloudCommands(commands.Cog):
    """EMA Cloud Signal Commands"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="ema", description="Get EMA cloud signals across multiple timeframes")
    @app_commands.describe(
        symbol="Stock symbol or list: tech, mega, ai, indices (default: mega)",
        show_neutral="Show neutral signals too (default: False)"
    )
    async def ema_command(self, interaction: discord.Interaction, symbol: str = "mega", show_neutral: bool = False):
        """Get EMA cloud signals"""
        await interaction.response.defer()
        
        try:
            symbol = symbol.upper()
            
            # Determine which symbols to scan
            if symbol in ['TECH', 'MEGA', 'AI', 'INDICES']:
                symbols = STOCK_LISTS[symbol.lower()]
                scan_type = f"{symbol.title()} Stocks"
            elif symbol in ALL_PREDEFINED or ',' not in symbol:
                # Single symbol or predefined
                symbols = [symbol] if symbol in ALL_PREDEFINED or ',' not in symbol else symbol.split(',')
                scan_type = f"{symbol} Analysis"
            else:
                # Multiple symbols
                symbols = [s.strip() for s in symbol.split(',')]
                scan_type = "Custom Scan"
            
            # Limit to 15 symbols to avoid timeout
            symbols = symbols[:15]
            
            await interaction.followup.send(f"🔍 Scanning {len(symbols)} symbols for EMA signals across 1H, 4H, and Daily timeframes...")
            
            # Collect signals for all timeframes
            results = {'1h': [], '4h': [], 'daily': []}
            
            for ticker in symbols:
                for tf in ['1h', '4h', '1d']:
                    signal_data = get_ema_signals(ticker, tf)
                    if signal_data:
                        tf_key = tf.replace('1d', 'daily')
                        results[tf_key].append(signal_data)
            
            # Filter results based on show_neutral
            if not show_neutral:
                for tf in results:
                    results[tf] = [r for r in results[tf] if r['signal'] != 'NEUTRAL']
            
            # Create embeds for each timeframe
            embeds = []
            
            for tf_name, tf_data in [('1 Hour', results['1h']), ('4 Hour', results['4h']), ('Daily', results['daily'])]:
                if not tf_data:
                    continue
                
                # Sort by signal strength
                signal_order = {'BULL_CROSS': 0, 'STRONG_BULL': 1, 'BULLISH': 2, 'NEUTRAL': 3, 
                               'BEARISH': 4, 'STRONG_BEAR': 5, 'BEAR_CROSS': 6}
                tf_data.sort(key=lambda x: signal_order.get(x['signal'], 3))
                
                embed = discord.Embed(
                    title=f"📊 EMA Cloud Signals - {tf_name} Timeframe",
                    description=f"**Scan Type:** {scan_type}\n**Symbols Analyzed:** {len(tf_data)}",
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                
                # Group by signal type
                bullish = [r for r in tf_data if 'BULL' in r['signal']]
                bearish = [r for r in tf_data if 'BEAR' in r['signal']]
                neutral = [r for r in tf_data if r['signal'] == 'NEUTRAL']
                
                # Add bullish signals
                if bullish:
                    bull_text = ""
                    for sig in bullish[:8]:  # Limit to 8 per section
                        bull_text += f"{sig['emoji']} **{sig['ticker']}** ${sig['price']:.2f}\n"
                        if sig['bullish_cross']:
                            bull_text += "   🚀 BULLISH CROSSOVER!\n"
                        else:
                            bull_text += f"   EMA 5/12: {sig['ema_5_dist']:+.2f}% / {sig['ema_12_dist']:+.2f}%\n"
                    
                    embed.add_field(
                        name=f"🟢 Bullish Signals ({len(bullish)})",
                        value=bull_text or "None",
                        inline=False
                    )
                
                # Add bearish signals
                if bearish:
                    bear_text = ""
                    for sig in bearish[:8]:
                        bear_text += f"{sig['emoji']} **{sig['ticker']}** ${sig['price']:.2f}\n"
                        if sig['bearish_cross']:
                            bear_text += "   💥 BEARISH CROSSOVER!\n"
                        else:
                            bear_text += f"   EMA 5/12: {sig['ema_5_dist']:+.2f}% / {sig['ema_12_dist']:+.2f}%\n"
                    
                    embed.add_field(
                        name=f"🔴 Bearish Signals ({len(bearish)})",
                        value=bear_text or "None",
                        inline=False
                    )
                
                # Add neutral if requested
                if show_neutral and neutral:
                    neutral_text = ", ".join([f"{r['ticker']}" for r in neutral[:10]])
                    embed.add_field(
                        name=f"⚪ Neutral ({len(neutral)})",
                        value=neutral_text[:1024] or "None",  # Discord field limit
                        inline=False
                    )
                
                embed.set_footer(text="EMA Cloud: 5/12/34/50 periods | Strong signals = all EMAs aligned")
                embeds.append(embed)
            
            if not embeds:
                await interaction.followup.send(f"❌ No significant EMA signals found for {scan_type}")
                return
            
            # Send embeds (max 3 timeframes)
            for embed in embeds:
                await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in ema command: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Setup function called by Discord.py"""
    await bot.add_cog(EMACloudCommands(bot))
