"""
Whale Score Command
Scans for whale activity using VALR formula from Whale Flows page
"""

import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)

# Predefined top tech stocks
TOP_TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'AMD',
    'CRM', 'GS', 'NFLX', 'IBIT', 'COIN',
    'APP', 'PLTR', 'SNOW', 'TEAM', 'CRWD',
    'LLY', 'ABBV', 'AXP', 'JPM', 'HD',  # Pharma, Financial, Retail
    'SPY', 'QQQ'
]


def get_next_friday():
    """Get next Friday for weekly expiry"""
    today = datetime.now().date()
    weekday = today.weekday()
    days_to_friday = (4 - weekday) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    return today + timedelta(days=days_to_friday)


def get_next_three_fridays():
    """Get next 3 weekly expiries (Fridays)"""
    first_friday = get_next_friday()
    return [
        first_friday,
        first_friday + timedelta(days=7),
        first_friday + timedelta(days=14)
    ]


def calculate_whale_score(option_data, underlying_price, underlying_volume):
    """
    Calculate VALR (Volatility Adjusted Leverage Ratio) whale score
    Formula: (Delta × Price / Mark) × IV × (Vol/OI) × (OptDollar/UndDollar) × 1000
    """
    try:
        delta = abs(option_data.get('delta', 0))
        mark = option_data.get('mark', 0)
        iv = option_data.get('volatility', 0) / 100 if option_data.get('volatility', 0) else 0.01
        volume = option_data.get('totalVolume', 0)
        oi = max(option_data.get('openInterest', 1), 1)  # Avoid division by zero
        
        if mark == 0 or delta == 0 or underlying_volume == 0:
            return 0
        
        # Calculate components
        leverage = delta * underlying_price
        leverage_ratio = leverage / mark
        valr = leverage_ratio * iv
        vol_oi = volume / oi
        dvolume_opt = volume * mark * 100
        dvolume_und = underlying_price * underlying_volume
        dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
        
        # VALR Formula
        whale_score = valr * vol_oi * dvolume_ratio * 1000
        
        return whale_score
        
    except Exception as e:
        logger.error(f"Error calculating whale score: {e}")
        return 0


def scan_stock_whale_flows(client, symbol, expiry_dates, min_whale_score=100):
    """Scan a stock for whale flows across multiple expiries (optimized for 3 weeks)"""
    try:
        # Get quote
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            return None
        
        underlying_price = quote[symbol]['quote']['lastPrice']
        underlying_volume = quote[symbol]['quote'].get('totalVolume', 0)
        
        # Skip if no underlying volume
        if underlying_volume == 0:
            return None
        
        # Get options chain for all 3 expiries in one call (optimized)
        first_expiry = expiry_dates[0].strftime("%Y-%m-%d")
        last_expiry = expiry_dates[-1].strftime("%Y-%m-%d")
        options_chain = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=first_expiry,
            to_date=last_expiry
        )
        
        if not options_chain:
            return None
        
        results = []
        
        # Filter to target expiries only (faster processing)
        target_expiry_strs = [exp.strftime("%Y-%m-%d") for exp in expiry_dates]
        
        # Process calls
        if 'callExpDateMap' in options_chain:
            for exp_date_full, strikes in options_chain['callExpDateMap'].items():
                exp_date = exp_date_full.split(':')[0]
                if exp_date not in target_expiry_strs:
                    continue
                    
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Filter: Only strikes within 5% of current price
                        if abs(strike - underlying_price) / underlying_price > 0.05:
                            continue
                        
                        whale_score = calculate_whale_score(contract, underlying_price, underlying_volume)
                        
                        if whale_score >= min_whale_score:
                            results.append({
                                'symbol': symbol,
                                'type': 'CALL',
                                'strike': strike,
                                'expiry': exp_date,
                                'whale_score': whale_score,
                                'volume': contract.get('totalVolume', 0),
                                'oi': contract.get('openInterest', 0),
                                'iv': contract.get('volatility', 0),
                                'delta': contract.get('delta', 0),
                                'underlying_price': underlying_price
                            })
        
        # Process puts
        if 'putExpDateMap' in options_chain:
            for exp_date_full, strikes in options_chain['putExpDateMap'].items():
                exp_date = exp_date_full.split(':')[0]
                if exp_date not in target_expiry_strs:
                    continue
                    
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        # Filter: Only strikes within 5% of current price
                        if abs(strike - underlying_price) / underlying_price > 0.05:
                            continue
                        
                        whale_score = calculate_whale_score(contract, underlying_price, underlying_volume)
                        
                        if whale_score >= min_whale_score:
                            results.append({
                                'symbol': symbol,
                                'type': 'PUT',
                                'strike': strike,
                                'expiry': exp_date,
                                'whale_score': whale_score,
                                'volume': contract.get('totalVolume', 0),
                                'oi': contract.get('openInterest', 0),
                                'iv': contract.get('volatility', 0),
                                'delta': contract.get('delta', 0),
                                'underlying_price': underlying_price
                            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None


class WhaleScoreCommands(commands.Cog):
    """Whale Score Analysis Commands"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="whalescan", description="Scan predefined stocks for whale activity")
    @app_commands.describe(min_score="Minimum whale score threshold (default: 50)")
    async def whale_scan(self, interaction: discord.Interaction, min_score: int = 50):
        """Scan predefined tech stocks for whale flows"""
        await interaction.response.defer()
        
        try:
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                await interaction.followup.send("❌ Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            expiry_dates = get_next_three_fridays()
            
            expiry_str = ", ".join([exp.strftime('%b %d') for exp in expiry_dates])
            await interaction.followup.send(f"🔍 Scanning {len(TOP_TECH_STOCKS)} stocks for whale activity...\nExpiries: {expiry_str}\nMin Score: {min_score:,}")
            
            all_flows = []
            
            # Scan each stock (with all 3 expiries in one call per stock)
            for symbol in TOP_TECH_STOCKS:
                flows = scan_stock_whale_flows(client, symbol, expiry_dates, min_score)
                if flows:
                    all_flows.extend(flows)
            
            if not all_flows:
                await interaction.followup.send(f"❌ No whale flows found with score >= {min_score}")
                return
            
            # Sort by whale score
            df = pd.DataFrame(all_flows)
            df = df.sort_values('whale_score', ascending=False).head(20)
            
            # Create embed
            expiry_display = ", ".join([exp.strftime('%b %d') for exp in expiry_dates])
            embed = discord.Embed(
                title="🐋 Whale Flows Scanner Results",
                description=f"**Expiries:** {expiry_display}\n**Results:** {len(df)} whale flows detected",
                color=discord.Color.purple(),
                timestamp=datetime.now()
            )
            
            # Add summary stats
            call_count = len(df[df['type'] == 'CALL'])
            put_count = len(df[df['type'] == 'PUT'])
            avg_score = df['whale_score'].mean()
            
            embed.add_field(
                name="📊 Summary",
                value=f"Avg Score: {avg_score:,.0f}\n"
                      f"Calls/Puts: {call_count}/{put_count}\n"
                      f"Total Vol: {df['volume'].sum():,.0f}",
                inline=False
            )
            
            # Add top flows (in chunks to avoid exceeding field limits)
            for i in range(0, min(len(df), 10), 5):
                chunk = df.iloc[i:i+5]
                flows_text = ""
                
                for _, row in chunk.iterrows():
                    emoji = "🟢" if row['type'] == 'CALL' else "🔴"
                    distance = ((row['strike'] - row['underlying_price']) / row['underlying_price'] * 100)
                    exp_date = datetime.strptime(row['expiry'], '%Y-%m-%d').strftime('%m/%d')
                    flows_text += f"{emoji} **{row['symbol']}** ${row['strike']:.2f} ({distance:+.1f}%) [{exp_date}]\n"
                    flows_text += f"   Score: {row['whale_score']:,.0f} | Vol: {row['volume']:,.0f} | IV: {row['iv']*100:.0f}%\n"
                
                embed.add_field(
                    name=f"Top Flows #{i+1}-{i+len(chunk)}",
                    value=flows_text,
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in whale_scan: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")
    
    @app_commands.command(name="whalestock", description="Get whale flows for a specific stock")
    @app_commands.describe(
        symbol="Stock symbol (e.g., NVDA, TSLA)",
        min_score="Minimum whale score threshold (default: 100)"
    )
    async def whale_stock(self, interaction: discord.Interaction, symbol: str, min_score: int = 100):
        """Get whale flows for a specific stock"""
        await interaction.response.defer()
        
        try:
            symbol = symbol.upper()
            
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                await interaction.followup.send("❌ Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            expiry_dates = get_next_three_fridays()
            
            # Scan stock
            flows = scan_stock_whale_flows(client, symbol, expiry_dates, min_score)
            
            if not flows:
                await interaction.followup.send(f"❌ No whale flows found for {symbol} with score >= {min_score}")
                return
            
            df = pd.DataFrame(flows)
            df = df.sort_values('whale_score', ascending=False)
            
            # Create embed
            expiry_display = ", ".join([exp.strftime('%b %d') for exp in expiry_dates])
            embed = discord.Embed(
                title=f"🐋 {symbol} - Whale Flows",
                description=f"**Current Price:** ${flows[0]['underlying_price']:,.2f}\n"
                           f"**Expiries:** {expiry_display}\n"
                           f"**Flows Found:** {len(df)}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            # Add flows
            for i, row in df.head(15).iterrows():
                emoji = "🟢" if row['type'] == 'CALL' else "🔴"
                distance = ((row['strike'] - row['underlying_price']) / row['underlying_price'] * 100)
                exp_date = datetime.strptime(row['expiry'], '%Y-%m-%d').strftime('%m/%d')
                
                embed.add_field(
                    name=f"{emoji} {row['type']} ${row['strike']:.2f} ({distance:+.1f}%) [{exp_date}]",
                    value=f"**Score:** {row['whale_score']:,.0f}\n"
                          f"Vol: {row['volume']:,.0f} | OI: {row['oi']:,.0f}\n"
                          f"IV: {row['iv']*100:.0f}% | Δ: {row['delta']:.2f}",
                    inline=True
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in whale_stock: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Setup function called by Discord.py"""
    await bot.add_cog(WhaleScoreCommands(bot))
