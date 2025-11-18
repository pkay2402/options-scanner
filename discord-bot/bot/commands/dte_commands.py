"""
0DTE Commands - SPY, QQQ, $SPX
Provides 0DTE analysis for index symbols using latest day expiry
For stocks like NVDA, TSLA, provides weekly expiry data
Based on Option Volume Walls page logic
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

# Index symbols that support 0DTE
INDEX_SYMBOLS = ['SPY', 'QQQ', '$SPX']

def get_next_expiry(symbol: str):
    """Get next expiry date - 0DTE for indices, weekly for stocks"""
    today = datetime.now().date()
    weekday = today.weekday()  # 0=Monday, 6=Sunday
    
    if symbol.upper() in INDEX_SYMBOLS:
        # For indices, use next trading day (0DTE)
        # Skip weekends: if Saturday (5) or Sunday (6), go to Monday
        if weekday == 5:  # Saturday
            return today + timedelta(days=2)
        elif weekday == 6:  # Sunday
            return today + timedelta(days=1)
        else:
            # Weekday - use today
            return today
    else:
        # For stocks, get next Friday (weekly expiry)
        days_to_friday = (4 - weekday) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        return today + timedelta(days=days_to_friday)

def calculate_option_metrics(options_chain, underlying_price, expiry_date):
    """Calculate key metrics from options chain"""
    try:
        metrics = {
            'total_call_volume': 0,
            'total_put_volume': 0,
            'total_call_oi': 0,
            'total_put_oi': 0,
            'max_pain': 0,
            'call_walls': [],
            'put_walls': [],
            'gamma_levels': []
        }
        
        options_list = []
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        
        logger.info(f"Looking for expiry: {expiry_str}")
        
        # Log available expiries for debugging
        if 'callExpDateMap' in options_chain:
            available_expiries = list(options_chain['callExpDateMap'].keys())
            logger.info(f"Available call expiries: {available_expiries[:5]}")
        
        # Process calls
        if 'callExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['callExpDateMap'].items():
                # Extract date from format like "2025-11-16:7"
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                # Compare dates (both as strings in YYYY-MM-DD format)
                if expiry != expiry_date.strftime("%Y-%m-%d"):
                    continue
                    
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        metrics['total_call_volume'] += volume
                        metrics['total_call_oi'] += oi
                        
                        options_list.append({
                            'strike': strike,
                            'type': 'CALL',
                            'volume': volume,
                            'oi': oi,
                            'gamma': gamma
                        })
        
        # Process puts
        if 'putExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['putExpDateMap'].items():
                # Extract date from format like "2025-11-16:7"
                expiry = exp_date.split(':')[0] if ':' in exp_date else exp_date
                # Compare dates (both as strings in YYYY-MM-DD format)
                if expiry != expiry_date.strftime("%Y-%m-%d"):
                    continue
                    
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        metrics['total_put_volume'] += volume
                        metrics['total_put_oi'] += oi
                        
                        options_list.append({
                            'strike': strike,
                            'type': 'PUT',
                            'volume': volume,
                            'oi': oi,
                            'gamma': gamma
                        })
        
        if not options_list:
            return None
        
        df = pd.DataFrame(options_list)
        
        # Calculate call/put walls (top 5 by OI)
        call_df = df[df['type'] == 'CALL'].sort_values('oi', ascending=False).head(5)
        put_df = df[df['type'] == 'PUT'].sort_values('oi', ascending=False).head(5)
        
        metrics['call_walls'] = [(row['strike'], row['oi']) for _, row in call_df.iterrows()]
        metrics['put_walls'] = [(row['strike'], row['oi']) for _, row in put_df.iterrows()]
        
        # Calculate gamma by strike
        gamma_by_strike = df.groupby('strike').apply(
            lambda x: (x[x['type'] == 'CALL']['gamma'].sum() * underlying_price - 
                      x[x['type'] == 'PUT']['gamma'].sum() * underlying_price)
        ).abs().sort_values(ascending=False).head(5)
        
        metrics['gamma_levels'] = list(gamma_by_strike.items())
        
        # Calculate Put/Call ratio
        metrics['pc_ratio'] = metrics['total_put_volume'] / max(metrics['total_call_volume'], 1)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

def create_summary_embed(symbol: str, underlying_price: float, metrics: dict, expiry_date):
    """Create Discord embed with summary data"""
    embed = discord.Embed(
        title=f"📊 {symbol} - {'0DTE' if symbol in INDEX_SYMBOLS else 'Weekly'} Analysis",
        description=f"**Expiry:** {expiry_date.strftime('%B %d, %Y')}\n**Current Price:** ${underlying_price:,.2f}",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    # Volume section
    total_volume = metrics['total_call_volume'] + metrics['total_put_volume']
    embed.add_field(
        name="📈 Volume",
        value=f"**Total:** {total_volume:,}\n"
              f"Calls: {metrics['total_call_volume']:,}\n"
              f"Puts: {metrics['total_put_volume']:,}\n"
              f"P/C Ratio: {metrics['pc_ratio']:.2f}",
        inline=True
    )
    
    # Open Interest section
    total_oi = metrics['total_call_oi'] + metrics['total_put_oi']
    embed.add_field(
        name="📊 Open Interest",
        value=f"**Total:** {total_oi:,}\n"
              f"Calls: {metrics['total_call_oi']:,}\n"
              f"Puts: {metrics['total_put_oi']:,}",
        inline=True
    )
    
    # Call Walls
    call_walls_text = "\n".join([f"${strike:,.2f}: {oi:,}" for strike, oi in metrics['call_walls'][:3]])
    embed.add_field(
        name="🟢 Top Call Walls",
        value=call_walls_text or "No data",
        inline=True
    )
    
    # Put Walls
    put_walls_text = "\n".join([f"${strike:,.2f}: {oi:,}" for strike, oi in metrics['put_walls'][:3]])
    embed.add_field(
        name="🔴 Top Put Walls",
        value=put_walls_text or "No data",
        inline=True
    )
    
    # Gamma Levels
    gamma_text = "\n".join([f"${strike:,.2f}: {gamma:,.0f}" for strike, gamma in metrics['gamma_levels'][:3]])
    embed.add_field(
        name="⚡ Top Gamma Strikes",
        value=gamma_text or "No data",
        inline=True
    )
    
    embed.set_footer(text=f"Data from Schwab API")
    
    return embed


class DTECommands(commands.Cog):
    """0DTE and Weekly Expiry Commands"""
    
    def __init__(self, bot):
        self.bot = bot
        
    @app_commands.command(name="0dte", description="Get 0DTE/weekly expiry analysis for any symbol")
    @app_commands.describe(symbol="Symbol: SPY, QQQ, $SPX (0DTE) or NVDA, TSLA (weekly)")
    async def dte_command(self, interaction: discord.Interaction, symbol: str):
        """0DTE or weekly expiry analysis"""
        await self._process_symbol(interaction, symbol.upper())
    
    async def _process_symbol(self, interaction: discord.Interaction, symbol: str):
        """Process symbol and send analysis"""
        await interaction.response.defer()
        
        try:
            # Get Schwab client from bot's service
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                await interaction.followup.send("❌ Schwab API not available")
                return
            
            client = self.bot.schwab_service.client
            
            # Get next expiry
            expiry_date = get_next_expiry(symbol)
            expiry_str = expiry_date.strftime("%Y-%m-%d")
            
            # Get quote
            quote_symbol = symbol if symbol.startswith('$') else symbol
            quote = client.get_quote(quote_symbol)
            
            if not quote or quote_symbol not in quote:
                await interaction.followup.send(f"❌ Could not fetch quote for {symbol}")
                return
            
            underlying_price = quote[quote_symbol]['quote']['lastPrice']
            
            # Get options chain
            chain_params = {
                'symbol': symbol,
                'contract_type': 'ALL',
                'from_date': expiry_str,
                'to_date': expiry_str
            }
            
            # For index symbols, limit strikes
            if symbol in INDEX_SYMBOLS:
                chain_params['strike_count'] = 50
            
            options_chain = client.get_options_chain(**chain_params)
            
            if not options_chain or 'callExpDateMap' not in options_chain:
                await interaction.followup.send(f"❌ No options data for {symbol} on {expiry_str}")
                return
            
            # Calculate metrics
            metrics = calculate_option_metrics(options_chain, underlying_price, expiry_date)
            
            if not metrics:
                await interaction.followup.send(f"❌ Could not calculate metrics for {symbol}")
                return
            
            # Create embed
            embed = create_summary_embed(symbol, underlying_price, metrics, expiry_date)
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Setup function called by Discord.py"""
    await bot.add_cog(DTECommands(bot))
