"""
Opening Move Alert Command
Sends top 3 trade opportunities every 15 minutes during market hours
Analyzes watchlist using market dynamics and big trades detection
"""

import asyncio
import discord
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys
import json
import pytz

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

try:
    from src.utils.droplet_api import DropletAPI
    HAS_DROPLET_API = True
except ImportError:
    HAS_DROPLET_API = False

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup opening move alert command"""
    await bot.add_cog(OpeningMoveCommands(bot))


class OpeningMoveCommands(discord.ext.commands.Cog):
    """Commands for 15-minute opening move alerts"""
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        self.scan_interval_minutes = 15
        
        # Load watchlist
        self.watchlist = self._load_watchlist()
        self.watchlist_data_map = {}  # Store full watchlist data with price changes
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Cache for performance
        self.last_scan_time = None
        self.last_analysis = None
    
    def _load_watchlist(self):
        """Load watchlist from Droplet API or fallback to user_preferences.json"""
        # Try Droplet API first (same as Trading Dashboard)
        if HAS_DROPLET_API:
            try:
                api = DropletAPI()
                watchlist_data = api.get_watchlist(order_by='daily_change_pct', limit=150)
                if watchlist_data:
                    symbols = [item['symbol'] for item in watchlist_data]
                    # Store full data for accessing daily_change_pct later
                    self.watchlist_data_map = {item['symbol']: item for item in watchlist_data}
                    logger.info(f"Loaded {len(symbols)} symbols from Droplet API watchlist")
                    return symbols
            except Exception as e:
                logger.warning(f"Failed to load from Droplet API: {e}, falling back to user_preferences.json")
        
        # Fallback to user_preferences.json
        try:
            prefs_path = project_root / "user_preferences.json"
            if prefs_path.exists():
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
                    symbols = prefs.get('watchlist', [])
                    self.watchlist_data_map = {}  # No pre-calculated changes
                    logger.info(f"Loaded {len(symbols)} symbols from user_preferences.json")
                    return symbols
            return []
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return []
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (ET)"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        current_time = now_et.time()
        
        # Skip weekends
        weekday = now_et.weekday()
        if weekday >= 5:
            return False
        
        return self.market_open <= current_time < self.market_close
    
    def _analyze_volume_walls(self, options_data, underlying_price):
        """Analyze call/put walls and flip level"""
        try:
            import numpy as np
            
            # Collect data from all strikes
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
            
            # Get all strikes
            all_strikes = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
            
            if not all_strikes:
                return None, None, None, None
            
            # Calculate net GEX for each strike
            gex_by_strike = {}
            for strike in all_strikes:
                call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price
                put_gex = put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * -1
                gex_by_strike[strike] = call_gex + put_gex
            
            # Find max GEX strike (flip level)
            max_gex_strike = max(gex_by_strike.items(), key=lambda x: abs(x[1]))[0] if gex_by_strike else None
            max_gex_value = gex_by_strike.get(max_gex_strike, 0) if max_gex_strike else 0
            
            # Find call wall (highest call OI above price)
            above_strikes = [s for s in all_strikes if s > underlying_price]
            call_wall = max([(s, call_oi.get(s, 0)) for s in above_strikes], 
                          key=lambda x: x[1])[0] if above_strikes else None
            
            # Find put wall (highest put OI below price)
            below_strikes = [s for s in all_strikes if s < underlying_price]
            put_wall = max([(s, put_oi.get(s, 0)) for s in below_strikes], 
                         key=lambda x: x[1])[0] if below_strikes else None
            
            return call_wall, put_wall, max_gex_strike, max_gex_value
            
        except Exception as e:
            logger.error(f"Error analyzing walls: {e}")
            return None, None, None, None
    
    def _calculate_momentum_score(self, symbol: str, quote_data: dict, options_data: dict) -> Dict:
        """Calculate momentum and opportunity score for a symbol"""
        try:
            price = quote_data.get('lastPrice', 0)
            volume = quote_data.get('totalVolume', 0)
            prev_close = quote_data.get('closePrice', price)
            
            # If quote price is zero (market closed), get from options chain
            if price == 0 and options_data and 'underlyingPrice' in options_data:
                price = options_data.get('underlyingPrice', 0)
                # Use price as prev_close if we don't have close data
                if prev_close == 0:
                    prev_close = price
            
            # Safety checks
            if price == 0:
                return None
            
            # Use Droplet API's daily_change_pct if available (more accurate)
            if symbol in self.watchlist_data_map and 'daily_change_pct' in self.watchlist_data_map[symbol]:
                change_pct = self.watchlist_data_map[symbol]['daily_change_pct']
            else:
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0
            
            # Analyze options activity
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            if 'callExpDateMap' in options_data:
                for exp_date, strikes in options_data['callExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            total_call_volume += contracts[0].get('totalVolume', 0)
                            total_call_oi += contracts[0].get('openInterest', 0)
            
            if 'putExpDateMap' in options_data:
                for exp_date, strikes in options_data['putExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            total_put_volume += contracts[0].get('totalVolume', 0)
                            total_put_oi += contracts[0].get('openInterest', 0)
            
            # Calculate put/call ratio (handle zero volumes)
            if total_call_volume == 0 and total_put_volume == 0:
                # No options activity - skip this symbol
                return None
            pcr = total_put_volume / total_call_volume if total_call_volume > 0 else 999
            
            # Options volume vs stock volume ratio
            options_dollar_volume = (total_call_volume + total_put_volume) * price * 100
            stock_dollar_volume = volume * price
            vol_ratio = options_dollar_volume / stock_dollar_volume if stock_dollar_volume > 0 else 0
            
            # Get walls
            call_wall, put_wall, max_gex, gex_value = self._analyze_volume_walls(options_data, price)
            
            # Calculate opportunity score (0-100)
            score = 0
            reasons = []
            
            # Factor 1: Price momentum (0-25 points)
            if abs(change_pct) > 2:
                momentum_score = min(abs(change_pct) * 5, 25)
                score += momentum_score
                reasons.append(f"Strong momentum: {change_pct:+.1f}%")
            
            # Factor 2: Options activity (0-25 points)
            if vol_ratio > 1:
                activity_score = min(vol_ratio * 10, 25)
                score += activity_score
                reasons.append(f"High options activity: {vol_ratio:.1f}x stock volume")
            
            # Factor 3: Put/Call imbalance (0-25 points)
            if pcr < 0.7:  # Bullish
                score += 20
                reasons.append(f"Bullish flow: PCR {pcr:.2f}")
            elif pcr > 1.3:  # Bearish
                score += 20
                reasons.append(f"Bearish flow: PCR {pcr:.2f}")
            
            # Factor 4: Near walls/flip level (0-25 points)
            if call_wall and price > 0 and abs(price - call_wall) / price < 0.03:
                score += 15
                reasons.append(f"Near call wall at ${call_wall:.2f}")
            if put_wall and price > 0 and abs(price - put_wall) / price < 0.03:
                score += 15
                reasons.append(f"Near put wall at ${put_wall:.2f}")
            
            # Determine direction based on price momentum AND options flow
            if abs(change_pct) > 1:  # Strong price movement
                # Price direction takes priority
                direction = "BULLISH" if change_pct > 0 else "BEARISH"
                # Add flow confirmation or divergence
                if (change_pct > 0 and pcr > 1.3):
                    direction = "BULLISH (⚠️ hedging)"  # Up move but heavy puts
                elif (change_pct < 0 and pcr < 0.7):
                    direction = "BEARISH (⚠️ bottom fishing)"  # Down move but heavy calls
            else:
                # No strong price movement, use options flow
                direction = "BULLISH" if pcr < 1 else "BEARISH"
            
            return {
                'symbol': symbol,
                'price': price,
                'change_pct': change_pct,
                'score': min(score, 100),
                'direction': direction,
                'pcr': pcr,
                'call_wall': call_wall,
                'put_wall': put_wall,
                'max_gex': max_gex,
                'gex_value': gex_value,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {e}")
            return None
    
    async def _scan_top_opportunities(self) -> List[Dict]:
        """Scan watchlist and find top 3 opportunities"""
        try:
            client = SchwabClient(interactive=False)
            opportunities = []
            
            # Smart selection: Watchlist is pre-sorted by daily_change_pct from Droplet API
            # Take top 20 biggest movers (up or down) + 10 from middle for diversity
            scan_count = 30
            
            if len(self.watchlist) > scan_count:
                # Top 15 biggest gainers + Top 10 biggest losers + 5 from middle
                top_gainers = self.watchlist[:15]  # Already sorted by change %
                bottom_losers = self.watchlist[-10:]  # Biggest losers at end
                middle = self.watchlist[len(self.watchlist)//2 - 2:len(self.watchlist)//2 + 3]  # 5 from middle
                symbols_to_scan = top_gainers + bottom_losers + middle
            else:
                symbols_to_scan = self.watchlist[:scan_count]
            
            logger.info(f"Smart scan: {len(symbols_to_scan)} candidates from {len(self.watchlist)} watchlist")
            
            for symbol in symbols_to_scan:
                try:
                    # Get quote
                    quote_data = client.get_quote(symbol)
                    if not quote_data or symbol not in quote_data:
                        continue
                    
                    # Get options data
                    options_data = client.get_options_chain(
                        symbol=symbol,
                        contract_type='ALL',
                        strike_count=20
                    )
                    
                    if not options_data:
                        continue
                    
                    # Calculate opportunity score
                    analysis = self._calculate_momentum_score(symbol, quote_data[symbol], options_data)
                    
                    if analysis and analysis['score'] > 30:  # Minimum threshold
                        opportunities.append(analysis)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by score and return top 5
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities[:5]
            
        except Exception as e:
            logger.error(f"Error in scan_top_opportunities: {e}")
            return []
    
    def _create_opportunity_embed(self, opportunities: List[Dict], scan_time: datetime) -> discord.Embed:
        """Create Discord embed for top opportunities"""
        eastern = pytz.timezone('US/Eastern')
        scan_time_et = scan_time.astimezone(eastern)
        
        embed = discord.Embed(
            title="📊 Opening Move Report",
            description=f"Top 5 Trade Opportunities • {scan_time_et.strftime('%I:%M %p ET')}",
            color=discord.Color.blue(),
            timestamp=scan_time
        )
        
        if not opportunities:
            embed.add_field(
                name="No Opportunities",
                value="No high-probability setups found at this time.",
                inline=False
            )
            return embed
        
        for i, opp in enumerate(opportunities, 1):
            # Build title
            emoji = "🟢" if opp['direction'] == "BULLISH" else "🔴"
            title = f"{emoji} #{i}: {opp['symbol']} - {opp['direction']}"
            
            # Build description
            lines = [
                f"**Price:** ${opp['price']:.2f} ({opp['change_pct']:+.1f}%)",
                f"**Score:** {opp['score']:.0f}/100",
                f"**Put/Call Ratio:** {opp['pcr']:.2f}",
                ""
            ]
            
            # Add walls info
            if opp['call_wall']:
                lines.append(f"📈 **Call Wall:** ${opp['call_wall']:.2f}")
            if opp['put_wall']:
                lines.append(f"📉 **Put Wall:** ${opp['put_wall']:.2f}")
            if opp['max_gex']:
                lines.append(f"⚡ **Flip Level:** ${opp['max_gex']:.2f}")
            
            lines.append("")
            
            # Add reasons
            if opp['reasons']:
                lines.append("**Why This Setup:**")
                for reason in opp['reasons'][:3]:
                    lines.append(f"• {reason}")
            
            embed.add_field(
                name=title,
                value="\n".join(lines),
                inline=False
            )
        
        embed.set_footer(text="Next scan in 15 minutes")
        
        return embed
    
    async def _scanner_loop(self):
        """Main scanner loop - runs every 15 minutes during market hours"""
        logger.info("Opening Move scanner started")
        
        while self.is_running:
            try:
                # Check if market is open
                if not self.is_market_hours():
                    logger.info("Market closed, scanner sleeping...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # Check if channel is set
                if not self.channel_id:
                    logger.warning("No channel set for Opening Move alerts")
                    await asyncio.sleep(60)
                    continue
                
                # Get channel
                channel = self.bot.get_channel(self.channel_id)
                if not channel:
                    logger.error(f"Channel {self.channel_id} not found")
                    await asyncio.sleep(60)
                    continue
                
                # Run scan
                logger.info("Running Opening Move scan...")
                scan_time = datetime.now(pytz.UTC)
                opportunities = await self._scan_top_opportunities()
                
                # Create and send embed
                embed = self._create_opportunity_embed(opportunities, scan_time)
                await channel.send(embed=embed)
                
                logger.info(f"Sent Opening Move alert with {len(opportunities)} opportunities")
                
                # Cache results
                self.last_scan_time = scan_time
                self.last_analysis = opportunities
                
                # Wait 15 minutes
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in scanner loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    @discord.app_commands.command(
        name="setup_opening_move",
        description="Set this channel for Opening Move alerts (every 15 min during market hours)"
    )
    async def setup_opening_move(self, interaction: discord.Interaction):
        """Set the current channel for opening move alerts"""
        try:
            self.channel_id = interaction.channel_id
            
            watchlist_preview = ", ".join(self.watchlist[:10])
            if len(self.watchlist) > 10:
                watchlist_preview += f" ... +{len(self.watchlist) - 10} more"
            
            embed = discord.Embed(
                title="📊 Opening Move Report Configured",
                description="This channel will receive top 5 trade opportunities every 15 minutes during market hours",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="📋 Watchlist",
                value=watchlist_preview or "No watchlist configured",
                inline=False
            )
            embed.add_field(
                name="🎯 Smart Scanning",
                value="Analyzes top 15 gainers + top 10 losers + 5 mid-cap for optimal coverage (~30 stocks in 15 seconds)",
                inline=False
            )
            embed.add_field(
                name="⏰ Schedule",
                value="Every 15 minutes from 9:30 AM - 4:00 PM ET (weekdays only)",
                inline=False
            )
            embed.add_field(
                name="📈 What's Analyzed",
                value="• Price momentum & volume\n• Options flow (puts vs calls)\n• Call/Put walls\n• Gamma flip levels\n• Opportunity score (0-100)",
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Opening Move alerts configured for channel {self.channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting up opening move: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error setting up Opening Move alerts: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="start_opening_move",
        description="Start the Opening Move scanner (must setup channel first)"
    )
    async def start_opening_move(self, interaction: discord.Interaction):
        """Start the opening move scanner"""
        try:
            if self.is_running:
                await interaction.response.send_message(
                    "⚠️ Opening Move scanner is already running!",
                    ephemeral=True
                )
                return
            
            if not self.channel_id:
                await interaction.response.send_message(
                    "❌ Please setup a channel first using `/setup_opening_move`",
                    ephemeral=True
                )
                return
            
            self.is_running = True
            self.scanner_task = asyncio.create_task(self._scanner_loop())
            
            await interaction.response.send_message(
                "✅ Opening Move scanner started! Will scan every 15 minutes during market hours.",
                ephemeral=True
            )
            logger.info("Opening Move scanner started by user")
            
        except Exception as e:
            logger.error(f"Error starting scanner: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error starting scanner: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="stop_opening_move",
        description="Stop the Opening Move scanner"
    )
    async def stop_opening_move(self, interaction: discord.Interaction):
        """Stop the opening move scanner"""
        try:
            if not self.is_running:
                await interaction.response.send_message(
                    "⚠️ Opening Move scanner is not running!",
                    ephemeral=True
                )
                return
            
            self.is_running = False
            if self.scanner_task:
                self.scanner_task.cancel()
                self.scanner_task = None
            
            await interaction.response.send_message(
                "✅ Opening Move scanner stopped.",
                ephemeral=True
            )
            logger.info("Opening Move scanner stopped by user")
            
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}", exc_info=True)
            await interaction.response.send_message(
                f"❌ Error stopping scanner: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="opening_move_now",
        description="Run Opening Move analysis immediately - shows top 5 plays (manual test)"
    )
    async def opening_move_now(self, interaction: discord.Interaction):
        """Run opening move analysis immediately - top 5 plays"""
        try:
            await interaction.response.defer(thinking=True)
            
            logger.info("Running manual Opening Move scan...")
            scan_time = datetime.now(pytz.UTC)
            opportunities = await self._scan_top_opportunities()
            
            # Create and send embed
            embed = self._create_opportunity_embed(opportunities, scan_time)
            
            await interaction.followup.send(embed=embed)
            logger.info(f"Manual Opening Move scan completed with {len(opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"Error in manual scan: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ Error running scan: {str(e)}"
            )
