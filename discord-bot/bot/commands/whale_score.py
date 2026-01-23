"""
Whale Score Command
Scans for whale activity using VALR formula from Whale Flows page
"""

import asyncio
import discord
from discord import app_commands
from discord.ext import commands
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, List
import json
import pytz
import yfinance as yf

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient
from ..services.signal_storage import get_storage

logger = logging.getLogger(__name__)

# Predefined top tech stocks
TOP_TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'AMD',
    'CRM', 'GS', 'NFLX', 'IBIT', 'COIN',
    'APP', 'PLTR', 'SNOW', 'TEAM', 'CRWD',
    'LLY', 'ABBV', 'AXP', 'JPM', 'HD',  # Pharma, Financial, Retail
    'SPY', 'QQQ','GLD','SLV','$VIX','NBIS','NOW','SNDK','SLV','GLD'
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


def analyze_price_action(symbol: str, current_price: float, direction: str, schwab_client=None) -> Dict:
    """
    Analyze 30-day price action and return key levels for whale flow context.
    Uses Schwab API if available, falls back to yfinance.
    
    Args:
        symbol: Stock ticker
        current_price: Current stock price
        direction: 'BULLISH' or 'BEARISH' based on flow type
        schwab_client: Optional Schwab client for price history
    
    Returns:
        Dict with key levels and analysis
    """
    try:
        hist = None
        
        # Try Schwab API first
        if schwab_client:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=35)  # Extra days for buffer
                
                price_history = schwab_client.get_price_history(
                    symbol=symbol,
                    period_type='month',
                    period=1,
                    frequency_type='daily',
                    frequency=1
                )
                
                if price_history and 'candles' in price_history:
                    candles = price_history['candles']
                    if candles and len(candles) >= 5:
                        hist = pd.DataFrame(candles)
                        hist['High'] = hist['high']
                        hist['Low'] = hist['low']
                        hist['Close'] = hist['close']
                        hist['Volume'] = hist['volume']
                        logger.debug(f"Got {len(hist)} candles from Schwab for {symbol}")
            except Exception as e:
                logger.warning(f"Schwab price history failed for {symbol}: {e}")
        
        # Fallback to yfinance
        if hist is None or hist.empty:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
            except Exception as e:
                logger.warning(f"yfinance failed for {symbol}: {e}")
                return None
        
        if hist is None or hist.empty or len(hist) < 5:
            return None
        
        # Calculate key metrics
        high_30d = hist['High'].max()
        low_30d = hist['Low'].min()
        avg_volume = hist['Volume'].mean()
        
        # Recent price action (last 5 days)
        recent = hist.tail(5)
        recent_high = recent['High'].max()
        recent_low = recent['Low'].min()
        
        # Moving averages
        if len(hist) >= 20:
            ma_20 = hist['Close'].tail(20).mean()
        else:
            ma_20 = hist['Close'].mean()
        
        if len(hist) >= 10:
            ma_10 = hist['Close'].tail(10).mean()
        else:
            ma_10 = hist['Close'].tail(5).mean()
        
        # Calculate support and resistance levels using pivots
        # Find swing highs and lows
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        
        # Simple pivot detection
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            # Swing high (resistance)
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                if highs[i] > current_price:
                    resistance_levels.append(highs[i])
            # Swing low (support)
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                if lows[i] < current_price:
                    support_levels.append(lows[i])
        
        # Add 30-day high/low as key levels
        if high_30d > current_price:
            resistance_levels.append(high_30d)
        if low_30d < current_price:
            support_levels.append(low_30d)
        
        # Add MA levels
        if ma_20 > current_price:
            resistance_levels.append(ma_20)
        else:
            support_levels.append(ma_20)
        
        if ma_10 > current_price:
            resistance_levels.append(ma_10)
        else:
            support_levels.append(ma_10)
        
        # Sort and deduplicate (within 0.5% considered same level)
        def dedupe_levels(levels, threshold_pct=0.5):
            if not levels:
                return []
            sorted_levels = sorted(set(levels))
            deduped = [sorted_levels[0]]
            for level in sorted_levels[1:]:
                if abs(level - deduped[-1]) / deduped[-1] > threshold_pct / 100:
                    deduped.append(level)
            return deduped
        
        resistance_levels = dedupe_levels(resistance_levels)
        support_levels = dedupe_levels(support_levels, threshold_pct=0.5)
        support_levels = sorted(support_levels, reverse=True)  # Highest support first
        
        # Calculate momentum
        if len(hist) >= 5:
            pct_change_5d = ((closes[-1] - closes[-5]) / closes[-5]) * 100
        else:
            pct_change_5d = 0
        
        # Trend analysis
        if current_price > ma_20 and current_price > ma_10:
            trend = "📈 Uptrend"
        elif current_price < ma_20 and current_price < ma_10:
            trend = "📉 Downtrend"
        else:
            trend = "↔️ Sideways"
        
        # Build analysis result
        result = {
            'trend': trend,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'high_30d': high_30d,
            'low_30d': low_30d,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'pct_change_5d': pct_change_5d,
            'resistance_levels': resistance_levels[:3],  # Top 3
            'support_levels': support_levels[:3],  # Top 3
        }
        
        # Generate target levels based on direction
        if direction == 'BULLISH':
            # For bullish flow, show upside targets
            targets = [r for r in resistance_levels if r > current_price][:3]
            if not targets and high_30d > current_price:
                targets = [high_30d]
            result['targets'] = targets
            result['target_label'] = "🎯 Upside Targets"
        else:
            # For bearish flow, show downside targets
            targets = [s for s in support_levels if s < current_price][:3]
            if not targets and low_30d < current_price:
                targets = [low_30d]
            result['targets'] = targets
            result['target_label'] = "🎯 Downside Targets"
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing price action for {symbol}: {e}")
        return None


def format_price_analysis(symbol: str, analysis: Dict, current_price: float) -> str:
    """Format price analysis into a readable string for Discord embed"""
    if not analysis:
        return "Price analysis unavailable"
    
    lines = []
    
    # Trend and momentum
    lines.append(f"{analysis['trend']} | 5D: {analysis['pct_change_5d']:+.1f}%")
    
    # Moving averages
    ma_10_dist = ((analysis['ma_10'] - current_price) / current_price) * 100
    ma_20_dist = ((analysis['ma_20'] - current_price) / current_price) * 100
    lines.append(f"MA10: ${analysis['ma_10']:.2f} ({ma_10_dist:+.1f}%)")
    lines.append(f"MA20: ${analysis['ma_20']:.2f} ({ma_20_dist:+.1f}%)")
    
    # Targets
    if analysis.get('targets'):
        target_strs = []
        for t in analysis['targets'][:2]:  # Show max 2 targets
            dist = ((t - current_price) / current_price) * 100
            target_strs.append(f"${t:.2f} ({dist:+.1f}%)")
        lines.append(f"{analysis['target_label']}: {', '.join(target_strs)}")
    
    return "\n".join(lines)


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
                                'mark': contract.get('mark', 0),
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
                                'mark': contract.get('mark', 0),
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
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        self.scan_interval_minutes = 10  # Scan every 10 minutes
        self.min_score_threshold = 50
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Track alerted flows to avoid duplicates
        self.alerted_today: Set[str] = set()
        
        # Config file for persistence
        self.config_file = project_root / "discord-bot" / "whale_score_config.json"
        self._load_config()
    
    def _load_config(self):
        """Load saved channel and scanner state"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.channel_id = config.get('channel_id')
                    self.min_score_threshold = config.get('min_score_threshold', 50)
                    was_running = config.get('is_running', False)
                    
                    if self.channel_id and was_running:
                        logger.info(f"Loaded whale score config: channel_id={self.channel_id}, auto-start enabled")
                    elif self.channel_id:
                        logger.info(f"Loaded whale score config: channel_id={self.channel_id}, was stopped")
        except Exception as e:
            logger.error(f"Error loading whale score config: {e}")
    
    def _save_config(self):
        """Save current channel and scanner state"""
        try:
            config = {
                'channel_id': self.channel_id,
                'is_running': self.is_running,
                'scan_interval_minutes': self.scan_interval_minutes,
                'min_score_threshold': self.min_score_threshold
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Whale score config saved: {config}")
        except Exception as e:
            logger.error(f"Error saving whale score config: {e}")
    
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
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Auto-start scanner if it was previously running"""
        if self.channel_id and not self.is_running:
            try:
                # Check if scanner was running before restart
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                        was_running = config.get('is_running', False)
                        
                        if was_running:
                            logger.info(f"Auto-starting whale score scanner for channel {self.channel_id}")
                            self.is_running = True
                            self.scanner_task = asyncio.create_task(self._scanner_loop())
                            
                            # Send notification to channel
                            channel = self.bot.get_channel(self.channel_id)
                            if channel:
                                embed = discord.Embed(
                                    title="🐋 Whale Scanner Auto-Resumed",
                                    description="Monitoring resumed after bot restart",
                                    color=discord.Color.green()
                                )
                                await channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Error auto-starting whale score scanner: {e}")
    
    async def _scan_and_alert(self):
        """Scan for whale flows and send alerts for new detections"""
        try:
            # Get Schwab client
            if not self.bot.schwab_service or not self.bot.schwab_service.client:
                logger.error("Schwab API not available for whale scanning")
                return
            
            client = self.bot.schwab_service.client
            expiry_dates = get_next_three_fridays()
            
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            all_flows = []
            
            # Scan each stock
            for symbol in TOP_TECH_STOCKS:
                flows = scan_stock_whale_flows(client, symbol, expiry_dates, self.min_score_threshold)
                if flows:
                    all_flows.extend(flows)
            
            if not all_flows:
                return
            
            # Filter for new flows not alerted today
            eastern = pytz.timezone('US/Eastern')
            today_str = datetime.now(eastern).strftime("%Y-%m-%d")
            
            new_flows = []
            for flow in all_flows:
                alert_key = f"{flow['symbol']}_{flow['type']}_{flow['strike']}_{flow['expiry']}_{today_str}"
                if alert_key not in self.alerted_today:
                    new_flows.append(flow)
                    self.alerted_today.add(alert_key)
            
            if not new_flows:
                return
            
            # Store signals in database
            try:
                storage = get_storage()
                for flow in new_flows:
                    # Determine direction based on call/put
                    direction = 'BULLISH' if flow['type'] == 'CALL' else 'BEARISH'
                    
                    # Store the signal
                    storage.store_signal(
                        symbol=flow['symbol'],
                        signal_type='WHALE',
                        signal_subtype=flow['type'],  # CALL or PUT
                        direction=direction,
                        price=flow['underlying_price'],
                        data={
                            'strike': flow['strike'],
                            'expiry': flow['expiry'],
                            'volume': flow['volume'],
                            'oi': flow['oi'],
                            'mark': flow['mark'],
                            'iv': flow['iv'],
                            'whale_score': flow['whale_score'],
                            'notional': flow['volume'] * flow['mark'] * 100
                        }
                    )
                logger.info(f"Stored {len(new_flows)} whale flow signals in database")
            except Exception as e:
                logger.error(f"Error storing whale flow signals: {e}")
            
            # Sort by whale score and take top 10
            df = pd.DataFrame(new_flows)
            df = df.sort_values('whale_score', ascending=False).head(10)
            
            # Calculate insights
            all_flows_df = pd.DataFrame(new_flows)
            
            # 1. Total Notional Value
            all_flows_df['notional'] = all_flows_df['volume'] * all_flows_df['mark'] * 100
            total_notional = all_flows_df['notional'].sum()
            
            # 2. Expiry Concentration
            expiry_counts = all_flows_df['expiry'].value_counts()
            week_labels = ['Week 1', 'Week 2', 'Week 3']
            expiry_breakdown = ""
            for i, exp_date in enumerate(expiry_dates):
                exp_str = exp_date.strftime('%Y-%m-%d')
                count = expiry_counts.get(exp_str, 0)
                expiry_breakdown += f"**{week_labels[i]}:** {count} flows\n"
            
            # 3. Call/Put Bias per Symbol
            symbol_bias = []
            for symbol in all_flows_df['symbol'].unique():
                symbol_flows = all_flows_df[all_flows_df['symbol'] == symbol]
                calls = len(symbol_flows[symbol_flows['type'] == 'CALL'])
                puts = len(symbol_flows[symbol_flows['type'] == 'PUT'])
                total = calls + puts
                call_pct = (calls / total * 100) if total > 0 else 0
                
                if call_pct >= 70:
                    bias = f"🟢 {symbol}: {call_pct:.0f}% Calls (Bullish)"
                elif call_pct <= 30:
                    bias = f"🔴 {symbol}: {100-call_pct:.0f}% Puts (Bearish)"
                else:
                    bias = f"⚪ {symbol}: {call_pct:.0f}% / {100-call_pct:.0f}% (Mixed)"
                symbol_bias.append(bias)
            
            # Send alert
            expiry_display = ", ".join([exp.strftime('%b %d') for exp in expiry_dates])
            embed = discord.Embed(
                title="🐋 New Whale Flows Detected",
                description=f"**Expiries:** {expiry_display}\n**New Flows:** {len(df)}\n**Total Notional:** ${total_notional/1e6:.1f}M",
                color=discord.Color.purple(),
                timestamp=datetime.now()
            )
            
            # Add expiry concentration
            embed.add_field(
                name="📅 Expiry Concentration",
                value=expiry_breakdown.strip(),
                inline=True
            )
            
            # Add call/put bias
            if symbol_bias:
                bias_text = "\n".join(symbol_bias[:5])  # Top 5 symbols
                embed.add_field(
                    name="📊 Directional Bias",
                    value=bias_text,
                    inline=True
                )
            
            embed.add_field(name="\u200b", value="\u200b", inline=False)  # Line break
            
            # Add flows with price analysis
            for i, row in df.iterrows():
                emoji = "🟢" if row['type'] == 'CALL' else "🔴"
                distance = ((row['strike'] - row['underlying_price']) / row['underlying_price'] * 100)
                exp_date = datetime.strptime(row['expiry'], '%Y-%m-%d').strftime('%m/%d')
                direction = 'BULLISH' if row['type'] == 'CALL' else 'BEARISH'
                
                notional = row['volume'] * row['mark'] * 100
                
                # Get price analysis for this symbol (pass Schwab client)
                analysis = analyze_price_action(row['symbol'], row['underlying_price'], direction, client)
                
                # Build field value
                field_value = f"**Score:** {row['whale_score']:,.0f} | **Notional:** ${notional/1e3:.0f}K\n"
                field_value += f"Vol: {row['volume']:,.0f} | OI: {row['oi']:,.0f} | IV: {row['iv']*100:.0f}%"
                
                # Add price analysis if available
                if analysis:
                    field_value += f"\n{analysis['trend']} | 5D: {analysis['pct_change_5d']:+.1f}%"
                    if analysis.get('targets'):
                        targets_str = ", ".join([f"${t:.2f}" for t in analysis['targets'][:2]])
                        field_value += f"\n{analysis['target_label']}: {targets_str}"
                
                embed.add_field(
                    name=f"{emoji} {row['symbol']} {row['type']} ${row['strike']:.2f} ({distance:+.1f}%) [{exp_date}]",
                    value=field_value,
                    inline=True
                )
            
            await channel.send(embed=embed)
            logger.info(f"Sent whale flow alert with {len(df)} new flows")
            
        except Exception as e:
            logger.error(f"Error in whale scanner: {e}", exc_info=True)
    
    async def _scanner_loop(self):
        """Main loop - scans during market hours"""
        logger.info("Whale score scanner loop started")
        
        while self.is_running:
            try:
                # Check if market hours
                if self.is_market_hours():
                    logger.info("Running whale score scan...")
                    await self._scan_and_alert()
                else:
                    # Clear alert cache at start of new day
                    eastern = pytz.timezone('US/Eastern')
                    now_et = datetime.now(eastern)
                    
                    if now_et.hour == 0 and now_et.minute < self.scan_interval_minutes:
                        self.alerted_today.clear()
                        logger.info("New day - cleared whale alerts cache")
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Whale scanner task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in whale scanner loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    @app_commands.command(name="setup_whale_scanner", description="Setup automated whale flow scanner for this channel")
    @app_commands.describe(min_score="Minimum whale score threshold (default: 50)")
    async def setup_whale_scanner(self, interaction: discord.Interaction, min_score: int = 50):
        """Configure automated whale flow scanner"""
        self.channel_id = interaction.channel_id
        self.min_score_threshold = min_score
        self._save_config()
        await interaction.response.send_message(
            f"✅ Whale flow scanner configured for this channel!\n"
            f"Min Score: {min_score:,}\n"
            f"Use `/start_whale_scanner` to begin monitoring.",
            ephemeral=True
        )
        logger.info(f"Whale scanner configured for channel {self.channel_id} with min_score={min_score}")
    
    @app_commands.command(name="start_whale_scanner", description="Start automated whale flow monitoring")
    async def start_whale_scanner(self, interaction: discord.Interaction):
        """Start the automated whale flow scanner"""
        if not self.channel_id:
            await interaction.response.send_message(
                "❌ Please run `/setup_whale_scanner` first to configure the channel.",
                ephemeral=True
            )
            return
        
        if self.is_running:
            await interaction.response.send_message(
                "⚠️ Whale scanner is already running!",
                ephemeral=True
            )
            return
        
        self.is_running = True
        self._save_config()
        self.scanner_task = asyncio.create_task(self._scanner_loop())
        
        embed = discord.Embed(
            title="🐋 Whale Flow Scanner Started",
            description="Monitoring whale activity during market hours",
            color=discord.Color.purple()
        )
        embed.add_field(name="Scan Interval", value=f"{self.scan_interval_minutes} minutes", inline=True)
        embed.add_field(name="Min Score", value=f"{self.min_score_threshold:,}", inline=True)
        embed.add_field(name="Stocks Monitored", value=f"{len(TOP_TECH_STOCKS)} stocks", inline=True)
        embed.set_footer(text="New whale flows will be alerted automatically")
        
        await interaction.response.send_message(embed=embed)
        logger.info("Whale scanner started")
    
    @app_commands.command(name="stop_whale_scanner", description="Stop automated whale flow monitoring")
    async def stop_whale_scanner(self, interaction: discord.Interaction):
        """Stop the automated whale flow scanner"""
        if not self.is_running:
            await interaction.response.send_message(
                "⚠️ Whale scanner is not running.",
                ephemeral=True
            )
            return
        
        self.is_running = False
        self._save_config()
        if self.scanner_task:
            self.scanner_task.cancel()
            self.scanner_task = None
        
        await interaction.response.send_message(
            "✅ Whale scanner stopped.",
            ephemeral=True
        )
        logger.info("Whale scanner stopped")
    
    @app_commands.command(name="whale_scanner_status", description="Check whale scanner status")
    async def whale_scanner_status(self, interaction: discord.Interaction):
        """Show current status of whale scanner"""
        embed = discord.Embed(
            title="📊 Whale Scanner Status",
            color=discord.Color.blue()
        )
        
        # Running status
        status_emoji = "🟢" if self.is_running else "🔴"
        status_text = "Running" if self.is_running else "Stopped"
        embed.add_field(name="Status", value=f"{status_emoji} {status_text}", inline=True)
        
        # Channel
        if self.channel_id:
            embed.add_field(name="Channel", value=f"<#{self.channel_id}>", inline=True)
        else:
            embed.add_field(name="Channel", value="Not configured", inline=True)
        
        # Market status
        market_status = "🟢 Open" if self.is_market_hours() else "🔴 Closed"
        embed.add_field(name="Market", value=market_status, inline=True)
        
        # Config
        embed.add_field(name="Min Score", value=f"{self.min_score_threshold:,}", inline=True)
        embed.add_field(name="Scan Interval", value=f"{self.scan_interval_minutes} min", inline=True)
        embed.add_field(name="Stocks", value=f"{len(TOP_TECH_STOCKS)}", inline=True)
        
        # Alert count today
        embed.add_field(name="Alerts Today", value=f"{len(self.alerted_today)}", inline=True)
        
        await interaction.response.send_message(embed=embed)
    
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
            
            # Get price analysis for the stock
            # Determine overall direction from top flows
            top_flows = df.head(5)
            calls = len(top_flows[top_flows['type'] == 'CALL'])
            puts = len(top_flows[top_flows['type'] == 'PUT'])
            overall_direction = 'BULLISH' if calls >= puts else 'BEARISH'
            
            price_analysis = analyze_price_action(symbol, flows[0]['underlying_price'], overall_direction, client)
            
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
            
            # Add price analysis section
            if price_analysis:
                analysis_text = f"{price_analysis['trend']} | 5D: {price_analysis['pct_change_5d']:+.1f}%\n"
                analysis_text += f"MA10: ${price_analysis['ma_10']:.2f} | MA20: ${price_analysis['ma_20']:.2f}\n"
                analysis_text += f"30D Range: ${price_analysis['low_30d']:.2f} - ${price_analysis['high_30d']:.2f}\n"
                
                if price_analysis.get('targets'):
                    targets_str = ", ".join([f"${t:.2f}" for t in price_analysis['targets'][:3]])
                    analysis_text += f"{price_analysis['target_label']}: {targets_str}"
                
                embed.add_field(
                    name="📊 Price Analysis (30D)",
                    value=analysis_text,
                    inline=False
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
