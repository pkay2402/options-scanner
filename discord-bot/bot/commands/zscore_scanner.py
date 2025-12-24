"""
Z-score Scanner Command for Discord Bot
Monitors watchlist stocks for z-score crossings every 15 minutes
Sends chart and alert summary when signals are detected
"""

import asyncio
import discord
import logging
from datetime import datetime, timedelta
from typing import Set, Optional, Dict, List
from pathlib import Path
import sys
import json
import pytz
import io
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to access existing code
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


async def setup(bot):
    """Setup z-score scanner commands"""
    await bot.add_cog(ZScoreScannerCommands(bot))


class ZScoreScannerCommands(discord.ext.commands.Cog):
    """Commands for monitoring watchlist z-score crossings"""
    
    def __init__(self, bot):
        self.bot = bot
        self.is_running = False
        self.scanner_task = None
        self.channel_id: Optional[int] = None
        self.scan_interval_minutes = 15
        
        # Track previous z-score crossings to avoid duplicate alerts
        self.alerted_symbols: Set[str] = set()
        
        # Load watchlist
        self.watchlist = self._load_watchlist()
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
    
    def _load_watchlist(self):
        """Load watchlist from user_preferences.json"""
        try:
            prefs_path = project_root / "user_preferences.json"
            if prefs_path.exists():
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
                    return prefs.get('watchlist', [])
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
    
    def calculate_zscore_with_quality(self, symbol: str, lookback: int = 20) -> Optional[Dict]:
        """
        Calculate z-score and quality indicators for a symbol
        Returns dict with z-score, quality, and indicators
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                return None
            
            df = hist.reset_index()[['Date', 'Close', 'Volume']]
            df.columns = ['datetime', 'close', 'volume']
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Calculate z-score
            df['ma'] = df['close'].rolling(window=lookback, min_periods=1).mean()
            df['std'] = df['close'].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0, 1e-8)
            df['zscore'] = (df['close'] - df['ma']) / df['std']
            
            # Additional quality indicators
            df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
            df['trend'] = (df['close'] / df['ma50'] - 1) * 100
            df['roc5'] = df['close'].pct_change(5) * 100
            df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest
            
            # Check for crossings
            crossed_m2 = prev['zscore'] >= -2 and latest['zscore'] < -2
            crossed_m3 = prev['zscore'] >= -3 and latest['zscore'] < -3
            crossed_p2 = prev['zscore'] <= 2 and latest['zscore'] > 2
            crossed_p3 = prev['zscore'] <= 3 and latest['zscore'] > 3
            
            # Determine signal type
            signal = None
            quality = None
            
            if crossed_m3 or crossed_m2:
                # Buy signal - check quality
                is_high_quality = (
                    latest['rsi'] < 40 and
                    latest['trend'] > -15 and
                    (latest['roc5'] > -10 or latest['vol_ratio'] > 1.5)
                )
                signal = f"-3σ" if crossed_m3 else "-2σ"
                quality = "⭐⭐⭐" if is_high_quality else "⚠️"
            elif crossed_p3 or crossed_p2:
                # Sell signal - always high quality
                signal = f"+3σ" if crossed_p3 else "+2σ"
                quality = "⭐⭐⭐"
            
            return {
                'symbol': symbol,
                'price': latest['close'],
                'zscore': latest['zscore'],
                'signal': signal,
                'quality': quality,
                'rsi': latest['rsi'],
                'trend': latest['trend'],
                'roc5': latest['roc5'],
                'vol_ratio': latest['vol_ratio'],
                'data': df,
                'has_alert': signal is not None
            }
            
        except Exception as e:
            logger.error(f"Error calculating z-score for {symbol}: {e}")
            return None
    
    def create_zscore_chart(self, data: Dict) -> io.BytesIO:
        """Create z-score chart with price and z-score"""
        try:
            df = data['data']
            symbol = data['symbol']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{symbol} Price', 'Z-Score')
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#7fdbca', width=2)
                ),
                row=1, col=1
            )
            
            # Z-score chart
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['zscore'],
                    mode='lines+markers',
                    name='Z-Score',
                    line=dict(color='#ff6b6b', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # Add threshold lines on z-score
            for level, color in [(-3, '#8b5cf6'), (-2, '#fbbf24'), (2, '#fbbf24'), (3, '#8b5cf6')]:
                fig.add_hline(y=level, line_dash='dash', line_color=color, line_width=1, row=2, col=1)
            
            # Update layout
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=2, col=1, range=[-4, 4])
            
            fig.update_layout(
                template='plotly_dark',
                height=600,
                showlegend=True,
                hovermode='x unified',
                title=f'{symbol} Z-Score Analysis'
            )
            
            # Save to bytes
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format='png', width=1200, height=600, scale=2)
            img_bytes.seek(0)
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None
    
    async def scan_watchlist(self) -> List[Dict]:
        """Scan watchlist for z-score crossings"""
        alerts = []
        
        for symbol in self.watchlist:
            try:
                result = await asyncio.to_thread(
                    self.calculate_zscore_with_quality, symbol
                )
                
                if result and result['has_alert']:
                    # Avoid duplicate alerts in same session
                    alert_key = f"{symbol}_{result['signal']}"
                    if alert_key not in self.alerted_symbols:
                        alerts.append(result)
                        self.alerted_symbols.add(alert_key)
                        logger.info(f"New z-score alert: {symbol} {result['signal']}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return alerts
    
    async def send_alert(self, alerts: List[Dict]):
        """Send z-score alert to configured channel"""
        if not self.channel_id:
            return
        
        channel = self.bot.get_channel(self.channel_id)
        if not channel:
            logger.error(f"Channel {self.channel_id} not found")
            return
        
        try:
            # Create summary embed
            embed = discord.Embed(
                title="📊 Z-Score Alerts Detected",
                description=f"Found {len(alerts)} new signal(s)",
                color=discord.Color.gold(),
                timestamp=datetime.utcnow()
            )
            
            # Add alert summary table
            for alert in alerts:
                quality_indicator = alert['quality']
                signal_type = "🟢 BUY" if alert['signal'] in ['-2σ', '-3σ'] else "🔴 SELL"
                
                value_text = (
                    f"**Signal:** {signal_type} ({alert['signal']})\n"
                    f"**Quality:** {quality_indicator}\n"
                    f"**Price:** ${alert['price']:.2f}\n"
                    f"**Z-Score:** {alert['zscore']:.2f}\n"
                    f"**RSI:** {alert['rsi']:.1f}\n"
                    f"**Trend:** {alert['trend']:.1f}%\n"
                    f"**Volume:** {alert['vol_ratio']:.2f}x"
                )
                
                embed.add_field(
                    name=f"{alert['symbol']}",
                    value=value_text,
                    inline=True
                )
            
            embed.set_footer(text="⭐⭐⭐ = High Quality | ⚠️ = Weak Signal - Wait for Confirmation")
            
            await channel.send(embed=embed)
            
            # Send charts for each alert (limit to 3 to avoid spam)
            for alert in alerts[:3]:
                try:
                    chart_bytes = await asyncio.to_thread(self.create_zscore_chart, alert)
                    if chart_bytes:
                        file = discord.File(chart_bytes, filename=f"{alert['symbol']}_zscore.png")
                        await channel.send(file=file)
                except Exception as e:
                    logger.error(f"Error sending chart for {alert['symbol']}: {e}")
            
            if len(alerts) > 3:
                await channel.send(f"_Showing charts for first 3 alerts. Total: {len(alerts)}_")
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _scanner_loop(self):
        """Main scanner loop"""
        logger.info("Z-score scanner loop started")
        
        while self.is_running:
            try:
                # Only scan during market hours
                if not self.is_market_hours():
                    logger.debug("Outside market hours, skipping scan")
                    await asyncio.sleep(60)  # Check every minute if market opens
                    continue
                
                logger.info("Starting z-score scan...")
                alerts = await self.scan_watchlist()
                
                if alerts:
                    await self.send_alert(alerts)
                    logger.info(f"Sent {len(alerts)} z-score alerts")
                else:
                    logger.info("No new z-score alerts")
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Scanner loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scanner loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    @discord.app_commands.command(
        name="setup_zscore_scanner",
        description="Set this channel for z-score alerts (scans every 15min during market hours)"
    )
    async def setup_zscore_scanner(self, interaction: discord.Interaction):
        """Set the current channel for z-score scanner alerts"""
        try:
            channel_id = interaction.channel_id
            self.channel_id = channel_id
            
            watchlist_str = ", ".join(self.watchlist[:10])
            if len(self.watchlist) > 10:
                watchlist_str += f" ... (+{len(self.watchlist) - 10} more)"
            
            embed = discord.Embed(
                title="📊 Z-Score Scanner Configured",
                description="This channel will receive alerts when watchlist stocks cross z-score thresholds",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="Watchlist",
                value=watchlist_str or "No stocks in watchlist",
                inline=False
            )
            embed.add_field(
                name="Scan Frequency",
                value="Every 15 minutes during market hours (9:30 AM - 4:00 PM ET)",
                inline=False
            )
            embed.add_field(
                name="Alert Triggers",
                value=(
                    "🟢 **Buy Signals:** -2σ, -3σ crossings\n"
                    "🔴 **Sell Signals:** +2σ, +3σ crossings"
                ),
                inline=False
            )
            embed.add_field(
                name="Signal Quality",
                value=(
                    "⭐⭐⭐ High Quality: RSI < 40, not in severe downtrend, stabilizing\n"
                    "⚠️ Weak: Wait for confirmation"
                ),
                inline=False
            )
            embed.add_field(
                name="Data Included",
                value="Chart + Price, Z-Score, RSI, Trend, Volume, Quality Rating",
                inline=False
            )
            embed.set_footer(text="Use /start_zscore_scanner to begin monitoring")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Z-score scanner channel set to: {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting z-score scanner channel: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="start_zscore_scanner",
        description="Start automated z-score monitoring"
    )
    async def start_zscore_scanner(self, interaction: discord.Interaction):
        """Start the z-score scanner service"""
        try:
            if not self.channel_id:
                await interaction.response.send_message(
                    "❌ Please setup the scanner channel first using `/setup_zscore_scanner`",
                    ephemeral=True
                )
                return
            
            if self.is_running:
                await interaction.response.send_message(
                    "⚠️ Z-score scanner is already running!",
                    ephemeral=True
                )
                return
            
            if not self.watchlist:
                await interaction.response.send_message(
                    "❌ No stocks in watchlist. Add stocks to user_preferences.json",
                    ephemeral=True
                )
                return
            
            self.is_running = True
            self.alerted_symbols.clear()  # Reset alert tracking
            self.scanner_task = asyncio.create_task(self._scanner_loop())
            
            embed = discord.Embed(
                title="✅ Z-Score Scanner Started",
                description=f"Monitoring {len(self.watchlist)} stocks for z-score crossings",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="Status",
                value="🟢 Active",
                inline=True
            )
            embed.add_field(
                name="Next Scan",
                value=f"In {self.scan_interval_minutes} minutes",
                inline=True
            )
            embed.add_field(
                name="Market Hours",
                value="9:30 AM - 4:00 PM ET (Mon-Fri)",
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Z-score scanner started")
            
        except Exception as e:
            logger.error(f"Error starting z-score scanner: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="stop_zscore_scanner",
        description="Stop automated z-score monitoring"
    )
    async def stop_zscore_scanner(self, interaction: discord.Interaction):
        """Stop the z-score scanner service"""
        try:
            if not self.is_running:
                await interaction.response.send_message(
                    "⚠️ Z-score scanner is not running",
                    ephemeral=True
                )
                return
            
            self.is_running = False
            if self.scanner_task:
                self.scanner_task.cancel()
                try:
                    await self.scanner_task
                except asyncio.CancelledError:
                    pass
            
            embed = discord.Embed(
                title="🛑 Z-Score Scanner Stopped",
                description="Z-score monitoring has been disabled",
                color=discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            
            await interaction.response.send_message(embed=embed)
            logger.info("Z-score scanner stopped")
            
        except Exception as e:
            logger.error(f"Error stopping z-score scanner: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="zscore_status",
        description="Check z-score scanner status"
    )
    async def zscore_status(self, interaction: discord.Interaction):
        """Show current z-score scanner status"""
        try:
            status = "🟢 Running" if self.is_running else "🔴 Stopped"
            channel_info = f"<#{self.channel_id}>" if self.channel_id else "Not configured"
            watchlist_str = ", ".join(self.watchlist[:5])
            if len(self.watchlist) > 5:
                watchlist_str += f" ... ({len(self.watchlist)} total)"
            
            embed = discord.Embed(
                title="📊 Z-Score Scanner Status",
                color=discord.Color.blue() if self.is_running else discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="Status", value=status, inline=True)
            embed.add_field(name="Channel", value=channel_info, inline=True)
            embed.add_field(name="Scan Interval", value=f"{self.scan_interval_minutes} min", inline=True)
            embed.add_field(name="Watchlist", value=watchlist_str or "No stocks", inline=False)
            embed.add_field(
                name="Market Hours",
                value="Active only during 9:30 AM - 4:00 PM ET",
                inline=False
            )
            
            if self.alerted_symbols:
                recent_alerts = ", ".join(list(self.alerted_symbols)[:10])
                embed.add_field(name="Recent Alerts", value=recent_alerts, inline=False)
            
            await interaction.response.send_message(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing z-score scanner status: {e}")
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
    
    @discord.app_commands.command(
        name="check_zscore",
        description="Manually check z-score for a specific symbol"
    )
    async def check_zscore(self, interaction: discord.Interaction, symbol: str):
        """Manually check z-score for a symbol"""
        try:
            await interaction.response.defer()
            
            result = await asyncio.to_thread(
                self.calculate_zscore_with_quality, symbol.upper()
            )
            
            if not result:
                await interaction.followup.send(
                    f"❌ Could not fetch data for {symbol.upper()}",
                    ephemeral=True
                )
                return
            
            # Create embed
            color = discord.Color.green() if result['zscore'] < -2 else (
                discord.Color.red() if result['zscore'] > 2 else discord.Color.blue()
            )
            
            embed = discord.Embed(
                title=f"📊 Z-Score Analysis: {result['symbol']}",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
            embed.add_field(name="Z-Score", value=f"{result['zscore']:.2f}", inline=True)
            embed.add_field(name="RSI", value=f"{result['rsi']:.1f}", inline=True)
            embed.add_field(name="Trend (50-MA)", value=f"{result['trend']:.1f}%", inline=True)
            embed.add_field(name="5-Day ROC", value=f"{result['roc5']:.1f}%", inline=True)
            embed.add_field(name="Volume Ratio", value=f"{result['vol_ratio']:.2f}x", inline=True)
            
            if result['has_alert']:
                signal_type = "🟢 BUY" if result['signal'] in ['-2σ', '-3σ'] else "🔴 SELL"
                embed.add_field(
                    name="Signal",
                    value=f"{signal_type} ({result['signal']}) {result['quality']}",
                    inline=False
                )
            else:
                embed.add_field(name="Signal", value="No crossing detected", inline=False)
            
            await interaction.followup.send(embed=embed)
            
            # Send chart
            chart_bytes = await asyncio.to_thread(self.create_zscore_chart, result)
            if chart_bytes:
                file = discord.File(chart_bytes, filename=f"{symbol}_zscore.png")
                await interaction.followup.send(file=file)
            
        except Exception as e:
            logger.error(f"Error checking z-score: {e}")
            await interaction.followup.send(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
