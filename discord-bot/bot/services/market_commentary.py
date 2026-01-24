"""
Market Commentary Service
Aggregates data from all scanners and generates AI-powered market commentary
Posts to Discord every 15 minutes during market hours
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available - install with: pip install groq")

# Try to import discord
try:
    import discord
except ImportError:
    pass


class MarketCommentaryService:
    """
    Aggregates market data from all scanners and generates AI commentary
    """
    
    def __init__(self, bot=None):
        self.bot = bot
        self.is_running = False
        self.commentary_task = None
        self.channel_id: Optional[int] = None
        self.commentary_interval_minutes = 15
        
        # Groq client
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = "llama-3.3-70b-versatile"
        self.client = None
        
        if GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq AI client initialized for market commentary")
        else:
            logger.warning("Groq not configured - market commentary will be limited")
        
        # Market hours (Eastern Time)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Config file for persistence
        project_root = Path(__file__).parent.parent.parent
        self.config_file = project_root / "market_commentary_config.json"
        self._load_config()
    
    def _load_config(self):
        """Load saved channel configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.channel_id = config.get('channel_id')
                    self.commentary_interval_minutes = config.get('interval_minutes', 15)
                    logger.info(f"Loaded market commentary config: channel_id={self.channel_id}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def _save_config(self):
        """Save current configuration"""
        try:
            config = {
                'channel_id': self.channel_id,
                'interval_minutes': self.commentary_interval_minutes,
                'is_running': self.is_running
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Market commentary config saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (ET)"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        current_time = now_et.time()
        
        # Skip weekends
        weekday = now_et.weekday()
        if weekday >= 5:
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def get_current_session_info(self) -> str:
        """Get current market session info"""
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        if not self.is_market_hours():
            return "Market Closed"
        
        # Determine session
        hour = now_et.hour
        minute = now_et.minute
        
        if hour == 9 and minute < 45:
            return "Opening Bell 🔔"
        elif hour < 10:
            return "Early Morning Session"
        elif hour < 12:
            return "Morning Session"
        elif hour < 14:
            return "Midday Session"
        elif hour < 15:
            return "Afternoon Session"
        else:
            return "Power Hour ⚡"
    
    async def collect_scanner_data(self, lookback_minutes: int = 30) -> Dict:
        """
        Collect recent data from all scanners
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'session': self.get_current_session_info(),
            'whale_flows': [],
            'tos_alerts': [],
            'zscore_signals': [],
            'etf_momentum': [],
            'market_levels': {},
            'watchlist_moves': []
        }
        
        try:
            # Get signal storage
            from .signal_storage import get_storage
            storage = get_storage()
            
            # Get all recent signals
            cutoff_hours = lookback_minutes / 60
            all_signals = storage.get_signals(days=cutoff_hours/24 + 0.01)  # Small buffer
            
            # Filter to last N minutes
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            
            for signal in all_signals:
                try:
                    signal_time = datetime.fromisoformat(signal['timestamp'].replace(' ', 'T'))
                    if signal_time < cutoff_time:
                        continue
                    
                    signal_type = signal.get('signal_type', '')
                    
                    if signal_type == 'WHALE':
                        data['whale_flows'].append({
                            'symbol': signal['symbol'],
                            'type': signal.get('signal_subtype', 'CALL'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price'),
                            'data': signal.get('data', {})
                        })
                    elif signal_type == 'TOS':
                        data['tos_alerts'].append({
                            'symbol': signal['symbol'],
                            'alert_type': signal.get('signal_subtype'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price')
                        })
                    elif signal_type == 'ZSCORE':
                        data['zscore_signals'].append({
                            'symbol': signal['symbol'],
                            'condition': signal.get('signal_subtype'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price'),
                            'data': signal.get('data', {})
                        })
                    elif signal_type == 'ETF_MOMENTUM':
                        data['etf_momentum'].append({
                            'symbol': signal['symbol'],
                            'signal': signal.get('signal_subtype'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price')
                        })
                except Exception as e:
                    logger.debug(f"Error parsing signal: {e}")
                    continue
            
            # Get market levels from Schwab if available
            if self.bot and hasattr(self.bot, 'schwab_service') and self.bot.schwab_service:
                try:
                    client = self.bot.schwab_service.client
                    if client:
                        # Get SPY, QQQ quotes
                        for symbol in ['SPY', 'QQQ', 'IWM']:
                            quote = client.get_quote(symbol)
                            if quote and symbol in quote:
                                q = quote[symbol]['quote']
                                data['market_levels'][symbol] = {
                                    'price': q.get('lastPrice'),
                                    'change': q.get('netChange'),
                                    'change_pct': q.get('netPercentChangeInDouble'),
                                    'high': q.get('highPrice'),
                                    'low': q.get('lowPrice'),
                                    'volume': q.get('totalVolume')
                                }
                except Exception as e:
                    logger.warning(f"Error getting market levels: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting scanner data: {e}")
        
        return data
    
    def generate_ai_commentary(self, data: Dict) -> str:
        """
        Generate AI-powered market commentary using Groq
        """
        if not self.client:
            return self._generate_basic_commentary(data)
        
        try:
            # Build the prompt
            prompt = self._build_commentary_prompt(data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional market analyst providing real-time market commentary for options traders. 
Your style is:
- Concise and actionable (max 300 words)
- Focus on significant moves and patterns
- Highlight unusual activity (whale flows, unusual volume)
- Use trader-friendly language
- Include relevant emojis for visual clarity
- End with a brief outlook or key levels to watch

Do NOT provide specific trading advice or recommendations. Focus on describing what's happening in the market."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            commentary = response.choices[0].message.content
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating AI commentary: {e}")
            return self._generate_basic_commentary(data)
    
    def _build_commentary_prompt(self, data: Dict) -> str:
        """Build the prompt for AI commentary"""
        parts = []
        
        parts.append(f"Market Session: {data['session']}")
        parts.append(f"Time: {datetime.now().strftime('%I:%M %p ET')}")
        parts.append("")
        
        # Market levels
        if data.get('market_levels'):
            parts.append("MARKET INDICES:")
            for symbol, levels in data['market_levels'].items():
                if levels.get('price'):
                    change_pct = levels.get('change_pct') or 0
                    direction = "↑" if change_pct >= 0 else "↓"
                    parts.append(f"  {symbol}: ${levels['price']:.2f} ({change_pct:+.2f}%) {direction}")
            parts.append("")
        
        # Whale flows
        if data.get('whale_flows'):
            parts.append(f"WHALE FLOWS ({len(data['whale_flows'])} detected):")
            
            # Group by direction
            bullish = [f for f in data['whale_flows'] if f.get('direction') == 'BULLISH']
            bearish = [f for f in data['whale_flows'] if f.get('direction') == 'BEARISH']
            
            if bullish:
                symbols = list(set([f['symbol'] for f in bullish]))[:5]
                parts.append(f"  Bullish: {', '.join(symbols)}")
            if bearish:
                symbols = list(set([f['symbol'] for f in bearish]))[:5]
                parts.append(f"  Bearish: {', '.join(symbols)}")
            
            # Highlight largest flows
            for flow in data['whale_flows'][:3]:
                flow_data = flow.get('data', {})
                notional = flow_data.get('notional', 0)
                if notional:
                    parts.append(f"  • {flow['symbol']} {flow['type']}: ${notional/1000:.0f}K notional")
            parts.append("")
        
        # TOS Alerts
        if data.get('tos_alerts'):
            parts.append(f"TOS ALERTS ({len(data['tos_alerts'])} signals):")
            for alert in data['tos_alerts'][:5]:
                parts.append(f"  • {alert['symbol']}: {alert.get('alert_type', 'Signal')} ({alert.get('direction', 'N/A')})")
            parts.append("")
        
        # Z-Score signals
        if data.get('zscore_signals'):
            parts.append(f"Z-SCORE EXTREMES ({len(data['zscore_signals'])} signals):")
            for sig in data['zscore_signals'][:5]:
                parts.append(f"  • {sig['symbol']}: {sig.get('condition', 'Signal')}")
            parts.append("")
        
        # ETF Momentum
        if data.get('etf_momentum'):
            parts.append(f"ETF MOMENTUM ({len(data['etf_momentum'])} signals):")
            for etf in data['etf_momentum'][:5]:
                parts.append(f"  • {etf['symbol']}: {etf.get('signal', 'Signal')} ({etf.get('direction', 'N/A')})")
            parts.append("")
        
        # Add request
        parts.append("Based on this data, provide a brief market commentary summarizing the key themes, unusual activity, and what traders should be watching.")
        
        return "\n".join(parts)
    
    def _generate_basic_commentary(self, data: Dict) -> str:
        """Generate basic commentary without AI"""
        parts = []
        
        parts.append(f"**{data['session']}** - {datetime.now().strftime('%I:%M %p ET')}")
        parts.append("")
        
        # Market levels
        if data.get('market_levels'):
            parts.append("📊 **Market Levels:**")
            for symbol, levels in data['market_levels'].items():
                if levels.get('price'):
                    change_pct = levels.get('change_pct') or 0
                    emoji = "🟢" if change_pct >= 0 else "🔴"
                    parts.append(f"{emoji} {symbol}: ${levels['price']:.2f} ({change_pct:+.2f}%)")
            parts.append("")
        
        # Summary counts
        whale_count = len(data.get('whale_flows', []))
        tos_count = len(data.get('tos_alerts', []))
        zscore_count = len(data.get('zscore_signals', []))
        
        if whale_count or tos_count or zscore_count:
            parts.append("📈 **Activity Summary:**")
            if whale_count:
                bullish = len([f for f in data['whale_flows'] if f.get('direction') == 'BULLISH'])
                bearish = whale_count - bullish
                parts.append(f"🐋 Whale Flows: {whale_count} (🟢{bullish} / 🔴{bearish})")
            if tos_count:
                parts.append(f"⚡ TOS Alerts: {tos_count}")
            if zscore_count:
                parts.append(f"📊 Z-Score Signals: {zscore_count}")
        
        return "\n".join(parts)
    
    async def generate_and_post_commentary(self):
        """Generate commentary and post to Discord"""
        try:
            if not self.channel_id:
                logger.warning("No channel configured for market commentary")
                return
            
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            # Collect data
            logger.info("Collecting scanner data for market commentary...")
            data = await self.collect_scanner_data(lookback_minutes=30)
            
            # Check if there's anything to report
            total_signals = (
                len(data.get('whale_flows', [])) +
                len(data.get('tos_alerts', [])) +
                len(data.get('zscore_signals', [])) +
                len(data.get('etf_momentum', []))
            )
            
            # Generate commentary
            logger.info(f"Generating AI commentary ({total_signals} signals)...")
            commentary = self.generate_ai_commentary(data)
            
            # Create embed
            session = data.get('session', 'Market Update')
            
            # Determine embed color based on market direction
            color = discord.Color.blue()
            spy_change = data.get('market_levels', {}).get('SPY', {}).get('change_pct') or 0
            if spy_change > 0.5:
                color = discord.Color.green()
            elif spy_change < -0.5:
                color = discord.Color.red()
            
            embed = discord.Embed(
                title=f"🎙️ Market Commentary - {session}",
                description=commentary,
                color=color,
                timestamp=datetime.now()
            )
            
            # Add market levels as footer
            if data.get('market_levels'):
                levels_str = " | ".join([
                    f"{sym}: ${info.get('price') or 0:.2f} ({(info.get('change_pct') or 0):+.2f}%)"
                    for sym, info in data['market_levels'].items()
                    if info.get('price')
                ])
                embed.set_footer(text=levels_str)
            
            # Add signal counts
            if total_signals > 0:
                signal_summary = []
                if data.get('whale_flows'):
                    signal_summary.append(f"🐋 {len(data['whale_flows'])}")
                if data.get('tos_alerts'):
                    signal_summary.append(f"⚡ {len(data['tos_alerts'])}")
                if data.get('zscore_signals'):
                    signal_summary.append(f"📊 {len(data['zscore_signals'])}")
                
                embed.add_field(
                    name="Signals (Last 30min)",
                    value=" | ".join(signal_summary),
                    inline=False
                )
            
            await channel.send(embed=embed)
            logger.info(f"Posted market commentary to channel {self.channel_id}")
            
        except Exception as e:
            logger.error(f"Error in generate_and_post_commentary: {e}", exc_info=True)
    
    async def _commentary_loop(self):
        """Main loop - generates commentary every N minutes during market hours"""
        logger.info("Market commentary loop started")
        
        while self.is_running:
            try:
                if self.is_market_hours():
                    await self.generate_and_post_commentary()
                else:
                    logger.debug("Market closed - skipping commentary")
                
                # Wait for next interval
                await asyncio.sleep(self.commentary_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Commentary loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in commentary loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def start(self, channel_id: int = None):
        """Start the commentary service"""
        if channel_id:
            self.channel_id = channel_id
        
        if not self.channel_id:
            logger.error("No channel ID configured")
            return False
        
        self.is_running = True
        self._save_config()
        self.commentary_task = asyncio.create_task(self._commentary_loop())
        logger.info(f"Market commentary service started for channel {self.channel_id}")
        return True
    
    async def stop(self):
        """Stop the commentary service"""
        self.is_running = False
        self._save_config()
        if self.commentary_task:
            self.commentary_task.cancel()
            self.commentary_task = None
        logger.info("Market commentary service stopped")
    
    def get_status(self) -> Dict:
        """Get current service status"""
        return {
            'is_running': self.is_running,
            'channel_id': self.channel_id,
            'interval_minutes': self.commentary_interval_minutes,
            'groq_available': self.client is not None,
            'market_open': self.is_market_hours()
        }
