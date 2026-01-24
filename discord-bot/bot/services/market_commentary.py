"""
Market Commentary Service
Aggregates data from all scanners and generates AI-powered market commentary
Posts to Discord every 15 minutes during market hours
"""

import asyncio
import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)

# Configuration
DROPLET_API_URL = os.environ.get("DROPLET_API_URL", "http://138.197.210.166:8000")

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
            'watchlist_movers': {'bullish': [], 'bearish': []},  # NEW: Top movers from watchlist
            'watchlist_moves': [],
            'options_activity': []  # New: Options data for signaled stocks
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
            
            # Track unique symbols from non-whale scanners for options lookup
            symbols_to_scan = set()
            
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
                        symbols_to_scan.add(signal['symbol'])
                    elif signal_type == 'ZSCORE':
                        data['zscore_signals'].append({
                            'symbol': signal['symbol'],
                            'condition': signal.get('signal_subtype'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price'),
                            'data': signal.get('data', {})
                        })
                        symbols_to_scan.add(signal['symbol'])
                    elif signal_type == 'ETF_MOMENTUM':
                        data['etf_momentum'].append({
                            'symbol': signal['symbol'],
                            'signal': signal.get('signal_subtype'),
                            'direction': signal.get('direction'),
                            'price': signal.get('price')
                        })
                        # Only scan non-leveraged ETFs for options
                        if not any(x in signal['symbol'] for x in ['3X', '2X', 'TQQQ', 'SQQQ', 'UVXY', 'SVXY']):
                            symbols_to_scan.add(signal['symbol'])
                except Exception as e:
                    logger.debug(f"Error parsing signal: {e}")
                    continue
            
            # Get market levels from Schwab if available
            client = None
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
            
            # NEW: Fetch watchlist top movers from droplet API
            try:
                response = requests.get(
                    f"{DROPLET_API_URL}/api/watchlist?order_by=daily_change_pct&limit=100",
                    timeout=10
                )
                if response.status_code == 200:
                    watchlist_data = response.json().get('data', [])
                    # Top 3 bullish (highest %)
                    top_bullish = watchlist_data[:3]
                    # Top 3 bearish (lowest %)
                    top_bearish = sorted(watchlist_data, key=lambda x: x.get('daily_change_pct', 0))[:3]
                    
                    data['watchlist_movers']['bullish'] = top_bullish
                    data['watchlist_movers']['bearish'] = top_bearish
                    
                    # Add movers to symbols_to_scan for options analysis
                    for item in top_bullish + top_bearish:
                        symbols_to_scan.add(item['symbol'])
                    
                    logger.info(f"Fetched watchlist movers - Bullish: {[s['symbol'] for s in top_bullish]}, Bearish: {[s['symbol'] for s in top_bearish]}")
            except Exception as e:
                logger.warning(f"Error fetching watchlist movers: {e}")
            
            # Scan whale flows for symbols from TOS, Z-Score, ETF scanners AND watchlist movers
            if client and symbols_to_scan:
                # Remove symbols already in whale flows
                whale_symbols = set(f['symbol'] for f in data['whale_flows'])
                symbols_to_scan = symbols_to_scan - whale_symbols
                
                # Limit to top 10 symbols to avoid API overload
                symbols_to_scan = list(symbols_to_scan)[:10]
                
                if symbols_to_scan:
                    logger.info(f"Scanning options activity for {len(symbols_to_scan)} signaled symbols: {symbols_to_scan}")
                    options_data = await self._scan_options_for_symbols(client, symbols_to_scan)
                    data['options_activity'] = options_data
            
        except Exception as e:
            logger.error(f"Error collecting scanner data: {e}")
        
        return data
    
    async def _scan_options_for_symbols(self, client, symbols: List[str]) -> List[Dict]:
        """
        Scan options activity for given symbols to enrich commentary
        """
        from bot.commands.whale_score import scan_stock_whale_flows, get_next_three_fridays
        
        results = []
        expiry_dates = get_next_three_fridays()
        
        for symbol in symbols:
            try:
                # Lower threshold (50) to catch more activity
                flows = scan_stock_whale_flows(client, symbol, expiry_dates, min_whale_score=50)
                
                if flows:
                    # Summarize the flows for this symbol
                    calls = [f for f in flows if f['type'] == 'CALL']
                    puts = [f for f in flows if f['type'] == 'PUT']
                    
                    total_call_volume = sum(f['volume'] for f in calls)
                    total_put_volume = sum(f['volume'] for f in puts)
                    
                    # Calculate put/call ratio
                    pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
                    
                    # Get top flow by whale score
                    top_flow = max(flows, key=lambda x: x['whale_score'])
                    
                    # Determine sentiment
                    if total_call_volume > total_put_volume * 1.5:
                        sentiment = "BULLISH"
                    elif total_put_volume > total_call_volume * 1.5:
                        sentiment = "BEARISH"
                    else:
                        sentiment = "MIXED"
                    
                    results.append({
                        'symbol': symbol,
                        'total_flows': len(flows),
                        'calls': len(calls),
                        'puts': len(puts),
                        'call_volume': total_call_volume,
                        'put_volume': total_put_volume,
                        'pc_ratio': pc_ratio,
                        'sentiment': sentiment,
                        'top_strike': top_flow['strike'],
                        'top_type': top_flow['type'],
                        'top_score': top_flow['whale_score'],
                        'underlying_price': top_flow['underlying_price']
                    })
                    
            except Exception as e:
                logger.debug(f"Error scanning options for {symbol}: {e}")
                continue
        
        return results
    
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
        
        # Watchlist Top Movers (NEW)
        watchlist = data.get('watchlist_movers', {})
        if watchlist.get('bullish') or watchlist.get('bearish'):
            parts.append("WATCHLIST TOP MOVERS:")
            if watchlist.get('bullish'):
                parts.append("  Bullish Leaders:")
                for s in watchlist['bullish']:
                    parts.append(f"    • {s['symbol']}: +{s.get('daily_change_pct', 0):.2f}% @ ${s.get('price', 0):.2f}")
            if watchlist.get('bearish'):
                parts.append("  Bearish Laggards:")
                for s in watchlist['bearish']:
                    parts.append(f"    • {s['symbol']}: {s.get('daily_change_pct', 0):.2f}% @ ${s.get('price', 0):.2f}")
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
        
        # Options Activity for signaled stocks with divergence analysis
        if data.get('options_activity'):
            # Build map of watchlist movers for divergence check
            mover_changes = {}
            for m in data.get('watchlist_movers', {}).get('bullish', []):
                mover_changes[m['symbol']] = m.get('daily_change_pct', 0)
            for m in data.get('watchlist_movers', {}).get('bearish', []):
                mover_changes[m['symbol']] = m.get('daily_change_pct', 0)
            
            parts.append(f"OPTIONS FLOW ANALYSIS ({len(data['options_activity'])} stocks scanned):")
            for opt in data['options_activity']:
                symbol = opt['symbol']
                sentiment = opt['sentiment']
                price_change = mover_changes.get(symbol)
                
                # Check for divergence
                divergence = ""
                if price_change is not None:
                    if price_change > 1 and sentiment == 'BEARISH':
                        divergence = " ⚠️ DIVERGENCE: Price up but options bearish"
                    elif price_change < -1 and sentiment == 'BULLISH':
                        divergence = " ⚠️ DIVERGENCE: Price down but options bullish"
                
                change_str = f" ({price_change:+.2f}% today)" if price_change is not None else ""
                parts.append(f"  • {symbol}{change_str}: {sentiment} options sentiment{divergence}")
                parts.append(f"    Calls: {opt['calls']} ({opt['call_volume']:,} vol) | Puts: {opt['puts']} ({opt['put_volume']:,} vol)")
                parts.append(f"    P/C Ratio: {opt['pc_ratio']:.2f} | Top Flow: {opt['top_type']} ${opt['top_strike']} (Score: {opt['top_score']:.0f})")
            parts.append("")
        
        # Add request
        parts.append("""Based on this data, provide market commentary that:
1. Summarizes key market themes and unusual activity
2. Highlights any DIVERGENCE between price action and options flow (this is critical)
3. Notes confirmation when price and options align
4. Suggests what to watch based on options positioning""")
        
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
        
        # Watchlist Top Movers
        watchlist = data.get('watchlist_movers', {})
        if watchlist.get('bullish') or watchlist.get('bearish'):
            parts.append("🚀 **Top Movers:**")
            if watchlist.get('bullish'):
                for s in watchlist['bullish'][:3]:
                    parts.append(f"🟢 {s['symbol']}: +{s.get('daily_change_pct', 0):.2f}%")
            if watchlist.get('bearish'):
                for s in watchlist['bearish'][:3]:
                    parts.append(f"🔴 {s['symbol']}: {s.get('daily_change_pct', 0):.2f}%")
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
        
        # Options activity for signaled stocks
        if data.get('options_activity'):
            parts.append("")
            parts.append("🔍 **Options Activity on Signaled Stocks:**")
            for opt in data['options_activity']:
                sentiment_emoji = "🟢" if opt['sentiment'] == 'BULLISH' else "🔴" if opt['sentiment'] == 'BEARISH' else "⚪"
                parts.append(f"{sentiment_emoji} {opt['symbol']}: {opt['sentiment']} (P/C: {opt['pc_ratio']:.2f})")
        
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
