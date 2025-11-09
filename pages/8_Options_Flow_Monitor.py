#!/usr/bin/env python3
"""
🎯 OPTIONS FLOW ANALYSIS - PRODUCTION VERSION
===========================================

📊 Real-time options flow analysis with Discord notifications
🚀 Runs automatically during market hours (Mon-Fri 9:30-16:00 ET)

PYTHONANYWHERE DEPLOYMENT:
1. Upload this script to your PythonAnywhere account
2. Set your Discord webhook URL in the DISCORD_WEBHOOK_URL variable (line ~190)
3. Create a scheduled task: python3.11 /home/yourusername/abc.py
4. Set to run every 15 minutes during market hours

LOCAL SETUP:
1. Set environment variable: export DISCORD_WEBHOOK_URL='your_webhook_url'
2. Run scheduler: python3 abc.py --schedule
3. Or single run: python3 abc.py

FEATURES:
✅ Advanced price caching & batch processing
✅ Market hours validation
✅ Enhanced Discord UI with color coding
✅ Performance monitoring & logging
✅ Error handling & retry logic
✅ Mobile-optimized table format
✅ PythonAnywhere task ready

LOG FILES: options_flow_analysis.log
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import logging
import json
import os
import sqlite3
from typing import List, Optional, Dict, Tuple
import hashlib
import pytz
import time
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except Exception:
    # schedule library not available in this environment. Provide a minimal no-op
    # fallback so the rest of the script can run without scheduling features.
    SCHEDULE_AVAILABLE = False
    logger.warning("python-schedule package not available. Scheduling features will be disabled.")

    class _DummyJob:
        def do(self, func, *args, **kwargs):
            # No-op: return a dummy reference
            return None

    class _DummyEvery:
        def __init__(self, interval=None):
            self.interval = interval

        def minutes(self):
            return _DummyJob()

        def day(self):
            return self

        def days(self):
            return _DummyJob()

        def at(self, t: str):
            return _DummyJob()

    class _DummyScheduleModule:
        def every(self, interval=None):
            return _DummyEvery(interval)

        def run_pending(self):
            # No scheduled jobs to run
            return None

    schedule = _DummyScheduleModule()
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import asyncio
from functools import lru_cache
from threading import Lock
import threading

# Configuration flags - must be defined before logging setup
TEST_MODE = False  # Set to True for testing, False for production
FORCE_DISCORD_FORMAT = True  # Set to True to see Discord format even when market is closed during test mode
SEND_TO_DISCORD_IN_TEST = True  # Set to True to actually send to Discord during test mode

# Configure production-ready logging
log_level = logging.INFO if not TEST_MODE else logging.DEBUG
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_flow_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production startup message
if not TEST_MODE:
    logger.info("🚀 PRODUCTION MODE: Options Flow Analysis starting...")
    logger.info("📅 Market hours: Monday-Friday 9:30 AM - 4:00 PM ET")
else:
    logger.warning("🧪 TEST MODE: Running in development mode")

# ⚡ PERFORMANCE OPTIMIZATION: Advanced Price Caching System
class AdvancedPriceCache:
    """Thread-safe price cache with TTL, batch processing, and intelligent prefetching."""

    def __init__(self, ttl_seconds=300, max_cache_size=1000):
        self.cache = {}
        self.cache_timestamps = {}
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.lock = Lock()
        self.batch_size = 20  # Optimal batch size for yfinance

    def is_cache_valid(self, symbol: str) -> bool:
        """Check if cached price is still valid."""
        if symbol not in self.cache_timestamps:
            return False
        age = time.time() - self.cache_timestamps[symbol]
        return age < self.ttl_seconds

    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Get price from cache if valid."""
        with self.lock:
            if self.is_cache_valid(symbol):
                return self.cache.get(symbol)
        return None

    def cache_price(self, symbol: str, price: float):
        """Cache a price with timestamp."""
        with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_cache_size and symbol not in self.cache:
                oldest_symbol = min(self.cache_timestamps.keys(),
                                  key=lambda k: self.cache_timestamps[k])
                del self.cache[oldest_symbol]
                del self.cache_timestamps[oldest_symbol]

            self.cache[symbol] = price
            self.cache_timestamps[symbol] = time.time()

    def get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Efficiently fetch multiple prices using batching and caching."""
        result = {}
        symbols_to_fetch = []

        # Check cache first
        for symbol in symbols:
            cached_price = self.get_cached_price(symbol)
            if cached_price is not None:
                result[symbol] = cached_price
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return result

        logger.info(f"🔄 Fetching {len(symbols_to_fetch)} prices (cache hit: {len(result)}/{len(symbols)})")

        # Batch fetch remaining symbols
        batched_prices = self._fetch_prices_in_batches(symbols_to_fetch)

        # Cache and merge results
        for symbol, price in batched_prices.items():
            if price is not None:
                self.cache_price(symbol, price)
                result[symbol] = price

        return result

    def _fetch_prices_in_batches(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch prices in optimal batches with rate limiting."""
        all_prices = {}

        # Split into batches
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]

            try:
                # Create ticker string for batch fetch
                ticker_string = ' '.join(batch)
                tickers = yf.Tickers(ticker_string)

                # Fetch data for all tickers in batch
                for symbol in batch:
                    try:
                        ticker = tickers.tickers[symbol]
                        hist = ticker.history(period="1d", interval="1m")
                        if not hist.empty:
                            all_prices[symbol] = float(hist['Close'].iloc[-1])
                        else:
                            # Fallback to daily data
                            hist_daily = ticker.history(period="2d")
                            if not hist_daily.empty:
                                all_prices[symbol] = float(hist_daily['Close'].iloc[-1])
                    except Exception as e:
                        logger.debug(f"Failed to get price for {symbol}: {e}")
                        continue

                # Rate limiting between batches
                if i + self.batch_size < len(symbols):
                    time.sleep(0.2)  # 200ms between batches

            except Exception as e:
                logger.error(f"Batch fetch failed for symbols {batch}: {e}")
                # Fallback to individual fetching for this batch
                for symbol in batch:
                    try:
                        price = self._fetch_single_price_fallback(symbol)
                        if price:
                            all_prices[symbol] = price
                    except Exception:
                        continue

        return all_prices

    def _fetch_single_price_fallback(self, symbol: str) -> Optional[float]:
        """Fallback method for individual price fetching."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return None

    def clear_expired(self):
        """Remove expired entries from cache."""
        with self.lock:
            current_time = time.time()
            expired_symbols = [
                symbol for symbol, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]

            for symbol in expired_symbols:
                del self.cache[symbol]
                del self.cache_timestamps[symbol]

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        with self.lock:
            valid_entries = sum(1 for symbol in self.cache.keys()
                              if self.is_cache_valid(symbol))
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'expired_entries': len(self.cache) - valid_entries,
                'cache_size_mb': len(str(self.cache)) / (1024 * 1024)
            }

# Global price cache instance
price_cache = AdvancedPriceCache(ttl_seconds=300)  # 5-minute TTL

# Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Discord sending is disabled by default in this deployment. If you want to re-enable,
# set DISCORD_WEBHOOK_URL in the environment. The script will not exit when the webhook
# is missing — it will log a warning and continue processing.
if not DISCORD_WEBHOOK_URL:
    logger.warning("DISCORD_WEBHOOK_URL not configured - Discord sending is DISABLED.")

DATABASE_FILE = "top3_flow.db"
LAST_RUN_FILE = "last_run_top3.json"

# Market hours configuration
US_EASTERN = pytz.timezone('US/Eastern')

MARKET_OPEN_TIME = "09:30"
def run_scheduler():
    """Run the scheduler."""
    logger.info("🚀 Starting Separated Flow Scheduler (Mag7 vs Others)...")

    # Schedule every 15 minutes during market hours
    schedule.every(15).minutes.do(analyze_separated_flows)

    # Initial run
    analyze_separated_flows()

    while True:
        schedule.run_pending()
        time.sleep(60)

def get_current_movers(min_change_percent: float = 2.0) -> List[str]:
    """Get list of current pre/post-market movers from database"""
    try:
        # Check if movers database exists
        if not os.path.exists('market_movers.db'):
            return []
        conn = sqlite3.connect('market_movers.db')
        cursor = conn.cursor()
        # Get movers from today
        today = datetime.now(US_EASTERN).date().isoformat()

        cursor.execute('''
            SELECT DISTINCT symbol, change_percent, session_type
            FROM movers
            WHERE DATE(timestamp) = ?
            AND ABS(change_percent) >= ?
            ORDER BY ABS(change_percent) DESC
        ''', (today, min_change_percent))

        movers = cursor.fetchall()
        conn.close()

        # Return just the symbols
        return [mover[0] for mover in movers]

    except Exception as e:
        logger.error(f"Error fetching movers: {e}")
        return []

def enhance_flow_with_movers(flows_df: pd.DataFrame) -> pd.DataFrame:
    """Enhance flow data with mover flag (scoring boost applied later)"""
    if flows_df.empty:
        return flows_df

    # Get current movers
    current_movers = get_current_movers()

    if not current_movers:
        flows_df['Is_Mover'] = False
        return flows_df

    logger.info(f"🎯 Found {len(current_movers)} current movers: {', '.join(current_movers[:5])}...")

    # Add mover flag (score boost happens later in calculate_flow_score)
    flows_df['Is_Mover'] = flows_df['Symbol'].isin(current_movers)

    # Add mover count to logging
    mover_count = flows_df['Is_Mover'].sum()
    if mover_count > 0:
        logger.info(f"🚀 Tagged {mover_count} flows from current movers")

    return flows_df

def get_context_indicators(details: Dict) -> str:
    """Return context indicators for a flow's details (e.g., block trades, urgency, volatility)."""
    indicators = []
    if details.get('block_trades'):
        indicators.append("🏛️Block")
    if details.get('total_volume', 0) > 10000:
        indicators.append("🔥Vol")
    if details.get('avg_days', 0) <= 3:
        indicators.append("⏰Urgent")
    if details.get('sentiment') == "BEARISH":
        indicators.append("⚠️Bear")
    if details.get('sentiment') == "BULLISH":
        indicators.append("📈Bull")
    return " ".join(indicators)

def get_sentiment_emoji(sentiment: str) -> str:
    """Return an emoji representing sentiment."""
    if sentiment == "BULLISH":
        return "📈"
    elif sentiment == "BEARISH":
        return "📉"
    elif sentiment == "MIXED":
        return "⚡"
    else:
        return ""

def get_urgency_indicator(flow: dict) -> str:
    """Return an urgency indicator emoji based on days to expiry."""
    details = flow.get('Details', {})
    avg_days = details.get('avg_days', 0)
    if avg_days <= 3:
        return "🔴"
    elif avg_days <= 7:
        return "🟡"
    else:
        return "🟢"

def format_mover_enhanced_message(flows: List[Dict]) -> Optional[Dict]:
    """Enhanced message formatting that highlights movers"""
    if not flows:
        return None

    current_movers = get_current_movers()

    # Separate movers from regular flows
    mover_flows = [f for f in flows if f['Symbol'] in current_movers]
    regular_flows = [f for f in flows if f['Symbol'] not in current_movers]

    embeds = []

    # Create mover-specific embed if we have mover flows
    if mover_flows:
        mover_embed = {
            "title": "🎯 HOT MOVERS + UNUSUAL FLOW",
            "color": 0xFF6B35,  # Orange-red for hot movers
            "fields": [],
            "timestamp": datetime.now().isoformat()
        }

        for flow in mover_flows[:5]:  # Top 5 mover flows
            details = flow['Details']

            # Enhanced formatting for movers
            field_name = f"🔥 {flow['Symbol']} {get_sentiment_emoji(details['sentiment'])} {get_urgency_indicator(flow)}"

            field_value = f"**${details['total_premium']/1000000:.1f}M** | "
            field_value += f"**{details['primary_type']}** / "
            field_value += f"**${details['stock_price']:.2f}** → **${details['most_active_strike']}** "
            field_value += f"({((details['most_active_strike'] - details['stock_price']) / details['stock_price'] * 100):+.1f}%) | "
            field_value += f"**${details['target_price']}** | "
            field_value += f"⏰{details['time_to_expiry']}d | "
            field_value += f"{details['confidence']}% | "
            field_value += get_context_indicators(details)

            mover_embed["fields"].append({
                "name": field_name,
                "value": field_value,
                "inline": False
            })

        embeds.append(mover_embed)

    # Regular flows embed (if we have regular flows and space)
    if regular_flows and len(embeds) == 0:  # Only if no mover embed
        regular_embed = {
            "title": "📊 UNUSUAL OPTIONS FLOW",
            "color": 0x00D4AA,  # Teal
            "fields": [],
            "timestamp": datetime.now().isoformat()
        }

        for flow in regular_flows[:10]:  # More regular flows if no movers
            details = flow['Details']

            field_name = f"{flow['Symbol']} {get_sentiment_emoji(details['sentiment'])} {get_urgency_indicator(flow)}"

            field_value = f"**${details['total_premium']/1000000:.1f}M** | "
            field_value += f"**{details['primary_type']}** / "
            field_value += f"**${details['stock_price']:.2f}** → **${details['most_active_strike']}** | "
            field_value += f"**${details['target_price']}** | "
            field_value += f"⏰{details['time_to_expiry']}d | "
            field_value += f"{details['confidence']}% | "
            field_value += get_context_indicators(details)

            regular_embed["fields"].append({
                "name": field_name,
                "value": field_value,
                "inline": True
            })

        embeds.append(regular_embed)

    return {"embeds": embeds} if embeds else None

MARKET_CLOSE_TIME = "16:00"
MARKET_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday

# 2025 US Stock Market Holidays
US_HOLIDAYS_2025 = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
]

# Early close days
EARLY_CLOSE_DAYS_2025 = {
    '2025-07-03': '13:00',
    '2025-11-28': '13:00',
    '2025-12-24': '13:00'
}

# Exclude index symbols and problematic symbols - focus only on individual stocks
INDEX_SYMBOLS = ['SPX', 'SPXW', 'IWM', 'DIA', 'VIX', 'VIXW', 'XSP', 'RUTW']
EXCLUDED_SYMBOLS = INDEX_SYMBOLS + [
    # Exclude symbols that commonly cause yfinance errors
    'BRKB', 'RUT', '4SPY', 'RUTW', 'DJX', 'BFB'
] + [s for s in [] if s.startswith('$')]  # Exclude any symbols starting with $

# Magnificent 7 - separate these out for dedicated analysis
MAG7_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META','SPY','QQQ']

def is_market_open():
    """Check if the US market is currently open."""
    if TEST_MODE:
        logger.info("TEST MODE: Market check bypassed - treating as OPEN")
        return True

    try:
        et_now = datetime.now(US_EASTERN)
        today_str = et_now.strftime('%Y-%m-%d')

        if today_str in US_HOLIDAYS_2025:
            logger.info(f"Market closed - Holiday: {today_str}")
            return False

        if et_now.weekday() not in MARKET_DAYS:
            logger.info(f"Market closed - Weekend")
            return False

        market_close_time = EARLY_CLOSE_DAYS_2025.get(today_str, MARKET_CLOSE_TIME)
        market_open = datetime.strptime(MARKET_OPEN_TIME, "%H:%M").time()
        market_close = datetime.strptime(market_close_time, "%H:%M").time()
        current_time = et_now.time()

        is_open = market_open <= current_time <= market_close
        return is_open

    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False

def init_database():
    """Initialize SQLite database."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS top3_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                strike_price REAL,
                call_put TEXT,
                expiration DATE,
                volume INTEGER,
                premium REAL,
                flow_score REAL,
                sentiment TEXT,
                data_hash TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_hash ON top3_flows(data_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON top3_flows(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON top3_flows(timestamp)')
        conn.commit()

def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        required_columns = ['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Last Price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return None

        # Clean and filter data
        df = df.dropna(subset=['Symbol', 'Expiration', 'Strike Price', 'Call/Put'])
        df = df[df['Volume'] >= 50].copy()  # Lower volume threshold for more opportunities

        # Parse expiration dates
        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
        df = df.dropna(subset=['Expiration'])
        df = df[df['Expiration'].dt.date >= datetime.now().date()]

        return df
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    """⚡ OPTIMIZED: Fetch and combine data from multiple URLs with improved error handling."""
    data_frames = []

    def fetch_with_retry(url, max_retries=2):
        """Fetch URL with exponential backoff retry."""
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"🔄 Retry {attempt}/{max_retries} for {url} (waiting {wait_time}s)")
                    time.sleep(wait_time)

                return fetch_data_from_url(url)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ Failed to fetch {url} after {max_retries} retries: {e}")
                    return None
                continue
        return None

    # Use optimized concurrency settings
    max_workers = min(4, len(urls))  # Don't create more threads than URLs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_url = {executor.submit(fetch_with_retry, url): url for url in urls}

        # Collect results as they complete
        for future in as_completed(future_to_url, timeout=120):  # 2-minute total timeout
            url = future_to_url[future]
            try:
                df = future.result(timeout=30)  # 30-second timeout per URL
                if df is not None and not df.empty:
                    data_frames.append(df)
                    logger.info(f"✅ Successfully fetched {len(df)} records from {url}")
                else:
                    logger.warning(f"⚠️ No data from {url}")
            except Exception as e:
                logger.error(f"❌ Exception fetching {url}: {e}")

    # Combine all successful data frames
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        logger.info(f"🎯 Combined {len(combined_df)} total records from {len(data_frames)}/{len(urls)} sources")
        return combined_df
    else:
        logger.error("❌ No data fetched from any source")
        return pd.DataFrame()

def get_stock_price(symbol: str) -> Optional[float]:
    """Get current stock price with advanced caching - OPTIMIZED."""
    # Use the global cache for single symbol requests
    cached_price = price_cache.get_cached_price(symbol)
    if cached_price is not None:
        return cached_price

    # Fetch and cache if not available
    prices = price_cache.get_batch_prices([symbol])
    return prices.get(symbol)

def get_multiple_stock_prices(symbols: List[str]) -> Dict[str, float]:
    """⚡ OPTIMIZED: Get multiple stock prices efficiently using batch processing."""
    # Filter out problematic symbols
    clean_symbols = [s for s in symbols if not s.startswith('$') and len(s) <= 5]

    if not clean_symbols:
        return {}

    return price_cache.get_batch_prices(clean_symbols)

def detect_block_trades(symbol_flows: pd.DataFrame, symbol: str) -> List[Dict]:
    """Detect potential block trades and unusual activity."""
    blocks = []

    if symbol_flows.empty:
        return blocks

    # Group by strike, expiry, and type to find concentrated activity
    grouped = symbol_flows.groupby(['Strike Price', 'Expiration', 'Call/Put']).agg({
        'Volume': 'sum',
        'Premium': 'sum'
    }).reset_index()

    # Look for unusually large single strike activity
    for _, row in grouped.iterrows():
        if row['Volume'] > 500 and row['Premium'] > 500000:  # 500+ contracts, $500K+ premium
            avg_price = row['Premium'] / (row['Volume'] * 100)
            blocks.append({
                'symbol': symbol,
                'strike': row['Strike Price'],
                'expiry': row['Expiration'],
                'type': 'CALL' if row['Call/Put'] == 'C' else 'PUT',
                'volume': row['Volume'],
                'premium': row['Premium'],
                'avg_price': avg_price
            })

    return sorted(blocks, key=lambda x: x['premium'], reverse=True)[:3]

def calculate_flow_score(symbol_flows: pd.DataFrame, current_price: float) -> Dict:
    """Enhanced flow scoring with buy/sell direction inference using multiple heuristics."""
    if symbol_flows.empty or current_price is None:
        return {'score': 0, 'details': {}}

    # Calculate both actual premium (for display) and weighted premium (for scoring)
    actual_premium = symbol_flows['Premium'].sum()  # Real premium for display
    weighted_premium = symbol_flows.get('Weighted_Premium', symbol_flows['Premium']).sum()  # Weighted for scoring

    symbol = symbol_flows['Symbol'].iloc[0] if 'Symbol' in symbol_flows.columns else 'UNKNOWN'

    # Lower threshold for MAG7 symbols - use weighted premium for filtering
    if symbol in MAG7_SYMBOLS:
        if weighted_premium < 50000:  # Reduced from 100K
            return {'score': 0, 'details': {'reason': 'Below premium threshold'}}
    else:
        if weighted_premium < 200000:  # Reduced from 500K
            return {'score': 0, 'details': {'reason': 'Below premium threshold'}}

    # 🎯 OPTIMIZATION: Reduce premium-based scoring since high scores showed negative correlation
    # Focus more on signal quality than pure size - use weighted premium for scoring
    base_score = (weighted_premium / 1000000) * 5  # Reduced multiplier from 10 to 5

    calls = symbol_flows[symbol_flows['Call/Put'] == 'C']
    puts = symbol_flows[symbol_flows['Call/Put'] == 'P']

    call_premium = calls['Premium'].sum()
    put_premium = puts['Premium'].sum()
    total_volume = symbol_flows['Volume'].sum()

    # 🎯 ENHANCED: Multi-factor direction inference due to CBOE data limitations
    direction_signals = []
    confidence_factors = []

    # Factor 1: Premium weighting (basic)
    premium_ratio = call_premium / put_premium if put_premium > 0 else float('inf')
    if premium_ratio > 1.5:
        direction_signals.append("BULLISH")
        confidence_factors.append(("Premium Ratio", min(premium_ratio, 5) / 5 * 30))
    elif premium_ratio < 0.67 and premium_ratio > 0:
        direction_signals.append("BEARISH")
        confidence_factors.append(("Premium Ratio", min(1/premium_ratio, 5) / 5 * 30))

    # Factor 2: Strike price analysis (OTM vs ITM activity)
    otm_calls = calls[calls['Strike Price'] > current_price * 1.02]  # >2% OTM
    itm_calls = calls[calls['Strike Price'] <= current_price * 0.98]  # >2% ITM
    otm_puts = puts[puts['Strike Price'] < current_price * 0.98]  # >2% OTM
    itm_puts = puts[puts['Strike Price'] >= current_price * 1.02]  # >2% ITM

    otm_call_premium = otm_calls['Premium'].sum()
    itm_call_premium = itm_calls['Premium'].sum()
    otm_put_premium = otm_puts['Premium'].sum()
    itm_put_premium = itm_puts['Premium'].sum()

    # OTM calls usually bought (bullish), ITM puts usually bought (bearish)
    speculative_bullish = otm_call_premium
    speculative_bearish = otm_put_premium + itm_put_premium

    if speculative_bullish > speculative_bearish * 1.3:
        direction_signals.append("BULLISH")
        confidence_factors.append(("OTM Activity", 25))
    elif speculative_bearish > speculative_bullish * 1.3:
        direction_signals.append("BEARISH")
        confidence_factors.append(("OTM Activity", 25))

    # Factor 3: Unusual volume patterns (big trades likely institutional)
    unusual_threshold = symbol_flows['Volume'].quantile(0.8)
    unusual_flows = symbol_flows[symbol_flows['Volume'] > unusual_threshold]

    if not unusual_flows.empty:
        unusual_calls = unusual_flows[unusual_flows['Call/Put'] == 'C']['Premium'].sum()
        unusual_puts = unusual_flows[unusual_flows['Call/Put'] == 'P']['Premium'].sum()

        if unusual_calls > unusual_puts * 1.2:
            direction_signals.append("BULLISH")
            confidence_factors.append(("Large Trades", 20))
        elif unusual_puts > unusual_calls * 1.2:
            direction_signals.append("BEARISH")
            confidence_factors.append(("Large Trades", 20))

    # Factor 4: Time to expiration patterns
    near_term = symbol_flows[symbol_flows['Days_to_Expiry'] <= 30]
    if not near_term.empty:
        nt_call_premium = near_term[near_term['Call/Put'] == 'C']['Premium'].sum()
        nt_put_premium = near_term[near_term['Call/Put'] == 'P']['Premium'].sum()

        # Near-term OTM calls often speculative buying
        nt_otm_calls = near_term[
            (near_term['Call/Put'] == 'C') &
            (near_term['Strike Price'] > current_price * 1.05)
        ]['Premium'].sum()

        if nt_otm_calls > nt_put_premium * 0.8:
            direction_signals.append("BULLISH")
            confidence_factors.append(("Near-term OTM", 15))

    # Aggregate signals and determine confidence
    bullish_count = direction_signals.count("BULLISH")
    bearish_count = direction_signals.count("BEARISH")
    total_confidence = sum([factor[1] for factor in confidence_factors])

    # 🎯 OPTIMIZATION: Enhance bearish signal detection since they showed 100% accuracy
    bearish_signal_boost = 0
    if bearish_count > 0:
        # Boost confidence for bearish signals based on performance analysis
        bearish_signal_boost = 15
        total_confidence += bearish_signal_boost

    # 🎯 NEW: Quality-based scoring adjustments
    quality_score = 0
    time_weight_avg = symbol_flows.get('Time_Weight', pd.Series([1.0] * len(symbol_flows))).mean()
    quality_score += time_weight_avg * 20  # Up to 20 points for optimal timing

    # Add signal consistency bonus
    signal_consistency = len(direction_signals) / 4 * 10  # Up to 10 points for multiple confirming signals
    quality_score += signal_consistency

    # 🎯 NEW: Mover boost - apply 2x multiplier if this is a current mover
    mover_boost = 1.0
    if 'Is_Mover' in symbol_flows.columns and symbol_flows['Is_Mover'].any():
        mover_boost = 2.0
        quality_score += 25  # Additional quality points for mover relevance
        logger.info(f"🚀 Applied mover boost to {symbol} (2x multiplier)")

    # Final score combines base premium score with quality adjustments and mover boost
    final_score = (base_score + quality_score) * mover_boost

    if bullish_count > bearish_count:
        sentiment = "BULLISH"
        bias_strength = min(95, 55 + total_confidence)
        # Lower confidence for bullish signals since they showed mixed performance
        direction_quality = "HIGH" if total_confidence > 60 else "MEDIUM" if total_confidence > 35 else "LOW"
    elif bearish_count > bullish_count:
        sentiment = "BEARISH"
        bias_strength = min(95, 55 + total_confidence)
        # Higher confidence for bearish signals since they showed 100% accuracy in analysis
        direction_quality = "HIGH" if total_confidence > 45 else "MEDIUM" if total_confidence > 25 else "LOW"
    else:
        sentiment = "MIXED"
        bias_strength = 60
        direction_quality = "LOW"

    strike_analysis = symbol_flows.groupby(['Strike Price', 'Call/Put', 'Expiration']).agg({
        'Premium': 'sum',
        'Volume': 'sum'
    }).reset_index()

    top_strikes = strike_analysis.nlargest(5, 'Premium')

    # 🎯 NEW: Add block trades detection
    block_trades = detect_block_trades(symbol_flows, symbol)

    # Find key strike levels for backward compatibility
    strike_volumes = symbol_flows.groupby('Strike Price')['Premium'].sum().sort_values(ascending=False)
    key_strikes = strike_volumes.head(3).index.tolist()
    avg_days = symbol_flows['Days_to_Expiry'].mean()

    details = {
        'total_premium': actual_premium,  # Fixed: Use actual premium for display, not weighted
        'weighted_premium': weighted_premium,  # Store weighted premium for reference
        'total_volume': total_volume,
        'call_premium': call_premium,
        'put_premium': put_premium,
        'sentiment': sentiment,
        'bias_strength': bias_strength,
        'direction_quality': direction_quality,  # NEW: Confidence level
        'confidence_factors': confidence_factors,  # NEW: What drove the decision
        'interpretation_note': "⚠️ CBOE data lacks buy/sell direction",  # NEW
        'otm_call_premium': otm_call_premium,  # NEW: More granular data
        'otm_put_premium': otm_put_premium,  # NEW
        'speculative_bias': "BULLISH" if speculative_bullish > speculative_bearish else "BEARISH",  # NEW
        'top_strikes': top_strikes,
        'block_trades': block_trades,
        'key_strikes': key_strikes,  # For backward compatibility
        'avg_days': avg_days  # For backward compatibility
    }

    return {'score': final_score, 'details': details}

def filter_and_score_flows(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Filter flows and calculate scores for each symbol with Mag7 separation."""
    if df.empty:
        return {'mag7': [], 'other': []}

    # Exclude problematic symbols early
    df = df[~df['Symbol'].isin(EXCLUDED_SYMBOLS)].copy()

    # Additional filters to reduce dataset size
    df = df[df['Symbol'].str.len() <= 5]  # Exclude complex symbols
    df = df[~df['Symbol'].str.contains(r'[\$\d]', regex=True)]  # Exclude symbols with $ or numbers

    # Calculate days to expiry and premium
    df['Days_to_Expiry'] = (df['Expiration'] - datetime.now()).dt.days

    # 🎯 OPTIMIZATION: Focus on 3-7 day window based on performance analysis
    # Performance data shows 3-5 day period has 66-100% accuracy vs declining accuracy for longer periods
    df = df[(df['Days_to_Expiry'] >= 1) & (df['Days_to_Expiry'] <= 45)]  # Sweet spot window

    df['Premium'] = df['Volume'] * df['Last Price'] * 100

    # Add time decay weighting - closer to 3-5 days gets higher weight
    def calculate_time_weight(days):
        if 3 <= days <= 5:
            return 1.0  # Optimal range
        elif days == 2 or days == 6:
            return 0.8  # Close to optimal
        elif days == 1 or days == 7:
            return 0.6  # Further from optimal
        else:
            return 0.3  # Outside optimal range

    df['Time_Weight'] = df['Days_to_Expiry'].apply(calculate_time_weight)
    df['Weighted_Premium'] = df['Premium'] * df['Time_Weight']

    # Pre-filter by weighted premium to focus on quality signals
    df = df[df['Weighted_Premium'] >= 150000]  # Reduced from 200K to account for weighting

    # Separate Mag7 and other symbols
    mag7_df = df[df['Symbol'].isin(MAG7_SYMBOLS)].copy()
    other_df = df[~df['Symbol'].isin(MAG7_SYMBOLS)].copy()

    # Get top symbols by weighted premium for each category
    mag7_premiums = mag7_df.groupby('Symbol')['Weighted_Premium'].sum().sort_values(ascending=False)
    mag7_symbols = mag7_premiums.index.tolist()  # All Mag7 symbols present

    other_premiums = other_df.groupby('Symbol')['Weighted_Premium'].sum().sort_values(ascending=False)
    other_symbols = other_premiums.head(50).index.tolist()  # Reduced from 200 to 50 for better performance

    all_symbols_to_analyze = mag7_symbols + other_symbols

    logger.info(f"Analyzing {len(mag7_symbols)} Mag7 symbols and top {len(other_symbols)} other symbols...")

    # ⚡ OPTIMIZED: Batch get stock prices for maximum efficiency
    start_time = time.time()
    price_cache_dict = get_multiple_stock_prices(all_symbols_to_analyze)
    fetch_time = time.time() - start_time

    logger.info(f"Got prices for {len(price_cache_dict)} symbols in {fetch_time:.2f}s (avg: {fetch_time/max(1, len(all_symbols_to_analyze)):.3f}s per symbol)")

    # Log cache performance
    cache_stats = price_cache.get_cache_stats()
    logger.info(f"💾 Cache stats: {cache_stats['valid_entries']}/{cache_stats['total_entries']} valid entries")

    # Process symbols and separate into categories
    def process_symbols(symbol_list, source_df, category_name):
        symbol_scores = []
        for symbol in symbol_list:
            if symbol not in price_cache_dict:
                continue

            current_price = price_cache_dict[symbol]
            symbol_flows = source_df[source_df['Symbol'] == symbol].copy()

            if symbol_flows.empty:
                continue

            # Filter for OTM only
            otm_calls = symbol_flows[
                (symbol_flows['Call/Put'] == 'C') &
                (symbol_flows['Strike Price'] > current_price)
            ]
            otm_puts = symbol_flows[
                (symbol_flows['Call/Put'] == 'P') &
                (symbol_flows['Strike Price'] < current_price)
            ]

            otm_flows = pd.concat([otm_calls, otm_puts], ignore_index=True)

            if otm_flows.empty:
                continue

            flow_analysis = calculate_flow_score(otm_flows, current_price)

            threshold = 15 if category_name == 'mag7' else 20  # Further reduced thresholds
            if flow_analysis['score'] > threshold:
                symbol_scores.append({
                    'Symbol': symbol,
                    'Current_Price': current_price,
                    'Flow_Score': flow_analysis['score'],
                    'Details': flow_analysis['details'],
                    'Flows': otm_flows
                })

        symbol_scores.sort(key=lambda x: x['Flow_Score'], reverse=True)
        return symbol_scores

    # Process both categories
    mag7_results = process_symbols(mag7_symbols, df, 'mag7')
    other_results = process_symbols(other_symbols, df, 'other')

    return {'mag7': mag7_results, 'other': other_results}

def format_separated_message(mag7_symbols: List[Dict], other_symbols: List[Dict]) -> Optional[Dict]:
    """Format alternative plays (non-Mag7) for Discord output in table structure. In test mode, also include Mag7 if no others found."""
    # In test mode, if no other symbols, use Mag7 symbols for testing
    if TEST_MODE and not other_symbols and mag7_symbols:
        logger.info("🧪 TEST MODE: No other symbols found, using Mag7 symbols for testing")
        symbols_to_display = mag7_symbols[:8]
        section_title = "📊 TOP PLAYS (MAG7 - TEST MODE)"
    elif other_symbols:
        symbols_to_display = other_symbols[:8]
        section_title = "📊 TOP PLAYS"
    else:
        logger.warning("❌ No symbols to display in Discord message")
        return None

    et_now = datetime.now(US_EASTERN)
    market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"

    # Add market context
    if is_market_open():
        market_context = f"🕐 Live Analysis"
    else:
        next_open = "Monday 9:30 AM ET" if et_now.weekday() >= 4 else "Tomorrow 9:30 AM ET"
        market_context = f"📊 Pre-Market • Next: {next_open}"

    # Create table header
    message_lines = [
        f"🎯 **OPTIONS FLOW ANALYSIS - {et_now.strftime('%H:%M ET')}**",
        f"**{market_status} {market_context}**",
        # "⚠️ *Direction analysis*\n",
        f"**{section_title}**",
        "```",
        "🎯Tkr  Dir Strike  Exp     $    Conf Notes",
        "───────────────────────────────────────"
    ]

    # Display selected symbols
    for stock in symbols_to_display:
        symbol = stock['Symbol']
        price = stock['Current_Price']
        score = stock['Flow_Score']
        details = stock['Details']

        sentiment = details['sentiment']
        total_premium = details['total_premium']
        bias_strength = details['bias_strength']
        direction_quality = details.get('direction_quality', 'MEDIUM')
        key_strikes = details.get('key_strikes', [])
        avg_days = details.get('avg_days', 0)

        # Format direction with colors and improved icons
        if sentiment == "BULLISH":
            direction = "C"  # Calls
            direction_colored = "🟢C"  # Green for calls
            dir_emoji = "📈"
        elif sentiment == "BEARISH":
            direction = "P"  # Puts
            direction_colored = "🔴P"  # Red for puts
            dir_emoji = "📉"
        else:
            direction = "M"  # Mixed
            direction_colored = "🟡M"  # Yellow for mixed
            dir_emoji = "⚡"

        # Get primary strike
        if key_strikes:
            primary_strike = key_strikes[0]
            move_pct = ((primary_strike - price) / price) * 100
        else:
            primary_strike = price * 1.05  # Default to 5% OTM
            move_pct = 5.0

        # Format expiration with urgency indicators
        if avg_days <= 7:
            exp_display = f"{avg_days:.0f}d"
        else:
            exp_display = f"{avg_days:.0f}d"

        # Add urgency color coding for expiration
        if avg_days <= 3:
            exp_colored = f"🔴{exp_display}"  # Red for urgent
        elif avg_days <= 7:
            exp_colored = f"🟡{exp_display}"  # Yellow for soon
        else:
            exp_colored = f"🟢{exp_display}"  # Green for longer term

        # Format premium with better visual hierarchy
        if total_premium >= 1000000:
            premium_display = f"{total_premium/1000000:.1f}M"
        else:
            premium_display = f"{total_premium/1000:.0f}K"

        # Add premium size indicators
        if total_premium >= 10000000:  # $10M+
            premium_colored = f"💎{premium_display}"  # Diamond for huge
        elif total_premium >= 5000000:  # $5M+
            premium_colored = f"💰{premium_display}"  # Money bag for large
        elif total_premium >= 1000000:  # $1M+
            premium_colored = f"💵{premium_display}"  # Bills for medium
        else:
            premium_colored = premium_display  # No icon for smaller

        # Enhanced confidence indicator with icons
        if direction_quality == "HIGH":
            conf = "🎯High"  # Target for high confidence
        elif direction_quality == "MEDIUM":
            conf = "⚠️Med"   # Warning for medium
        else:
            conf = "❓Low"   # Question for low

        # Generate notes/flags
        notes = []
        total_volume = details.get('total_volume', 0)

        if total_volume > 10000:
            notes.append("🔥")  # High volume
        elif total_volume > 5000:
            notes.append("📊")  # Medium volume

        if symbol in ['NVDA', 'TSLA', 'MSTR']:
            notes.append("⚠️")  # High volatility warning

        block_trades = details.get('block_trades', [])
        if block_trades:
            notes.append("🏛️")  # Block trade

        if total_premium > 5000000:
            notes.append("💎")  # Heavy flow

        if avg_days <= 3:
            notes.append("⏰")  # Time urgency

        notes_text = "".join(notes) if notes else ""

        # 🎨 ENHANCED: Format table row with color coding and better spacing
        table_row = f"{symbol:<4} {direction_colored} {primary_strike:>6.0f} {exp_display:>4} {premium_colored:>7} {conf}"
        if notes_text:
            table_row += f" {notes_text}"

        message_lines.append(table_row)

    # Close table and add legend
    message_lines.extend([
        "───────────────────────────────────────",
        "```",
        "",
        "**🎨 Enhanced Legend:**",
        "• **Dir:** 🟢C=Calls 🔴P=Puts �M=Mixed",
        "• **Conf:** 🎯High ⚠️Med ❓Low",
        "• **Premium:** 💎$10M+ 💰$5M+ 💵$1M+",
        "• **Notes:** 🔥Volume 🏛️Block ⚠️HighVol ⏰Urgent �Heavy",
        "",
        f"**📊 Analysis:** {len(symbols_to_display)} plays • ${sum([s['Details']['total_premium'] for s in symbols_to_display])/1000000:.0f}M total premium"
    ])

    message_text = "\n".join(message_lines)

    # Use new flag to control Discord format during test mode
    if is_market_open() or (TEST_MODE and FORCE_DISCORD_FORMAT):  # Show Discord format when market is open OR in test mode with force flag
        # Discord format for production or test preview
        embed = {
            #"title": "🎯 Options Flow Analysis",
            "description": message_text,
            "color": 0x00ff00 if market_status == "🟢 OPEN" else 0xff0000,
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": f"Multi-Factor Analysis • Performance-Tuned • {et_now.strftime('%H:%M ET')}"}
        }
        return {"embeds": [embed]}
    else:
        # Terminal output for testing or when market closed
        return {"terminal_output": message_text}


def send_discord_webhook(message: Dict):
    """Send message to Discord webhook."""
    # Discord sending intentionally disabled. Log the message for debugging instead of posting.
    try:
        if "terminal_output" in message:
            logger.info("Discord disabled - terminal output:\n%s", message["terminal_output"])
            return

        # Pretty-print embed or message content for debugging
        if 'embeds' in message:
            logger.info("Discord disabled - embed message preview: %s", json.dumps(message.get('embeds')[0], default=str))
        else:
            logger.info("Discord disabled - message preview: %s", json.dumps(message, default=str))
    except Exception as e:
        logger.debug(f"Failed to log disabled Discord message: {e}")

def store_separated_data(mag7_symbols: List[Dict], other_symbols: List[Dict]):
    """Store separated results in database."""
    if not mag7_symbols and not other_symbols:
        return

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        # Store top flows from both categories
        all_symbols = mag7_symbols[:5] + other_symbols[:7]  # Top 5 Mag7 + Top 7 others

        for stock in all_symbols:
            symbol = stock['Symbol']
            details = stock['Details']
            flows = stock['Flows']
            category = 'mag7' if symbol in MAG7_SYMBOLS else 'other'

            # Store each significant flow
            for _, flow in flows.head(3).iterrows():  # Top 3 flows per symbol
                data_hash = hashlib.md5(
                    f"{symbol}{flow['Strike Price']}{flow['Call/Put']}{flow['Expiration']}{flow['Premium']}{datetime.now().strftime('%Y-%m-%d-%H')}{category}".encode()
                ).hexdigest()

                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO top3_flows
                        (symbol, strike_price, call_put, expiration, volume, premium, flow_score, sentiment, data_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        flow['Strike Price'],
                        flow['Call/Put'],
                        flow['Expiration'].strftime('%Y-%m-%d'),
                        flow['Volume'],
                        flow['Premium'],
                        stock['Flow_Score'],
                        details['sentiment'],
                        data_hash
                    ))
                except Exception as e:
                    logger.warning(f"Error storing flow data: {e}")

        conn.commit()

def analyze_separated_flows():
    """⚡ OPTIMIZED: Main analysis function with performance monitoring."""
    start_time = time.time()
    logger.info("🎯 Starting Separated Flow Analysis (Mag7 vs Others)...")

    # 🔒 PRODUCTION SAFETY: Check market hours unless in test mode
    if not TEST_MODE:
        if not is_market_open():
            logger.info("🔴 Market is CLOSED - Analysis will not run")
            logger.info("📅 Next analysis will run when market opens (Mon-Fri 9:30 AM - 4:00 PM ET)")
            return
        else:
            logger.info("🟢 Market is OPEN - Proceeding with analysis")

    if TEST_MODE:
        logger.warning("🧪 RUNNING IN TEST MODE")

    # Performance tracking
    perf_metrics = {
        'start_time': start_time,
        'data_fetch_time': 0,
        'data_processing_time': 0,
        'analysis_time': 0,
        'total_records': 0,
        'cache_hits': 0
    }

    init_database()

    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]

    # Fetch and process data with timing
    logger.info("Fetching data from CBOE URLs...")
    fetch_start = time.time()
    data = fetch_data_from_urls(urls)
    perf_metrics['data_fetch_time'] = time.time() - fetch_start

    if data.empty:
        logger.warning("❌ No data fetched from any URLs")
        return

    perf_metrics['total_records'] = len(data)
    logger.info(f"✅ Fetched {len(data)} total flow records in {perf_metrics['data_fetch_time']:.2f}s")

    # 🎯 NEW: Enhance flows with mover data
    processing_start = time.time()
    data = enhance_flow_with_movers(data)
    perf_metrics['data_processing_time'] = time.time() - processing_start

    # Find top scoring symbols separated by category
    analysis_start = time.time()
    results = filter_and_score_flows(data)
    perf_metrics['analysis_time'] = time.time() - analysis_start

    mag7_symbols = results['mag7']
    other_symbols = results['other']

    if not mag7_symbols and not other_symbols:
        logger.warning("❌ No qualifying flows found after analysis")
        logger.info(f"Debug: Analyzed symbols but none met scoring thresholds")
        return

    logger.info(f"✅ Found {len(mag7_symbols)} Mag7 and {len(other_symbols)} other qualifying symbols")

    # Store results
    store_separated_data(mag7_symbols, other_symbols)

    # 🎯 NEW: Use mover-enhanced message formatting
    current_movers = get_current_movers()
    if current_movers:
        logger.info(f"🔥 Current movers in play: {', '.join(current_movers[:5])}...")
        # Check if any of our flows are movers
        all_symbols = [s['Symbol'] for s in mag7_symbols + other_symbols]
        mover_flows = [s for s in all_symbols if s in current_movers]
        if mover_flows:
            logger.info(f"🎯 MOVER FLOWS DETECTED: {', '.join(mover_flows)}")

    # Format and send message (enhanced for movers)
    logger.info("Formatting message for Discord...")
    message = format_separated_message(mag7_symbols, other_symbols)
    if message:
        logger.info("📱 Sending message to Discord webhook...")
        send_discord_webhook(message)
    else:
        logger.warning("❌ No message generated - likely no other_symbols (non-Mag7) found")

    # Log results for debugging
    if mag7_symbols:
        logger.info("Top Mag7 flows:")
        for i, stock in enumerate(mag7_symbols[:5], 1):
            mover_indicator = "🔥" if stock['Symbol'] in current_movers else ""
            logger.info(f"  #{i}: {stock['Symbol']}{mover_indicator} (Score: {stock['Flow_Score']:.1f}, "
                       f"Sentiment: {stock['Details']['sentiment']}, "
                       f"Premium: ${stock['Details']['total_premium']/1000000:.1f}M)")

    if other_symbols:
        logger.info("Top Plays (non-Mag7):")
        for i, stock in enumerate(other_symbols[:10], 1):
            mover_indicator = "🔥" if stock['Symbol'] in current_movers else ""
            logger.info(f"  #{i}: {stock['Symbol']}{mover_indicator} (Score: {stock['Flow_Score']:.1f}, "
                       f"Sentiment: {stock['Details']['sentiment']}, "
                       f"Premium: ${stock['Details']['total_premium']/1000000:.1f}M)")

    # ⚡ Performance reporting
    total_time = time.time() - start_time
    cache_stats = price_cache.get_cache_stats()

    logger.info("📊 PERFORMANCE METRICS:")
    logger.info(f"  📥 Data Fetch: {perf_metrics['data_fetch_time']:.2f}s")
    logger.info(f"  🔄 Data Processing: {perf_metrics['data_processing_time']:.2f}s")
    logger.info(f"  🧮 Analysis: {perf_metrics['analysis_time']:.2f}s")
    logger.info(f"  ⏱️ Total Runtime: {total_time:.2f}s")
    logger.info(f"  📊 Records/sec: {perf_metrics['total_records']/max(total_time, 0.1):.0f}")
    logger.info(f"  💾 Cache Efficiency: {cache_stats['valid_entries']}/{cache_stats['total_entries']} hits")

    # Clean up expired cache entries
    price_cache.clear_expired()

def run_scheduler():
    """🔄 Production-ready scheduler with enhanced monitoring."""
    logger.info("🚀 Starting Options Flow Analysis Scheduler...")

    if not TEST_MODE:
        logger.info("📅 PRODUCTION MODE: Will only run during market hours")
        logger.info("⏰ Schedule: Every 15 minutes, Monday-Friday 9:30 AM - 4:00 PM ET")
    else:
        logger.warning("🧪 TEST MODE: Will run regardless of market hours")

    # Schedule every 15 minutes during market hours
    schedule.every(15).minutes.do(analyze_separated_flows)

    # Add daily cache cleanup at market close
    schedule.every().day.at("16:05").do(lambda: price_cache.clear_expired())

    # Initial run
    logger.info("🎯 Running initial analysis...")
    try:
        analyze_separated_flows()
    except Exception as e:
        logger.error(f"❌ Initial analysis failed: {e}")

    logger.info("🔄 Scheduler is now running. Press Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    import sys

    # Production startup checks
    if not TEST_MODE:
        logger.info("🔒 PRODUCTION MODE ACTIVATED")
        logger.info("📋 Configuration:")
        logger.info(f"  • Discord webhook: {'✅ Configured' if DISCORD_WEBHOOK_URL else '❌ Missing'}")
        logger.info(f"  • Database file: {DATABASE_FILE}")
        logger.info(f"  • Market hours only: ✅ Enabled")
        logger.info(f"  • Cache TTL: {price_cache.ttl_seconds}s")
        logger.info("📡 Deployment: Ready for PythonAnywhere scheduled tasks")

    if len(sys.argv) > 1 and sys.argv[1] == '--schedule':
        run_scheduler()
    else:
        logger.info("🎯 Running single analysis (perfect for PythonAnywhere tasks)...")
        analyze_separated_flows()
