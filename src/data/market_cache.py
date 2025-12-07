"""
Market Data Cache - SQLite Database Layer
Stores watchlist and whale flows data to reduce API calls
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import threading

logger = logging.getLogger(__name__)

class MarketCache:
    """Thread-safe SQLite cache for market data"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.db_path = Path(__file__).parent.parent.parent / 'data' / 'market_cache.db'
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            self._init_db()
    
    def _get_connection(self):
        """Get thread-safe database connection"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Watchlist table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    daily_change REAL NOT NULL,
                    daily_change_pct REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            
            # Create index on symbol
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_watchlist_symbol 
                ON watchlist(symbol)
            """)
            
            # Create index on updated_at
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_watchlist_updated 
                ON watchlist(updated_at DESC)
            """)
            
            # Whale flows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS whale_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    strike REAL NOT NULL,
                    whale_score REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    open_interest INTEGER NOT NULL,
                    vol_oi REAL NOT NULL,
                    premium REAL NOT NULL,
                    delta REAL NOT NULL,
                    expiry DATE NOT NULL,
                    dte INTEGER NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for whale flows
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_whale_symbol 
                ON whale_flows(symbol)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_whale_score 
                ON whale_flows(whale_score DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_whale_detected 
                ON whale_flows(detected_at DESC)
            """)
            
            # MACD Scanner table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macd_scanner (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    price_change REAL NOT NULL,
                    price_change_pct REAL NOT NULL,
                    macd REAL NOT NULL,
                    signal REAL NOT NULL,
                    histogram REAL NOT NULL,
                    bullish_cross BOOLEAN NOT NULL,
                    bearish_cross BOOLEAN NOT NULL,
                    trend TEXT NOT NULL,
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_macd_symbol 
                ON macd_scanner(symbol)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_macd_bullish 
                ON macd_scanner(bullish_cross, scanned_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_macd_bearish 
                ON macd_scanner(bearish_cross, scanned_at DESC)
            """)
            
            # Volume-Price Break Scanner table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vpb_scanner (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    price_change REAL NOT NULL,
                    price_change_pct REAL NOT NULL,
                    buy_signal BOOLEAN NOT NULL,
                    sell_signal BOOLEAN NOT NULL,
                    volume_surge BOOLEAN NOT NULL,
                    current_volume INTEGER NOT NULL,
                    volume_ma30 INTEGER NOT NULL,
                    volume_surge_pct REAL NOT NULL,
                    highest_high_7 REAL NOT NULL,
                    lowest_low_7 REAL NOT NULL,
                    breakout_distance_pct REAL NOT NULL,
                    breakdown_distance_pct REAL NOT NULL,
                    pattern TEXT NOT NULL,
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vpb_symbol 
                ON vpb_scanner(symbol)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vpb_buy 
                ON vpb_scanner(buy_signal, scanned_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vpb_sell 
                ON vpb_scanner(sell_signal, scanned_at DESC)
            """)
            
            # Metadata table for tracking last update
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def upsert_watchlist(self, data: List[Dict]):
        """Insert or update watchlist data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute("""
                    INSERT INTO watchlist (symbol, price, daily_change, daily_change_pct, volume, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(symbol) DO UPDATE SET
                        price = excluded.price,
                        daily_change = excluded.daily_change,
                        daily_change_pct = excluded.daily_change_pct,
                        volume = excluded.volume,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    item['symbol'],
                    item['price'],
                    item['daily_change'],
                    item['daily_change_pct'],
                    item['volume']
                ))
            
            conn.commit()
            logger.info(f"Updated {len(data)} watchlist items")
    
    def get_watchlist(self, order_by: str = 'daily_change_pct') -> List[Dict]:
        """Get watchlist data sorted by specified column"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Validate order_by to prevent SQL injection
            allowed_orders = ['daily_change_pct', 'volume', 'symbol', 'price']
            if order_by not in allowed_orders:
                order_by = 'daily_change_pct'
            
            cursor.execute(f"""
                SELECT symbol, price, daily_change, daily_change_pct, volume, updated_at
                FROM watchlist
                ORDER BY {order_by} DESC
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def insert_whale_flows(self, flows: List[Dict]):
        """Insert new whale flows (keeps all historical flows)"""
        if not flows:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for flow in flows:
                cursor.execute("""
                    INSERT INTO whale_flows 
                    (symbol, type, strike, whale_score, volume, open_interest, vol_oi, 
                     premium, delta, expiry, dte, detected_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    flow['symbol'],
                    flow['type'],
                    flow['strike'],
                    flow['whale_score'],
                    flow['volume'],
                    flow.get('open_interest', 0),
                    flow['vol_oi'],
                    flow['premium'],
                    flow['delta'],
                    flow['expiry'].strftime('%Y-%m-%d') if isinstance(flow['expiry'], datetime) else flow['expiry'],
                    flow['dte'],
                    flow.get('timestamp', datetime.now())
                ))
            
            conn.commit()
            logger.info(f"Inserted {len(flows)} whale flows")
    
    def get_whale_flows(self, sort_by: str = 'score', limit: int = 10, 
                        hours_lookback: int = 6) -> List[Dict]:
        """
        Get whale flows sorted by score or recency
        
        Args:
            sort_by: 'score' or 'time'
            limit: Number of results to return
            hours_lookback: Only return flows from last N hours
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if sort_by == 'time':
                order_clause = "detected_at DESC"
            else:
                order_clause = "whale_score DESC"
            
            cursor.execute(f"""
                SELECT 
                    symbol, type, strike, whale_score, volume, open_interest,
                    vol_oi, premium, delta, expiry, dte, detected_at
                FROM whale_flows
                WHERE detected_at >= datetime('now', '-{hours_lookback} hours')
                ORDER BY {order_clause}
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def cleanup_old_whale_flows(self, days_to_keep: int = 1):
        """Remove whale flows older than specified days"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM whale_flows
                WHERE detected_at < datetime('now', '-' || ? || ' days')
            """, (days_to_keep,))
            
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted} old whale flows")
    
    def get_last_update_time(self, key: str) -> Optional[str]:
        """Get last update timestamp for a specific cache key"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value, updated_at
                FROM cache_metadata
                WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def set_last_update_time(self, key: str, value: str = None):
        """Set last update timestamp for a cache key"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if value is None:
                value = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO cache_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value))
            
            conn.commit()
    
    def set_macd_scanner(self, results: Dict):
        """Store MACD scanner results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Store all signals
            for item in results['all_signals']:
                cursor.execute("""
                    INSERT INTO macd_scanner (
                        symbol, price, price_change, price_change_pct,
                        macd, signal, histogram, bullish_cross, bearish_cross,
                        trend, scanned_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(symbol) DO UPDATE SET
                        price = excluded.price,
                        price_change = excluded.price_change,
                        price_change_pct = excluded.price_change_pct,
                        macd = excluded.macd,
                        signal = excluded.signal,
                        histogram = excluded.histogram,
                        bullish_cross = excluded.bullish_cross,
                        bearish_cross = excluded.bearish_cross,
                        trend = excluded.trend,
                        scanned_at = CURRENT_TIMESTAMP
                """, (
                    item['symbol'],
                    item['price'],
                    item['price_change'],
                    item['price_change_pct'],
                    item['macd'],
                    item['signal'],
                    item['histogram'],
                    item['bullish_cross'],
                    item['bearish_cross'],
                    item['trend']
                ))
            
            conn.commit()
            
            # Update metadata
            self.set_metadata('macd_scanner_last_update', datetime.now().isoformat())
            self.set_metadata('macd_bullish_count', str(len(results['bullish_crosses'])))
            self.set_metadata('macd_bearish_count', str(len(results['bearish_crosses'])))
            
            logger.info(f"Stored MACD scanner results: {len(results['all_signals'])} stocks")
    
    def get_macd_scanner(self, filter_type: str = 'all') -> List[Dict]:
        """
        Get MACD scanner results
        filter_type: 'all', 'bullish', 'bearish'
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if filter_type == 'bullish':
                cursor.execute("""
                    SELECT * FROM macd_scanner
                    WHERE bullish_cross = 1
                    ORDER BY price_change_pct DESC
                """)
            elif filter_type == 'bearish':
                cursor.execute("""
                    SELECT * FROM macd_scanner
                    WHERE bearish_cross = 1
                    ORDER BY price_change_pct ASC
                """)
            else:
                cursor.execute("""
                    SELECT * FROM macd_scanner
                    ORDER BY scanned_at DESC
                """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def set_vpb_scanner(self, results: Dict) -> None:
        """
        Store Volume-Price Break scanner results
        results: dict with 'all_signals', 'bullish_breakouts', 'bearish_breakdowns'
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM vpb_scanner")
            
            # Insert new results
            for signal in results.get('all_signals', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO vpb_scanner (
                        symbol, price, price_change, price_change_pct,
                        buy_signal, sell_signal, volume_surge,
                        current_volume, volume_ma30, volume_surge_pct,
                        highest_high_7, lowest_low_7,
                        breakout_distance_pct, breakdown_distance_pct,
                        pattern, scanned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal['symbol'],
                    signal['price'],
                    signal['price_change'],
                    signal['price_change_pct'],
                    1 if signal['buy_signal'] else 0,
                    1 if signal['sell_signal'] else 0,
                    1 if signal['volume_surge'] else 0,
                    signal['current_volume'],
                    signal['volume_ma30'],
                    signal['volume_surge_pct'],
                    signal['highest_high_7'],
                    signal['lowest_low_7'],
                    signal['breakout_distance_pct'],
                    signal['breakdown_distance_pct'],
                    signal['pattern'],
                    signal['scanned_at']
                ))
            
            conn.commit()
            
            # Update metadata
            self.set_metadata('vpb_last_scan', datetime.now().isoformat())
            self.set_metadata('vpb_bullish_count', str(len(results['bullish_breakouts'])))
            self.set_metadata('vpb_bearish_count', str(len(results['bearish_breakdowns'])))
            
            logger.info(f"Stored VPB scanner results: {len(results['all_signals'])} stocks")
    
    def get_vpb_scanner(self, filter_type: str = 'all') -> List[Dict]:
        """
        Get VPB scanner results
        filter_type: 'all', 'bullish', 'bearish'
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if filter_type == 'bullish':
                cursor.execute("""
                    SELECT * FROM vpb_scanner
                    WHERE buy_signal = 1
                    ORDER BY volume_surge_pct DESC
                """)
            elif filter_type == 'bearish':
                cursor.execute("""
                    SELECT * FROM vpb_scanner
                    WHERE sell_signal = 1
                    ORDER BY volume_surge_pct DESC
                """)
            else:
                cursor.execute("""
                    SELECT * FROM vpb_scanner
                    ORDER BY scanned_at DESC
                """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Watchlist count
            cursor.execute("SELECT COUNT(*) as count FROM watchlist")
            watchlist_count = cursor.fetchone()['count']
            
            # Whale flows count
            cursor.execute("SELECT COUNT(*) as count FROM whale_flows")
            whale_count = cursor.fetchone()['count']
            
            # MACD scanner count
            cursor.execute("SELECT COUNT(*) as count FROM macd_scanner")
            macd_count = cursor.fetchone()['count']
            
            # Last updates
            cursor.execute("""
                SELECT key, value, updated_at 
                FROM cache_metadata
                ORDER BY updated_at DESC
            """)
            metadata = [dict(row) for row in cursor.fetchall()]
            
            return {
                'watchlist_count': watchlist_count,
                'whale_flows_count': whale_count,
                'macd_scanner_count': macd_count,
                'last_updates': metadata
            }
