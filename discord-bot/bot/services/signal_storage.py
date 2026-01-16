"""
Signal Storage Service
Stores and retrieves trading signals from all scanners
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SignalStorage:
    """Store and retrieve trading signals with 5-day rolling window"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to discord-bot/data/signals.db
            db_path = Path(__file__).parent.parent.parent / "data" / "signals.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_subtype TEXT,
                    direction TEXT,
                    price REAL,
                    data JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON signals(symbol, timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON signals(timestamp DESC)
            """)
            
            conn.commit()
            logger.info(f"Signal storage initialized at {self.db_path}")
    
    def store_signal(self, 
                     symbol: str,
                     signal_type: str,
                     signal_subtype: str = None,
                     direction: str = None,
                     price: float = None,
                     data: dict = None):
        """
        Store a trading signal
        
        Args:
            symbol: Stock symbol
            signal_type: Type of signal (WHALE, ZSCORE, TOS, ETF_MOMENTUM)
            signal_subtype: Subtype (e.g., BUY, SELL, OVERBOUGHT, etc.)
            direction: Direction (BULLISH, BEARISH, NEUTRAL)
            price: Price at time of signal
            data: Additional data as dict
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO signals (symbol, signal_type, signal_subtype, direction, price, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol.upper(),
                    signal_type,
                    signal_subtype,
                    direction,
                    price,
                    json.dumps(data) if data else None
                ))
                conn.commit()
                logger.info(f"Stored {signal_type} signal for {symbol}")
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    def get_signals(self, 
                    symbol: str = None,
                    signal_type: str = None,
                    days: int = 5) -> List[Dict]:
        """
        Retrieve signals
        
        Args:
            symbol: Filter by symbol (optional)
            signal_type: Filter by signal type (optional)
            days: Number of days to look back
            
        Returns:
            List of signal dictionaries
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = "SELECT * FROM signals WHERE timestamp >= ?"
            params = [cutoff_date]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                signals = []
                for row in cursor:
                    signal = dict(row)
                    if signal['data']:
                        signal['data'] = json.loads(signal['data'])
                    signals.append(signal)
                
                return signals
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            return []
    
    def cleanup_old_signals(self, days: int = 5):
        """Remove signals older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "DELETE FROM signals WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted = result.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old signals")
                
                return deleted
        except Exception as e:
            logger.error(f"Error cleaning up signals: {e}")
            return 0
    
    def get_summary(self, symbol: str, days: int = 1) -> Dict:
        """
        Get aggregated summary for a symbol
        
        Returns:
            Dictionary with signal counts and latest signals by type
        """
        signals = self.get_signals(symbol=symbol, days=days)
        
        summary = {
            'symbol': symbol.upper(),
            'period_days': days,
            'total_signals': len(signals),
            'by_type': {},
            'by_direction': {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0},
            'latest_signals': [],
            'price_range': {'high': None, 'low': None, 'current': None}
        }
        
        # Aggregate by type
        for signal in signals:
            sig_type = signal['signal_type']
            direction = signal.get('direction')
            
            if sig_type not in summary['by_type']:
                summary['by_type'][sig_type] = {
                    'count': 0,
                    'subtypes': {},
                    'latest': None
                }
            
            summary['by_type'][sig_type]['count'] += 1
            
            # Track subtypes
            subtype = signal.get('signal_subtype')
            if subtype:
                if subtype not in summary['by_type'][sig_type]['subtypes']:
                    summary['by_type'][sig_type]['subtypes'][subtype] = 0
                summary['by_type'][sig_type]['subtypes'][subtype] += 1
            
            # Track direction
            if direction in summary['by_direction']:
                summary['by_direction'][direction] += 1
            
            # Store latest of each type
            if summary['by_type'][sig_type]['latest'] is None:
                summary['by_type'][sig_type]['latest'] = signal
            
            # Track price range
            if signal.get('price'):
                price = signal['price']
                if summary['price_range']['high'] is None or price > summary['price_range']['high']:
                    summary['price_range']['high'] = price
                if summary['price_range']['low'] is None or price < summary['price_range']['low']:
                    summary['price_range']['low'] = price
        
        # Get latest 5 signals
        summary['latest_signals'] = signals[:5]
        
        # Current price is most recent signal with price
        for signal in signals:
            if signal.get('price'):
                summary['price_range']['current'] = signal['price']
                break
        
        return summary
    
    def get_stock_activity_timeline(self, symbol: str, days: int = 5) -> List[Dict]:
        """Get chronological timeline of all activity for a stock"""
        signals = self.get_signals(symbol=symbol, days=days)
        
        # Group by date
        timeline = {}
        for signal in signals:
            date = signal['timestamp'].split(' ')[0]
            if date not in timeline:
                timeline[date] = []
            timeline[date].append(signal)
        
        # Sort by date (newest first)
        sorted_timeline = []
        for date in sorted(timeline.keys(), reverse=True):
            sorted_timeline.append({
                'date': date,
                'signals': sorted(timeline[date], key=lambda x: x['timestamp'], reverse=True)
            })
        
        return sorted_timeline

# Global instance
_storage = None

def get_storage() -> SignalStorage:
    """Get or create global storage instance"""
    global _storage
    if _storage is None:
        _storage = SignalStorage()
    return _storage
