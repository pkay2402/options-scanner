#!/usr/bin/env python3
"""
Market Sentiment Worker - Fetches SPY/QQQ options data and calculates market sentiment
Runs every 5 minutes during market hours
"""
import os
import sys
import time
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/options-scanner/logs/sentiment_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketSentimentWorker:
    def __init__(self):
        self.client = SchwabClient()
        self.db_path = '/root/options-scanner/data/market_cache.db'
        self.symbols = ['SPY', 'QQQ']
        self.init_database()
        
    def init_database(self):
        """Initialize database table for sentiment data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price REAL,
                    total_call_volume INTEGER,
                    total_put_volume INTEGER,
                    pc_volume_ratio REAL,
                    total_call_oi INTEGER,
                    total_put_oi INTEGER,
                    pc_oi_ratio REAL,
                    net_gamma REAL,
                    call_gamma REAL,
                    put_gamma REAL,
                    iv_rank REAL,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    data_quality TEXT
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp 
                ON market_sentiment(symbol, timestamp DESC)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_next_n_expiries(self, n=5):
        """Get next N trading days (daily expiries for SPY/QQQ)"""
        expiries = []
        today = datetime.now().date()
        current = today
        
        while len(expiries) < n:
            # Skip weekends
            if current.weekday() < 5:  # Monday=0, Friday=4
                expiries.append(current)
            current += timedelta(days=1)
        
        return expiries
    
    def calculate_sentiment(self, symbol, options_data, underlying_price):
        """Calculate market sentiment based on options metrics"""
        try:
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            call_gamma_total = 0
            put_gamma_total = 0
            iv_values = []
            
            # Process calls
            if 'callExpDateMap' in options_data:
                for exp_date, strikes in options_data['callExpDateMap'].items():
                    for strike_price, contracts in strikes.items():
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0) or 0
                            oi = contract.get('openInterest', 0) or 0
                            gamma = contract.get('gamma', 0) or 0
                            iv = contract.get('volatility', 0) or 0
                            
                            total_call_volume += volume
                            total_call_oi += oi
                            call_gamma_total += gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            if iv > 0:
                                iv_values.append(iv)
            
            # Process puts
            if 'putExpDateMap' in options_data:
                for exp_date, strikes in options_data['putExpDateMap'].items():
                    for strike_price, contracts in strikes.items():
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0) or 0
                            oi = contract.get('openInterest', 0) or 0
                            gamma = contract.get('gamma', 0) or 0
                            iv = contract.get('volatility', 0) or 0
                            
                            total_put_volume += volume
                            total_put_oi += oi
                            put_gamma_total += gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            if iv > 0:
                                iv_values.append(iv)
            
            # Calculate ratios
            pc_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            net_gamma = call_gamma_total - put_gamma_total
            avg_iv = sum(iv_values) / len(iv_values) if iv_values else 0
            
            # Calculate sentiment score (0-100)
            # Lower P/C ratio = more bullish, Higher net gamma = more bullish
            volume_score = max(0, min(100, (1.2 - pc_volume_ratio) * 100))  # 0.8 P/C = 40, 0.5 P/C = 70
            oi_score = max(0, min(100, (1.2 - pc_oi_ratio) * 100))
            gamma_score = 50 + (net_gamma / abs(net_gamma + 1)) * 50 if net_gamma != 0 else 50
            
            sentiment_score = (volume_score * 0.4 + oi_score * 0.3 + gamma_score * 0.3)
            
            # Determine sentiment label
            if sentiment_score >= 65:
                sentiment_label = "BULLISH"
            elif sentiment_score >= 55:
                sentiment_label = "SLIGHTLY_BULLISH"
            elif sentiment_score >= 45:
                sentiment_label = "NEUTRAL"
            elif sentiment_score >= 35:
                sentiment_label = "SLIGHTLY_BEARISH"
            else:
                sentiment_label = "BEARISH"
            
            # Data quality check
            total_volume = total_call_volume + total_put_volume
            if total_volume < 1000:
                data_quality = "LOW"
            elif total_volume < 10000:
                data_quality = "MEDIUM"
            else:
                data_quality = "HIGH"
            
            return {
                'symbol': symbol,
                'price': underlying_price,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'pc_volume_ratio': round(pc_volume_ratio, 3),
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'pc_oi_ratio': round(pc_oi_ratio, 3),
                'net_gamma': round(net_gamma, 2),
                'call_gamma': round(call_gamma_total, 2),
                'put_gamma': round(put_gamma_total, 2),
                'iv_rank': round(avg_iv * 100, 2),
                'sentiment_score': round(sentiment_score, 2),
                'sentiment_label': sentiment_label,
                'data_quality': data_quality
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return None
    
    def fetch_sentiment_for_symbol(self, symbol):
        """Fetch options data and calculate sentiment for a symbol"""
        try:
            logger.info(f"Fetching sentiment data for {symbol}...")
            
            # Authenticate
            if not self.client.authenticate():
                logger.error(f"Failed to authenticate for {symbol}")
                return None
            
            # Get quote for underlying price
            quote = self.client.get_quote(symbol)
            if not quote or symbol not in quote:
                logger.error(f"Failed to get quote for {symbol}")
                return None
            
            underlying_price = quote[symbol]['quote']['lastPrice']
            logger.info(f"{symbol} price: ${underlying_price:.2f}")
            
            # Get next 5 expiries
            expiries = self.get_next_n_expiries(5)
            from_date = expiries[0].strftime('%Y-%m-%d')
            to_date = expiries[-1].strftime('%Y-%m-%d')
            
            logger.info(f"Fetching options from {from_date} to {to_date}")
            
            # Fetch options chain
            options = self.client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                from_date=from_date,
                to_date=to_date
            )
            
            if not options or 'callExpDateMap' not in options:
                logger.error(f"Failed to get options chain for {symbol}")
                return None
            
            logger.info(f"Got options data for {symbol}")
            
            # Calculate sentiment
            sentiment = self.calculate_sentiment(symbol, options, underlying_price)
            
            if sentiment:
                # Store in database
                self.store_sentiment(sentiment)
                logger.info(f"{symbol} Sentiment: {sentiment['sentiment_label']} (Score: {sentiment['sentiment_score']:.1f})")
                return sentiment
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}", exc_info=True)
            return None
    
    def store_sentiment(self, sentiment):
        """Store sentiment data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_sentiment (
                    symbol, price, total_call_volume, total_put_volume, pc_volume_ratio,
                    total_call_oi, total_put_oi, pc_oi_ratio, net_gamma, call_gamma, put_gamma,
                    iv_rank, sentiment_score, sentiment_label, data_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sentiment['symbol'],
                sentiment['price'],
                sentiment['total_call_volume'],
                sentiment['total_put_volume'],
                sentiment['pc_volume_ratio'],
                sentiment['total_call_oi'],
                sentiment['total_put_oi'],
                sentiment['pc_oi_ratio'],
                sentiment['net_gamma'],
                sentiment['call_gamma'],
                sentiment['put_gamma'],
                sentiment['iv_rank'],
                sentiment['sentiment_score'],
                sentiment['sentiment_label'],
                sentiment['data_quality']
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored sentiment for {sentiment['symbol']}")
            
        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")
    
    def is_market_hours(self):
        """Check if current time is within market hours (9:30 AM - 4:00 PM ET)"""
        now = datetime.now()
        # Convert to ET (approximately)
        et_hour = (now.hour - 5) % 24  # Rough ET conversion
        
        # Market hours: 9:30 AM to 4:00 PM ET (weekdays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        if et_hour < 9 or (et_hour == 9 and now.minute < 30):
            return False
        if et_hour >= 16:
            return False
        
        return True
    
    def run(self):
        """Main run loop - fetch sentiment every 5 minutes"""
        logger.info("Market Sentiment Worker started")
        
        while True:
            try:
                if self.is_market_hours():
                    logger.info("Market hours - fetching sentiment data...")
                    
                    # Fetch both symbols in parallel
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = {
                            executor.submit(self.fetch_sentiment_for_symbol, symbol): symbol 
                            for symbol in self.symbols
                        }
                        
                        for future in as_completed(futures):
                            symbol = futures[future]
                            try:
                                result = future.result()
                                if result:
                                    logger.info(f"✓ {symbol} sentiment updated")
                            except Exception as e:
                                logger.error(f"Error processing {symbol}: {e}")
                    
                    logger.info("Sleeping for 5 minutes...")
                    time.sleep(300)  # 5 minutes
                else:
                    logger.info("Outside market hours - sleeping for 30 minutes")
                    time.sleep(1800)  # 30 minutes
                    
            except KeyboardInterrupt:
                logger.info("Shutting down Market Sentiment Worker...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                logger.info("Sleeping for 1 minute before retry...")
                time.sleep(60)

if __name__ == '__main__':
    worker = MarketSentimentWorker()
    worker.run()
