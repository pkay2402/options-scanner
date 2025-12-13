#!/usr/bin/env python3
"""
MACD Scanner Service
Scans watchlist for MACD bullish and bearish crossovers on daily charts
Runs continuously and updates database
"""

import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.data.market_cache import MarketCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/macd_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Comprehensive watchlist (150 stocks from market_data_worker)
WATCHLIST = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    
    # Large Cap Growth
    'CRM', 'NFLX', 'AMD', 'QCOM', 'INTC', 'CSCO', 'PYPL', 'SHOP', 'SQ', 'UBER',
    'LYFT', 'ABNB', 'COIN', 'RBLX', 'U', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS',
    
    # AI & Cloud
    'PLTR', 'AI', 'SMCI', 'ARM', 'MRVL', 'ANET', 'NOW', 'WDAY', 'MDB', 'PANW',
    'FTNT', 'OKTA', 'DDOG', 'S', 'TEAM', 'ZM', 'DOCN', 'GTLB', 'CFLT', 'ESTC',
    
    # Semiconductors
    'TSM', 'ASML', 'MU', 'LRCX', 'KLAC', 'AMAT', 'MCHP', 'ADI', 'NXPI', 'TXN',
    
    # EVs & Clean Energy
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'ENPH', 'SEDG', 'RUN',
    
    # Fintech & Payments
    'V', 'MA', 'HOOD', 'SOFI', 'AFRM', 'NU', 'MELI', 'SE', 'UPST',
    
    # E-commerce & Consumer
    'BABA', 'JD', 'PDD', 'BKNG', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA',
    
    # Entertainment & Gaming
    'DIS', 'WBD', 'PARA', 'EA', 'TTWO', 'RBLX', 'DKNG', 'PENN', 'LYV',
    
    # Health Tech
    'TDOC', 'DOCS', 'VEEV', 'DXCM', 'ISRG', 'PODD', 'ALGN', 'ILMN',
    
    # Communications
    'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR',
    
    # Traditional Value
    'XOM', 'CVX', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
    'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR',
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'CMG',
    'BA', 'CAT', 'DE', 'GE', 'MMM', 'HON', 'UPS', 'FDX',
    
    # Index ETFs
    'SPY', 'QQQ', 'IWM', 'DIA'
]


def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Dict:
    """
    Calculate MACD indicators
    Returns: dict with macd, signal, histogram, and crossover info
    """
    try:
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd - signal_line
        
        # Get last 2 values to detect crossovers
        macd_current = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_current = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        
        # Detect crossovers
        bullish_cross = False
        bearish_cross = False
        
        if macd_prev <= signal_prev and macd_current > signal_current:
            bullish_cross = True
        elif macd_prev >= signal_prev and macd_current < signal_current:
            bearish_cross = True
        
        return {
            'macd': macd_current,
            'signal': signal_current,
            'histogram': histogram.iloc[-1],
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross,
            'trend': 'bullish' if macd_current > signal_current else 'bearish'
        }
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None


def scan_stock(client: SchwabClient, symbol: str) -> Dict:
    """
    Scan a single stock for MACD signals
    """
    try:
        # Get daily data using period (avoids timestamp issues)
        price_history = client.get_price_history(
            symbol=symbol,
            period_type='month',
            period=3,  # 3 months of data
            frequency_type='daily',
            frequency=1,
            need_extended_hours=False
        )
        
        if not price_history or 'candles' not in price_history or not price_history['candles']:
            logger.warning(f"No price history for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(price_history['candles'])
        
        if len(df) < 30:  # Need at least 30 days for MACD
            logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
            return None
        
        # Calculate MACD
        macd_data = calculate_macd(df['close'])
        
        if not macd_data:
            return None
        
        # Get current price and change
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        return {
            'symbol': symbol,
            'price': float(current_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'macd': float(macd_data['macd']),
            'signal': float(macd_data['signal']),
            'histogram': float(macd_data['histogram']),
            'bullish_cross': macd_data['bullish_cross'],
            'bearish_cross': macd_data['bearish_cross'],
            'trend': macd_data['trend'],
            'scanned_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None


def scan_watchlist():
    """
    Scan entire watchlist for MACD signals
    """
    logger.info("Starting MACD scan...")
    
    client = SchwabClient()
    if not client.authenticate():
        logger.error("Failed to authenticate with Schwab API")
        return None
    
    cache = MarketCache()
    results = {
        'bullish_crosses': [],
        'bearish_crosses': [],
        'all_signals': []
    }
    
    for i, symbol in enumerate(WATCHLIST):
        try:
            logger.info(f"Scanning {symbol} ({i+1}/{len(WATCHLIST)})")
            
            scan_result = scan_stock(client, symbol)
            
            if scan_result:
                results['all_signals'].append(scan_result)
                
                if scan_result['bullish_cross']:
                    results['bullish_crosses'].append(scan_result)
                    logger.info(f"🟢 BULLISH CROSS: {symbol} @ ${scan_result['price']:.2f}")
                
                if scan_result['bearish_cross']:
                    results['bearish_crosses'].append(scan_result)
                    logger.info(f"🔴 BEARISH CROSS: {symbol} @ ${scan_result['price']:.2f}")
            
            # Rate limiting - reduced for faster scans
            if (i + 1) % 30 == 0:
                logger.info(f"Processed {i+1}/{len(WATCHLIST)}, sleeping for rate limit...")
                time.sleep(1)
            else:
                time.sleep(0.3)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    # Store results in cache
    cache.set_macd_scanner(results)
    
    logger.info(f"Scan complete: {len(results['bullish_crosses'])} bullish, {len(results['bearish_crosses'])} bearish")
    
    return results


def main():
    """Main scanner loop"""
    logger.info("🚀 MACD Scanner Service starting...")
    
    # Run initial scan
    scan_watchlist()
    
    # Run every 1 hour
    while True:
        try:
            sleep_time = 3600  # 1 hour
            
            logger.info(f"Sleeping for {sleep_time/60:.0f} minutes...")
            time.sleep(sleep_time)
            
            scan_watchlist()
            
        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
