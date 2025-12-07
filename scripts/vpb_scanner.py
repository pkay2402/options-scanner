#!/usr/bin/env python3
"""
Volume-Price Break (VPB) Scanner Service
Detects bullish breakouts and bearish breakdowns based on volume surge and price action

Buy Signal (Bullish):
- Current volume > 30-day volume MA
- Current close > highest high of last 7 days (excluding current bar)

Sell Signal (Bearish):
- Current volume > 30-day volume MA
- Current close < lowest low of last 7 days (excluding current bar)
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
        logging.FileHandler('logs/vpb_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Comprehensive watchlist (150 stocks)
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


def calculate_vpb_signals(df: pd.DataFrame) -> Dict:
    """
    Calculate Volume-Price Break signals
    
    Returns: dict with buy/sell signals and related metrics
    """
    try:
        if len(df) < 30:
            return None
        
        # Calculate 30-day volume MA
        df['volume_ma30'] = df['volume'].rolling(window=30).mean()
        
        # Check volume surge (current volume > 30-day MA)
        current_volume = df['volume'].iloc[-1]
        volume_ma30 = df['volume_ma30'].iloc[-1]
        volume_surge = current_volume > volume_ma30
        
        # Calculate highest high of last 7 days (excluding current bar)
        highest_high_7 = df['high'].iloc[-8:-1].max()  # Last 7 bars before current
        
        # Calculate lowest low of last 7 days (excluding current bar)
        lowest_low_7 = df['low'].iloc[-8:-1].min()  # Last 7 bars before current
        
        # Get current bar data
        current_close = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # Check price breakout (close > highest high of last 7 days)
        price_breakout = current_close > highest_high_7
        
        # Check price breakdown (close < lowest low of last 7 days)
        price_breakdown = current_close < lowest_low_7
        
        # Generate signals
        buy_signal = volume_surge and price_breakout
        sell_signal = volume_surge and price_breakdown
        
        # Calculate volume surge percentage
        volume_surge_pct = ((current_volume - volume_ma30) / volume_ma30) * 100 if volume_ma30 > 0 else 0
        
        # Calculate distance from breakout levels
        breakout_distance_pct = ((current_close - highest_high_7) / highest_high_7) * 100
        breakdown_distance_pct = ((current_close - lowest_low_7) / lowest_low_7) * 100
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'volume_surge': volume_surge,
            'current_volume': int(current_volume),
            'volume_ma30': int(volume_ma30),
            'volume_surge_pct': float(volume_surge_pct),
            'current_close': float(current_close),
            'highest_high_7': float(highest_high_7),
            'lowest_low_7': float(lowest_low_7),
            'breakout_distance_pct': float(breakout_distance_pct),
            'breakdown_distance_pct': float(breakdown_distance_pct),
            'pattern': 'bullish_breakout' if buy_signal else ('bearish_breakdown' if sell_signal else 'none')
        }
        
    except Exception as e:
        logger.error(f"Error calculating VPB signals: {e}")
        return None


def scan_stock(client: SchwabClient, symbol: str) -> Dict:
    """
    Scan a single stock for Volume-Price Break signals
    """
    try:
        # Get 60 days of daily data (need 30+ for volume MA)
        price_history = client.get_price_history(
            symbol=symbol,
            period_type='month',
            period=2,  # 2 months of data
            frequency_type='daily',
            frequency=1,
            need_extended_hours=False
        )
        
        if not price_history or 'candles' not in price_history or not price_history['candles']:
            logger.warning(f"No price history for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(price_history['candles'])
        
        if len(df) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
            return None
        
        # Calculate VPB signals
        vpb_data = calculate_vpb_signals(df)
        
        if not vpb_data:
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
            'buy_signal': vpb_data['buy_signal'],
            'sell_signal': vpb_data['sell_signal'],
            'volume_surge': vpb_data['volume_surge'],
            'current_volume': vpb_data['current_volume'],
            'volume_ma30': vpb_data['volume_ma30'],
            'volume_surge_pct': vpb_data['volume_surge_pct'],
            'highest_high_7': vpb_data['highest_high_7'],
            'lowest_low_7': vpb_data['lowest_low_7'],
            'breakout_distance_pct': vpb_data['breakout_distance_pct'],
            'breakdown_distance_pct': vpb_data['breakdown_distance_pct'],
            'pattern': vpb_data['pattern'],
            'scanned_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None


def scan_watchlist():
    """
    Scan entire watchlist for Volume-Price Break signals
    """
    logger.info("Starting VPB scan...")
    
    client = SchwabClient()
    if not client.authenticate():
        logger.error("Failed to authenticate with Schwab API")
        return None
    
    cache = MarketCache()
    results = {
        'bullish_breakouts': [],
        'bearish_breakdowns': [],
        'all_signals': []
    }
    
    for i, symbol in enumerate(WATCHLIST):
        try:
            logger.info(f"Scanning {symbol} ({i+1}/{len(WATCHLIST)})")
            
            scan_result = scan_stock(client, symbol)
            
            if scan_result:
                results['all_signals'].append(scan_result)
                
                if scan_result['buy_signal']:
                    results['bullish_breakouts'].append(scan_result)
                    logger.info(f"🟢 BULLISH BREAKOUT: {symbol} @ ${scan_result['price']:.2f} (Vol: +{scan_result['volume_surge_pct']:.1f}%)")
                
                if scan_result['sell_signal']:
                    results['bearish_breakdowns'].append(scan_result)
                    logger.info(f"🔴 BEARISH BREAKDOWN: {symbol} @ ${scan_result['price']:.2f} (Vol: +{scan_result['volume_surge_pct']:.1f}%)")
            
            # Rate limiting
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i+1}/{len(WATCHLIST)}, sleeping for rate limit...")
                time.sleep(2)
            else:
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    # Store results in cache
    cache.set_vpb_scanner(results)
    
    logger.info(f"Scan complete: {len(results['bullish_breakouts'])} breakouts, {len(results['bearish_breakdowns'])} breakdowns")
    
    return results


def main():
    """Main scanner loop"""
    logger.info("🚀 Volume-Price Break Scanner Service starting...")
    
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
