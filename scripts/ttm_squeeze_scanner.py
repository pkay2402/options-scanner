#!/usr/bin/env python3
"""
TTM Squeeze Scanner Service
Scans watchlist for TTM Squeeze setups and fires
Identifies low volatility compression followed by explosive breakouts
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
        logging.FileHandler('logs/ttm_squeeze_scanner.log'),
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


def calculate_ttm_squeeze(df: pd.DataFrame, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5) -> Dict:
    """
    Calculate TTM Squeeze indicator
    
    Squeeze ON: When Bollinger Bands are inside Keltner Channels (low volatility)
    Squeeze OFF: When Bollinger Bands break outside Keltner Channels (breakout)
    
    Parameters:
    - bb_length: Bollinger Bands period (default 20)
    - bb_mult: Bollinger Bands standard deviation multiplier (default 2.0)
    - kc_length: Keltner Channels period (default 20)
    - kc_mult: Keltner Channels ATR multiplier (default 1.5)
    
    Returns: dict with squeeze status, duration, momentum, and recent fire info
    """
    try:
        if len(df) < max(bb_length, kc_length) + 20:
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate Bollinger Bands
        bb_basis = close.rolling(window=bb_length).mean()
        bb_std = close.rolling(window=bb_length).std()
        bb_upper = bb_basis + (bb_mult * bb_std)
        bb_lower = bb_basis - (bb_mult * bb_std)
        
        # Calculate Keltner Channels
        kc_basis = close.rolling(window=kc_length).mean()
        
        # True Range for ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=kc_length).mean()
        
        kc_upper = kc_basis + (kc_mult * atr)
        kc_lower = kc_basis - (kc_mult * atr)
        
        # Squeeze condition: BB inside KC
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Calculate momentum (Linear Regression)
        # Use close price minus moving average, normalized
        momentum = close - close.rolling(window=kc_length).mean()
        momentum = momentum / atr  # Normalize by ATR
        
        # Current status
        current_squeeze = squeeze_on.iloc[-1]
        current_momentum = momentum.iloc[-1]
        current_price = close.iloc[-1]
        
        # Detect squeeze fire (OFF after being ON)
        # Look back up to 5 days for recent fires
        recent_fire = False
        fire_date = None
        fire_direction = None
        squeeze_duration = 0
        
        # Count how many consecutive days squeeze has been ON
        if current_squeeze:
            for i in range(len(squeeze_on) - 1, -1, -1):
                if squeeze_on.iloc[i]:
                    squeeze_duration += 1
                else:
                    break
        
        # Check for recent fires (last 5 days)
        for i in range(1, min(6, len(squeeze_on))):
            if not squeeze_on.iloc[-i] and squeeze_on.iloc[-(i+1)]:
                # Squeeze turned OFF (fired)
                recent_fire = True
                fire_date = df.index[-i]
                # Determine direction based on momentum at fire time
                fire_momentum = momentum.iloc[-i]
                fire_direction = 'bullish' if fire_momentum > 0 else 'bearish'
                break
        
        # Determine current signal
        if current_squeeze:
            signal = 'active'  # Squeeze is ON, waiting for breakout
        elif recent_fire:
            signal = 'fired'  # Recently fired, breakout happening
        else:
            signal = 'none'  # No squeeze
        
        return {
            'signal': signal,
            'squeeze_on': current_squeeze,
            'momentum': current_momentum,
            'momentum_direction': 'bullish' if current_momentum > 0 else 'bearish',
            'squeeze_duration': squeeze_duration,
            'recent_fire': recent_fire,
            'fire_date': fire_date.strftime('%Y-%m-%d') if fire_date is not None else None,
            'fire_direction': fire_direction,
            'price': current_price,
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'kc_upper': kc_upper.iloc[-1],
            'kc_lower': kc_lower.iloc[-1]
        }
        
    except Exception as e:
        logger.error(f"Error calculating TTM squeeze: {e}")
        return None


def scan_stock(symbol: str, client: SchwabClient) -> Dict:
    """Scan a single stock for TTM Squeeze"""
    try:
        logger.info(f"Scanning {symbol}...")
        
        # Get 60 days of daily data (need enough for calculations)
        price_history = client.get_price_history(
            symbol=symbol,
            period_type='month',
            period=3,  # 3 months to ensure 60+ trading days
            frequency_type='daily',
            frequency=1
        )
        
        if not price_history or 'candles' not in price_history:
            logger.warning(f"No price history for {symbol}")
            return None
        
        # Convert to DataFrame
        candles = price_history['candles']
        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Take last 60 trading days
        df = df.tail(60)
        
        if len(df) < 40:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
            return None
        
        # Calculate TTM Squeeze
        squeeze_data = calculate_ttm_squeeze(df)
        
        if not squeeze_data:
            return None
        
        return {
            'symbol': symbol,
            **squeeze_data
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None


def scan_watchlist(watchlist: List[str]) -> List[Dict]:
    """Scan entire watchlist for TTM Squeeze setups"""
    logger.info(f"Starting TTM Squeeze scan for {len(watchlist)} symbols...")
    
    client = SchwabClient()
    results = []
    
    for i, symbol in enumerate(watchlist):
        try:
            result = scan_stock(symbol, client)
            if result:
                results.append(result)
                logger.info(
                    f"{symbol}: {result['signal'].upper()} - "
                    f"{'Squeeze ON' if result['squeeze_on'] else 'Squeeze OFF'} - "
                    f"Momentum: {result['momentum_direction']} ({result['momentum']:.4f})"
                )
            
            # Rate limiting
            time.sleep(0.3)  # 300ms between requests
            
            # Extra delay every 20 stocks
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{len(watchlist)} stocks, cooling down...")
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    logger.info(f"Scan complete. Found {len(results)} results")
    return results


def main():
    """Main scanner loop"""
    logger.info("TTM Squeeze Scanner starting...")
    
    # Initialize market cache
    cache = MarketCache()
    
    while True:
        try:
            # Run scan
            results = scan_watchlist(WATCHLIST)
            
            # Store results in database
            if results:
                cache.set_ttm_squeeze_scanner(results)
                
                # Log summary
                active_squeezes = [r for r in results if r['signal'] == 'active']
                recent_fires = [r for r in results if r['signal'] == 'fired']
                bullish_fires = [r for r in recent_fires if r['fire_direction'] == 'bullish']
                bearish_fires = [r for r in recent_fires if r['fire_direction'] == 'bearish']
                
                logger.info(
                    f"Scan summary: {len(active_squeezes)} active squeezes, "
                    f"{len(recent_fires)} recent fires "
                    f"({len(bullish_fires)} bullish, {len(bearish_fires)} bearish)"
                )
            
            # Wait 1 hour before next scan
            logger.info("Waiting 1 hour until next scan...")
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in scanner loop: {e}", exc_info=True)
            logger.info("Retrying in 5 minutes...")
            time.sleep(300)


if __name__ == "__main__":
    main()
