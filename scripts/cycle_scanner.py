"""
Cycle Indicator Scanner
Scans watchlist for cycle peaks and bottoms using Ehlers methodology
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Comprehensive watchlist (150 stocks from MACD scanner)
WATCHLIST = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    
    # Large Cap Growth
    'CRM', 'NFLX', 'AMD', 'QCOM', 'INTC', 'CSCO', 'PYPL', 'SHOP', 'SQ', 'UBER',
    'LYFT', 'ABNB', 'COIN', 'RBLX', 'U', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS',
    
    # AI & Cloud
    'PLTR', 'AI', 'SMCI', 'ARM', 'MRVL', 'ANET', 'NOW', 'WDAY', 'MDB', 'PANW',
    'FTNT', 'OKTA', 'S', 'TEAM', 'ZM', 'DOCN', 'GTLB', 'CFLT', 'ESTC',
    
    # Semiconductors
    'TSM', 'ASML', 'MU', 'LRCX', 'KLAC', 'AMAT', 'MCHP', 'ADI', 'NXPI', 'TXN',
    
    # EVs & Clean Energy
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'ENPH', 'SEDG', 'RUN',
    
    # Fintech & Payments
    'V', 'MA', 'HOOD', 'SOFI', 'AFRM', 'NU', 'MELI', 'SE', 'UPST',
    
    # E-commerce & Consumer
    'BABA', 'JD', 'PDD', 'BKNG', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA',
    
    # Entertainment & Gaming
    'DIS', 'WBD', 'PARA', 'EA', 'TTWO', 'DKNG', 'PENN', 'LYV',
    
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
    'SPY', 'QQQ', 'IWM', 'DIA','IBIT','ETHA'
]

# Output file for results
RESULTS_FILE = Path(__file__).parent.parent / 'data' / 'cycle_signals.json'


def calculate_cycle_indicator(df, detrend_period=20):
    """Calculate Ehlers cycle indicator with phase and signals"""
    price = df['close'].values
    n = len(price)
    
    # Initialize arrays
    smooth = np.zeros(n)
    inphase = np.zeros(n)
    quadrature = np.zeros(n)
    phase = np.zeros(n)
    
    # Smooth prices
    for i in range(3, n):
        smooth[i] = (price[i] + 2*price[i-1] + 2*price[i-2] + price[i-3]) / 6
    
    # Calculate Hilbert Transform components
    for i in range(7, n):
        inphase[i] = (smooth[i] - smooth[i-7]) / 2
        
    for i in range(3, n):
        quadrature[i] = (smooth[i] - smooth[i-2] + 2*(smooth[i-1] - smooth[i-3])) / 4
    
    # Calculate phase
    for i in range(1, n):
        if inphase[i] != 0:
            raw_phase = np.arctan(quadrature[i] / inphase[i]) * 180 / np.pi
        else:
            raw_phase = 0
            
        # Accumulate phase
        if raw_phase < phase[i-1] - 270:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1] + 360)
        elif raw_phase > phase[i-1] + 270:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1] - 360)
        else:
            phase[i] = phase[i-1] + (raw_phase - phase[i-1])
    
    # Wrap phase to 0-360
    wrapped_phase = phase % 360
    
    # Detrended price
    detrended = np.zeros(n)
    for i in range(detrend_period, n):
        trend = np.mean(smooth[i-detrend_period:i])
        detrended[i] = smooth[i] - trend
    
    # Normalize detrended price
    normalized_cycle = np.zeros(n)
    for i in range(detrend_period*2, n):
        window = detrended[i-detrend_period:i]
        std = np.std(window)
        if std > 0:
            normalized_cycle[i] = detrended[i] / std
    
    # Calculate momentum
    momentum = np.zeros(n)
    for i in range(5, n):
        momentum[i] = smooth[i] - smooth[i-5]
    
    # Cycle strength
    cycle_strength = np.abs(normalized_cycle)
    
    # Add to dataframe
    df['smooth'] = smooth
    df['normalized_cycle'] = normalized_cycle
    df['phase'] = wrapped_phase
    df['momentum'] = momentum
    df['cycle_strength'] = cycle_strength
    
    # Local extrema detection
    window = 5
    df['is_local_max'] = False
    df['is_local_min'] = False
    
    for i in range(window, n - window):
        if normalized_cycle[i] == max(normalized_cycle[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('is_local_max')] = True
        if normalized_cycle[i] == min(normalized_cycle[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('is_local_min')] = True
    
    # Signal detection
    df['is_peak'] = (
        ((df['phase'] >= 315) | (df['phase'] <= 45)) & 
        (df['normalized_cycle'] > 1.5) & 
        (df['cycle_strength'] > 1.0) &
        df['is_local_max']
    )
    
    df['is_bottom'] = (
        ((df['phase'] >= 120) & (df['phase'] <= 240)) & 
        (df['normalized_cycle'] < -1.3) &
        (df['cycle_strength'] > 0.8) &
        df['is_local_min'] &
        (df['momentum'] < 0)
    )
    
    df['approaching_peak'] = (
        ((df['phase'] >= 270) & (df['phase'] < 315)) & 
        (df['normalized_cycle'] > 1.2) &
        (df['cycle_strength'] > 0.8)
    )
    
    df['approaching_bottom'] = (
        ((df['phase'] >= 75) & (df['phase'] < 120)) &
        (df['normalized_cycle'] < -1.0) &
        (df['cycle_strength'] > 0.6) &
        (df['momentum'] < 0)
    )
    
    return df


def scan_stock(symbol: str) -> dict:
    """Scan a single stock for cycle signals"""
    try:
        logger.info(f"Scanning {symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='6mo', interval='1d')
        
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Ensure timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df[['Close']].copy()
        df.columns = ['close']
        
        # Calculate indicator
        df = calculate_cycle_indicator(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Determine signal type
        signal_type = None
        signal_strength = 'NONE'
        action = 'HOLD'
        
        if latest['is_peak']:
            signal_type = 'PEAK'
            signal_strength = 'STRONG'
            action = 'SELL'
        elif latest['is_bottom']:
            signal_type = 'BOTTOM'
            signal_strength = 'STRONG'
            action = 'BUY'
        elif latest['approaching_peak']:
            signal_type = 'APPROACHING_PEAK'
            signal_strength = 'MODERATE'
            action = 'PREPARE_SELL'
        elif latest['approaching_bottom']:
            signal_type = 'APPROACHING_BOTTOM'
            signal_strength = 'MODERATE'
            action = 'PREPARE_BUY'
        
        # Only return if there's a signal
        if signal_type:
            return {
                'symbol': symbol,
                'price': float(latest['close']),
                'cycle_value': float(latest['normalized_cycle']),
                'phase': float(latest['phase']),
                'strength': float(latest['cycle_strength']),
                'momentum': float(latest['momentum']),
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'action': action,
                'scanned_at': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None


def scan_watchlist():
    """Scan entire watchlist for cycle signals"""
    logger.info("=" * 80)
    logger.info("CYCLE SCANNER STARTED")
    logger.info(f"Scanning {len(WATCHLIST)} stocks...")
    logger.info("=" * 80)
    
    signals = {
        'peak': [],
        'bottom': [],
        'approaching_peak': [],
        'approaching_bottom': [],
        'metadata': {
            'scan_time': datetime.now().isoformat(),
            'total_stocks': len(WATCHLIST),
            'stocks_scanned': 0
        }
    }
    
    for i, symbol in enumerate(WATCHLIST):
        try:
            result = scan_stock(symbol)
            
            if result:
                signal_type = result['signal_type']
                
                if signal_type == 'PEAK':
                    signals['peak'].append(result)
                    logger.info(f"🔴 PEAK SIGNAL: {symbol} @ ${result['price']:.2f} (cycle: {result['cycle_value']:.2f}σ)")
                elif signal_type == 'BOTTOM':
                    signals['bottom'].append(result)
                    logger.info(f"🟢 BOTTOM SIGNAL: {symbol} @ ${result['price']:.2f} (cycle: {result['cycle_value']:.2f}σ)")
                elif signal_type == 'APPROACHING_PEAK':
                    signals['approaching_peak'].append(result)
                    logger.info(f"🟠 APPROACHING PEAK: {symbol} @ ${result['price']:.2f}")
                elif signal_type == 'APPROACHING_BOTTOM':
                    signals['approaching_bottom'].append(result)
                    logger.info(f"🟢 APPROACHING BOTTOM: {symbol} @ ${result['price']:.2f}")
            
            signals['metadata']['stocks_scanned'] = i + 1
            
            # Rate limiting
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(WATCHLIST)} stocks scanned")
                time.sleep(1)  # Brief pause every 10 stocks
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(signals, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("SCAN COMPLETE")
    logger.info(f"Peak signals: {len(signals['peak'])}")
    logger.info(f"Bottom signals: {len(signals['bottom'])}")
    logger.info(f"Approaching peak: {len(signals['approaching_peak'])}")
    logger.info(f"Approaching bottom: {len(signals['approaching_bottom'])}")
    logger.info(f"Results saved to: {RESULTS_FILE}")
    logger.info("=" * 80)
    
    return signals


if __name__ == '__main__':
    scan_watchlist()
