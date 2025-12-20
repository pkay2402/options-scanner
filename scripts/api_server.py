"""
Simple REST API to serve cached market data
Runs on droplet and exposes data to Streamlit Cloud
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from functools import wraps
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_cache import MarketCache
from src.api.schwab_client import SchwabClient

app = Flask(__name__)
CORS(app)  # Allow requests from Streamlit Cloud

cache = MarketCache()
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Timeout decorator
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timeout")

def with_timeout(seconds=30):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set alarm signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable alarm
            return result
        return wrapper
    return decorator

@app.route('/')
def home():
    """API status endpoint"""
    stats = cache.get_cache_stats()
    return jsonify({
        'status': 'running',
        'service': 'Options Scanner API',
        'version': '1.0',
        'cache_stats': stats
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/watchlist')
def get_watchlist():
    """
    Get watchlist data
    Query params:
        - order_by: daily_change_pct (default), volume, symbol, price
        - limit: number of results (default 20)
    """
    order_by = request.args.get('order_by', 'daily_change_pct')
    limit = int(request.args.get('limit', 20))
    data = cache.get_watchlist(order_by=order_by)
    # Apply limit
    data = data[:limit] if limit > 0 else data
    return jsonify({
        'success': True,
        'count': len(data),
        'data': data
    })

@app.route('/api/whale_flows')
def get_whale_flows():
    """
    Get whale flows
    Query params:
        - sort_by: score (default) or time
        - limit: number of results (default 10)
        - hours: lookback hours (default 6)
    """
    sort_by = request.args.get('sort_by', 'score')
    limit = int(request.args.get('limit', 10))
    hours = int(request.args.get('hours', 6))
    
    flows = cache.get_whale_flows(
        sort_by=sort_by,
        limit=limit,
        hours_lookback=hours
    )
    
    return jsonify({
        'success': True,
        'count': len(flows),
        'sort_by': sort_by,
        'data': flows
    })

@app.route('/api/stats')
def get_stats():
    """Get cache statistics"""
    stats = cache.get_cache_stats()
    return jsonify({
        'success': True,
        'stats': stats
    })

@app.route('/api/last_update')
def get_last_update():
    """Get last update times"""
    watchlist_update = cache.get_last_update_time('watchlist')
    whale_flows_update = cache.get_last_update_time('whale_flows')
    
    return jsonify({
        'success': True,
        'watchlist': watchlist_update,
        'whale_flows': whale_flows_update
    })

@app.route('/api/macd_scanner')
def get_macd_scanner():
    """
    Get MACD scanner results
    Query params:
        - filter: all (default), bullish, bearish
        - limit: number of results (default 20)
    """
    filter_type = request.args.get('filter', 'all')
    limit = int(request.args.get('limit', 20))
    
    data = cache.get_macd_scanner(filter_type=filter_type)
    
    # Apply limit
    data = data[:limit] if limit > 0 else data
    
    return jsonify({
        'success': True,
        'filter': filter_type,
        'count': len(data),
        'data': data
    })

@app.route('/api/vpb_scanner')
def get_vpb_scanner():
    """
    Get Volume-Price Break scanner results
    Query params:
        - filter: all (default), bullish, bearish
        - limit: number of results (default 20)
    """
    filter_type = request.args.get('filter', 'all')
    limit = int(request.args.get('limit', 20))
    
    data = cache.get_vpb_scanner(filter_type=filter_type)
    
    # Apply limit
    data = data[:limit] if limit > 0 else data
    
    return jsonify({
        'success': True,
        'filter': filter_type,
        'count': len(data),
        'data': data
    })

@app.route('/api/ttm_squeeze_scanner')
def get_ttm_squeeze_scanner():
    """
    Get TTM Squeeze scanner results
    Query params:
        - filter: all (default), active, fired, bullish, bearish
        - limit: number of results (default 20)
    """
    filter_type = request.args.get('filter', 'all')
    limit = int(request.args.get('limit', 20))
    
    data = cache.get_ttm_squeeze_scanner(filter_type=filter_type)
    
    # Apply limit
    data = data[:limit] if limit > 0 else data
    
    return jsonify({
        'success': True,
        'filter': filter_type,
        'count': len(data),
        'data': data
    })

@app.route('/api/market_sentiment')
def get_market_sentiment():
    """
    Get market sentiment for SPY and QQQ
    Returns latest sentiment scores, labels, and metrics
    """
    conn = None
    try:
        # Query latest sentiment for SPY and QQQ with timeout
        import sqlite3
        conn = sqlite3.connect('/root/options-scanner/data/market_cache.db', timeout=5.0)
        cursor = conn.cursor()
        
        results = {}
        for symbol in ['SPY', 'QQQ']:
            cursor.execute('''
                SELECT symbol, timestamp, price, total_call_volume, total_put_volume, 
                       pc_volume_ratio, total_call_oi, total_put_oi, pc_oi_ratio,
                       net_gamma, call_gamma, put_gamma, iv_rank, 
                       sentiment_score, sentiment_label, data_quality
                FROM market_sentiment
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            row = cursor.fetchone()
            if row:
                results[symbol] = {
                    'symbol': row[0],
                    'timestamp': row[1],
                    'price': row[2],
                    'total_call_volume': row[3],
                    'total_put_volume': row[4],
                    'pc_volume_ratio': row[5],
                    'total_call_oi': row[6],
                    'total_put_oi': row[7],
                    'pc_oi_ratio': row[8],
                    'net_gamma': row[9],
                    'call_gamma': row[10],
                    'put_gamma': row[11],
                    'iv_rank': row[12],
                    'sentiment_score': row[13],
                    'sentiment_label': row[14],
                    'data_quality': row[15]
                }
        
        return jsonify({
            'success': True,
            'count': len(results),
            'data': results,
            'fetched_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching market sentiment: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/market_snapshot')
def get_market_snapshot():
    """
    Get market snapshot with price history and options chain
    Query params:
        - symbol: ticker symbol (required)
        - expiry: expiration date YYYY-MM-DD (required)
        - timeframe: intraday or daily (default: intraday)
    """
    symbol = request.args.get('symbol')
    expiry = request.args.get('expiry')
    timeframe = request.args.get('timeframe', 'intraday')
    
    if not symbol or not expiry:
        return jsonify({
            'success': False,
            'error': 'symbol and expiry parameters are required'
        }), 400
    
    client = SchwabClient()
    
    if not client.authenticate():
        return jsonify({
            'success': False,
            'error': 'Failed to authenticate with Schwab API'
        }), 500
    
    try:
        # Get quote
        quote = client.get_quote(symbol)
        if not quote:
            return jsonify({
                'success': False,
                'error': f'Failed to get quote for {symbol}'
            }), 404
        
        underlying_price = quote.get(symbol, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            return jsonify({
                'success': False,
                'error': f'No price found for {symbol}'
            }), 404
        
        # Get options chain
        options_chain = client.get_options_chain(
            symbol=symbol,
            from_date=expiry,
            to_date=expiry
        )
        
        # Get price history based on timeframe
        price_history = None
        if timeframe == 'intraday':
            # Get 2 trading days of 5-minute data (today + previous trading day)
            try:
                end_date = datetime.now()
                # Go back 5 days to ensure we get at least 2 trading days (accounting for weekends)
                start_date = end_date - timedelta(days=5)
                start_ms = int(start_date.timestamp() * 1000)
                end_ms = int(end_date.timestamp() * 1000)
                
                logger.info(f"Fetching intraday data for {symbol} from {start_date} to {end_date}")
                
                # Get data with extended hours, then filter manually
                price_history = client.get_price_history(
                    symbol=symbol,
                    frequency_type='minute',
                    frequency=5,
                    start_date=start_ms,
                    end_date=end_ms,
                    need_extended_hours=True  # Get all data, filter below
                )
                
                # ALWAYS filter to market hours (9:30 AM - 4:00 PM ET) regardless of source
                if price_history and 'candles' in price_history:
                    from collections import defaultdict
                    import pytz
                    eastern = pytz.timezone('America/New_York')
                    
                    all_candles = price_history['candles']
                    logger.info(f"Received {len(all_candles)} candles (with extended hours)")
                    
                    # Filter to market hours ONLY
                    market_hours_candles = []
                    for candle in all_candles:
                        dt = datetime.fromtimestamp(candle['datetime'] / 1000, tz=pytz.UTC)
                        dt_eastern = dt.astimezone(eastern)
                        hour = dt_eastern.hour
                        minute = dt_eastern.minute
                        
                        # Market hours: 9:30 AM to 4:00 PM ET
                        if (hour == 9 and minute >= 30) or (hour >= 10 and hour < 16):
                            market_hours_candles.append(candle)
                    
                    logger.info(f"Filtered to {len(market_hours_candles)} market-hours candles")
                    
                    # Now group by trading day and get last 2 days
                    days = defaultdict(list)
                    for candle in market_hours_candles:
                        dt = datetime.fromtimestamp(candle['datetime'] / 1000, tz=pytz.UTC)
                        dt_eastern = dt.astimezone(eastern)
                        day_key = dt_eastern.strftime('%Y-%m-%d')
                        days[day_key].append(candle)
                    
                    # Get last 2 trading days
                    trading_days = sorted(days.keys())[-2:] if len(days) >= 2 else sorted(days.keys())
                    filtered_candles = []
                    for day in trading_days:
                        filtered_candles.extend(days[day])
                    
                    price_history['candles'] = sorted(filtered_candles, key=lambda x: x['datetime'])
                    logger.info(f"Final: {len(filtered_candles)} candles across {len(trading_days)} trading days")
                
            except Exception as e:
                logger.error(f"Schwab intraday data failed for {symbol}: {e}")
                # Fallback to yfinance for intraday (2 days, 5-min intervals)
                try:
                    import yfinance as yf
                    yf_symbol = symbol.replace('$', '^') if symbol.startswith('$') else symbol
                    ticker = yf.Ticker(yf_symbol)
                    # Get 5 days of data to ensure we have at least 2 trading days
                    hist = ticker.history(period="5d", interval="5m")
                    
                    if not hist.empty:
                        # Group by date to get last 2 trading days, filter market hours
                        from collections import defaultdict
                        import pytz
                        eastern = pytz.timezone('America/New_York')
                        
                        days = defaultdict(list)
                        for idx, row in hist.iterrows():
                            # Convert to Eastern time
                            if idx.tz is None:
                                idx_eastern = eastern.localize(idx)
                            else:
                                idx_eastern = idx.astimezone(eastern)
                            
                            # Filter to market hours only (9:30 AM - 4:00 PM ET)
                            hour = idx_eastern.hour
                            minute = idx_eastern.minute
                            
                            if (hour == 9 and minute >= 30) or (hour >= 10 and hour < 16):
                                day_key = idx_eastern.strftime('%Y-%m-%d')
                                days[day_key].append((idx, row))
                        
                        # Get last 2 trading days
                        trading_days = sorted(days.keys())[-2:]
                        candles = []
                        for day in trading_days:
                            for idx, row in days[day]:
                                candles.append({
                                    'datetime': int(idx.timestamp() * 1000),
                                    'open': float(row['Open']),
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'close': float(row['Close']),
                                    'volume': int(row['Volume'])
                                })
                        
                        price_history = {'candles': sorted(candles, key=lambda x: x['datetime'])}
                        logger.info(f"yfinance: Fetched {len(candles)} market-hours candles across {len(trading_days)} trading days for {symbol}")
                    else:
                        logger.error(f"No intraday data from yfinance for {symbol}")
                except Exception as yf_error:
                    logger.error(f"yfinance intraday fallback failed for {symbol}: {yf_error}")
        else:
            # Daily data - get 30 trading days
            try:
                end_date = datetime.now()
                # Request ~45 calendar days to ensure we get 30 trading days
                start_date = end_date - timedelta(days=45)
                start_ms = int(start_date.timestamp() * 1000)
                end_ms = int(end_date.timestamp() * 1000)
                price_history = client.get_price_history(
                    symbol=symbol,
                    period_type='month',
                    period=2,
                    frequency_type='daily',
                    frequency=1,
                    start_date=start_ms,
                    end_date=end_ms
                )
                
                # Filter to last 30 candles (30 trading days)
                if price_history and 'candles' in price_history:
                    candles = price_history['candles']
                    if len(candles) > 30:
                        price_history['candles'] = candles[-30:]
                        logger.info(f"Filtered to last 30 trading days for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Schwab daily data failed for {symbol}, trying yfinance: {e}")
                # Fallback to yfinance for daily data (30 trading days)
                try:
                    import yfinance as yf
                    yf_symbol = symbol.replace('$', '^') if symbol.startswith('$') else symbol
                    ticker = yf.Ticker(yf_symbol)
                    # Request ~45 days to ensure we get 30 trading days
                    hist = ticker.history(period="2mo", interval="1d")
                    
                    if not hist.empty:
                        candles = []
                        for idx, row in hist.iterrows():
                            candles.append({
                                'datetime': int(idx.timestamp() * 1000),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'close': float(row['Close']),
                                'volume': int(row['Volume'])
                            })
                        
                        # Keep last 30 trading days
                        if len(candles) > 30:
                            candles = candles[-30:]
                        
                        price_history = {'candles': candles}
                        logger.info(f"yfinance: Fetched {len(candles)} daily candles for {symbol}")
                    else:
                        logger.error(f"No daily data from yfinance for {symbol}")
                except Exception as yf_error:
                    logger.error(f"yfinance daily fallback failed for {symbol}: {yf_error}")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'underlying_price': underlying_price,
            'quote': quote,
            'options_chain': options_chain,
            'price_history': price_history,
            'timeframe': timeframe,
            'fetched_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching market snapshot: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/key_levels')
def get_key_levels():
    """
    Calculate key option levels: walls, flip level, max GEX
    Query params:
        - symbol: ticker symbol (required)
        - expiry: expiration date YYYY-MM-DD (required)
    """
    symbol = request.args.get('symbol')
    expiry = request.args.get('expiry')
    
    if not symbol or not expiry:
        return jsonify({
            'success': False,
            'error': 'symbol and expiry parameters are required'
        }), 400
    
    client = SchwabClient()
    
    if not client.authenticate():
        return jsonify({
            'success': False,
            'error': 'Failed to authenticate with Schwab API'
        }), 500
    
    try:
        # Get quote for underlying price
        quote = client.get_quote(symbol)
        if not quote:
            return jsonify({
                'success': False,
                'error': f'Failed to get quote for {symbol}'
            }), 404
        
        underlying_price = quote.get(symbol, {}).get('quote', {}).get('lastPrice', 0)
        if not underlying_price:
            return jsonify({
                'success': False,
                'error': f'No price found for {symbol}'
            }), 404
        
        # Get options chain
        options_chain = client.get_options_chain(
            symbol=symbol,
            from_date=expiry,
            to_date=expiry
        )
        
        if not options_chain or 'callExpDateMap' not in options_chain:
            return jsonify({
                'success': False,
                'error': f'No options chain data for {symbol}'
            }), 404
        
        # Calculate key levels
        levels = calculate_option_levels(options_chain, underlying_price)
        
        if not levels:
            return jsonify({
                'success': False,
                'error': 'Failed to calculate key levels'
            }), 500
        
        # Convert DataFrame to list of dicts
        strike_data = levels['strike_data'].to_dict('records') if 'strike_data' in levels else []
        
        # Convert Series to dict for key levels
        result = {
            'success': True,
            'symbol': symbol,
            'underlying_price': underlying_price,
            'call_wall': levels['call_wall'].to_dict() if levels['call_wall'] is not None and not levels['call_wall'].empty else None,
            'put_wall': levels['put_wall'].to_dict() if levels['put_wall'] is not None and not levels['put_wall'].empty else None,
            'max_gex': levels['max_gex'].to_dict() if levels['max_gex'] is not None and not levels['max_gex'].empty else None,
            'flip_level': float(levels['flip_level']) if levels['flip_level'] else None,
            'pc_ratio': float(levels['pc_ratio']),
            'total_call_vol': float(levels['total_call_vol']),
            'total_put_vol': float(levels['total_put_vol']),
            'strike_data': strike_data,
            'fetched_at': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating key levels: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def calculate_option_levels(options_data, underlying_price):
    """Calculate key option levels: walls, flip level, max GEX"""
    try:
        call_data = {}
        put_data = {}
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in call_data:
                            call_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0}
                        call_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        call_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        call_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        call_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    for contract in contracts:
                        if strike not in put_data:
                            put_data[strike] = {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0}
                        put_data[strike]['volume'] += contract.get('totalVolume', 0) or 0
                        put_data[strike]['oi'] += contract.get('openInterest', 0) or 0
                        put_data[strike]['gamma'] += contract.get('gamma', 0) or 0
                        put_data[strike]['premium'] += (contract.get('mark', 0) or 0) * (contract.get('totalVolume', 0) or 0) * 100
        
        # Calculate metrics by strike
        all_strikes = sorted(set(call_data.keys()) | set(put_data.keys()))
        strike_analysis = []
        
        for strike in all_strikes:
            call = call_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0})
            put = put_data.get(strike, {'volume': 0, 'oi': 0, 'gamma': 0, 'premium': 0})
            
            # GEX calculation
            call_gex = call['gamma'] * call['oi'] * 100 * underlying_price * underlying_price * 0.01
            put_gex = -put['gamma'] * put['oi'] * 100 * underlying_price * underlying_price * 0.01
            net_gex = call_gex + put_gex
            
            # Net volume
            net_volume = put['volume'] - call['volume']
            
            # Distance from current price
            distance_pct = abs(strike - underlying_price) / underlying_price * 100
            
            strike_analysis.append({
                'strike': strike,
                'call_vol': call['volume'],
                'put_vol': put['volume'],
                'net_vol': net_volume,
                'call_oi': call['oi'],
                'put_oi': put['oi'],
                'call_gex': call_gex,
                'put_gex': put_gex,
                'net_gex': net_gex,
                'call_premium': call['premium'],
                'put_premium': put['premium'],
                'distance_pct': distance_pct
            })
        
        df = pd.DataFrame(strike_analysis)
        
        # Return early if no data
        if len(df) == 0:
            return {
                'call_wall': None,
                'put_wall': None,
                'max_gex': None,
                'flip_level': None,
                'pc_ratio': 0,
                'total_call_vol': 0,
                'total_put_vol': 0,
                'strike_data': df
            }
        
        # Find key levels
        call_wall = df.loc[df['call_vol'].idxmax()] if df['call_vol'].max() > 0 else None
        put_wall = df.loc[df['put_vol'].idxmax()] if df['put_vol'].max() > 0 else None
        max_gex = df.loc[df['net_gex'].abs().idxmax()] if len(df) > 0 else None
        
        # Find flip level (where net volume crosses zero)
        flip_level = None
        if 'distance_pct' in df.columns and len(df) > 0:
            nearby = df[df['distance_pct'] < 2.0].sort_values('strike')
            for i in range(len(nearby) - 1):
                if (nearby.iloc[i]['net_vol'] > 0 and nearby.iloc[i+1]['net_vol'] < 0) or \
                   (nearby.iloc[i]['net_vol'] < 0 and nearby.iloc[i+1]['net_vol'] > 0):
                    flip_level = nearby.iloc[i]['strike']
                    break
        
        # Calculate P/C ratio
        total_call_vol = df['call_vol'].sum()
        total_put_vol = df['put_vol'].sum()
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        return {
            'call_wall': call_wall,
            'put_wall': put_wall,
            'max_gex': max_gex,
            'flip_level': flip_level,
            'pc_ratio': pc_ratio,
            'total_call_vol': total_call_vol,
            'total_put_vol': total_put_vol,
            'strike_data': df
        }
        
    except Exception as e:
        logger.error(f"Error calculating levels: {e}", exc_info=True)
        return None

@app.route('/api/top_opportunities')
def top_opportunities():
    """Get top trading opportunities from SQLite with composite scoring"""
    try:
        import sqlite3
        db_path = Path(__file__).parent.parent / "market_data.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        limit = request.args.get('limit', 50, type=int)
        
        # Query to calculate composite scores
        query = """
        WITH latest_whale AS (
            SELECT symbol, expiry, strike, option_type, MAX(whale_score) as max_whale_score
            FROM whale_flows
            WHERE timestamp > datetime('now', '-30 minutes')
            GROUP BY symbol, expiry, strike, option_type
        ),
        latest_oi AS (
            SELECT symbol, expiry, MAX(vol_oi_ratio) as max_vol_oi
            FROM oi_flows
            WHERE timestamp > datetime('now', '-30 minutes')
            GROUP BY symbol, expiry
        ),
        latest_skew AS (
            SELECT symbol, expiry, skew_25d, put_call_ratio
            FROM skew_metrics
            WHERE timestamp > datetime('now', '-30 minutes')
        )
        SELECT 
            COALESCE(w.symbol, o.symbol, s.symbol) as symbol,
            COALESCE(w.expiry, o.expiry, s.expiry) as expiry,
            w.strike,
            w.option_type,
            COALESCE(w.max_whale_score, 0) as whale_score,
            COALESCE(o.max_vol_oi, 0) as vol_oi_ratio,
            s.skew_25d,
            s.put_call_ratio,
            -- Composite score calculation
            (CASE 
                WHEN COALESCE(w.max_whale_score, 0) > 200 THEN 35
                WHEN COALESCE(w.max_whale_score, 0) > 100 THEN 25
                WHEN COALESCE(w.max_whale_score, 0) > 50 THEN 15
                ELSE CAST(COALESCE(w.max_whale_score, 0) / 10 AS INTEGER)
            END +
            CASE
                WHEN COALESCE(o.max_vol_oi, 0) > 8.0 THEN 35
                WHEN COALESCE(o.max_vol_oi, 0) > 6.0 THEN 30
                WHEN COALESCE(o.max_vol_oi, 0) > 4.0 THEN 20
                ELSE CAST(COALESCE(o.max_vol_oi, 0) * 3 AS INTEGER)
            END +
            CASE
                WHEN s.skew_25d > 6.0 THEN 30
                WHEN s.skew_25d < -1.0 THEN 25
                ELSE 5
            END) as composite_score
        FROM latest_whale w
        FULL OUTER JOIN latest_oi o ON w.symbol = o.symbol AND w.expiry = o.expiry
        FULL OUTER JOIN latest_skew s ON COALESCE(w.symbol, o.symbol) = s.symbol 
            AND COALESCE(w.expiry, o.expiry) = s.expiry
        ORDER BY composite_score DESC
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        opportunities = []
        for row in rows:
            opportunities.append({
                'symbol': row['symbol'],
                'expiry': row['expiry'],
                'strike': row['strike'],
                'option_type': row['option_type'],
                'composite_score': row['composite_score'],
                'whale_score': row['whale_score'],
                'vol_oi_ratio': round(row['vol_oi_ratio'], 2),
                'skew_25d': round(row['skew_25d'], 2) if row['skew_25d'] else None,
                'put_call_ratio': round(row['put_call_ratio'], 2) if row['put_call_ratio'] else None
            })
        
        conn.close()
        
        return jsonify({
            'opportunities': opportunities,
            'count': len(opportunities),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching top opportunities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_fear_greed')
def market_fear_greed():
    """Get market sentiment from skew metrics"""
    try:
        import sqlite3
        db_path = Path(__file__).parent.parent / "market_data.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(skew_25d) as avg_skew,
                AVG(put_call_ratio) as avg_pc_ratio,
                COUNT(DISTINCT symbol) as symbols_count
            FROM skew_metrics
            WHERE timestamp > datetime('now', '-30 minutes')
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] is not None:
            avg_skew = round(row[0], 2)
            avg_pc = round(row[1], 2) if row[1] else None
            
            # Sentiment classification
            if avg_skew > 5:
                sentiment = "EXTREME_FEAR"
            elif avg_skew > 3:
                sentiment = "FEAR"
            elif avg_skew > 1:
                sentiment = "SLIGHT_FEAR"
            elif avg_skew < -2:
                sentiment = "GREED"
            else:
                sentiment = "NEUTRAL"
            
            return jsonify({
                'sentiment': sentiment,
                'avg_skew': avg_skew,
                'avg_put_call_ratio': avg_pc,
                'symbols_analyzed': row[2],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'No recent data'}), 404
    
    except Exception as e:
        logger.error(f"Error fetching market sentiment: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on port 8000, accessible from external IPs with threading enabled
    print("=" * 60)
    print("Options Scanner API Server")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:8000 (threaded mode)")
    print("\nEndpoints:")
    print("  GET /                     - API status")
    print("  GET /health               - Health check")
    print("  GET /api/watchlist        - Get watchlist data")
    print("  GET /api/whale_flows      - Get whale flows")
    print("  GET /api/top_opportunities - Get top trading opportunities")
    print("  GET /api/market_fear_greed - Get market fear/greed sentiment")
    print("  GET /api/market_snapshot  - Get market snapshot with chart data")
    print("  GET /api/key_levels       - Get key option levels")
    print("  GET /api/stats            - Cache statistics")
    print("  GET /api/last_update      - Last update times")
    print("=" * 60)
    
    # Enable threading to handle multiple concurrent requests
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
