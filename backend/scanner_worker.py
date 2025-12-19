"""
Options Flow Scanner - Background Worker
Scans options data and stores in PostgreSQL with rate limiting for Schwab API

Rate Limit: 120 calls/minute
Strategy: Split 44 stocks into 2 batches of 22, process 1 minute apart
Each stock = 4 API calls (1 per expiry)
Batch 1: 22 stocks × 4 expiries = 88 calls (73% of limit)
Batch 2: 22 stocks × 4 expiries = 88 calls (73% of limit)
Total time: ~2 minutes per complete scan
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid
import time

import psycopg2
from psycopg2.extras import execute_batch
from psycopg2 import pool
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.schwab_client import SchwabClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/options-scanner/scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Stock lists
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'AMD',
    'CRM', 'GS', 'NFLX', 'IBIT', 'COIN',
    'APP', 'PLTR', 'SNOW', 'TEAM', 'CRWD', 'SPY', 'QQQ'
]

VALUE_STOCKS = [
    'AXP', 'JPM', 'C', 'WFC', 'XOM',
    'CVX', 'PG', 'JNJ', 'UNH', 'V',
    'MA', 'HD', 'WMT', 'KO', 'PEP',
    'MRK', 'ABBV', 'CAT', 'TMO', 'LLY',
    'DIA', 'IWM'
]

ALL_STOCKS = TECH_STOCKS + VALUE_STOCKS  # 44 stocks total

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'options_scanner',
    'user': 'options_user',
    'password': 'your_secure_password',  # Replace in production
    'port': 5432
}

# Connection pool
db_pool = None


class RateLimiter:
    """Rate limiter to ensure we don't exceed Schwab API limits"""
    
    def __init__(self, max_calls_per_minute: int = 120):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to stay under rate limit"""
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call) + 0.1  # Add 0.1s buffer
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                # Re-clean after waiting
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            self.calls.append(now)
            return True


def init_db_pool():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


def get_db_connection():
    """Get connection from pool"""
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()


def return_db_connection(conn):
    """Return connection to pool"""
    if db_pool:
        db_pool.putconn(conn)


def get_next_friday():
    """Get next Friday date for weekly expiry"""
    today = datetime.now().date()
    weekday = today.weekday()
    days_to_friday = (4 - weekday) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    return today + timedelta(days=days_to_friday)


def get_next_4_fridays():
    """Get next 4 weekly Friday dates"""
    fridays = []
    first_friday = get_next_friday()
    for i in range(4):
        fridays.append(first_friday + timedelta(weeks=i))
    return fridays


def calculate_skew_metrics(symbol: str, expiry_date: str, underlying_price: float, 
                          call_data: dict, put_data: dict) -> Optional[Dict]:
    """Calculate put-call skew and implied move metrics"""
    try:
        # Find ATM strike (closest to underlying)
        call_strikes = sorted([float(s) for s in call_data.keys()])
        put_strikes = sorted([float(s) for s in put_data.keys()])
        
        if not call_strikes or not put_strikes:
            return None
        
        atm_strike = min(call_strikes, key=lambda x: abs(x - underlying_price))
        
        # Get ATM straddle for implied move
        atm_call = None
        atm_put = None
        
        if str(atm_strike) in call_data and call_data[str(atm_strike)]:
            atm_call = call_data[str(atm_strike)][0]
        if str(atm_strike) in put_data and put_data[str(atm_strike)]:
            atm_put = put_data[str(atm_strike)][0]
        
        if not atm_call or not atm_put:
            return None
        
        # Calculate implied move
        call_mark = atm_call.get('mark', atm_call.get('last', 0))
        put_mark = atm_put.get('mark', atm_put.get('last', 0))
        straddle_price = call_mark + put_mark
        implied_move_dollars = straddle_price
        implied_move_pct = (straddle_price / underlying_price) * 100
        
        # Find 25-delta strikes for skew
        call_25d = None
        put_25d = None
        
        for strike_str, contracts in call_data.items():
            if contracts:
                delta = contracts[0].get('delta', 0)
                if 0.20 <= delta <= 0.30:
                    call_25d = contracts[0]
                    break
        
        for strike_str, contracts in put_data.items():
            if contracts:
                delta = contracts[0].get('delta', 0)
                if -0.30 <= delta <= -0.20:
                    put_25d = contracts[0]
                    break
        
        # Calculate 25-delta skew
        skew_25d = 0
        if call_25d and put_25d:
            call_iv = call_25d.get('volatility', 0)
            put_iv = put_25d.get('volatility', 0)
            skew_25d = put_iv - call_iv
        
        # Calculate ATM skew
        atm_call_iv = atm_call.get('volatility', 0)
        atm_put_iv = atm_put.get('volatility', 0)
        atm_skew = atm_put_iv - atm_call_iv
        
        # Calculate average IV
        all_call_ivs = []
        all_put_ivs = []
        
        for strike_str, contracts in call_data.items():
            strike = float(strike_str)
            if abs(strike - underlying_price) / underlying_price <= 0.10 and contracts:
                iv = contracts[0].get('volatility', 0)
                if iv > 0:
                    all_call_ivs.append(iv)
        
        for strike_str, contracts in put_data.items():
            strike = float(strike_str)
            if abs(strike - underlying_price) / underlying_price <= 0.10 and contracts:
                iv = contracts[0].get('volatility', 0)
                if iv > 0:
                    all_put_ivs.append(iv)
        
        avg_call_iv = np.mean(all_call_ivs) if all_call_ivs else 0
        avg_put_iv = np.mean(all_put_ivs) if all_put_ivs else 0
        avg_iv = (avg_call_iv + avg_put_iv) / 2
        
        # Calculate put/call ratios
        total_call_oi = sum([c[0].get('openInterest', 0) for c in call_data.values() if c])
        total_put_oi = sum([c[0].get('openInterest', 0) for c in put_data.values() if c])
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_call_vol = sum([c[0].get('totalVolume', 0) for c in call_data.values() if c])
        total_put_vol = sum([c[0].get('totalVolume', 0) for c in put_data.values() if c])
        put_call_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # Calculate breakout levels
        upper_breakout = underlying_price + implied_move_dollars
        lower_breakout = underlying_price - implied_move_dollars
        
        # Calculate expected range (68% probability)
        days_to_exp = (datetime.strptime(expiry_date, '%Y-%m-%d').date() - datetime.now().date()).days
        days_to_exp = max(days_to_exp, 1)
        upper_1sd = underlying_price * (1 + (avg_iv / 100) * np.sqrt(days_to_exp / 365))
        lower_1sd = underlying_price * (1 - (avg_iv / 100) * np.sqrt(days_to_exp / 365))
        
        return {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'atm_strike': atm_strike,
            'implied_move_dollars': implied_move_dollars,
            'implied_move_pct': implied_move_pct,
            'skew_25d': skew_25d,
            'atm_skew': atm_skew,
            'avg_iv': avg_iv,
            'avg_call_iv': avg_call_iv,
            'avg_put_iv': avg_put_iv,
            'put_call_oi_ratio': put_call_oi_ratio,
            'put_call_vol_ratio': put_call_vol_ratio,
            'upper_breakout': upper_breakout,
            'lower_breakout': lower_breakout,
            'upper_1sd': upper_1sd,
            'lower_1sd': lower_1sd,
            'straddle_price': straddle_price,
            'atm_call_iv': atm_call_iv,
            'atm_put_iv': atm_put_iv
        }
        
    except Exception as e:
        logger.error(f"Error calculating skew for {symbol}: {e}")
        return None


async def scan_stock_combined(client: SchwabClient, symbol: str, expiry_date: str, 
                              rate_limiter: RateLimiter) -> Optional[Dict]:
    """
    Combined scanner: fetch API data, calculate whale, OI, and skew metrics
    Uses rate limiter to respect API constraints
    """
    try:
        # Rate limit: Get quote
        await rate_limiter.acquire()
        quote_response = client.get_quotes([symbol])
        if not quote_response or symbol not in quote_response:
            return None
        
        underlying_price = quote_response[symbol]['quote']['lastPrice']
        underlying_volume = quote_response[symbol]['quote'].get('totalVolume', 0)
        
        if underlying_volume == 0:
            return None
        
        # Rate limit: Get options chain
        await rate_limiter.acquire()
        options_response = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            from_date=expiry_date,
            to_date=expiry_date
        )
        
        if not options_response or 'callExpDateMap' not in options_response:
            return None
        
        # Process options
        whale_options = []
        oi_options = []
        
        # Extract call/put data for skew
        call_data_for_skew = options_response.get('callExpDateMap', {})
        put_data_for_skew = options_response.get('putExpDateMap', {})
        
        call_strikes = {}
        put_strikes = {}
        
        if call_data_for_skew:
            for exp_date_key, strikes_map in call_data_for_skew.items():
                call_strikes.update(strikes_map)
        
        if put_data_for_skew:
            for exp_date_key, strikes_map in put_data_for_skew.items():
                put_strikes.update(strikes_map)
        
        # Process calls for whale and OI flows
        if 'callExpDateMap' in options_response:
            for exp_date, strikes in options_response['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        mark_price = contract.get('mark', contract.get('last', 1))
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        gamma = contract.get('gamma', 0)
                        
                        if volume == 0 or mark_price == 0:
                            continue
                        
                        # WHALE SCORE CALCULATION
                        if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                            leverage = delta * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            whale_options.append({
                                'symbol': symbol,
                                'strike': strike,
                                'type': 'CALL',
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'gamma': gamma,
                                'gex': gex,
                                'mark': mark_price,
                                'delta': delta,
                                'iv': contract.get('volatility', 0),
                                'underlying_price': underlying_price
                            })
                        
                        # OI SCORE CALCULATION
                        if abs(strike - underlying_price) / underlying_price <= 0.10:
                            vol_oi_ratio = volume / oi
                            
                            if vol_oi_ratio >= 3.0:
                                notional_value = volume * mark_price * 100
                                oi_score = vol_oi_ratio * notional_value / 1000
                                
                                oi_options.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'type': 'CALL',
                                    'volume': volume,
                                    'open_interest': oi,
                                    'vol_oi_ratio': vol_oi_ratio,
                                    'oi_score': round(oi_score, 0),
                                    'mark': mark_price,
                                    'notional': notional_value,
                                    'delta': delta,
                                    'iv': contract.get('volatility', 0),
                                    'underlying_price': underlying_price
                                })
        
        # Process puts (similar logic)
        if 'putExpDateMap' in options_response:
            for exp_date, strikes in options_response['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        contract = contracts[0]
                        strike = float(strike_str)
                        
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                        mark_price = contract.get('mark', contract.get('last', 1))
                        delta = contract.get('delta', 0)
                        ivol = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                        gamma = contract.get('gamma', 0)
                        
                        if volume == 0 or mark_price == 0:
                            continue
                        
                        # WHALE SCORE
                        if abs(strike - underlying_price) / underlying_price <= 0.05 and delta != 0:
                            leverage = abs(delta) * underlying_price
                            leverage_ratio = leverage / mark_price
                            valr = leverage_ratio * ivol
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark_price * 100
                            dvolume_und = underlying_price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            gex = gamma * oi * 100 * underlying_price * underlying_price * 0.01
                            
                            whale_options.append({
                                'symbol': symbol,
                                'strike': strike,
                                'type': 'PUT',
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'gamma': gamma,
                                'gex': gex,
                                'mark': mark_price,
                                'delta': delta,
                                'iv': contract.get('volatility', 0),
                                'underlying_price': underlying_price
                            })
                        
                        # OI SCORE
                        if abs(strike - underlying_price) / underlying_price <= 0.10:
                            vol_oi_ratio = volume / oi
                            
                            if vol_oi_ratio >= 3.0:
                                notional_value = volume * mark_price * 100
                                oi_score = vol_oi_ratio * notional_value / 1000
                                
                                oi_options.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'type': 'PUT',
                                    'volume': volume,
                                    'open_interest': oi,
                                    'vol_oi_ratio': vol_oi_ratio,
                                    'oi_score': round(oi_score, 0),
                                    'mark': mark_price,
                                    'notional': notional_value,
                                    'delta': delta,
                                    'iv': contract.get('volatility', 0),
                                    'underlying_price': underlying_price
                                })
        
        # Calculate summary stats for whale flows
        result = {'whale': None, 'oi': None, 'skew': None}
        
        if whale_options:
            call_vol = sum([opt['volume'] for opt in whale_options if opt['type'] == 'CALL'])
            put_vol = sum([opt['volume'] for opt in whale_options if opt['type'] == 'PUT'])
            
            max_gex_opt = max(whale_options, key=lambda x: abs(x['gex']))
            call_opts = [opt for opt in whale_options if opt['type'] == 'CALL']
            put_opts = [opt for opt in whale_options if opt['type'] == 'PUT']
            
            call_wall = max(call_opts, key=lambda x: x['volume']) if call_opts else None
            put_wall = max(put_opts, key=lambda x: x['volume']) if put_opts else None
            
            summary = {
                'call_volume': call_vol,
                'put_volume': put_vol,
                'vol_ratio': (put_vol - call_vol) / max(call_vol, put_vol, 1) * 100,
                'max_gex_strike': max_gex_opt['strike'],
                'max_gex_value': max_gex_opt['gex'],
                'call_wall_strike': call_wall['strike'] if call_wall else None,
                'put_wall_strike': put_wall['strike'] if put_wall else None
            }
            
            result['whale'] = {
                'options': whale_options,
                'summary': summary
            }
        
        if oi_options:
            result['oi'] = oi_options
        
        # Calculate skew metrics
        skew_metrics = calculate_skew_metrics(symbol, expiry_date, underlying_price, call_strikes, put_strikes)
        if skew_metrics:
            result['skew'] = skew_metrics
        
        return result if (result['whale'] or result['oi'] or result['skew']) else None
        
    except Exception as e:
        logger.error(f"Error scanning {symbol} for {expiry_date}: {e}")
        return None


def save_to_database(scan_id: str, expiry_date: str, whale_data: List[Dict], 
                    oi_data: List[Dict], skew_data: List[Dict], timestamp: datetime):
    """Save scan results to PostgreSQL database"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert whale flows
        if whale_data:
            whale_insert = """
                INSERT INTO whale_flows (
                    scan_id, timestamp, symbol, expiry, strike, type, whale_score,
                    volume, open_interest, gamma, gex, mark, delta, iv, underlying_price,
                    call_volume, put_volume, vol_ratio, max_gex_strike, max_gex_value,
                    call_wall_strike, put_wall_strike
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
            """
            whale_values = [
                (
                    scan_id, timestamp, w['symbol'], expiry_date, w['strike'], w['type'],
                    w['whale_score'], w['volume'], w['open_interest'], w['gamma'], w['gex'],
                    w['mark'], w['delta'], w['iv'], w['underlying_price'],
                    w.get('call_volume'), w.get('put_volume'), w.get('vol_ratio'),
                    w.get('max_gex_strike'), w.get('max_gex_value'),
                    w.get('call_wall_strike'), w.get('put_wall_strike')
                )
                for w in whale_data
            ]
            execute_batch(cur, whale_insert, whale_values)
        
        # Insert OI flows
        if oi_data:
            oi_insert = """
                INSERT INTO oi_flows (
                    scan_id, timestamp, symbol, expiry, strike, type, oi_score,
                    vol_oi_ratio, volume, open_interest, notional, mark, delta, iv, underlying_price
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            oi_values = [
                (
                    scan_id, timestamp, o['symbol'], expiry_date, o['strike'], o['type'],
                    o['oi_score'], o['vol_oi_ratio'], o['volume'], o['open_interest'],
                    o['notional'], o['mark'], o['delta'], o['iv'], o['underlying_price']
                )
                for o in oi_data
            ]
            execute_batch(cur, oi_insert, oi_values)
        
        # Insert skew metrics
        if skew_data:
            skew_insert = """
                INSERT INTO skew_metrics (
                    scan_id, timestamp, symbol, expiry, underlying_price, atm_strike,
                    skew_25d, atm_skew, avg_iv, avg_call_iv, avg_put_iv, atm_call_iv, atm_put_iv,
                    put_call_oi_ratio, put_call_vol_ratio, implied_move_dollars, implied_move_pct,
                    straddle_price, upper_breakout, lower_breakout, upper_1sd, lower_1sd
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            skew_values = [
                (
                    scan_id, timestamp, s['symbol'], expiry_date, s['underlying_price'],
                    s['atm_strike'], s['skew_25d'], s['atm_skew'], s['avg_iv'],
                    s['avg_call_iv'], s['avg_put_iv'], s['atm_call_iv'], s['atm_put_iv'],
                    s['put_call_oi_ratio'], s['put_call_vol_ratio'], s['implied_move_dollars'],
                    s['implied_move_pct'], s['straddle_price'], s['upper_breakout'],
                    s['lower_breakout'], s['upper_1sd'], s['lower_1sd']
                )
                for s in skew_data
            ]
            execute_batch(cur, skew_insert, skew_values)
        
        conn.commit()
        logger.info(f"Saved to DB: {len(whale_data)} whale, {len(oi_data)} OI, {len(skew_data)} skew")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error saving to database: {e}")
        raise
    finally:
        if conn:
            cur.close()
            return_db_connection(conn)


async def scan_batch(client: SchwabClient, stocks: List[str], expiries: List[str],
                    rate_limiter: RateLimiter, scan_id: str, timestamp: datetime):
    """Scan a batch of stocks with rate limiting"""
    logger.info(f"Starting batch scan of {len(stocks)} stocks across {len(expiries)} expiries")
    
    all_whale_data = []
    all_oi_data = []
    all_skew_data = []
    api_calls = 0
    
    for stock in stocks:
        for expiry_date in expiries:
            try:
                result = await scan_stock_combined(client, stock, expiry_date.strftime('%Y-%m-%d'), rate_limiter)
                api_calls += 2  # 1 quote + 1 options chain
                
                if result:
                    # Process whale flows
                    if result['whale']:
                        for opt in result['whale']['options']:
                            opt.update(result['whale']['summary'])
                            all_whale_data.append(opt)
                    
                    # Process OI flows
                    if result['oi']:
                        all_oi_data.extend(result['oi'])
                    
                    # Process skew
                    if result['skew']:
                        all_skew_data.append(result['skew'])
                
            except Exception as e:
                logger.error(f"Error scanning {stock} {expiry_date}: {e}")
                continue
    
    # Save batch to database
    for expiry_date in expiries:
        expiry_str = expiry_date.strftime('%Y-%m-%d')
        whale_for_expiry = [w for w in all_whale_data if expiry_str in str(w.get('symbol'))]
        oi_for_expiry = [o for o in all_oi_data if expiry_str in str(o.get('symbol'))]
        skew_for_expiry = [s for s in all_skew_data if expiry_str in str(s.get('symbol'))]
        
        if whale_for_expiry or oi_for_expiry or skew_for_expiry:
            save_to_database(scan_id, expiry_str, whale_for_expiry, oi_for_expiry, skew_for_expiry, timestamp)
    
    logger.info(f"Batch complete: {api_calls} API calls, {len(all_whale_data)} whale, {len(all_oi_data)} OI, {len(all_skew_data)} skew")
    return api_calls


async def run_full_scan():
    """
    Run complete scan of all stocks with rate limiting
    Splits 44 stocks into 2 batches to respect 120 calls/min limit
    """
    scan_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    logger.info(f"Starting full scan {scan_id} at {timestamp}")
    
    # Create scan run record
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO scan_runs (scan_id, start_time, status, stocks_scanned, expiries_scanned)
            VALUES (%s, %s, %s, %s, %s)
        """, (scan_id, timestamp, 'running', len(ALL_STOCKS), 4))
        conn.commit()
        cur.close()
        return_db_connection(conn)
    except Exception as e:
        logger.error(f"Error creating scan run: {e}")
        if conn:
            return_db_connection(conn)
        return
    
    # Initialize
    client = SchwabClient()
    if not client.authenticate():
        logger.error("Failed to authenticate with Schwab API")
        return
    
    expiries = get_next_4_fridays()
    rate_limiter = RateLimiter(max_calls_per_minute=115)  # Leave 5-call buffer
    
    # Split stocks into 2 batches
    batch_size = 22
    batch1 = ALL_STOCKS[:batch_size]  # First 22 stocks (tech)
    batch2 = ALL_STOCKS[batch_size:]  # Last 22 stocks (value)
    
    total_api_calls = 0
    
    try:
        # Process batch 1
        logger.info(f"Processing BATCH 1: {len(batch1)} stocks")
        api_calls_1 = await scan_batch(client, batch1, expiries, rate_limiter, scan_id, timestamp)
        total_api_calls += api_calls_1
        
        # Wait 60 seconds to reset rate limit window
        logger.info("Waiting 60 seconds before batch 2...")
        await asyncio.sleep(60)
        
        # Process batch 2
        logger.info(f"Processing BATCH 2: {len(batch2)} stocks")
        api_calls_2 = await scan_batch(client, batch2, expiries, rate_limiter, scan_id, timestamp)
        total_api_calls += api_calls_2
        
        # Update scan run as completed
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE scan_runs
            SET end_time = %s, status = %s, total_api_calls = %s
            WHERE scan_id = %s
        """, (datetime.now(), 'completed', total_api_calls, scan_id))
        conn.commit()
        cur.close()
        return_db_connection(conn)
        
        logger.info(f"Scan {scan_id} completed successfully. Total API calls: {total_api_calls}")
        
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        
        # Update scan run as failed
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE scan_runs
                SET end_time = %s, status = %s, error_message = %s, total_api_calls = %s
                WHERE scan_id = %s
            """, (datetime.now(), 'failed', str(e), total_api_calls, scan_id))
            conn.commit()
            cur.close()
            return_db_connection(conn)
        except Exception as db_error:
            logger.error(f"Error updating failed scan: {db_error}")


def main():
    """Main entry point for scanner"""
    init_db_pool()
    asyncio.run(run_full_scan())


if __name__ == '__main__':
    main()
