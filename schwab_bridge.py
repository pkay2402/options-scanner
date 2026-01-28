"""
FastAPI bridge for Trading Hub to access Schwab API
Run alongside the Next.js app: python schwab_bridge.py

Rate Limit: 120 calls/minute to Schwab API
"""

import sys
import time
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from collections import deque
import logging

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.api.schwab_client import SchwabClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Hub - Schwab Bridge", version="1.0.0")

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Rate Limiter ====================
class RateLimiter:
    """Simple sliding window rate limiter for Schwab API (120 calls/min)"""
    def __init__(self, max_calls: int = 100, window_seconds: int = 60):
        self.max_calls = max_calls  # Leave headroom below 120
        self.window = window_seconds
        self.calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Returns True if call is allowed, False if rate limited"""
        with self.lock:
            now = time.time()
            # Remove old calls outside window
            while self.calls and self.calls[0] < now - self.window:
                self.calls.popleft()
            
            if len(self.calls) >= self.max_calls:
                logger.warning(f"Rate limit hit: {len(self.calls)} calls in last {self.window}s")
                return False
            
            self.calls.append(now)
            return True
    
    def calls_remaining(self) -> int:
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.window:
                self.calls.popleft()
            return self.max_calls - len(self.calls)

rate_limiter = RateLimiter(max_calls=100, window_seconds=60)

# ==================== Schwab Client ====================
_client = None

def get_schwab_client():
    global _client
    if _client is None:
        _client = SchwabClient(interactive=False)
        if not _client.authenticate():
            logger.warning("Schwab authentication failed - some features may be limited")
    return _client


@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "rate_limit_remaining": rate_limiter.calls_remaining()
    }


@app.get("/api/market-pulse")
def market_pulse():
    """Get SPY, QQQ, VIX quotes"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        symbols = ['SPY', 'QQQ', '$VIX']
        quotes = client.get_quotes(symbols)
        
        if not quotes:
            raise HTTPException(status_code=503, detail="Failed to fetch quotes")
        
        result = {}
        symbol_map = {'SPY': 'spy', 'QQQ': 'qqq', '$VIX': 'vix'}
        
        for symbol, key in symbol_map.items():
            if symbol in quotes:
                q = quotes[symbol].get('quote', {})
                last_price = q.get('lastPrice', q.get('mark', 0))
                close_price = q.get('closePrice', 0)
                net_change = q.get('netChange', 0)
                
                # Calculate percent change if not provided
                pct_change = q.get('netPercentChangeInDouble', 0)
                if pct_change == 0 and close_price > 0:
                    pct_change = (net_change / close_price) * 100
                
                result[key] = {
                    'symbol': symbol.replace('$', ''),
                    'lastPrice': last_price,
                    'netChange': net_change,
                    'netPercentChange': round(pct_change, 2),
                    'bidPrice': q.get('bidPrice', 0),
                    'askPrice': q.get('askPrice', 0),
                    'volume': q.get('totalVolume', 0),
                    'high': q.get('highPrice', 0),
                    'low': q.get('lowPrice', 0),
                    'open': q.get('openPrice', 0),
                    'close': close_price,
                }
            else:
                result[key] = None
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market pulse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quote")
def get_quote(symbol: str = Query(...)):
    """Get single stock quote"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        quote = client.get_quote(symbol)
        if not quote or symbol not in quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
        
        q = quote[symbol].get('quote', {})
        return {
            'symbol': symbol,
            'lastPrice': q.get('lastPrice', q.get('mark', 0)),
            'netChange': q.get('netChange', 0),
            'netPercentChange': q.get('netPercentChangeInDouble', 0),
            'bidPrice': q.get('bidPrice', 0),
            'askPrice': q.get('askPrice', 0),
            'volume': q.get('totalVolume', 0),
            'high': q.get('highPrice', 0),
            'low': q.get('lowPrice', 0),
            'open': q.get('openPrice', 0),
            'close': q.get('closePrice', 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price-history")
def price_history(
    symbol: str = Query(...),
    period: str = Query("1D")  # 1D, 5D, 1M, 3M
):
    """Get price history candles"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        
        # Configure based on period
        if period == '1D':
            start_time = int((now - timedelta(days=1)).timestamp() * 1000)
            freq_type = 'minute'
            freq = 5
        elif period == '5D':
            start_time = int((now - timedelta(days=5)).timestamp() * 1000)
            freq_type = 'minute'
            freq = 15
        elif period == '1M':
            start_time = int((now - timedelta(days=30)).timestamp() * 1000)
            freq_type = 'daily'
            freq = 1
        else:  # 3M
            start_time = int((now - timedelta(days=90)).timestamp() * 1000)
            freq_type = 'daily'
            freq = 1
        
        history = client.get_price_history(
            symbol=symbol,
            frequency_type=freq_type,
            frequency=freq,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        candles = []
        for c in (history or {}).get('candles', []):
            candles.append({
                'time': c['datetime'],
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close'],
                'volume': c.get('volume', 0),
            })
        
        return {'candles': candles}
    except Exception as e:
        logger.error(f"Price history error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/options-chain")
def options_chain(
    symbol: str = Query(...),
    expiry: str = Query(None)
):
    """Get options chain for symbol"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        params = {
            'symbol': symbol,
            'contract_type': 'ALL',
            'strike_count': 50
        }
        
        if expiry:
            params['from_date'] = expiry
            params['to_date'] = expiry
        
        chain = client.get_options_chain(**params)
        if not chain or chain.get('status') != 'SUCCESS':
            raise HTTPException(status_code=404, detail="Options chain not found")
        
        return {
            'symbol': symbol,
            'underlyingPrice': chain.get('underlyingPrice', 0),
            'callExpDateMap': chain.get('callExpDateMap', {}),
            'putExpDateMap': chain.get('putExpDateMap', {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Options chain error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/volume-walls")
def volume_walls(
    symbol: str = Query(...),
    expiry: str = Query(None)
):
    """Calculate volume walls from options chain"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        params = {
            'symbol': symbol,
            'contract_type': 'ALL',
            'strike_count': 50
        }
        
        if expiry:
            params['from_date'] = expiry
            params['to_date'] = expiry
        
        chain = client.get_options_chain(**params)
        if not chain or chain.get('status') != 'SUCCESS':
            raise HTTPException(status_code=404, detail="Options chain not found")
        
        underlying_price = chain.get('underlyingPrice', 0)
        
        # Aggregate volume by strike
        strike_data = {}
        
        for exp_date, strikes in chain.get('callExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    strike = float(strike_str)
                    if strike not in strike_data:
                        strike_data[strike] = {'callVolume': 0, 'putVolume': 0, 'callOI': 0, 'putOI': 0}
                    strike_data[strike]['callVolume'] += c.get('totalVolume', 0)
                    strike_data[strike]['callOI'] += c.get('openInterest', 0)
        
        for exp_date, strikes in chain.get('putExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    strike = float(strike_str)
                    if strike not in strike_data:
                        strike_data[strike] = {'callVolume': 0, 'putVolume': 0, 'callOI': 0, 'putOI': 0}
                    strike_data[strike]['putVolume'] += c.get('totalVolume', 0)
                    strike_data[strike]['putOI'] += c.get('openInterest', 0)
        
        # Find walls (highest volume strikes)
        sorted_strikes = sorted(strike_data.items(), key=lambda x: x[0])
        
        call_wall = max(strike_data.items(), key=lambda x: x[1]['callOI'], default=(underlying_price, {}))[0] if strike_data else underlying_price
        put_wall = max(strike_data.items(), key=lambda x: x[1]['putOI'], default=(underlying_price, {}))[0] if strike_data else underlying_price
        
        # Build walls list
        walls = []
        for strike, data in sorted_strikes:
            if underlying_price > 0 and abs(strike - underlying_price) / underlying_price < 0.1:  # Within 10%
                walls.append({
                    'strike': strike,
                    'callVolume': data['callVolume'],
                    'putVolume': data['putVolume'],
                    'callOI': data['callOI'],
                    'putOI': data['putOI'],
                    'netGamma': (data['callOI'] - data['putOI']) * 100,  # Simplified
                })
        
        return {
            'symbol': symbol,
            'underlyingPrice': underlying_price,
            'callWall': call_wall,
            'putWall': put_wall,
            'walls': walls,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volume walls error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gex")
def gex_data(symbol: str = Query(...)):
    """Calculate GEX (Gamma Exposure) for symbol"""
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        chain = client.get_options_chain(symbol=symbol, contract_type='ALL', strike_count=50)
        if not chain or chain.get('status') != 'SUCCESS':
            raise HTTPException(status_code=404, detail="Options chain not found")
        
        underlying_price = chain.get('underlyingPrice', 0)
        total_gex = 0
        strike_gex = {}
        
        # Calculate GEX per strike
        for exp_date, strikes in chain.get('callExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    strike = float(strike_str)
                    gamma = c.get('gamma', 0)
                    oi = c.get('openInterest', 0)
                    gex = gamma * oi * 100 * underlying_price  # Call gamma is positive
                    total_gex += gex
                    strike_gex[strike] = strike_gex.get(strike, 0) + gex
        
        for exp_date, strikes in chain.get('putExpDateMap', {}).items():
            for strike_str, contracts in strikes.items():
                if contracts:
                    c = contracts[0]
                    strike = float(strike_str)
                    gamma = c.get('gamma', 0)
                    oi = c.get('openInterest', 0)
                    gex = -gamma * oi * 100 * underlying_price  # Put gamma is negative for dealers
                    total_gex += gex
                    strike_gex[strike] = strike_gex.get(strike, 0) + gex
        
        # Find flip price (where GEX crosses zero)
        sorted_gex = sorted(strike_gex.items(), key=lambda x: x[0])
        flip_price = underlying_price
        for i in range(len(sorted_gex) - 1):
            if sorted_gex[i][1] * sorted_gex[i+1][1] < 0:  # Sign change
                flip_price = (sorted_gex[i][0] + sorted_gex[i+1][0]) / 2
                break
        
        # Find max gamma strikes
        call_wall = max(strike_gex.items(), key=lambda x: x[1] if x[1] > 0 else 0, default=(underlying_price, 0))[0] if strike_gex else underlying_price
        put_wall = max(strike_gex.items(), key=lambda x: -x[1] if x[1] < 0 else 0, default=(underlying_price, 0))[0] if strike_gex else underlying_price
        
        return {
            'symbol': symbol,
            'totalGex': total_gex,
            'flipPrice': flip_price,
            'callWall': call_wall,
            'putWall': put_wall,
            'expectedMove': underlying_price * 0.01,  # Placeholder
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GEX error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/whale-flows")
def whale_flows(
    symbols: str = Query("SPY,QQQ,NVDA,TSLA,AAPL"),  # Reduced default symbols
    limit: int = Query(20)
):
    """Scan for whale options activity across symbols"""
    # This endpoint makes multiple calls, check budget first
    symbol_list = [s.strip() for s in symbols.split(',')][:5]  # Max 5 symbols
    
    if rate_limiter.calls_remaining() < len(symbol_list) * 2:
        raise HTTPException(status_code=429, detail="Insufficient rate limit budget for whale scan")
    
    try:
        client = get_schwab_client()
        if not client:
            raise HTTPException(status_code=503, detail="Schwab client not available")
        
        symbol_list = [s.strip() for s in symbols.split(',')]
        all_flows = []
        
        # Get next Friday expiry
        today = datetime.now().date()
        days_to_friday = (4 - today.weekday()) % 7 or 7
        expiry = (today + timedelta(days=days_to_friday)).strftime('%Y-%m-%d')
        
        for symbol in symbol_list[:5]:  # Max 5 symbols to stay within rate limits
            if not rate_limiter.acquire():
                logger.warning(f"Rate limit hit during whale scan at {symbol}")
                break
                
            try:
                quote = client.get_quote(symbol)
                if not quote or symbol not in quote:
                    continue
                
                price = quote[symbol].get('quote', {}).get('lastPrice', 0)
                if not price:
                    continue
                
                if not rate_limiter.acquire():
                    logger.warning(f"Rate limit hit before options chain for {symbol}")
                    continue
                    
                chain = client.get_options_chain(
                    symbol=symbol,
                    contract_type='ALL',
                    from_date=expiry,
                    to_date=expiry
                )
                
                if not chain:
                    continue
                
                # Find high vol/OI options
                for opt_type, exp_map_key in [('CALL', 'callExpDateMap'), ('PUT', 'putExpDateMap')]:
                    exp_map = chain.get(exp_map_key, {})
                    for exp_key, strikes in exp_map.items():
                        exp_date = exp_key.split(':')[0]
                        for strike_str, contracts in strikes.items():
                            if not contracts:
                                continue
                            c = contracts[0]
                            vol = c.get('totalVolume', 0)
                            oi = max(c.get('openInterest', 1), 1)
                            mark = c.get('mark', 0)
                            
                            if vol > 500 and mark > 0:
                                vol_oi = vol / oi
                                premium = vol * mark * 100
                                
                                if vol_oi > 2 or premium > 500000:
                                    all_flows.append({
                                        'symbol': symbol,
                                        'type': opt_type,
                                        'strike': float(strike_str),
                                        'expiry': exp_date,
                                        'volume': vol,
                                        'openInterest': oi,
                                        'volOiRatio': round(vol_oi, 2),
                                        'premium': premium,
                                        'whaleScore': int(vol * vol_oi),
                                        'timestamp': datetime.now().isoformat(),
                                    })
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by whale score and limit
        all_flows.sort(key=lambda x: x['whaleScore'], reverse=True)
        return {'data': all_flows[:limit]}
    
    except Exception as e:
        logger.error(f"Whale flows error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting Schwab Bridge on http://localhost:8502")
    uvicorn.run(app, host="0.0.0.0", port=8502, log_level="info")
