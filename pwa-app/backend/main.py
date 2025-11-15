"""
FastAPI Backend for Options Flow PWA
Integrates with existing Schwab API and analysis modules
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional, List
import asyncio
import json
import sys
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schwab_client import SchwabClient
from src.analysis.big_trades import BigTradesDetector
from typing import Dict

app = FastAPI(
    title="Options Flow Pro API",
    description="Real-time options trading intelligence API",
    version="1.0.0"
)

# CORS configuration for PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bootstrap: allow tokens via env var for hosted deployments
def _ensure_token_file_from_env():
    try:
        env_json = os.getenv('SCHWAB_CLIENT_JSON')
        if not env_json:
            return
        # Write to the location expected by SchwabClient (repo root)
        root_path = Path(__file__).parent.parent.parent
        token_path = root_path / 'schwab_client.json'
        token_path.write_text(env_json)
        logger.info(f"Wrote schwab_client.json from env to {token_path}")
    except Exception as e:
        logger.warning(f"Could not write schwab_client.json from env: {e}")

_ensure_token_file_from_env()

# Global Schwab client instance
schwab_client = None

def get_schwab_client():
    """Get or create authenticated Schwab client"""
    global schwab_client
    try:
        if schwab_client is None:
            schwab_client = SchwabClient()
        
        # Try to ensure we have a valid session
        if not schwab_client.ensure_valid_session():
            logger.warning("Session validation failed, attempting re-authentication...")
            if not schwab_client.authenticate():
                raise HTTPException(
                    status_code=401, 
                    detail="Failed to authenticate with Schwab API. Token may be expired. Please run setup again."
                )
        return schwab_client
    except Exception as e:
        logger.error(f"Error getting Schwab client: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication error: {str(e)}"
        )


# ============= REST Endpoints =============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Options Flow Pro API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/auth/status")
async def auth_status():
    """Check Schwab authentication status"""
    try:
        global schwab_client
        
        # Try to get/create client
        if schwab_client is None:
            schwab_client = SchwabClient()
        
        # Check if session is valid
        is_valid = schwab_client.check_session()
        
        if not is_valid:
            # Try to refresh
            logger.info("Session invalid, attempting refresh...")
            is_valid = schwab_client.refresh_token()
        
        return {
            "authenticated": is_valid,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Schwab API" if is_valid else "Token expired or invalid. Please re-authenticate."
        }
    except Exception as e:
        logger.error(f"Auth status check failed: {str(e)}")
        return {
            "authenticated": False,
            "error": str(e),
            "message": "Unable to connect to Schwab API. Please check credentials and re-authenticate."
        }

@app.post("/api/auth/refresh")
async def refresh_auth():
    """Manually trigger token refresh"""
    try:
        global schwab_client
        
        if schwab_client is None:
            schwab_client = SchwabClient()
        
        success = schwab_client.refresh_token()
        
        if success:
            return {
                "success": True,
                "message": "Token refreshed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Failed to refresh token. You may need to re-authenticate using the setup script."
            )
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing token: {str(e)}"
        )

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get current quote for a symbol"""
    try:
        client = get_schwab_client()
        quote = client.get_quote(symbol.upper())
        
        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
        
        # Extract price data
        symbol_data = quote.get(symbol.upper(), {})
        quote_data = symbol_data.get('quote', {})
        
        return {
            "symbol": symbol.upper(),
            "price": quote_data.get('lastPrice', 0),
            "change": quote_data.get('netChange', 0),
            "changePercent": quote_data.get('netPercentChange', 0),
            "volume": quote_data.get('totalVolume', 0),
            "bid": quote_data.get('bidPrice', 0),
            "ask": quote_data.get('askPrice', 0),
            "high": quote_data.get('highPrice', 0),
            "low": quote_data.get('lowPrice', 0),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/expiries/{symbol}")
async def get_expiries(symbol: str):
    """Return available expiration dates for a symbol (YYYY-MM-DD)."""
    try:
        client = get_schwab_client()
        # Request a reasonable window: next 60 days
        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')

        options = client.get_options_chain(
            symbol=symbol.upper(),
            contract_type='ALL',
            from_date=from_date,
            to_date=to_date
        )

        expiries = set()
        # Keys may look like '2025-11-21:7' → take left side of ':'
        for key in (options.get('callExpDateMap') or {}).keys():
            expiries.add(key.split(':')[0])
        for key in (options.get('putExpDateMap') or {}).keys():
            expiries.add(key.split(':')[0])

        if not expiries:
            return {"symbol": symbol.upper(), "expiries": []}

        sorted_exp = sorted(expiries)
        return {
            "symbol": symbol.upper(),
            "expiries": sorted_exp,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/options-chain/{symbol}")
async def get_options_chain(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """Get options chain for a symbol"""
    try:
        client = get_schwab_client()
        
        # Default to next week if no dates provided
        if not from_date:
            from_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = from_date
        
        options = client.get_options_chain(
            symbol=symbol.upper(),
            contract_type='ALL',
            from_date=from_date,
            to_date=to_date
        )
        
        if not options:
            raise HTTPException(status_code=404, detail=f"Options chain not found for {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "underlying_price": options.get('underlyingPrice', 0),
            "options": options,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volume-walls/{symbol}")
async def get_volume_walls(
    symbol: str,
    expiry_date: str = Query(..., description="Expiration date in YYYY-MM-DD format"),
    strike_spacing: float = Query(5.0, description="Strike spacing"),
    num_strikes: int = Query(20, description="Number of strikes each side")
):
    """Calculate volume walls for a symbol"""
    try:
        import numpy as np
        
        client = get_schwab_client()
        
        # Get current price
        quote = client.get_quote(symbol.upper())
        underlying_price = quote.get(symbol.upper(), {}).get('quote', {}).get('lastPrice', 0)
        
        if not underlying_price:
            raise HTTPException(status_code=404, detail=f"Could not get price for {symbol}")
        
        # Get options chain
        options = client.get_options_chain(
            symbol=symbol.upper(),
            contract_type='ALL',
            from_date=expiry_date,
            to_date=expiry_date
        )
        
        if not options or 'callExpDateMap' not in options:
            raise HTTPException(status_code=404, detail="No options data available")
        
        # Calculate walls (reuse logic from pages/3_🧱_Option_Volume_Walls.py)
        base_strike = np.floor(underlying_price / 10) * 10
        strikes_above = [base_strike + strike_spacing * i for i in range(num_strikes + 1)]
        strikes_below = [base_strike - strike_spacing * i for i in range(1, num_strikes + 1)]
        all_strikes = sorted(strikes_below + strikes_above)
        
        call_volumes = {}
        put_volumes = {}
        call_oi = {}
        put_oi = {}
        call_gamma = {}
        put_gamma = {}
        
        # Extract call data
        if 'callExpDateMap' in options:
            for exp_date, strikes in options['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if strike in all_strikes and contracts:
                        contract = contracts[0]
                        call_volumes[strike] = call_volumes.get(strike, 0) + contract.get('totalVolume', 0)
                        call_oi[strike] = call_oi.get(strike, 0) + contract.get('openInterest', 0)
                        call_gamma[strike] = call_gamma.get(strike, 0) + contract.get('gamma', 0)
        
        # Extract put data
        if 'putExpDateMap' in options:
            for exp_date, strikes in options['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if strike in all_strikes and contracts:
                        contract = contracts[0]
                        put_volumes[strike] = put_volumes.get(strike, 0) + contract.get('totalVolume', 0)
                        put_oi[strike] = put_oi.get(strike, 0) + contract.get('openInterest', 0)
                        put_gamma[strike] = put_gamma.get(strike, 0) + contract.get('gamma', 0)
        
        # Calculate net volumes and GEX
        net_volumes = {}
        gex_by_strike = {}
        for strike in all_strikes:
            call_vol = call_volumes.get(strike, 0)
            put_vol = put_volumes.get(strike, 0)
            net_volumes[strike] = put_vol - call_vol
            
            # GEX calculation
            call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01
            put_gex = put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * underlying_price * 0.01 * -1
            gex_by_strike[strike] = call_gex + put_gex
        
        # Find walls
        call_wall_strike = max(call_volumes.items(), key=lambda x: x[1])[0] if call_volumes else None
        put_wall_strike = max(put_volumes.items(), key=lambda x: x[1])[0] if put_volumes else None
        
        # Find net walls
        bullish_strikes = {k: abs(v) for k, v in net_volumes.items() if v < 0}
        bearish_strikes = {k: abs(v) for k, v in net_volumes.items() if v > 0}
        
        net_call_wall_strike = max(bullish_strikes.items(), key=lambda x: x[1])[0] if bullish_strikes else None
        net_put_wall_strike = max(bearish_strikes.items(), key=lambda x: x[1])[0] if bearish_strikes else None
        
        # Find flip level
        strikes_near_price = [s for s in all_strikes if abs(s - underlying_price) < strike_spacing * 5]
        flip_strike = None
        for i in range(len(strikes_near_price) - 1):
            s1, s2 = strikes_near_price[i], strikes_near_price[i + 1]
            if net_volumes.get(s1, 0) * net_volumes.get(s2, 0) < 0:
                flip_strike = s1 if abs(s1 - underlying_price) < abs(s2 - underlying_price) else s2
                break
        
        return {
            "symbol": symbol.upper(),
            "underlying_price": underlying_price,
            "expiry_date": expiry_date,
            "call_wall": {
                "strike": call_wall_strike,
                "volume": call_volumes.get(call_wall_strike, 0) if call_wall_strike else 0,
                "oi": call_oi.get(call_wall_strike, 0) if call_wall_strike else 0,
                "gex": gex_by_strike.get(call_wall_strike, 0) if call_wall_strike else 0
            },
            "put_wall": {
                "strike": put_wall_strike,
                "volume": put_volumes.get(put_wall_strike, 0) if put_wall_strike else 0,
                "oi": put_oi.get(put_wall_strike, 0) if put_wall_strike else 0,
                "gex": gex_by_strike.get(put_wall_strike, 0) if put_wall_strike else 0
            },
            "net_call_wall": {
                "strike": net_call_wall_strike,
                "volume": net_volumes.get(net_call_wall_strike, 0) if net_call_wall_strike else 0
            },
            "net_put_wall": {
                "strike": net_put_wall_strike,
                "volume": net_volumes.get(net_put_wall_strike, 0) if net_put_wall_strike else 0
            },
            "flip_level": flip_strike,
            "all_strikes": all_strikes,
            "call_volumes": call_volumes,
            "put_volumes": put_volumes,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "net_volumes": net_volumes,
            "gex_by_strike": gex_by_strike,
            "totals": {
                "call_vol": sum(call_volumes.values()),
                "put_vol": sum(put_volumes.values()),
                "net_vol": sum(put_volumes.values()) - sum(call_volumes.values()),
                "total_gex": sum(gex_by_strike.values())
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/option-finder/{symbol}")
async def option_finder(
    symbol: str,
    num_expiries: int = Query(5, ge=1, le=12),
    option_type: str = Query("ALL", description="ALL|CALLS|PUTS"),
    min_open_interest: int = Query(0, ge=0),
    moneyness_min: float = Query(-50.0),
    moneyness_max: float = Query(50.0),
    top_n: int = Query(5, ge=1, le=50)
):
    """Compute gamma metrics and Net GEX heatmap across expiries.

    Returns:
    - underlying_price
    - expiries, strikes
    - heatmap: x=expiries, y=strikes, z=net_gex values
    - top_calls, top_puts: top strikes by absolute dollar gamma
    """
    try:
        client = get_schwab_client()

        # Get quote for underlying price
        quote = client.get_quote(symbol.upper())
        underlying_price = quote.get(symbol.upper(), {}).get('quote', {}).get('lastPrice', 0) or 0

        # Fetch options for next 60 days and then slice expiries
        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        options = client.get_options_chain(
            symbol=symbol.upper(),
            contract_type='ALL',
            from_date=from_date,
            to_date=to_date
        )

        if not options:
            raise HTTPException(status_code=404, detail="No options data available")

        call_map: Dict[str, Dict[str, list]] = options.get('callExpDateMap') or {}
        put_map: Dict[str, Dict[str, list]] = options.get('putExpDateMap') or {}

        # Collect sorted expiries (strip ':days')
        exp_set = set()
        for k in call_map.keys():
            exp_set.add(k.split(':')[0])
        for k in put_map.keys():
            exp_set.add(k.split(':')[0])
        expiries = sorted(exp_set)[:num_expiries]
        if not expiries:
            return {"symbol": symbol.upper(), "underlying_price": underlying_price, "expiries": [], "strikes": [], "heatmap": {"x": [], "y": [], "z": []}, "top_calls": [], "top_puts": []}

        # Helper to iterate contracts for chosen expiries
        def iter_contracts(exp_map: Dict[str, Dict[str, list]]):
            for exp_key, strikes in exp_map.items():
                exp_date = exp_key.split(':')[0]
                if exp_date not in expiries:
                    continue
                for strike_str, contracts in (strikes or {}).items():
                    if not contracts:
                        continue
                    yield exp_date, float(strike_str), contracts[0]

        rows = []
        # Process calls
        for exp_date, strike, c in iter_contracts(call_map):
            rows.append((
                strike,
                exp_date,
                'Call',
                float(c.get('gamma', 0) or 0),
                float(c.get('delta', 0) or 0),
                float(c.get('vega', 0) or 0),
                int(c.get('totalVolume', 0) or 0),
                int(c.get('openInterest', 0) or 0),
                float(c.get('bid', 0) or 0),
                float(c.get('ask', 0) or 0),
                float(c.get('last', 0) or 0),
                float(c.get('volatility', 0) or 0)
            ))
        # Process puts
        for exp_date, strike, c in iter_contracts(put_map):
            rows.append((
                strike,
                exp_date,
                'Put',
                float(c.get('gamma', 0) or 0),
                float(c.get('delta', 0) or 0),
                float(c.get('vega', 0) or 0),
                int(c.get('totalVolume', 0) or 0),
                int(c.get('openInterest', 0) or 0),
                float(c.get('bid', 0) or 0),
                float(c.get('ask', 0) or 0),
                float(c.get('last', 0) or 0),
                float(c.get('volatility', 0) or 0)
            ))

        # Compute metrics and filter
        def days_to(exp_date: str) -> int:
            try:
                return (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
            except Exception:
                return 0

        results = []
        for (strike, exp_date, opt_type, gamma, delta, vega, volume, oi, bid, ask, last, vol) in rows:
            if underlying_price and underlying_price > 0:
                moneyness = (strike / underlying_price - 1) * 100.0
            else:
                moneyness = 0.0

            if option_type.upper() == 'CALLS' and opt_type != 'Call':
                continue
            if option_type.upper() == 'PUTS' and opt_type != 'Put':
                continue
            if oi < min_open_interest:
                continue
            if not (moneyness_min <= moneyness <= moneyness_max):
                continue

            # Dollar GEX (professional convention)
            dollar_gex = gamma * 100.0 * oi * (underlying_price ** 2) * 0.01 if underlying_price else gamma * 100.0 * oi * 100
            signed_gex = dollar_gex if opt_type == 'Call' else -dollar_gex
            results.append({
                "strike": strike,
                "expiry": exp_date,
                "days_to_exp": days_to(exp_date),
                "option_type": opt_type,
                "gamma": gamma,
                "delta": delta,
                "vega": vega,
                "volume": volume,
                "open_interest": oi,
                "bid": bid,
                "ask": ask,
                "last": last,
                "notional_gamma": abs(dollar_gex),
                "signed_notional_gamma": signed_gex,
                "moneyness": moneyness,
                "implied_volatility": (vol or 0) * 100.0,
            })

        # Prepare heatmap across chosen expiries/strikes
        strike_set = sorted({r[0] for r in rows})
        # For practicality, limit strikes to 60 around current price
        if underlying_price:
            around = [s for s in strike_set if underlying_price * 0.85 <= s <= underlying_price * 1.15]
            if around:
                strike_set = around
        strike_set = sorted(strike_set)[:120]

        # Build z matrix: sum signed_notional_gamma per (strike, expiry)
        z = []
        for s in strike_set:
            row_vals = []
            for exp in expiries:
                val = 0.0
                for item in results:
                    if item['strike'] == s and item['expiry'] == exp:
                        val += float(item['signed_notional_gamma'])
                row_vals.append(val)
            z.append(row_vals)

        # Top lists
        top_calls = [r for r in results if r['option_type'] == 'Call']
        top_calls.sort(key=lambda x: x['notional_gamma'], reverse=True)
        top_calls = top_calls[:top_n]

        top_puts = [r for r in results if r['option_type'] == 'Put']
        top_puts.sort(key=lambda x: x['notional_gamma'], reverse=True)
        top_puts = top_puts[:top_n]

        return {
            "symbol": symbol.upper(),
            "underlying_price": underlying_price,
            "expiries": expiries,
            "strikes": strike_set,
            "heatmap": {"x": expiries, "y": strike_set, "z": z},
            "top_calls": top_calls,
            "top_puts": top_puts,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/net-premium-heatmap/{symbol}")
async def net_premium_heatmap(
    symbol: str,
    num_expiries: int = Query(4, ge=1, le=8)
):
    """Compute Net Premium (Call Premium - Put Premium) heatmap across strikes and expiries.

    Premium per contract approximated as volume × mark × 100.
    Returns expiries, strikes, and z matrix for plotting.
    """
    try:
        client = get_schwab_client()
        # Underlying for labeling and filtering range
        quote = client.get_quote(symbol.upper())
        underlying_price = quote.get(symbol.upper(), {}).get('quote', {}).get('lastPrice', 0) or 0

        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        options = client.get_options_chain(
            symbol=symbol.upper(),
            contract_type='ALL',
            from_date=from_date,
            to_date=to_date
        )
        if not options:
            raise HTTPException(status_code=404, detail="No options data available")

        call_map = options.get('callExpDateMap') or {}
        put_map = options.get('putExpDateMap') or {}

        # Build premium matrix keyed by (strike, expiry)
        premium: Dict[tuple, Dict[str, float]] = {}
        def add_premium(exp_map: Dict[str, Dict[str, list]], side: str):
            for exp_key, strikes in exp_map.items():
                exp_date = exp_key.split(':')[0]
                for strike_str, contracts in (strikes or {}).items():
                    if not contracts:
                        continue
                    c = contracts[0]
                    try:
                        strike = float(strike_str)
                    except Exception:
                        continue
                    vol = float(c.get('totalVolume', 0) or 0)
                    mark = float(c.get('mark', 0) or 0)
                    notional = vol * mark * 100.0
                    key = (strike, exp_date)
                    if key not in premium:
                        premium[key] = {"call": 0.0, "put": 0.0}
                    premium[key][side] += notional

        add_premium(call_map, 'call')
        add_premium(put_map, 'put')

        if not premium:
            return {"symbol": symbol.upper(), "underlying_price": underlying_price, "expiries": [], "strikes": [], "heatmap": {"x": [], "y": [], "z": []}}

        all_strikes = sorted({k[0] for k in premium.keys()})
        all_expiries = sorted({k[1] for k in premium.keys()})[:num_expiries]

        # Focus strikes around spot for readability
        if underlying_price:
            strikes = [s for s in all_strikes if underlying_price * 0.95 <= s <= underlying_price * 1.05]
            if not strikes:
                strikes = sorted(all_strikes, key=lambda s: abs(s - underlying_price))[:12]
        else:
            strikes = all_strikes[:12]
        strikes = sorted(strikes)

        # Build z matrix
        z = []
        for s in strikes:
            row = []
            for exp in all_expiries:
                val = 0.0
                key = (s, exp)
                if key in premium:
                    val = float(premium[key]["call"]) - float(premium[key]["put"])
                row.append(val)
            z.append(row)

        return {
            "symbol": symbol.upper(),
            "underlying_price": underlying_price,
            "expiries": all_expiries,
            "strikes": strikes,
            "heatmap": {"x": all_expiries, "y": strikes, "z": z},
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price-history/{symbol}")
async def get_price_history(
    symbol: str,
    frequency: int = Query(5, description="Frequency in minutes"),
    hours: int = Query(24, description="Hours of history")
):
    """Get intraday price history"""
    try:
        client = get_schwab_client()
        
        now = datetime.now()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(hours=hours)).timestamp() * 1000)
        
        price_history = client.get_price_history(
            symbol=symbol.upper(),
            frequency_type='minute',
            frequency=frequency,
            start_date=start_time,
            end_date=end_time,
            need_extended_hours=False
        )
        
        if not price_history or 'candles' not in price_history:
            raise HTTPException(status_code=404, detail="No price history available")
        
        return {
            "symbol": symbol.upper(),
            "candles": price_history['candles'],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/big-trades/{symbol}")
async def scan_big_trades(
    symbol: str,
    min_premium: float = Query(100000, description="Minimum premium in dollars")
):
    """Scan for big trades in a symbol"""
    try:
        client = get_schwab_client()
        detector = BigTradesDetector(client)
        
        # Get next expiry
        exp_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        big_trades = detector.scan_for_big_trades(
            symbols=[symbol.upper()],
            min_premium=min_premium
        )
        
        return {
            "symbol": symbol.upper(),
            "big_trades": big_trades,
            "min_premium": min_premium,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= WebSocket Endpoints =============

class ConnectionManager:
    """Manage WebSocket connections"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

# Connection managers for different streams
flow_manager = ConnectionManager()
alerts_manager = ConnectionManager()


@app.websocket("/ws/flow")
async def flow_scanner_websocket(websocket: WebSocket):
    """WebSocket for real-time options flow"""
    await flow_manager.connect(websocket)
    
    try:
        while True:
            # Poll for new flow data every 10 seconds
            await asyncio.sleep(10)
            
            try:
                client = get_schwab_client()
                
                # Get flow for watchlist (SPY, QQQ, etc.)
                watchlist = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
                
                for symbol in watchlist:
                    # Get recent big trades
                    detector = BigTradesDetector(client)
                    big_trades = detector.scan_for_big_trades([symbol], min_premium=50000)
                    
                    if big_trades:
                        await websocket.send_json({
                            "type": "flow_update",
                            "symbol": symbol,
                            "trades": big_trades,
                            "timestamp": datetime.now().isoformat()
                        })
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        flow_manager.disconnect(websocket)


@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket for real-time alerts"""
    await alerts_manager.connect(websocket)
    
    try:
        # Receive alert configuration from client
        config = await websocket.receive_json()
        watchlist = config.get('symbols', ['SPY'])
        
        while True:
            await asyncio.sleep(180)  # Check every 3 minutes
            
            try:
                client = get_schwab_client()
                
                for symbol in watchlist:
                    # Get volume walls
                    quote = client.get_quote(symbol)
                    underlying_price = quote.get(symbol, {}).get('quote', {}).get('lastPrice', 0)
                    
                    # Check if price near key levels (simplified check)
                    # In production, call the full volume walls endpoint
                    
                    # Example alert
                    await websocket.send_json({
                        "type": "alert",
                        "priority": "HIGH",
                        "symbol": symbol,
                        "title": f"{symbol} Price Update",
                        "message": f"Current price: ${underlying_price:.2f}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        alerts_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
