"""
FastAPI REST Service for Options Scanner
Provides endpoints for accessing scanned options data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.market_cache import MarketCache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'options_scanner',
    'user': 'options_user',
    'password': 'your_secure_password',
    'port': 5432
}

app = FastAPI(
    title="Options Scanner API",
    description="Real-time options flow analysis with whale flows, fresh positioning, and skew metrics",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Data Models
# ============================================================================

class OpportunityScore(BaseModel):
    symbol: str
    expiry: str
    strike: float
    type: str
    underlying_price: float
    composite_score: int
    signal_type: str
    whale_score: int
    vol_oi_ratio: float
    skew_25d: float
    implied_move_pct: float
    volume: int
    notional: float
    put_call_oi_ratio: float
    call_wall_strike: Optional[float]
    put_wall_strike: Optional[float]
    upper_breakout: float
    lower_breakout: float
    data_timestamp: datetime


class MarketSentiment(BaseModel):
    stocks_analyzed: int
    avg_skew: float
    skew_volatility: float
    avg_iv: float
    avg_pc_ratio: float
    avg_implied_move: float
    extreme_fear_count: int
    extreme_greed_count: int
    high_vol_count: int
    market_sentiment: str
    data_timestamp: datetime


class StockAnalysis(BaseModel):
    symbol: str
    expiry: str
    whale_flows: List[Dict[str, Any]]
    oi_flows: List[Dict[str, Any]]
    skew_metrics: Dict[str, Any]
    composite_score: Optional[int]
    data_timestamp: datetime


class HistoricalSkew(BaseModel):
    symbol: str
    expiry: str
    timestamp: datetime
    skew_25d: float
    avg_iv: float
    put_call_oi_ratio: float
    implied_move_pct: float
    underlying_price: float
    skew_change: Optional[float]
    iv_change: Optional[float]


class ScanStatus(BaseModel):
    scan_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    stocks_scanned: int
    expiries_scanned: int
    total_api_calls: Optional[int]
    duration_seconds: Optional[float]


# ============================================================================
# Database Helpers
# ============================================================================

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute query and return results as list of dicts"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "service": "Options Scanner API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "top_opportunities": "/api/top-opportunities",
            "market_sentiment": "/api/market-sentiment",
            "stock_analysis": "/api/stock/{symbol}",
            "historical_skew": "/api/historical/{symbol}",
            "scan_status": "/api/scan-status",
            "ttm_squeeze": "/api/ttm_squeeze_scanner",
            "vpb_scanner": "/api/vpb_scanner",
            "macd_scanner": "/api/macd_scanner"
        }
    }


@app.get("/api/top-opportunities", response_model=List[OpportunityScore])
def get_top_opportunities(
    limit: int = Query(50, ge=1, le=200, description="Number of results to return"),
    min_composite_score: int = Query(0, ge=0, le=100, description="Minimum composite score"),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    expiry: Optional[str] = Query(None, description="Filter by expiry date (YYYY-MM-DD)")
):
    """
    Get top trading opportunities ranked by composite score
    
    Composite score (0-100) combines:
    - Whale score (institutional positioning)
    - Fresh OI (new capital deployment)
    - Skew alignment (contrarian signals)
    """
    query = """
        SELECT 
            symbol, expiry::text, strike, type, underlying_price,
            composite_score, signal_type, whale_score, vol_oi_ratio,
            skew_25d, implied_move_pct, volume, notional, put_call_oi_ratio,
            call_wall_strike, put_wall_strike, upper_breakout, lower_breakout,
            data_timestamp
        FROM top_opportunities
        WHERE composite_score >= %s
    """
    params = [min_composite_score]
    
    if signal_type:
        query += " AND signal_type = %s"
        params.append(signal_type)
    
    if expiry:
        query += " AND expiry = %s"
        params.append(expiry)
    
    query += " ORDER BY composite_score DESC LIMIT %s"
    params.append(limit)
    
    results = execute_query(query, tuple(params))
    return results


@app.get("/api/market-sentiment", response_model=MarketSentiment)
def get_market_sentiment():
    """
    Get overall market sentiment based on options skew
    
    Analyzes:
    - Average skew across all stocks
    - Put/Call ratios
    - Implied volatility levels
    - Extreme fear/greed counts
    """
    query = "SELECT * FROM market_sentiment LIMIT 1"
    results = execute_query(query)
    
    if not results:
        raise HTTPException(status_code=404, detail="No market sentiment data available")
    
    return results[0]


@app.get("/api/stock/{symbol}", response_model=StockAnalysis)
def get_stock_analysis(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Filter by expiry (YYYY-MM-DD)")
):
    """
    Get complete analysis for a specific stock
    
    Returns:
    - All whale flows
    - All fresh OI positions
    - Skew metrics
    - Composite scoring
    """
    # Get latest scan
    scan_query = "SELECT scan_id FROM latest_scan"
    scan_result = execute_query(scan_query)
    
    if not scan_result:
        raise HTTPException(status_code=404, detail="No scan data available")
    
    scan_id = scan_result[0]['scan_id']
    
    # Get whale flows
    whale_query = """
        SELECT * FROM whale_flows
        WHERE scan_id = %s AND symbol = %s
    """
    whale_params = [scan_id, symbol.upper()]
    
    if expiry:
        whale_query += " AND expiry = %s"
        whale_params.append(expiry)
    
    whale_query += " ORDER BY whale_score DESC"
    whale_flows = execute_query(whale_query, tuple(whale_params))
    
    # Get OI flows
    oi_query = """
        SELECT * FROM oi_flows
        WHERE scan_id = %s AND symbol = %s
    """
    oi_params = [scan_id, symbol.upper()]
    
    if expiry:
        oi_query += " AND expiry = %s"
        oi_params.append(expiry)
    
    oi_query += " ORDER BY oi_score DESC"
    oi_flows = execute_query(oi_query, tuple(oi_params))
    
    # Get skew metrics
    skew_query = """
        SELECT * FROM skew_metrics
        WHERE scan_id = %s AND symbol = %s
    """
    skew_params = [scan_id, symbol.upper()]
    
    if expiry:
        skew_query += " AND expiry = %s"
        skew_params.append(expiry)
    
    skew_query += " ORDER BY timestamp DESC LIMIT 1"
    skew_metrics = execute_query(skew_query, tuple(skew_params))
    
    # Get composite score if available
    composite_query = """
        SELECT composite_score, data_timestamp, expiry::text
        FROM top_opportunities
        WHERE symbol = %s
    """
    composite_params = [symbol.upper()]
    
    if expiry:
        composite_query += " AND expiry = %s"
        composite_params.append(expiry)
    
    composite_query += " ORDER BY composite_score DESC LIMIT 1"
    composite = execute_query(composite_query, tuple(composite_params))
    
    if not whale_flows and not oi_flows and not skew_metrics:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    
    return {
        "symbol": symbol.upper(),
        "expiry": expiry or (composite[0]['expiry'] if composite else None),
        "whale_flows": whale_flows,
        "oi_flows": oi_flows,
        "skew_metrics": skew_metrics[0] if skew_metrics else {},
        "composite_score": composite[0]['composite_score'] if composite else None,
        "data_timestamp": composite[0]['data_timestamp'] if composite else datetime.now()
    }


@app.get("/api/historical/{symbol}", response_model=List[HistoricalSkew])
def get_historical_skew(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of history to return"),
    expiry: Optional[str] = Query(None, description="Filter by expiry (YYYY-MM-DD)")
):
    """
    Get historical skew data for a stock
    
    Track changes in:
    - 25-delta skew
    - Implied volatility
    - Put/Call ratios
    - Implied moves
    """
    query = """
        SELECT 
            symbol, expiry::text, timestamp, skew_25d, avg_iv,
            put_call_oi_ratio, implied_move_pct, underlying_price,
            skew_change, iv_change
        FROM stock_skew_history
        WHERE symbol = %s
        AND timestamp >= %s
    """
    params = [symbol.upper(), datetime.now() - timedelta(hours=hours)]
    
    if expiry:
        query += " AND expiry = %s"
        params.append(expiry)
    
    query += " ORDER BY timestamp DESC"
    
    results = execute_query(query, tuple(params))
    
    if not results:
        raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
    
    return results


@app.get("/api/scan-status", response_model=List[ScanStatus])
def get_scan_status(
    limit: int = Query(10, ge=1, le=100, description="Number of recent scans to return")
):
    """
    Get status of recent scans
    
    Shows:
    - Scan timing and duration
    - Success/failure status
    - API call counts
    - Stocks scanned
    """
    query = """
        SELECT 
            scan_id::text,
            start_time,
            end_time,
            status,
            stocks_scanned,
            expiries_scanned,
            total_api_calls,
            EXTRACT(EPOCH FROM (end_time - start_time)) as duration_seconds
        FROM scan_runs
        ORDER BY start_time DESC
        LIMIT %s
    """
    
    results = execute_query(query, (limit,))
    return results


@app.get("/api/signal-types")
def get_signal_types():
    """Get available signal types with descriptions"""
    return {
        "signal_types": {
            "CONTRARIAN_BULL": {
                "description": "Extreme fear (high skew) + call buying = Bottom signal",
                "interpretation": "Institutions buying calls into panic. Contrarian buy opportunity.",
                "typical_edge": "65-70% win rate on extremes"
            },
            "CONTRARIAN_BEAR": {
                "description": "Greed (low/negative skew) + put buying = Top signal",
                "interpretation": "Institutions buying puts into euphoria. Contrarian short opportunity.",
                "typical_edge": "60-65% win rate on extremes"
            },
            "MOMENTUM_BULL": {
                "description": "Heavy call volume + fresh call OI = Bullish momentum",
                "interpretation": "Sustained institutional call buying. Follow the trend.",
                "typical_edge": "55-60% win rate with confirmation"
            },
            "MOMENTUM_BEAR": {
                "description": "Heavy put volume + fresh put OI = Bearish momentum",
                "interpretation": "Sustained institutional put buying. Follow the trend down.",
                "typical_edge": "55-60% win rate with confirmation"
            },
            "NEUTRAL": {
                "description": "Mixed signals or low conviction",
                "interpretation": "No clear directional bias. Range-bound or unclear.",
                "typical_edge": "Consider premium selling strategies"
            }
        }
    }


@app.get("/api/expiries")
def get_available_expiries():
    """Get all available expiry dates from latest scan"""
    query = """
        SELECT DISTINCT expiry::text
        FROM whale_flows
        WHERE scan_id = (SELECT scan_id FROM latest_scan)
        ORDER BY expiry
    """
    
    results = execute_query(query)
    return {"expiries": [r['expiry'] for r in results]}


@app.get("/api/symbols")
def get_available_symbols():
    """Get all symbols scanned in latest scan"""
    query = """
        SELECT DISTINCT symbol
        FROM whale_flows
        WHERE scan_id = (SELECT scan_id FROM latest_scan)
        ORDER BY symbol
    """
    
    results = execute_query(query)
    return {
        "symbols": [r['symbol'] for r in results],
        "count": len(results)
# ============================================================================
# Technical Scanner Endpoints (TTM, VPB, MACD)
# These read from SQLite cache populated by scanner services
# ============================================================================

@app.get("/api/ttm_squeeze_scanner")
def get_ttm_squeeze_scanner(
    filter: str = Query("all", description="Filter: all, active, fired, bullish, bearish"),
    limit: int = Query(150, ge=1, le=500, description="Number of results to return")
):
    """
    Get TTM Squeeze scanner results
    
    Identifies compression setups and breakouts:
    - Active: Squeeze ON (compression, waiting for breakout)
    - Fired: Recently broke out (bullish or bearish)
    """
    try:
        cache = MarketCache()
        results = cache.get_ttm_squeeze_scanner(filter_type=filter)
        
        # Limit results
        results = results[:limit]
        
        return {
            "status": "success",
            "filter": filter,
            "count": len(results),
            "data": results,
            "last_scan": cache.get_metadata('ttm_last_scan')
        }
    except Exception as e:
        logger.error(f"Error fetching TTM Squeeze data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vpb_scanner")
def get_vpb_scanner(
    filter: str = Query("all", description="Filter: all, buy, sell, volume_surge"),
    limit: int = Query(150, ge=1, le=500, description="Number of results to return")
):
    """
    Get Volume-Price Break (VPB) scanner results
    
    Identifies volume breakouts:
    - Buy: Volume surge + price breakout above 7-day high
    - Sell: Volume surge + price breakdown below 7-day low
    """
    try:
        cache = MarketCache()
        results = cache.get_vpb_scanner(filter_type=filter)
        
        # Limit results
        results = results[:limit]
        
        return {
            "status": "success",
            "filter": filter,
            "count": len(results),
            "data": results,
            "last_scan": cache.get_metadata('vpb_last_scan')
        }
    except Exception as e:
        logger.error(f"Error fetching VPB scanner data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/macd_scanner")
def get_macd_scanner(
    filter: str = Query("all", description="Filter: all, bullish, bearish, crossover"),
    limit: int = Query(150, ge=1, le=500, description="Number of results to return")
):
    """
    Get MACD scanner results
    
    Identifies momentum shifts:
    - Bullish: MACD crossed above signal line
    - Bearish: MACD crossed below signal line
    """
    try:
        cache = MarketCache()
        results = cache.get_macd_scanner(filter_type=filter)
        
        # Limit results
        results = results[:limit]
        
        return {
            "status": "success",
            "filter": filter,
            "count": len(results),
            "data": results,
            "last_scan": cache.get_metadata('macd_last_scan')
        }
    except Exception as e:
        logger.error(f"Error fetching MACD scanner data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
