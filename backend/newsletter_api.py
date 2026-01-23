#!/usr/bin/env python3
"""
Newsletter Scanner API - Lightweight FastAPI service for AI Copilot
Provides endpoints for newsletter scanner data (SQLite only, no PostgreSQL)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sqlite3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Newsletter Scanner API",
    description="API for AI Copilot to fetch scanner data",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
NEWSLETTER_DB_PATH = Path(__file__).parent.parent / "data" / "newsletter_scanner.db"


def get_newsletter_db():
    """Get SQLite connection to newsletter scanner database"""
    if not NEWSLETTER_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Newsletter scanner database not found. Run scanner first.")
    conn = sqlite3.connect(NEWSLETTER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "service": "Newsletter Scanner API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "plays": "/api/newsletter/plays",
            "improving": "/api/newsletter/improving",
            "summary": "/api/newsletter/summary",
            "stock": "/api/newsletter/stock/{symbol}"
        }
    }


@app.get("/api/newsletter/plays")
def get_newsletter_plays(
    query_type: str = Query("all", description="Filter: all, bullish, bearish, weekly, monthly"),
    min_score: int = Query(60, ge=0, le=100, description="Minimum opportunity score"),
    limit: int = Query(30, ge=1, le=100, description="Number of results")
):
    """
    Get trade plays from newsletter scanner database
    Used by AI Copilot to generate recommendations
    """
    try:
        conn = get_newsletter_db()
        
        if query_type == "bullish":
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE setup_type = 'BULLISH' AND opportunity_score >= ?
                ORDER BY opportunity_score DESC
                LIMIT ?
            """, (min_score, limit))
        elif query_type == "bearish":
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE setup_type = 'BEARISH' AND opportunity_score >= ?
                ORDER BY opportunity_score DESC
                LIMIT ?
            """, (min_score, limit))
        elif query_type == "weekly":
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE timeframe = 'WEEKLY' AND opportunity_score >= 70
                ORDER BY opportunity_score DESC
                LIMIT ?
            """, (limit,))
        elif query_type == "monthly":
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE timeframe = 'MONTHLY' AND opportunity_score >= 65
                ORDER BY opportunity_score DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE opportunity_score >= ?
                ORDER BY opportunity_score DESC
                LIMIT ?
            """, (min_score, limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {
            "status": "success",
            "query_type": query_type,
            "min_score": min_score,
            "count": len(results),
            "data": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching newsletter plays: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/newsletter/improving")
def get_improving_stocks(
    limit: int = Query(15, ge=1, le=50, description="Number of results")
):
    """
    Get stocks with improving momentum (score rising over recent scans)
    """
    try:
        conn = get_newsletter_db()
        cursor = conn.execute("""
            WITH recent AS (
                SELECT ticker, AVG(opportunity_score) as recent_avg
                FROM scans
                WHERE scan_time >= datetime('now', '-1 days')
                GROUP BY ticker
            ),
            older AS (
                SELECT ticker, AVG(opportunity_score) as older_avg
                FROM scans  
                WHERE scan_time >= datetime('now', '-3 days')
                  AND scan_time < datetime('now', '-1 days')
                GROUP BY ticker
            )
            SELECT 
                r.ticker,
                r.recent_avg,
                COALESCE(o.older_avg, r.recent_avg) as older_avg,
                (r.recent_avg - COALESCE(o.older_avg, r.recent_avg)) as improvement
            FROM recent r
            LEFT JOIN older o ON r.ticker = o.ticker
            WHERE r.recent_avg > COALESCE(o.older_avg, r.recent_avg - 1)
            ORDER BY improvement DESC
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {
            "status": "success",
            "count": len(results),
            "data": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching improving stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/newsletter/summary")
def get_newsletter_summary():
    """
    Get summary stats from newsletter scanner
    """
    try:
        conn = get_newsletter_db()
        
        # Get counts by setup type
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN setup_type = 'BULLISH' THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN setup_type = 'BEARISH' THEN 1 ELSE 0 END) as bearish_count,
                SUM(CASE WHEN timeframe = 'WEEKLY' THEN 1 ELSE 0 END) as weekly_count,
                SUM(CASE WHEN timeframe = 'MONTHLY' THEN 1 ELSE 0 END) as monthly_count,
                AVG(opportunity_score) as avg_score,
                MAX(last_updated) as last_scan
            FROM latest_scores
        """)
        summary = dict(cursor.fetchone())
        
        # Get top 5 by score
        cursor = conn.execute("""
            SELECT ticker, opportunity_score, setup_type, timeframe, current_price
            FROM latest_scores
            ORDER BY opportunity_score DESC
            LIMIT 5
        """)
        top_5 = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "status": "success",
            "summary": summary,
            "top_5": top_5
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching newsletter summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/newsletter/stock/{symbol}")
def get_newsletter_stock(symbol: str):
    """
    Get scanner data for a specific stock
    """
    try:
        conn = get_newsletter_db()
        symbol = symbol.upper()
        
        # Get latest score
        cursor = conn.execute("""
            SELECT * FROM latest_scores WHERE ticker = ?
        """, (symbol,))
        latest = cursor.fetchone()
        
        if not latest:
            conn.close()
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Get historical scans (last 7 days)
        cursor = conn.execute("""
            SELECT scan_time, opportunity_score, setup_type, current_price, rsi
            FROM scans
            WHERE ticker = ?
            ORDER BY scan_time DESC
            LIMIT 20
        """, (symbol,))
        history = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "status": "success",
            "symbol": symbol,
            "latest": dict(latest),
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
