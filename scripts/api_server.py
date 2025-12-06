"""
Simple REST API to serve cached market data
Runs on droplet and exposes data to Streamlit Cloud
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_cache import MarketCache

app = Flask(__name__)
CORS(app)  # Allow requests from Streamlit Cloud

cache = MarketCache()

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
    """
    order_by = request.args.get('order_by', 'daily_change_pct')
    data = cache.get_watchlist(order_by=order_by)
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

if __name__ == '__main__':
    # Run on port 8000, accessible from external IPs
    print("=" * 60)
    print("Options Scanner API Server")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  GET /                     - API status")
    print("  GET /health               - Health check")
    print("  GET /api/watchlist        - Get watchlist data")
    print("  GET /api/whale_flows      - Get whale flows")
    print("  GET /api/stats            - Cache statistics")
    print("  GET /api/last_update      - Last update times")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=False)
