#!/usr/bin/env python3
"""
Update watchlist for Discord bot from Droplet API
Run this periodically to sync the bot's watchlist with the Trading Dashboard
"""

import requests
import json
from pathlib import Path

def update_watchlist():
    """Fetch watchlist from Droplet API and save to local file"""
    try:
        # Fetch from Droplet API
        api_url = "http://138.197.210.166:8000/api/watchlist"
        response = requests.get(api_url, params={'order_by': 'daily_change_pct', 'limit': 150}, timeout=10)
        response.raise_for_status()
        api_response = response.json()
        
        # API returns {count: X, data: [...]}
        watchlist_data = api_response.get('data', [])
        
        if not watchlist_data:
            print("❌ No data received from Droplet API")
            return False
        
        # Extract symbols
        symbols = [item['symbol'] for item in watchlist_data]
        
        # Save to bot's watchlist file
        bot_dir = Path(__file__).parent
        watchlist_file = bot_dir / 'bot_watchlist.json'
        
        watchlist_config = {
            'symbols': symbols,
            'count': len(symbols),
            'last_updated': watchlist_data[0].get('last_updated', 'unknown') if watchlist_data else 'unknown'
        }
        
        with open(watchlist_file, 'w') as f:
            json.dump(watchlist_config, f, indent=2)
        
        print(f"✅ Updated watchlist with {len(symbols)} symbols")
        print(f"📁 Saved to: {watchlist_file}")
        print(f"🔝 Top 5 gainers: {', '.join(symbols[:5])}")
        print(f"🔻 Top 5 losers: {', '.join(symbols[-5:])}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating watchlist: {e}")
        return False

if __name__ == '__main__':
    update_watchlist()
