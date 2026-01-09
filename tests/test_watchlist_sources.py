#!/usr/bin/env python3
"""
Test loading watchlist from Droplet API
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("TESTING WATCHLIST SOURCES")
print("="*80 + "\n")

# Test 1: Droplet API
print("1. Testing Droplet API watchlist...")
try:
    from src.utils.droplet_api import DropletAPI
    
    api = DropletAPI()
    watchlist_data = api.get_watchlist(order_by='daily_change_pct', limit=150)
    
    if watchlist_data:
        symbols = [item['symbol'] for item in watchlist_data]
        print(f"   ✅ Loaded {len(symbols)} symbols from Droplet API")
        print(f"   Top 10: {', '.join(symbols[:10])}")
        print(f"   Sample data for {symbols[0]}:")
        print(f"      Price: ${watchlist_data[0].get('price', 'N/A'):.2f}")
        print(f"      Change: {watchlist_data[0].get('daily_change_pct', 'N/A'):+.2f}%")
        print(f"      Volume: {watchlist_data[0].get('volume', 'N/A'):,}")
    else:
        print("   ❌ No data returned from Droplet API")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: user_preferences.json
print("\n2. Testing user_preferences.json watchlist...")
try:
    import json
    prefs_path = project_root / "user_preferences.json"
    if prefs_path.exists():
        with open(prefs_path, 'r') as f:
            prefs = json.load(f)
            symbols = prefs.get('watchlist', [])
            print(f"   ✅ Loaded {len(symbols)} symbols from user_preferences.json")
            print(f"   First 10: {', '.join(symbols[:10])}")
    else:
        print("   ❌ File not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nThe bot will:")
print("  1. Try Droplet API first (same as Trading Dashboard)")
print("  2. Fallback to user_preferences.json if API fails")
print("\nBenefit: Always uses the freshest watchlist from your database!")
print("="*80 + "\n")
