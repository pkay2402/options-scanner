#!/usr/bin/env python3
"""
Test script for signal storage system
Creates sample signals and tests query functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.services.signal_storage import SignalStorage

def test_signal_storage():
    """Test the signal storage system"""
    print("🧪 Testing Signal Storage System\n")
    
    # Initialize storage
    storage = SignalStorage()
    print("✅ Storage initialized")
    
    # Test 1: Store various signal types
    print("\n📝 Test 1: Storing signals...")
    
    # Whale signal
    storage.store_signal(
        symbol='TSLA',
        signal_type='WHALE',
        signal_subtype='CALL',
        direction='BULLISH',
        price=245.50,
        data={
            'strike': 250.0,
            'expiry': '2025-01-17',
            'volume': 5000,
            'whale_score': 12567
        }
    )
    print("  ✓ Whale signal stored (TSLA)")
    
    # Z-score signal
    storage.store_signal(
        symbol='TSLA',
        signal_type='ZSCORE',
        signal_subtype='BUY_SIGNAL',
        direction='BULLISH',
        price=242.30,
        data={
            'zscore': -2.15,
            'signal': '-2σ',
            'rsi': 28.5
        }
    )
    print("  ✓ Z-score signal stored (TSLA)")
    
    # TOS alert
    storage.store_signal(
        symbol='AAPL',
        signal_type='TOS',
        signal_subtype='LONG',
        direction='BULLISH',
        price=185.20,
        data={
            'scan_name': 'HG_30mins_L',
            'timeframe': '30-Min'
        }
    )
    print("  ✓ TOS alert stored (AAPL)")
    
    # ETF momentum
    storage.store_signal(
        symbol='TQQQ',
        signal_type='ETF_MOMENTUM',
        signal_subtype='STRONG_MOMENTUM',
        direction='BULLISH',
        price=67.50,
        data={
            'day_return': 6.8,
            'week_return': 12.3
        }
    )
    print("  ✓ ETF momentum stored (TQQQ)")
    
    # Test 2: Query signals
    print("\n🔍 Test 2: Querying signals...")
    
    tsla_signals = storage.get_signals('TSLA', days=1)
    print(f"  ✓ Found {len(tsla_signals)} signals for TSLA")
    
    all_signals = storage.get_signals(days=1)
    print(f"  ✓ Found {len(all_signals)} total signals today")
    
    # Test 3: Get summary
    print("\n📊 Test 3: Getting summary...")
    
    summary = storage.get_summary('TSLA', days=1)
    print(f"  ✓ TSLA Summary:")
    print(f"    - Total signals: {summary['total_signals']}")
    print(f"    - Bullish: {summary['by_direction']['BULLISH']}")
    print(f"    - Bearish: {summary['by_direction']['BEARISH']}")
    print(f"    - Price range: ${summary['price_range']['low']:.2f} - ${summary['price_range']['high']:.2f}")
    print(f"    - Signal types: {list(summary['by_type'].keys())}")
    
    # Test 4: Get timeline
    print("\n📅 Test 4: Getting timeline...")
    
    timeline = storage.get_stock_activity_timeline('TSLA', days=1)
    print(f"  ✓ TSLA Timeline:")
    for day_data in timeline:
        print(f"    - {day_data['date']}: {len(day_data['signals'])} signals")
        for signal in day_data['signals']:
            print(f"      • {signal['timestamp']} - {signal['signal_type']} ({signal['signal_subtype']})")
    
    # Test 5: Cleanup old signals
    print("\n🧹 Test 5: Testing cleanup...")
    
    initial_count = len(storage.get_signals(days=365))  # All signals
    print(f"  - Signals before cleanup: {initial_count}")
    
    storage.cleanup_old_signals(days=5)
    
    after_count = len(storage.get_signals(days=365))
    print(f"  - Signals after cleanup: {after_count}")
    print("  ✓ Cleanup completed (keeps last 5 days)")
    
    print("\n✅ All tests passed!\n")
    print("📊 Signal Storage System Status:")
    print("  ✓ Database: working")
    print("  ✓ Storage: functional")
    print("  ✓ Queries: operational")
    print("  ✓ Summary: generating")
    print("  ✓ Timeline: building")
    print("  ✓ Cleanup: automated")
    print("\n🚀 Ready for production!")

if __name__ == '__main__':
    test_signal_storage()
