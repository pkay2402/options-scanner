#!/usr/bin/env python3
"""
Test script for ETF Momentum Scanner
Tests the scanning logic without requiring Discord connection
"""
import sys
from pathlib import Path
import asyncio

# Set up paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from bot.commands.etf_momentum import ETFMomentumCommands


class MockBot:
    """Mock bot for testing"""
    def __init__(self):
        self.schwab_service = None
    
    def get_channel(self, channel_id):
        return None


async def test_etf_scanner():
    """Test the ETF momentum scanner"""
    print("=" * 60)
    print("🧪 Testing ETF Momentum Scanner")
    print("=" * 60)
    
    # Create mock bot and command instance
    bot = MockBot()
    scanner = ETFMomentumCommands(bot)
    
    print(f"\n📋 Loaded {len(scanner.etf_list)} ETFs from CSV")
    print(f"   Sample ETFs: {', '.join(scanner.etf_list[:10])}")
    
    print("\n⏰ Scan Schedule:")
    for scan_time in scanner.first_hour_scans:
        print(f"   • {scan_time.strftime('%I:%M %p ET')}")
    print(f"   • {scanner.close_scan_time.strftime('%I:%M %p ET')} (Close)")
    
    print("\n🔍 Testing momentum calculation on sample ETFs...")
    print("-" * 60)
    
    test_symbols = ['SPY', 'QQQ', 'TQQQ', 'SOXL', 'TNA']
    results = []
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...", end=" ")
        try:
            result = scanner.calculate_etf_momentum(symbol)
            if result:
                print(f"✅ PASS")
                print(f"   Month: {result['month_change']:+.1f}% | Week: {result['week_change']:+.1f}% | Day: {result['day_change']:+.1f}%")
                print(f"   Score: {result['momentum_score']:.2f}")
                results.append(result)
            else:
                print(f"❌ Does not meet criteria")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Test Complete! Found {len(results)} qualifying ETFs from sample")
    
    if results:
        print("\n🏆 Top Result:")
        top = max(results, key=lambda x: x['momentum_score'])
        print(f"   {top['symbol']}: Score {top['momentum_score']:.2f}")
        print(f"   Month: {top['month_change']:+.1f}% | Week: {top['week_change']:+.1f}% | Day: {top['day_change']:+.1f}%")
    
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_etf_scanner())
