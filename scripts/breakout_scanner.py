#!/usr/bin/env python3
"""
Breakout Candidate Scanner
Filters watchlist stocks based on:
- Green candle (Close > Open)
- Close within 10-20% of 52-week high
- Volume > 1.2-1.5x 20-day average
- Relative strength vs SPY
- Avoid huge gaps (move already priced in)
"""

import sys
import os
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Configuration
DROPLET_API_URL = os.environ.get("DROPLET_API_URL", "http://138.197.210.166:8000")

# Criteria thresholds
MIN_VOLUME_RATIO = 1.2  # Volume must be 1.2x 20-day avg
MAX_VOLUME_RATIO = 5.0  # Avoid extreme volume spikes
DISTANCE_FROM_HIGH_MIN = 0.0  # At least 0% from high (can be at high)
DISTANCE_FROM_HIGH_MAX = 20.0  # Max 20% from 52-week high
MAX_GAP_PERCENT = 5.0  # Avoid stocks gapping more than 5%
MIN_RS_VS_SPY = 0.0  # Must outperform SPY on the day


def fetch_watchlist() -> List[str]:
    """Fetch watchlist symbols from droplet API"""
    try:
        response = requests.get(
            f"{DROPLET_API_URL}/api/watchlist?order_by=daily_change_pct&limit=100",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json().get('data', [])
            return [item['symbol'] for item in data]
    except Exception as e:
        print(f"Error fetching watchlist: {e}")
    
    # Fallback to local watchlist
    watchlist_file = project_root / "discord-bot" / "bot_watchlist.json"
    if watchlist_file.exists():
        import json
        with open(watchlist_file) as f:
            return json.load(f).get('symbols', [])
    
    return []


def get_price_history(client: SchwabClient, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """Get price history for a symbol"""
    try:
        history = client.get_price_history(
            symbol,
            period_type='month',
            period=3,  # 3 months for 52-week high calculation
            frequency_type='daily',
            frequency=1
        )
        
        if history and 'candles' in history:
            df = pd.DataFrame(history['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            return df
    except Exception as e:
        print(f"  Error getting history for {symbol}: {e}")
    return None


def calculate_metrics_from_history(df: pd.DataFrame) -> Dict:
    """Calculate volume and price metrics from historical data"""
    if df is None or len(df) < 20:
        return {}
    
    # Get latest data
    latest = df.iloc[-1]
    
    # Calculate 20-day average volume
    avg_volume_20d = df['volume'].tail(20).mean()
    
    # Calculate 52-week high (or max from available data)
    high_52w = df['high'].max()
    
    # Get previous day for gap calculation
    if len(df) >= 2:
        prev_close = df.iloc[-2]['close']
    else:
        prev_close = latest['open']
    
    return {
        'open': latest['open'],
        'close': latest['close'],
        'high': latest['high'],
        'low': latest['low'],
        'volume': latest['volume'],
        'avg_volume_20d': avg_volume_20d,
        'high_52w': high_52w,
        'prev_close': prev_close
    }


def analyze_stock(client: SchwabClient, symbol: str, spy_change: float) -> Optional[Dict]:
    """
    Analyze a single stock against breakout criteria
    Returns dict with analysis if it passes, None if it fails
    """
    result = {
        'symbol': symbol,
        'passed': False,
        'reasons': [],
        'failed_reasons': []
    }
    
    try:
        # Get price history for volume calculations
        df = get_price_history(client, symbol)
        metrics = calculate_metrics_from_history(df) if df is not None else {}
        
        # Get current quote for real-time data
        quote_data = client.get_quote(symbol)
        if not quote_data or symbol not in quote_data:
            return None
        
        quote = quote_data[symbol].get('quote', {})
        
        # Extract key data - prefer real-time, fallback to historical
        open_price = quote.get('openPrice') or metrics.get('open', 0)
        close_price = quote.get('lastPrice') or metrics.get('close', 0)
        high_price = quote.get('highPrice') or metrics.get('high', 0)
        low_price = quote.get('lowPrice') or metrics.get('low', 0)
        prev_close = quote.get('closePrice') or metrics.get('prev_close', 0)
        high_52w = quote.get('52WeekHigh') or metrics.get('high_52w', 0)
        volume = quote.get('totalVolume') or metrics.get('volume', 0)
        
        # Use calculated 20-day avg volume from history (more reliable)
        avg_volume = metrics.get('avg_volume_20d') or quote.get('averageVolume10Day') or quote.get('averageVolume', 0)
        
        net_change = quote.get('netChange', 0)
        change_pct = quote.get('netPercentChangeInDouble', 0)
        
        # If no change data, calculate from price
        if change_pct == 0 and prev_close > 0:
            change_pct = ((close_price - prev_close) / prev_close) * 100
        
        if not all([open_price, close_price, high_52w, volume]):
            return None
        
        result['price'] = close_price
        result['change_pct'] = change_pct
        result['volume'] = volume
        
        # ===== CRITERIA CHECKS =====
        
        # 1. Green Candle (Close > Open)
        is_green = close_price > open_price
        if is_green:
            result['reasons'].append(f"✅ Green candle (Close ${close_price:.2f} > Open ${open_price:.2f})")
        else:
            result['failed_reasons'].append(f"❌ Red candle (Close ${close_price:.2f} < Open ${open_price:.2f})")
        
        # 2. Distance from 52-week high (within 10-20%)
        distance_from_high = ((high_52w - close_price) / high_52w) * 100 if high_52w > 0 else 100
        result['distance_from_high'] = distance_from_high
        result['high_52w'] = high_52w
        
        if DISTANCE_FROM_HIGH_MIN <= distance_from_high <= DISTANCE_FROM_HIGH_MAX:
            result['reasons'].append(f"✅ {distance_from_high:.1f}% from 52w high (${high_52w:.2f})")
        else:
            if distance_from_high > DISTANCE_FROM_HIGH_MAX:
                result['failed_reasons'].append(f"❌ Too far from 52w high ({distance_from_high:.1f}%)")
            else:
                result['reasons'].append(f"⚠️ At/near 52w high ({distance_from_high:.1f}%)")
        
        # 3. Volume vs 20-day average
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            result['volume_ratio'] = volume_ratio
            
            if MIN_VOLUME_RATIO <= volume_ratio <= MAX_VOLUME_RATIO:
                result['reasons'].append(f"✅ Volume {volume_ratio:.1f}x avg ({volume:,.0f} vs {avg_volume:,.0f})")
            elif volume_ratio < MIN_VOLUME_RATIO:
                result['failed_reasons'].append(f"❌ Low volume ({volume_ratio:.1f}x avg)")
            else:
                result['failed_reasons'].append(f"⚠️ Extreme volume spike ({volume_ratio:.1f}x avg)")
        else:
            result['volume_ratio'] = 0
            result['failed_reasons'].append("❌ No avg volume data")
        
        # 4. Relative Strength vs SPY
        rs_vs_spy = change_pct - spy_change
        result['rs_vs_spy'] = rs_vs_spy
        
        if rs_vs_spy >= MIN_RS_VS_SPY:
            result['reasons'].append(f"✅ Outperforming SPY by {rs_vs_spy:.2f}%")
        else:
            result['failed_reasons'].append(f"❌ Underperforming SPY by {abs(rs_vs_spy):.2f}%")
        
        # 5. Gap Analysis (avoid huge gaps)
        if prev_close > 0:
            gap_pct = ((open_price - prev_close) / prev_close) * 100
            result['gap_pct'] = gap_pct
            
            if abs(gap_pct) <= MAX_GAP_PERCENT:
                if gap_pct > 1:
                    result['reasons'].append(f"✅ Modest gap up ({gap_pct:.1f}%)")
                elif gap_pct > 0:
                    result['reasons'].append(f"✅ Small gap ({gap_pct:.1f}%)")
                else:
                    result['reasons'].append(f"✅ No gap / gap down filled ({gap_pct:.1f}%)")
            else:
                result['failed_reasons'].append(f"⚠️ Large gap ({gap_pct:.1f}%) - move may be priced in")
        
        # 6. Check if making new high today
        if close_price >= high_52w * 0.98:  # Within 2% of 52w high
            result['reasons'].append(f"🔥 Near/At 52-week high!")
            result['at_high'] = True
        else:
            result['at_high'] = False
        
        # 7. Intraday strength (close near high of day)
        if high_price > low_price:
            intraday_position = (close_price - low_price) / (high_price - low_price)
            result['intraday_position'] = intraday_position
            
            if intraday_position >= 0.7:
                result['reasons'].append(f"✅ Strong close (top {(1-intraday_position)*100:.0f}% of range)")
            elif intraday_position >= 0.5:
                result['reasons'].append(f"⚪ Mid-range close")
            else:
                result['failed_reasons'].append(f"❌ Weak close (bottom {intraday_position*100:.0f}% of range)")
        
        # ===== SCORING =====
        # Must pass: green candle, volume, RS vs SPY
        critical_passed = (
            is_green and
            result.get('volume_ratio', 0) >= MIN_VOLUME_RATIO and
            rs_vs_spy >= MIN_RS_VS_SPY
        )
        
        # Nice to have: distance from high, no huge gap
        secondary_score = 0
        if DISTANCE_FROM_HIGH_MIN <= distance_from_high <= DISTANCE_FROM_HIGH_MAX:
            secondary_score += 2
        if abs(result.get('gap_pct', 0)) <= MAX_GAP_PERCENT:
            secondary_score += 1
        if result.get('at_high', False):
            secondary_score += 2
        if result.get('intraday_position', 0) >= 0.7:
            secondary_score += 1
        
        result['score'] = secondary_score
        result['passed'] = critical_passed and secondary_score >= 2
        
        return result
        
    except Exception as e:
        print(f"  Error analyzing {symbol}: {e}")
        return None


def run_scanner(verbose: bool = True):
    """Run the breakout scanner"""
    print("=" * 60)
    print("🔍 BREAKOUT CANDIDATE SCANNER")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize Schwab client
    print("\n📡 Connecting to Schwab API...")
    client = SchwabClient()
    
    # Get SPY change for RS comparison
    print("📊 Getting SPY benchmark...")
    spy_quote = client.get_quote("SPY")
    spy_change = 0
    if spy_quote and "SPY" in spy_quote:
        spy_change = spy_quote["SPY"].get("quote", {}).get("netPercentChangeInDouble", 0)
        print(f"   SPY: {spy_change:+.2f}%")
    
    # Fetch watchlist
    print("\n📋 Fetching watchlist...")
    symbols = fetch_watchlist()
    print(f"   Found {len(symbols)} symbols")
    
    # Analyze each stock
    print("\n🔬 Analyzing stocks...")
    results = []
    passed = []
    
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"   [{i+1}/{len(symbols)}] {symbol}...", end=" ")
        
        result = analyze_stock(client, symbol, spy_change)
        
        if result:
            results.append(result)
            if result['passed']:
                passed.append(result)
                if verbose:
                    print(f"✅ PASSED (Score: {result['score']})")
            else:
                if verbose:
                    print(f"❌ Failed")
        else:
            if verbose:
                print("⚠️ No data")
    
    # Sort passed stocks by score
    passed.sort(key=lambda x: (x['score'], x.get('rs_vs_spy', 0)), reverse=True)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"🎯 BREAKOUT CANDIDATES ({len(passed)} found)")
    print("=" * 60)
    
    if not passed:
        print("\n⚠️ No stocks passed all criteria today.")
        print("\nTop near-misses:")
        # Show top 5 that almost passed
        almost = [r for r in results if r and not r['passed']]
        almost.sort(key=lambda x: len(x.get('reasons', [])), reverse=True)
        for r in almost[:5]:
            print(f"\n{r['symbol']} ${r.get('price', 0):.2f} ({r.get('change_pct', 0):+.2f}%)")
            for reason in r.get('reasons', []):
                print(f"  {reason}")
            for fail in r.get('failed_reasons', [])[:2]:
                print(f"  {fail}")
    else:
        for r in passed:
            print(f"\n{'🔥' if r.get('at_high') else '📈'} {r['symbol']} - Score: {r['score']}")
            print(f"   Price: ${r.get('price', 0):.2f} ({r.get('change_pct', 0):+.2f}%)")
            print(f"   52w High: ${r.get('high_52w', 0):.2f} ({r.get('distance_from_high', 0):.1f}% away)")
            print(f"   Volume: {r.get('volume_ratio', 0):.1f}x average")
            print(f"   RS vs SPY: {r.get('rs_vs_spy', 0):+.2f}%")
            print(f"   Gap: {r.get('gap_pct', 0):+.1f}%")
            print("   Reasons:")
            for reason in r.get('reasons', []):
                print(f"     {reason}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Total Scanned: {len(symbols)}")
    print(f"   Passed: {len(passed)}")
    print(f"   SPY Today: {spy_change:+.2f}%")
    
    # Return for programmatic use
    return passed


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Breakout Candidate Scanner")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--min-volume", type=float, default=1.2, help="Min volume ratio (default: 1.2)")
    parser.add_argument("--max-distance", type=float, default=20.0, help="Max distance from 52w high (default: 20%)")
    args = parser.parse_args()
    
    global MIN_VOLUME_RATIO, DISTANCE_FROM_HIGH_MAX
    MIN_VOLUME_RATIO = args.min_volume
    DISTANCE_FROM_HIGH_MAX = args.max_distance
    
    run_scanner(verbose=not args.quiet)


if __name__ == "__main__":
    main()
