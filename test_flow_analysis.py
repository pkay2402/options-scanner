"""
Flow Analysis Test Script
Analyzes options flow to reverse-engineer the criteria for "Session Leader" detection

Based on screenshot showing:
- AMZN $247.5C 01-30 | $2.79M | +28sw/hr | SESSION LEADER - +$1.47M in 30min!
- IREN $60C 01-30 | $541K | +10sw/hr | Crypto miner momentum
- GOOG $337.5C 01-30 | $1.36M | NEW | New strike adding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.cached_client import get_client
from datetime import datetime, timedelta
import pandas as pd

# Target stocks from the screenshot
TEST_SYMBOLS = ['AMZN', 'IREN', 'GOOG']
TARGET_EXPIRY = '2025-01-31'  # 01-30 likely means Jan 30/31 expiry

def get_next_friday():
    """Get next Friday for weekly expiry"""
    today = datetime.now().date()
    weekday = today.weekday()
    days_to_friday = (4 - weekday) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    return today + timedelta(days=days_to_friday)

def analyze_options_flow(symbol: str, target_strike: float = None, target_type: str = 'CALL'):
    """
    Fetch and analyze options flow for a symbol
    Returns detailed metrics to understand the criteria
    """
    client = get_client()
    if not client:
        print(f"❌ Could not get client")
        return None
    
    try:
        # Get quote
        quote_response = client.get_quotes([symbol])
        if not quote_response or symbol not in quote_response:
            print(f"❌ Could not get quote for {symbol}")
            return None
        
        quote = quote_response[symbol]['quote']
        underlying_price = quote.get('lastPrice', 0)
        underlying_volume = quote.get('totalVolume', 0)
        
        print(f"\n{'='*60}")
        print(f"📊 {symbol} Analysis")
        print(f"{'='*60}")
        print(f"Current Price: ${underlying_price:.2f}")
        print(f"Stock Volume: {underlying_volume:,}")
        
        # Get options chain for next 2 weeks
        expiry_date = get_next_friday()
        options_response = client.get_options_chain(
            symbol,
            strike_count=50,
            from_date=expiry_date.strftime('%Y-%m-%d'),
            to_date=(expiry_date + timedelta(days=14)).strftime('%Y-%m-%d')
        )
        
        if not options_response:
            print(f"❌ No options data for {symbol}")
            return None
        
        # Analyze call options
        all_flows = []
        
        call_map = options_response.get('callExpDateMap', {})
        put_map = options_response.get('putExpDateMap', {})
        
        for opt_type, exp_map in [('CALL', call_map), ('PUT', put_map)]:
            for exp_date_key, strikes_map in exp_map.items():
                exp_date = exp_date_key.split(':')[0]  # Extract just the date
                
                for strike_str, contracts in strikes_map.items():
                    if not contracts:
                        continue
                    
                    contract = contracts[0]
                    strike = float(strike_str)
                    volume = contract.get('totalVolume', 0)
                    oi = max(contract.get('openInterest', 0), 1)
                    mark = contract.get('mark', contract.get('last', 0))
                    bid = contract.get('bid', 0)
                    ask = contract.get('ask', 0)
                    delta = abs(contract.get('delta', 0))
                    iv = contract.get('volatility', 0)
                    
                    if volume == 0 or mark == 0:
                        continue
                    
                    # Calculate key metrics
                    distance_pct = ((strike - underlying_price) / underlying_price) * 100
                    premium_total = volume * mark * 100  # Total $ premium
                    vol_oi_ratio = volume / oi
                    notional = volume * underlying_price * 100 * delta  # Delta-adjusted notional
                    
                    # Sweep indicator: tight spread + high volume
                    spread = ask - bid if bid > 0 else 0
                    spread_pct = (spread / mark * 100) if mark > 0 else 0
                    is_sweep_like = spread_pct < 5 and volume > 500
                    
                    all_flows.append({
                        'symbol': symbol,
                        'expiry': exp_date,
                        'strike': strike,
                        'type': opt_type,
                        'volume': volume,
                        'oi': oi,
                        'vol_oi': vol_oi_ratio,
                        'mark': mark,
                        'bid': bid,
                        'ask': ask,
                        'spread_pct': spread_pct,
                        'delta': delta,
                        'iv': iv,
                        'distance_pct': distance_pct,
                        'premium_total': premium_total,
                        'notional': notional,
                        'is_sweep_like': is_sweep_like,
                        'underlying_price': underlying_price
                    })
        
        if not all_flows:
            print(f"❌ No flows found for {symbol}")
            return None
        
        df = pd.DataFrame(all_flows)
        
        # Sort by premium (total $ spent)
        df = df.sort_values('premium_total', ascending=False)
        
        print(f"\n🔥 TOP FLOWS BY PREMIUM:")
        print("-" * 80)
        
        top_flows = df.head(10)
        for _, row in top_flows.iterrows():
            premium_str = f"${row['premium_total']/1e6:.2f}M" if row['premium_total'] >= 1e6 else f"${row['premium_total']/1e3:.0f}K"
            notional_str = f"${row['notional']/1e6:.2f}M" if row['notional'] >= 1e6 else f"${row['notional']/1e3:.0f}K"
            
            sweep_flag = "🔥" if row['is_sweep_like'] else ""
            
            print(f"  {row['type']:<4} ${row['strike']:<8.1f} {row['expiry']} | "
                  f"Premium: {premium_str:>8} | Vol: {row['volume']:>6,} | OI: {row['oi']:>6,} | "
                  f"Vol/OI: {row['vol_oi']:>5.1f}x | Δ: {row['delta']:.2f} | "
                  f"Spread: {row['spread_pct']:.1f}% {sweep_flag}")
        
        # Identify what makes a "Session Leader"
        print(f"\n📈 SESSION LEADER CRITERIA ANALYSIS:")
        print("-" * 80)
        
        # Get the highest premium flow
        top_flow = df.iloc[0]
        print(f"  Highest Premium Flow:")
        print(f"    Position: {top_flow['type']} ${top_flow['strike']:.1f} exp {top_flow['expiry']}")
        print(f"    Premium: ${top_flow['premium_total']/1e6:.2f}M")
        print(f"    Volume: {top_flow['volume']:,} contracts")
        print(f"    Vol/OI Ratio: {top_flow['vol_oi']:.1f}x")
        print(f"    Delta: {top_flow['delta']:.2f}")
        print(f"    IV: {top_flow['iv']:.1f}%")
        print(f"    Spread %: {top_flow['spread_pct']:.2f}%")
        print(f"    Distance from price: {top_flow['distance_pct']:+.1f}%")
        
        # Aggregate by strike to see total positioning
        print(f"\n📊 AGGREGATE BY STRIKE (Top Positioning):")
        print("-" * 80)
        
        agg = df.groupby(['type', 'strike', 'expiry']).agg({
            'volume': 'sum',
            'premium_total': 'sum',
            'oi': 'sum',
            'delta': 'mean'
        }).reset_index()
        agg = agg.sort_values('premium_total', ascending=False).head(5)
        
        for _, row in agg.iterrows():
            premium_str = f"${row['premium_total']/1e6:.2f}M" if row['premium_total'] >= 1e6 else f"${row['premium_total']/1e3:.0f}K"
            print(f"  {row['type']:<4} ${row['strike']:<8.1f} {row['expiry']} | Premium: {premium_str:>8} | Vol: {row['volume']:>6,}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main analysis function"""
    print("=" * 80)
    print("🐋 FLOW ANALYSIS - Reverse Engineering Session Leader Criteria")
    print("=" * 80)
    print(f"Target symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    for symbol in TEST_SYMBOLS:
        result = analyze_options_flow(symbol)
        if result is not None:
            all_results[symbol] = result
    
    # Cross-symbol comparison
    if all_results:
        print("\n" + "=" * 80)
        print("📊 CROSS-SYMBOL COMPARISON - Top Premium Flows")
        print("=" * 80)
        
        combined = pd.concat([
            df.assign(symbol=sym).head(3) 
            for sym, df in all_results.items()
        ])
        combined = combined.sort_values('premium_total', ascending=False)
        
        print(f"\n{'Symbol':<6} {'Type':<4} {'Strike':<8} {'Expiry':<12} {'Premium':>10} {'Volume':>8} {'Vol/OI':>7} {'Spread%':>8}")
        print("-" * 80)
        
        for _, row in combined.iterrows():
            premium_str = f"${row['premium_total']/1e6:.2f}M" if row['premium_total'] >= 1e6 else f"${row['premium_total']/1e3:.0f}K"
            print(f"{row['symbol']:<6} {row['type']:<4} ${row['strike']:<7.1f} {row['expiry']:<12} {premium_str:>10} {row['volume']:>8,} {row['vol_oi']:>6.1f}x {row['spread_pct']:>7.1f}%")
        
        print("\n" + "=" * 80)
        print("🎯 INFERRED CRITERIA FOR SESSION LEADER:")
        print("=" * 80)
        print("""
Based on analysis, likely criteria:

1. PREMIUM THRESHOLD:
   - Minimum $500K+ total premium on single strike
   - "Session Leader" = Highest premium gain in rolling 30-min window

2. VELOCITY (Sweeps/Hour):
   - Track premium change over time windows
   - +$X per hour = velocity metric
   - Tight spread (<5%) + high volume = sweep detection

3. TIER CLASSIFICATION:
   - Tier 1 (Actionable): Premium > $500K + Velocity > 10sw/hr
   - Session Leader: Largest $ gain in 30-min window
   - "NEW" flag: Strike with <100 OI but high today's volume

4. SWEEP DETECTION:
   - Bid-Ask spread < 5%
   - Volume > 500 contracts
   - Multiple smaller orders hitting ask = sweep behavior

5. MOMENTUM SIGNALS:
   - Crypto correlation (IREN -> "crypto miner momentum")
   - Sector tagging based on symbol
""")


if __name__ == "__main__":
    main()
