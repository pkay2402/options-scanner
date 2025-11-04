#!/usr/bin/env python3
"""
Test the official GEX formula implementation
"""
import sys
sys.path.insert(0, '.')
from src.api.schwab_client import SchwabClient

def test_official_gex_formula():
    """Test our implementation against the official formula"""
    
    print("Testing Official GEX Formula:")
    print("GEX = Gamma × Open Interest × Contract Size × Spot Price² × 0.01")
    print("Net GEX = Call GEX - Put GEX")
    print("=" * 70)
    
    client = SchwabClient()
    options = client.get_options_chain('PLTR', contract_type='ALL', strike_count=20)
    
    if not options or 'callExpDateMap' not in options:
        print("No options data available")
        return
    
    underlying = options.get('underlyingPrice', 200.47)
    print(f"PLTR Price: ${underlying:.2f}")
    print()
    
    # Test first expiry
    exp_dates = list(options['callExpDateMap'].keys())
    if not exp_dates:
        print("No expiry dates found")
        return
        
    first_exp = exp_dates[0]
    print(f"Testing expiry: {first_exp}")
    print()
    
    print("Strike   Type   Gamma    OI      Official_GEX    Professional_Approx")
    print("-" * 75)
    
    # Test specific strikes that we can compare with professional tool
    test_strikes = ['195.0', '200.0', '205.0', '210.0']
    
    for strike_str in test_strikes:
        if strike_str in options['callExpDateMap'][first_exp]:
            strike = float(strike_str)
            call_data = options['callExpDateMap'][first_exp][strike_str][0]
            gamma = call_data.get('gamma', 0)
            oi = call_data.get('openInterest', 0)
            
            # Official formula: GEX = Γ × OI × Contract_Size × S² × 0.01
            contract_size = 100  # Standard options
            official_gex = gamma * oi * contract_size * underlying * underlying * 0.01
            
            # Professional tool values (approximate from heat map)
            professional_values = {
                195.0: "~14M (light green)",
                200.0: "~630M (dark green)", 
                205.0: "~85M (medium green)",
                210.0: "~20M (light green)"
            }
            
            prof_approx = professional_values.get(strike, "Unknown")
            
            print(f"{strike:6.1f}  Call  {gamma:7.4f}  {oi:6,}  {official_gex/1e6:10.1f}M       {prof_approx}")
    
    print()
    print("Formula Breakdown for $200 strike:")
    if '200.0' in options['callExpDateMap'][first_exp]:
        call_data = options['callExpDateMap'][first_exp]['200.0'][0]
        gamma = call_data.get('gamma', 0)
        oi = call_data.get('openInterest', 0)
        
        gex_value = gamma * oi * 100 * underlying * underlying * 0.01
        
        print(f"Gamma: {gamma:.4f}")
        print(f"Open Interest: {oi:,}")
        print(f"Contract Size: 100")
        print(f"Spot Price: ${underlying:.2f}")
        print(f"Calculation: {gamma:.4f} × {oi:,} × 100 × {underlying:.2f}² × 0.01")
        print(f"Result: ${gex_value/1e6:.1f}M")
        print()
        print("Comparison with Professional Tool:")
        print(f"Professional: ~630M")
        print(f"Our calculation: {gex_value/1e6:.1f}M")
        print(f"Ratio: {(gex_value/1e6)/630:.2f}x")

if __name__ == "__main__":
    test_official_gex_formula()