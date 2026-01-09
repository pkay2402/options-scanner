#!/usr/bin/env python3
"""
Debug script to check what options data we're actually getting
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

print("\n" + "="*80)
print("DEBUGGING OPTIONS DATA RETRIEVAL")
print("="*80 + "\n")

client = SchwabClient(interactive=False)

# Test with SPY
symbol = "SPY"
print(f"Fetching data for {symbol}...\n")

# Get quote
quote_data = client.get_quote(symbol)
if quote_data and symbol in quote_data:
    quote = quote_data[symbol]
    print(f"QUOTE DATA:")
    print(f"  Last Price: ${quote.get('lastPrice', 0):.2f}")
    print(f"  Total Volume: {quote.get('totalVolume', 0):,}")
    print(f"  Close Price: ${quote.get('closePrice', 0):.2f}")
    print(f"  Bid: ${quote.get('bidPrice', 0):.2f}")
    print(f"  Ask: ${quote.get('askPrice', 0):.2f}")
else:
    print("❌ No quote data")

# Get options
print(f"\nFetching options chain...\n")
options_data = client.get_options_chain(
    symbol=symbol,
    contract_type='ALL',
    strike_count=5  # Just a few to test
)

if options_data:
    print(f"OPTIONS CHAIN DATA:")
    print(f"  Has callExpDateMap: {'callExpDateMap' in options_data}")
    print(f"  Has putExpDateMap: {'putExpDateMap' in options_data}")
    print(f"  Has underlying: {'underlying' in options_data}")
    print(f"  Has underlyingPrice: {'underlyingPrice' in options_data}")
    
    if 'underlying' in options_data and options_data['underlying']:
        print(f"\n  UNDERLYING DATA:")
        for key, value in options_data['underlying'].items():
            print(f"    {key}: {value}")
    
    if 'underlyingPrice' in options_data:
        print(f"\n  underlyingPrice: ${options_data['underlyingPrice']}")
    
    if 'callExpDateMap' in options_data:
        num_expirations = len(options_data['callExpDateMap'])
        print(f"  Call expirations: {num_expirations}")
        
        if num_expirations > 0:
            first_exp = list(options_data['callExpDateMap'].keys())[0]
            print(f"  First expiration: {first_exp}")
            
            strikes_data = options_data['callExpDateMap'][first_exp]
            num_strikes = len(strikes_data)
            print(f"  Strikes for first exp: {num_strikes}")
            
            if num_strikes > 0:
                first_strike = list(strikes_data.keys())[0]
                contract = strikes_data[first_strike][0] if strikes_data[first_strike] else {}
                print(f"\n  SAMPLE CALL CONTRACT (Strike {first_strike}):")
                print(f"    totalVolume: {contract.get('totalVolume', 'N/A')}")
                print(f"    openInterest: {contract.get('openInterest', 'N/A')}")
                print(f"    delta: {contract.get('delta', 'N/A')}")
                print(f"    gamma: {contract.get('gamma', 'N/A')}")
                print(f"    mark: {contract.get('mark', 'N/A')}")
                print(f"    bid: {contract.get('bid', 'N/A')}")
                print(f"    ask: {contract.get('ask', 'N/A')}")
    
    if 'putExpDateMap' in options_data:
        num_expirations = len(options_data['putExpDateMap'])
        print(f"\n  Put expirations: {num_expirations}")
        
        if num_expirations > 0:
            first_exp = list(options_data['putExpDateMap'].keys())[0]
            strikes_data = options_data['putExpDateMap'][first_exp]
            num_strikes = len(strikes_data)
            print(f"  Strikes for first exp: {num_strikes}")
            
            if num_strikes > 0:
                first_strike = list(strikes_data.keys())[0]
                contract = strikes_data[first_strike][0] if strikes_data[first_strike] else {}
                print(f"\n  SAMPLE PUT CONTRACT (Strike {first_strike}):")
                print(f"    totalVolume: {contract.get('totalVolume', 'N/A')}")
                print(f"    openInterest: {contract.get('openInterest', 'N/A')}")
                print(f"    delta: {contract.get('delta', 'N/A')}")
                print(f"    gamma: {contract.get('gamma', 'N/A')}")
                print(f"    mark: {contract.get('mark', 'N/A')}")
else:
    print("❌ No options data")

print("\n" + "="*80 + "\n")
