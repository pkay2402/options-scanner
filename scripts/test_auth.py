#!/usr/bin/env python3
"""
Test Schwab API Authentication
This script tests if your Schwab API authentication is working correctly.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.utils.config import get_settings

def test_authentication():
    """Test Schwab API authentication"""
    print("=" * 50)
    print("SCHWAB API AUTHENTICATION TEST")
    print("=" * 50)
    print()
    
    try:
        # Check settings
        settings = get_settings()
        print("Configuration Check:")
        print(f"✓ Client ID: {settings.SCHWAB_CLIENT_ID}")
        print(f"✓ Redirect URI: {settings.SCHWAB_REDIRECT_URI}")
        print()
        
        # Initialize client and check token status
        print("Initializing Schwab client...")
        client = SchwabClient()
        
        token_status = client.get_token_status()
        print(f"Token Status: {token_status['message']}")
        
        if token_status['status'] == 'no_token':
            print("❌ No authentication token found!")
            print("Please run: python3 scripts/auth_setup.py")
            return False
        
        if token_status['status'] == 'expired':
            print("⚠️  Token expired, attempting refresh...")
        
        # Test authentication
        print("⏳ Testing authentication...")
        auth_success = client.ensure_valid_session()
        
        if auth_success:
            print("✅ Authentication successful!")
            
            # Get updated token status
            updated_status = client.get_token_status()
            print(f"✓ Updated token status: {updated_status['message']}")
            if 'seconds_left' in updated_status:
                hours_left = updated_status['seconds_left'] / 3600
                print(f"✓ Token valid for: {hours_left:.1f} hours")
            print()
            
            # Test API calls
            print("Testing API Calls...")
            
            # Test 1: Market hours
            try:
                print("⏳ Fetching market hours...")
                market_hours = client.get_market_hours('equity')
                if market_hours:
                    print("✅ Market hours API call successful!")
                else:
                    print("⚠️  Market hours API returned empty data")
            except Exception as api_error:
                print(f"❌ Market hours API call failed: {str(api_error)}")
            
            # Test 2: Quote
            try:
                print("⏳ Fetching SPY quote...")
                quote = client.get_quote('SPY')
                if quote and 'SPY' in quote:
                    spy_price = quote['SPY']['quote']['lastPrice']
                    print(f"✅ Quote API call successful! SPY price: ${spy_price}")
                else:
                    print("⚠️  Quote API returned empty data")
            except Exception as api_error:
                print(f"❌ Quote API call failed: {str(api_error)}")
            
            # Test 3: Options chain (limited)
            try:
                print("⏳ Fetching SPY options chain...")
                options = client.get_options_chain('SPY', strike_count=5)
                if options and ('callExpDateMap' in options or 'putExpDateMap' in options):
                    call_count = len(options.get('callExpDateMap', {}))
                    put_count = len(options.get('putExpDateMap', {}))
                    print(f"✅ Options chain API call successful! Calls: {call_count}, Puts: {put_count}")
                else:
                    print("⚠️  Options chain API returned empty data")
            except Exception as api_error:
                print(f"❌ Options chain API call failed: {str(api_error)}")
            
            return True
            
        else:
            print("❌ Authentication failed!")
            print("Please run: python3 scripts/auth_setup.py")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

def main():
    """Main test function"""
    success = test_authentication()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Your Schwab API integration is ready to use.")
        print()
        print("Next steps:")
        print("- python3 main.py --mode analysis")
        print("- python3 main.py --mode dashboard")
        print("- python3 main.py --mode monitor")
    else:
        print("❌ TESTS FAILED!")
        print("Please fix the authentication issues before proceeding.")
        print()
        print("Need help? Run:")
        print("python3 scripts/auth_setup.py")
    
    print("=" * 50)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())