#!/usr/bin/env python3
"""
Schwab API Authentication Setup Script
This script helps you obtain and test the initial authorization for Schwab API access.
"""

import os
import sys
import webbrowser
import urllib.parse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.utils.config import get_settings

def main():
    """Main authentication setup process"""
    print("=" * 60)
    print("SCHWAB API AUTHENTICATION SETUP")
    print("=" * 60)
    print()
    
    try:
        print("✓ Initializing Schwab client...")
        client = SchwabClient()
        
        print(f"✓ Client ID: {client.settings.SCHWAB_CLIENT_ID}")
        print(f"✓ Redirect URI: {client.settings.SCHWAB_REDIRECT_URI}")
        print()
        
        # Check current token status
        token_status = client.get_token_status()
        print(f"Current token status: {token_status['message']}")
        print()
        
        if token_status['status'] == 'valid':
            print("✅ You already have a valid token!")
            choice = input("Do you want to re-authenticate anyway? (y/N): ").lower()
            if choice != 'y':
                print("Exiting...")
                return 0
        
        print("Starting OAuth2 authentication flow...")
        print()
        
        # Run the setup process
        success = client.setup()
        
        if success:
            print()
            print("🎉 AUTHENTICATION SUCCESSFUL!")
            print()
            
            # Test the connection
            print("Testing API connection...")
            try:
                market_hours = client.get_market_hours('equity')
                if market_hours:
                    print("✅ API test successful!")
                    print("Your Schwab API is ready to use.")
                else:
                    print("⚠️  API test returned empty data")
            except Exception as e:
                print(f"⚠️  API test failed: {e}")
                print("Authentication worked, but API calls may have issues.")
            
            print()
            print("Next steps:")
            print("- python3 main.py --mode dashboard")
            print("- python3 main.py --mode analysis")
            print("- python3 scripts/test_auth.py")
            
        else:
            print("❌ Authentication failed!")
            print("Please check your credentials and try again.")
            return 1
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())