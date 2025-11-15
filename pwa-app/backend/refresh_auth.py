#!/usr/bin/env python3
"""
Quick authentication refresh script for Schwab API
Run this when your token expires
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schwab_client import SchwabClient

def main():
    print("=" * 60)
    print("Schwab API Token Refresh")
    print("=" * 60)
    print()
    
    client = SchwabClient()
    
    # Check current session status
    print("Checking current session status...")
    is_valid = client.check_session()
    
    if is_valid:
        print("✅ Session is already valid!")
        print("You're good to go!")
        return
    
    print("❌ Session is invalid or expired")
    print()
    
    # Try to refresh
    print("Attempting to refresh token...")
    success = client.refresh_token()
    
    if success:
        print("✅ Token refreshed successfully!")
        print("You can now use the API")
        return
    
    # If refresh failed, need full re-auth
    print("❌ Token refresh failed")
    print()
    print("You need to re-authenticate. This will open a browser window.")
    print()
    
    response = input("Would you like to re-authenticate now? (y/n): ").strip().lower()
    
    if response == 'y':
        print()
        print("Starting authentication flow...")
        print()
        success = client.setup()
        
        if success:
            print()
            print("=" * 60)
            print("✅ Authentication successful!")
            print("=" * 60)
            print("You can now use the Options Flow Pro PWA")
        else:
            print()
            print("=" * 60)
            print("❌ Authentication failed")
            print("=" * 60)
            print("Please check your credentials and try again")
            sys.exit(1)
    else:
        print()
        print("Authentication cancelled.")
        print("Run this script again when you're ready to authenticate.")
        sys.exit(1)

if __name__ == "__main__":
    main()
