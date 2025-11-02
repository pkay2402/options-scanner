#!/usr/bin/env python3
"""
Check Schwab API Token Status
Warns when token is about to expire
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def check_token_status():
    """Check if Schwab refresh token needs renewal"""
    
    token_file = Path(__file__).parent.parent / 'schwab_client.json'
    
    if not token_file.exists():
        print("❌ Token file not found!")
        print(f"   Expected location: {token_file}")
        print("\n💡 Run: python scripts/auth_setup.py")
        return False
    
    try:
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        # Check if we have token creation date
        created_at = tokens.get('refresh_token_created_at')
        
        if not created_at:
            print("⚠️  No token creation date found")
            print("   Token might be from old format")
            print("\n💡 Recommended: Re-run auth_setup.py to update token format")
            return False
        
        # Parse creation date
        created = datetime.fromisoformat(created_at)
        now = datetime.now()
        
        # Schwab refresh tokens expire after 7 days
        expires_at = created + timedelta(days=7)
        time_left = expires_at - now
        days_left = time_left.days
        hours_left = time_left.seconds // 3600
        
        # Display status
        print("=" * 60)
        print("🔑 SCHWAB API TOKEN STATUS")
        print("=" * 60)
        print(f"\n📅 Token Created:  {created.strftime('%Y-%m-%d at %H:%M:%S')}")
        print(f"⏰ Token Expires:  {expires_at.strftime('%Y-%m-%d at %H:%M:%S')}")
        print(f"\n⏳ Time Remaining: {days_left} days, {hours_left} hours")
        
        # Status indicator
        if days_left < 0:
            print("\n🔴 STATUS: EXPIRED")
            print("   Your token has expired!")
            print("\n💡 ACTION REQUIRED:")
            print("   Run: python scripts/auth_setup.py")
            return False
        elif days_left == 0:
            print("\n🟠 STATUS: EXPIRES TODAY")
            print("   Your token expires in less than 24 hours!")
            print("\n💡 ACTION REQUIRED:")
            print("   Run: python scripts/auth_setup.py")
            return False
        elif days_left == 1:
            print("\n🟡 STATUS: EXPIRES TOMORROW")
            print("   Your token expires in ~1 day")
            print("\n💡 RECOMMENDED:")
            print("   Run: python scripts/auth_setup.py (to refresh)")
            return True
        elif days_left <= 2:
            print("\n🟡 STATUS: EXPIRES SOON")
            print("   Your token expires in ~2 days")
            print("\n💡 RECOMMENDED:")
            print("   Consider refreshing: python scripts/auth_setup.py")
            return True
        else:
            print("\n🟢 STATUS: VALID")
            print("   Your token is active and healthy")
            print("\n💡 NEXT ACTION:")
            print(f"   Refresh in {days_left - 1} days (around {(expires_at - timedelta(days=1)).strftime('%Y-%m-%d')})")
            return True
        
    except json.JSONDecodeError:
        print("❌ Token file is corrupted!")
        print("\n💡 Run: python scripts/auth_setup.py")
        return False
    except Exception as e:
        print(f"❌ Error checking token: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n")
    valid = check_token_status()
    print("\n" + "=" * 60 + "\n")
    
    # Exit code: 0 = valid, 1 = needs refresh
    exit(0 if valid else 1)
