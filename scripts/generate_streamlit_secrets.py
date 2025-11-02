#!/usr/bin/env python3
"""
Generate Streamlit secrets format from schwab_client.json
Run this after refreshing your tokens to get the format for Streamlit Cloud secrets
"""

import json
import os

def generate_streamlit_secrets():
    """Convert schwab_client.json to Streamlit secrets format"""
    
    token_file = 'schwab_client.json'
    
    if not os.path.exists(token_file):
        print("❌ Error: schwab_client.json not found!")
        print("Run: python scripts/auth_setup.py")
        return
    
    with open(token_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("📋 COPY THIS TO STREAMLIT CLOUD SECRETS")
    print("="*60)
    print()
    print("[schwab]")
    print(f'app_key = "{data.get("app_key", "")}"')
    print(f'app_secret = "{data.get("app_secret", "")}"')
    print(f'redirect_uri = "{data.get("redirect_uri", "https://127.0.0.1:8182")}"')
    print(f'access_token = "{data.get("access_token", "")}"')
    print(f'refresh_token = "{data.get("refresh_token", "")}"')
    print(f'id_token = "{data.get("id_token", "")}"')
    print(f'refresh_token_created_at = "{data.get("refresh_token_created_at", "")}"')
    print()
    print("="*60)
    print("📝 INSTRUCTIONS:")
    print("1. Go to your Streamlit Cloud app")
    print("2. Click ⚙️ Settings → Secrets")
    print("3. Copy-paste the section above")
    print("4. Click Save")
    print("5. App will auto-restart with new tokens!")
    print("="*60)
    print()
    
    # Check token expiration
    from datetime import datetime, timedelta
    try:
        created_at = datetime.fromisoformat(data.get("refresh_token_created_at", ""))
        expires_at = created_at + timedelta(days=7)
        days_left = (expires_at - datetime.now()).days
        
        if days_left < 0:
            print("🔴 TOKEN EXPIRED! Run: python scripts/auth_setup.py")
        elif days_left <= 2:
            print(f"🟡 Token expires in {days_left} days - refresh soon!")
        else:
            print(f"🟢 Token valid for {days_left} more days")
    except:
        print("⚠️  Could not check token expiration")
    
    print()

if __name__ == "__main__":
    generate_streamlit_secrets()
