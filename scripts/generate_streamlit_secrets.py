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
    
    # Extract from nested structure
    client_data = data.get('client', {})
    token_data = data.get('token', {})
    
    print("\n" + "="*60)
    print("📋 COPY THIS TO STREAMLIT CLOUD SECRETS")
    print("="*60)
    print()
    print("[schwab]")
    print(f'app_key = "{client_data.get("api_key", "")}"')
    print(f'app_secret = "{client_data.get("app_secret", "")}"')
    print(f'redirect_uri = "{client_data.get("callback", "https://127.0.0.1:8182")}"')
    print(f'access_token = "{token_data.get("access_token", "")}"')
    print(f'refresh_token = "{token_data.get("refresh_token", "")}"')
    print(f'id_token = "{token_data.get("id_token", "")}"')
    print(f'token_type = "{token_data.get("token_type", "")}"')
    print(f'expires_in = {token_data.get("expires_in", "")}')
    print(f'scope = "{token_data.get("scope", "")}"')
    print(f'expires_at = {token_data.get("expires_at", "")}')
    print(f'setup = "{client_data.get("setup", "")}"')
    print()
    print("[alerts]")
    print('discord_webhook = "https://discord.com/api/webhooks/1332242383458406401/6KXAsHFsvTKgDZyDimQ_ncrBx9vePgsOYxSRjga0mK-Zg2m404r65zzqdyL1bKCQRwVO"')
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
        created_str = client_data.get("setup", "")
        created_at = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
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
