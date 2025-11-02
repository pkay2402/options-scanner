#!/usr/bin/env python3
"""
Extract tokens from schwab_client.json and format for Streamlit secrets
"""

import json
from pathlib import Path

def format_for_streamlit_secrets():
    """Format tokens for Streamlit Cloud secrets"""
    
    token_file = Path(__file__).parent.parent / 'schwab_client.json'
    
    if not token_file.exists():
        print("❌ schwab_client.json not found!")
        print("Run: python scripts/auth_setup.py")
        return
    
    with open(token_file, 'r') as f:
        config = json.load(f)
    
    client = config.get('client', {})
    token = config.get('token', {})
    
    # Format for Streamlit secrets.toml
    secrets_toml = f"""# Copy this into Streamlit Cloud Settings → Secrets
# Go to: https://share.streamlit.io → Your App → Settings → Secrets

[schwab]
app_key = "{client.get('api_key', '')}"
app_secret = "{client.get('app_secret', '')}"
redirect_uri = "{client.get('callback', 'https://127.0.0.1:8182')}"
access_token = "{token.get('access_token', '')}"
refresh_token = "{token.get('refresh_token', '')}"
id_token = "{token.get('id_token', '')}"
token_type = "{token.get('token_type', 'Bearer')}"
expires_in = {token.get('expires_in', 1800)}
expires_at = {token.get('expires_at', 0)}
scope = "{token.get('scope', 'api')}"
refresh_token_expires_in = {token.get('refresh_token_expires_in', 604800)}
"""
    
    if 'refresh_token_created_at' in token:
        secrets_toml += f'refresh_token_created_at = "{token["refresh_token_created_at"]}"\n'
    
    print("=" * 70)
    print("📋 STREAMLIT CLOUD SECRETS")
    print("=" * 70)
    print(secrets_toml)
    print("=" * 70)
    print("\n✅ Copy everything above (including [schwab]) and paste into:")
    print("   Streamlit Cloud → Your App → ⚙️ Settings → Secrets")
    print("\n⚠️  IMPORTANT: Keep these secrets private!")
    
    # Also save to a file for easy copy-paste
    output_file = Path(__file__).parent.parent / 'streamlit_secrets.txt'
    with open(output_file, 'w') as f:
        f.write(secrets_toml)
    
    print(f"\n💾 Also saved to: {output_file}")
    print("   (This file is in .gitignore - safe to keep locally)")

if __name__ == '__main__':
    format_for_streamlit_secrets()
