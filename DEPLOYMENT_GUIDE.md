# 🚀 Deployment Guide - Options Scanner for Friends

## Overview
This guide helps you deploy the Options Scanner so your friends can access it, and manage the Schwab API token refresh (needed every 7 days).

---

## 📋 Table of Contents
1. [Local Testing](#local-testing)
2. [Token Management](#token-management)
3. [Deploy on Your Local Network](#deploy-on-local-network)
4. [Deploy to Cloud (Streamlit Cloud)](#deploy-to-cloud-streamlit-cloud)
5. [Token Refresh Automation](#token-refresh-automation)

---

## 🧪 Local Testing

### 1. Test Locally First
```bash
cd /Users/piyushkhaitan/schwab/options
streamlit run Main_Dashboard.py
```

The app will open at `http://localhost:8501`

### 2. Test the Max Gamma Scanner
- Navigate to "Max Gamma Scanner" in the sidebar
- Enter a symbol (e.g., TSLA, AAPL, SPY)
- Verify all sections work:
  - ✅ Current price with EMAs
  - ✅ Gamma heatmap
  - ✅ Options flow
  - ✅ Top strikes by expiry
  - ✅ Call/Put lists

---

## 🔑 Token Management

### Understanding Schwab API Tokens
- **Access Token**: Expires in 30 minutes (auto-refreshed by the app)
- **Refresh Token**: Expires in 7 days (needs manual refresh)

### Current Token Location
Your tokens are stored in:
```
/Users/piyushkhaitan/schwab/options/schwab_client.json
```

### How to Check Token Status
Create a simple checker script:

```python
# scripts/check_token.py
import json
from datetime import datetime, timedelta

with open('schwab_client.json', 'r') as f:
    tokens = json.load(f)

# Tokens expire 7 days after creation
refresh_token_date = tokens.get('refresh_token_created_at', None)
if refresh_token_date:
    created = datetime.fromisoformat(refresh_token_date)
    expires = created + timedelta(days=7)
    days_left = (expires - datetime.now()).days
    
    print(f"📅 Refresh token created: {created.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏰ Expires in: {days_left} days")
    
    if days_left < 2:
        print("⚠️  WARNING: Token expires soon! Run refresh script.")
    else:
        print("✅ Token is valid")
else:
    print("❌ No token date found - run auth_setup.py")
```

Run it:
```bash
python scripts/check_token.py
```

### How to Refresh Tokens (Every 7 Days)

**Method 1: Quick Refresh**
```bash
cd /Users/piyushkhaitan/schwab/options
python scripts/auth_setup.py
```

Follow the prompts:
1. Browser window opens with Schwab login
2. Log in to your Schwab account
3. Authorize the app
4. You'll be redirected - copy the URL
5. Paste it into the terminal
6. Done! ✅

**Method 2: Set a Calendar Reminder**
- Set a recurring reminder every 6 days
- Title: "Refresh Schwab API Token"
- Run `python scripts/auth_setup.py`

---

## 🌐 Deploy on Your Local Network

### For Friends on Same WiFi

1. **Find Your Local IP**
```bash
ipconfig getifaddr en0  # Mac WiFi
# or
ipconfig getifaddr en1  # Mac Ethernet
```

Let's say it returns: `192.168.1.100`

2. **Run Streamlit with Network Access**
```bash
streamlit run Main_Dashboard.py --server.address 0.0.0.0 --server.port 8501
```

3. **Share with Friends**
Tell them to visit: `http://192.168.1.100:8501`

**Note**: They must be on your WiFi network!

### Keep It Running 24/7 (Mac)

**Option A: Use `screen`**
```bash
screen -S options-scanner
cd /Users/piyushkhaitan/schwab/options
streamlit run Main_Dashboard.py --server.address 0.0.0.0
```

Press `Ctrl+A` then `D` to detach. App keeps running!

To reattach:
```bash
screen -r options-scanner
```

**Option B: Create a Launch Agent**
Create file: `~/Library/LaunchAgents/com.optionsscanner.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.optionsscanner</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/streamlit</string>
        <string>run</string>
        <string>/Users/piyushkhaitan/schwab/options/Main_Dashboard.py</string>
        <string>--server.address</string>
        <string>0.0.0.0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.optionsscanner.plist
```

---

## ☁️ Deploy to Streamlit Cloud (FREE!)

**Good News**: You CAN use Streamlit Cloud with a workaround for token refresh!

### How It Works:
1. Deploy app to Streamlit Cloud (free)
2. Store tokens in Streamlit Secrets (secure)
3. Refresh tokens from your local machine
4. Update secrets via GitHub

### Step-by-Step Deployment:

#### 1. Prepare Your Repository

**A. Create `.gitignore` (DON'T commit tokens!)**
```gitignore
schwab_client.json
*.pyc
__pycache__/
.env
.streamlit/secrets.toml
```

**B. Push to GitHub**
```bash
cd /Users/piyushkhaitan/schwab/options
git init
git add .
git commit -m "Initial commit - Options Scanner"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/options-scanner.git
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click**: "New app"
4. **Fill in**:
   - Repository: `YOUR-USERNAME/options-scanner`
   - Branch: `main`
   - Main file: `Main_Dashboard.py`
5. **Click**: "Deploy!"

#### 3. Setup Secrets (For Schwab Tokens)

**A. Get Your Current Token**
```bash
cd /Users/piyushkhaitan/schwab/options
cat schwab_client.json
```

Copy the entire JSON content.

**B. Add to Streamlit Secrets**
1. In Streamlit Cloud dashboard, click your app
2. Click "⚙️ Settings" → "Secrets"
3. Paste this format:

```toml
[schwab]
app_key = "YOUR_APP_KEY_HERE"
app_secret = "YOUR_APP_SECRET_HERE"
redirect_uri = "https://127.0.0.1:8182"
access_token = "YOUR_ACCESS_TOKEN"
refresh_token = "YOUR_REFRESH_TOKEN"
id_token = "YOUR_ID_TOKEN"
refresh_token_created_at = "2025-11-02T10:30:00"
```

4. **Save**

**C. Update Code to Use Secrets**

Modify `src/api/schwab_client.py` to check for Streamlit secrets:

```python
import streamlit as st
import os
import json

def load_tokens():
    """Load tokens from Streamlit secrets or local file"""
    
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'schwab' in st.secrets:
            return {
                'app_key': st.secrets['schwab']['app_key'],
                'app_secret': st.secrets['schwab']['app_secret'],
                'redirect_uri': st.secrets['schwab']['redirect_uri'],
                'access_token': st.secrets['schwab']['access_token'],
                'refresh_token': st.secrets['schwab']['refresh_token'],
                'id_token': st.secrets['schwab']['id_token'],
                'refresh_token_created_at': st.secrets['schwab']['refresh_token_created_at']
            }
    except:
        pass
    
    # Fallback to local file (for local development)
    token_file = 'schwab_client.json'
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            return json.load(f)
    
    raise Exception("No tokens found! Run auth_setup.py")
```

#### 4. Token Refresh Workflow (Every 7 Days)

**A. Refresh Locally**
```bash
cd /Users/piyushkhaitan/schwab/options
python scripts/auth_setup.py
```

**B. Get New Token**
```bash
cat schwab_client.json
```

**C. Update Streamlit Secrets**
1. Go to Streamlit Cloud dashboard
2. Settings → Secrets
3. Update the token values
4. Save
5. App auto-restarts with new tokens!

**⏰ Set Calendar Reminder**: Every 6 days to refresh tokens

---

### 🎯 Alternative: Automated Token Refresh

**Option 1: GitHub Actions (Automated)**

Create `.github/workflows/refresh-token.yml`:

```yaml
name: Refresh Schwab Token

on:
  schedule:
    # Run every 6 days at 9 AM UTC
    - cron: '0 9 */6 * *'
  workflow_dispatch:  # Manual trigger

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - name: Notify to Refresh Token
        run: |
          echo "⚠️ Schwab token needs refresh!"
          echo "Run locally: python scripts/auth_setup.py"
          echo "Then update Streamlit secrets"
```

This sends you a GitHub notification every 6 days.

**Option 2: Make it a 2-Minute Task**

1. **Bookmark this checklist**:
   - [ ] Run `python scripts/auth_setup.py` (2 min)
   - [ ] Copy token from `schwab_client.json`
   - [ ] Update Streamlit Secrets (1 min)
   - [ ] Done! ✅

2. **Set phone reminder**: Every 6 days

---

### 📱 Share with Friends

Once deployed on Streamlit Cloud:

1. **Your URL**: `https://YOUR-APP-NAME.streamlit.app`
2. **Share** this URL with friends
3. **That's it!** They can access from anywhere

### 🔒 Add Password Protection (Optional)

**For private access**, add to your app:

```python
# Add to Main_Dashboard.py at the top, before main()

import hmac
import streamlit as st

def check_password():
    """Returns `True` if user entered correct password."""

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    
    return False

if not check_password():
    st.stop()

# Rest of your app code...
```

Add to Streamlit Secrets:
```toml
password = "your-secret-password-123"
```

---

### 💰 Cost Comparison

| Platform | Cost | Pros | Cons |
|----------|------|------|------|
| **Streamlit Cloud** | FREE | Easy, no server management | Manual token refresh every 7 days |
| **PythonAnywhere** | $5/month | Persistent files | Need to setup server |
| **DigitalOcean/Linode** | $10-20/month | Full control | More technical setup |
| **AWS EC2** | $10-30/month | Scalable | Most complex |

**Recommendation**: Start with **Streamlit Cloud (FREE)** → Set 6-day reminder → Takes 3 minutes to refresh!

---

### Better Option: Use a VPS or Cloud VM

### Deploy on AWS/DigitalOcean/Linode

1. **Spin up Ubuntu Server**
   - Size: 2GB RAM minimum
   - Cost: ~$10-20/month

2. **Install Dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-venv -y

# Clone your repo or upload files
cd /home/ubuntu
# Upload your options folder

cd options
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Setup Schwab Tokens**
```bash
# Run auth from your local machine first, then copy token file
# From your Mac:
scp schwab_client.json ubuntu@your-server-ip:/home/ubuntu/options/
```

4. **Run with Systemd**
Create `/etc/systemd/system/options-scanner.service`:

```ini
[Unit]
Description=Options Scanner
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/options
ExecStart=/home/ubuntu/options/venv/bin/streamlit run Main_Dashboard.py --server.address 0.0.0.0 --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Start it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable options-scanner
sudo systemctl start options-scanner
```

5. **Setup Domain & SSL**
- Point domain to server IP
- Use Nginx as reverse proxy
- Get free SSL with Let's Encrypt

---

## 🔄 Token Refresh Automation

### Automated Token Refresh Script

Create `scripts/auto_refresh_token.sh`:

```bash
#!/bin/bash
# Auto-refresh Schwab token every 6 days

cd /Users/piyushkhaitan/schwab/options

# Check if token needs refresh
python3 << EOF
import json
from datetime import datetime, timedelta
import sys

try:
    with open('schwab_client.json', 'r') as f:
        tokens = json.load(f)
    
    created = datetime.fromisoformat(tokens.get('refresh_token_created_at'))
    days_left = 7 - (datetime.now() - created).days
    
    if days_left <= 1:
        print("NEEDS_REFRESH")
        sys.exit(1)
    else:
        print(f"OK - {days_left} days left")
        sys.exit(0)
except:
    print("NEEDS_REFRESH")
    sys.exit(1)
EOF

if [ $? -eq 1 ]; then
    echo "⚠️  Token needs refresh!"
    echo "📧 Sending notification..."
    
    # Send yourself an email or text
    osascript -e 'display notification "Schwab API token needs refresh! Run auth_setup.py" with title "Options Scanner Alert"'
fi
```

Make it executable:
```bash
chmod +x scripts/auto_refresh_token.sh
```

### Schedule with Cron

```bash
crontab -e
```

Add this line (runs daily at 9 AM):
```cron
0 9 * * * cd /Users/piyushkhaitan/schwab/options && ./scripts/auto_refresh_token.sh
```

---

## 📱 Easy Token Refresh for You

### Create a Desktop Shortcut

**Mac - Create Automator App**

1. Open Automator → New Document → Application
2. Add "Run Shell Script" action:

```bash
cd /Users/piyushkhaitan/schwab/options
python3 scripts/auth_setup.py

# Show success message
osascript -e 'display notification "Schwab token refreshed successfully!" with title "Options Scanner"'
```

3. Save as "Refresh Schwab Token.app" on Desktop
4. Double-click when you need to refresh!

---

## 🎯 Recommended Setup for Friends

### Best Approach: VPS + Password Protection

1. **Deploy on $10/month VPS** (DigitalOcean/Linode)
2. **Add password protection** to Streamlit:

Create `.streamlit/secrets.toml`:
```toml
[passwords]
# Simple password protection
password = "your-secret-password-123"
```

Add to `Main_Dashboard.py`:
```python
import streamlit as st

def check_password():
    """Returns `True` if user has entered correct password."""
    
    def password_entered():
        """Checks whether password entered by user is correct."""
        if st.session_state["password"] == st.secrets["passwords"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if check_password():
    # Rest of your main() code here
    main()
```

3. **Share URL + Password** with friends
4. **Refresh token every 6 days** (set calendar reminder)

---

## 📊 Monitoring & Maintenance

### Check if App is Running
```bash
curl http://localhost:8501
# Should return HTML
```

### View Streamlit Logs
```bash
# If using systemd on server:
sudo journalctl -u options-scanner -f

# If using screen locally:
screen -r options-scanner
```

### Restart App
```bash
# Systemd:
sudo systemctl restart options-scanner

# Screen: Ctrl+C then restart
streamlit run Main_Dashboard.py
```

---

## 🚨 Troubleshooting

### "API connection failed"
→ Token expired. Run `python scripts/auth_setup.py`

### "Symbol not found"
→ Check Schwab API is working: `python scripts/test_auth.py`

### Friends can't access
→ Check firewall allows port 8501
→ Verify they're on same network (local) or use correct public IP (VPS)

### App crashes
→ Check logs for Python errors
→ Restart app

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Start locally | `streamlit run Main_Dashboard.py` |
| Start for network | `streamlit run Main_Dashboard.py --server.address 0.0.0.0` |
| Check token | `python scripts/check_token.py` |
| Refresh token | `python scripts/auth_setup.py` |
| Test API | `python scripts/test_auth.py` |

---

## ✅ Deployment Checklist

- [ ] Test app locally
- [ ] Verify all scanners work
- [ ] Check token expiration
- [ ] Set calendar reminder for token refresh (every 6 days)
- [ ] Choose deployment method (local network or VPS)
- [ ] Setup password protection (if sharing)
- [ ] Test access from friend's device
- [ ] Document the shared URL
- [ ] Create token refresh shortcut on desktop

---

**Need help?** Check the logs or re-run `python scripts/auth_setup.py` to fix auth issues.
