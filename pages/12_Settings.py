"""
⚙️ Settings & Configuration
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.utils.cached_client import get_client

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Settings & Configuration")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔐 Schwab API", "🔔 Alerts", "🎨 Preferences", "ℹ️ System Info"])

with tab1:
    st.header("Schwab API Configuration")
    
    # Check current status
    config_file = Path("schwab_client.json")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        st.success("✅ Configuration file found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Client Info")
            st.code(f"""
API Key: {config['client']['api_key'][:20]}...
Callback: {config['client']['callback']}
Setup Date: {config['client']['setup']}
            """)
        
        with col2:
            st.subheader("Token Status")
            
            if 'token' in config:
                token = config['token']
                expires_at = token.get('expires_at', 0)
                expires_dt = datetime.fromtimestamp(expires_at)
                is_expired = expires_at < datetime.now().timestamp()
                
                if is_expired:
                    st.error(f"🔴 Token Expired")
                    st.caption(f"Expired: {expires_dt.strftime('%Y-%m-%d %H:%M')}")
                else:
                    time_left = expires_dt - datetime.now()
                    st.success(f"🟢 Token Valid")
                    st.caption(f"Expires: {expires_dt.strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Time left: {time_left}")
        
        st.markdown("---")
        
        # Test connection
        if st.button("🧪 Test Connection"):
            with st.spinner("Testing Schwab API connection..."):
                try:
                    client = get_client()
                    if client:
                        st.success("✅ Connection successful!")
                        
                        # Try a simple API call
                        quote = client.get_quote("SPY")
                        if quote:
                            st.json(quote)
                    else:
                        st.error("❌ Authentication failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.button("🔄 Re-authenticate"):
            st.info("Run the authentication script: `python scripts/auth_setup.py`")
    
    else:
        st.warning("⚠️ No configuration file found")
        st.markdown("""
        ### First Time Setup:
        
        1. Run the authentication script:
           ```bash
           python scripts/auth_setup.py
           ```
        
        2. Follow the browser OAuth flow
        
        3. Tokens will be saved to `schwab_client.json`
        
        4. Return here to verify connection
        """)

with tab2:
    st.header("Alert Configuration")
    
    alert_file = Path("alerts_config.json")
    
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    
    if enable_alerts:
        st.subheader("Email Alerts")
        email_enabled = st.checkbox("Enable Email Notifications")
        
        if email_enabled:
            email = st.text_input("Email Address")
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            smtp_user = st.text_input("SMTP Username")
            smtp_pass = st.text_input("SMTP Password", type="password")
        
        st.subheader("Webhook Alerts")
        webhook_enabled = st.checkbox("Enable Webhook Notifications")
        
        if webhook_enabled:
            webhook_url = st.text_input("Webhook URL (Discord/Slack)")
        
        st.subheader("Alert Triggers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_thrust = st.checkbox("Thrust Buy Signals", value=True)
            alert_cap = st.checkbox("Capitulation Signals", value=True)
            alert_trin = st.checkbox("TRIN Signals", value=True)
        
        with col2:
            alert_flow = st.checkbox("Large Flow Alerts", value=True)
            alert_gamma = st.checkbox("Gamma Extremes", value=True)
            alert_opportunities = st.checkbox("New Opportunities", value=True)
        
        if st.button("💾 Save Alert Settings"):
            config = {
                'enabled': True,
                'email': {
                    'enabled': email_enabled if 'email_enabled' in locals() else False,
                    'address': email if 'email' in locals() else '',
                    'smtp_server': smtp_server if 'smtp_server' in locals() else '',
                    'smtp_port': smtp_port if 'smtp_port' in locals() else 587
                },
                'webhook': {
                    'enabled': webhook_enabled if 'webhook_enabled' in locals() else False,
                    'url': webhook_url if 'webhook_url' in locals() else ''
                },
                'triggers': {
                    'thrust_buy': alert_thrust,
                    'capitulation': alert_cap,
                    'trin': alert_trin,
                    'flow': alert_flow,
                    'gamma': alert_gamma,
                    'opportunities': alert_opportunities
                }
            }
            
            with open(alert_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("✅ Alert settings saved!")

with tab3:
    st.header("User Preferences")
    
    st.subheader("Default Watchlist")
    default_watchlist = st.text_area(
        "Symbols (comma-separated)",
        value="SPY,QQQ,IWM,DIA,AAPL,MSFT,NVDA,TSLA,AMZN,META",
        height=100
    )
    
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        chart_style = st.selectbox("Chart Style", ["Plotly Dark", "Plotly", "Seaborn"])
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 60)
    
    st.subheader("Data Settings")
    
    default_lookback = st.selectbox("Default Lookback Period", ["1y", "2y", "5y", "10y"], index=1)
    cache_duration = st.slider("Cache duration (seconds)", 60, 600, 300)
    
    if st.button("💾 Save Preferences"):
        prefs = {
            'watchlist': [s.strip() for s in default_watchlist.split(',')],
            'display': {
                'theme': theme,
                'chart_style': chart_style,
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval
            },
            'data': {
                'default_lookback': default_lookback,
                'cache_duration': cache_duration
            }
        }
        
        with open('user_preferences.json', 'w') as f:
            json.dump(prefs, f, indent=2)
        
        st.success("✅ Preferences saved!")

with tab4:
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Platform Info")
        st.code(f"""
Version: 1.0.0
Python: {sys.version.split()[0]}
Streamlit: {st.__version__}
Platform: Multi-page App
        """)
    
    with col2:
        st.subheader("Data Sources")
        st.code("""
Primary: Schwab API
Fallback: yfinance
Historical: Up to 10 years
Real-time: Yes (when authenticated)
        """)
    
    st.subheader("Available Tools")
    
    tools = [
        "Max Gamma Scanner",
        "Index Positioning",
        "Boundary Scanner",
        "Flow Scanner",
        "Opportunity Scanner",
        "Options Flow Monitor",
        "Immediate Dashboard",
        "Newsletter Generator",
        "Report Generator"
    ]
    
    cols = st.columns(3)
    for idx, tool in enumerate(tools):
        with cols[idx % 3]:
            st.success(f"✅ {tool}")
    
    st.markdown("---")
    
    st.subheader("Quick Links")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Documentation", use_container_width=True):
            st.info("Documentation coming soon!")
    
    with col2:
        if st.button("🐛 Report Issue", use_container_width=True):
            st.info("Issue tracker coming soon!")
    
    with col3:
        if st.button("💬 Feedback", use_container_width=True):
            st.info("Feedback form coming soon!")
