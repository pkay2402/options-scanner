#!/usr/bin/env python3
"""
Discord Power Inflow Scanner
Scans for significant options flow and sends alerts to Discord
Designed to run every 3 minutes during market hours
"""

import pandas as pd
import numpy as np
import requests
import json
import sys
import os
from datetime import datetime, timedelta
from io import StringIO
import logging
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.schwab_client import SchwabClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State file to track reported flows
STATE_FILE = os.path.join(os.path.dirname(__file__), 'reported_flows.json')

# CBOE URLs
CBOE_URLS = [
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone',
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo',
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt'
]

# Watchlist - customize as needed
WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mag 7
    'SPY', 'QQQ', 'IWM', 'IBIT', # ETFs
    'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'TSM','CRWD','ZS', # Tech
    'JPM', 'BAC', 'GS', 'C', # Finance
    'XOM', 'CVX',  # Energy
]


def generate_flow_signature(row):
    """Generate unique signature for a flow to detect duplicates"""
    sig_string = f"{row['Symbol']}_{row['Call/Put']}_{row['Strike Price']}_{row['Expiration'][:10]}_{row['Volume']}_{row['Last Price']}"
    return hashlib.md5(sig_string.encode()).hexdigest()


def load_reported_flows():
    """Load previously reported flows"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                # Clean up old entries (older than 24 hours)
                cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
                data['flows'] = {k: v for k, v in data['flows'].items() if v > cutoff}
                return data
        except:
            return {'flows': {}, 'last_run': None}
    return {'flows': {}, 'last_run': None}


def save_reported_flows(state):
    """Save reported flows state"""
    state['last_run'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def fetch_cboe_data():
    """Fetch CBOE data"""
    all_data = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for url in CBOE_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def fetch_schwab_data(symbol):
    """Fetch Schwab options data for bid/ask/OI"""
    try:
        client = SchwabClient()
        chain = client.get_options_chain(symbol, strike_count=50)
        
        if not chain:
            return None
        
        data_map = {}
        
        for opt_type, exp_map_key in [('C', 'callExpDateMap'), ('P', 'putExpDateMap')]:
            if exp_map_key in chain:
                for exp_date, strikes in chain[exp_map_key].items():
                    exp_key = exp_date.split(':')[0]
                    for strike_str, contracts in strikes.items():
                        if contracts:
                            c = contracts[0]
                            data_map[(float(strike_str), exp_key, opt_type)] = {
                                'bid': c.get('bid', 0),
                                'ask': c.get('ask', 0),
                                'oi': c.get('openInterest', 0)
                            }
        
        return data_map
    except Exception as e:
        logger.error(f"Error fetching Schwab data for {symbol}: {e}")
        return None


def analyze_stock_flow(df, symbol, schwab_data, reported_flows):
    """Analyze flow for a single stock and return Discord-ready alerts"""
    
    # Filter for symbol and active contracts
    stock_df = df[df['Symbol'].str.upper() == symbol].copy()
    if stock_df.empty:
        return []
    
    # Filter expired
    today = pd.Timestamp.now()
    stock_df['Expiration_dt'] = pd.to_datetime(stock_df['Expiration'], errors='coerce')
    stock_df = stock_df[stock_df['Expiration_dt'] >= today].copy()
    
    if stock_df.empty:
        return []
    
    # Convert numeric
    for col in ['Volume', 'Strike Price', 'Last Price', 'Bid Price', 'Ask Price']:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
    
    # Initialize columns
    stock_df['Open Interest'] = 0
    stock_df['Bid Price_schwab'] = stock_df['Bid Price']
    stock_df['Ask Price_schwab'] = stock_df['Ask Price']
    
    # Update with Schwab data
    if schwab_data:
        for idx, row in stock_df.iterrows():
            key = (row['Strike Price'], row['Expiration'][:10], row['Call/Put'])
            if key in schwab_data:
                stock_df.at[idx, 'Bid Price_schwab'] = schwab_data[key]['bid']
                stock_df.at[idx, 'Ask Price_schwab'] = schwab_data[key]['ask']
                stock_df.at[idx, 'Open Interest'] = schwab_data[key]['oi']
    
    # Calculate metrics
    stock_df['Premium'] = stock_df['Volume'] * stock_df['Last Price'] * 100
    stock_df['Vol/OI'] = 0.0
    stock_df.loc[stock_df['Open Interest'] > 0, 'Vol/OI'] = stock_df['Volume'] / stock_df['Open Interest']
    
    # Classify trade side
    def get_trade_side(row):
        last = row['Last Price']
        bid = row['Bid Price_schwab']
        ask = row['Ask Price_schwab']
        
        if pd.isna(last) or last == 0 or bid == 0 or ask == 0:
            return 'UNKNOWN'
        
        if last >= ask * 0.95:
            return 'BUY'
        elif last <= bid * 1.05:
            return 'SELL'
        else:
            return 'MID'
    
    stock_df['Trade Side'] = stock_df.apply(get_trade_side, axis=1)
    
    # Calculate DTE
    stock_df['DTE'] = (stock_df['Expiration_dt'] - today).dt.days
    
    # Filter for significant flows only
    # Criteria: Large volume OR large premium OR high Vol/OI with clear direction
    significant = stock_df[
        (stock_df['Volume'] >= 500) &  # Minimum volume
        (
            (stock_df['Premium'] >= 100000) |  # $100K+ premium
            (stock_df['Volume'] >= 1000) |  # 1000+ contracts
            ((stock_df['Vol/OI'] > 2.0) & (stock_df['Trade Side'].isin(['BUY', 'SELL'])))  # Opening positions
        )
    ].copy()
    
    # Generate alerts
    alerts = []
    
    for _, row in significant.iterrows():
        # Generate signature
        sig = generate_flow_signature(row)
        
        # Skip if already reported
        if sig in reported_flows:
            continue
        
        # Mark as reported
        reported_flows[sig] = datetime.now().isoformat()
        
        # Determine direction
        if row['Call/Put'] == 'C' and row['Trade Side'] == 'BUY':
            direction = '🟢 BULLISH'
        elif row['Call/Put'] == 'P' and row['Trade Side'] == 'BUY':
            direction = '🔴 BEARISH'
        elif row['Call/Put'] == 'P' and row['Trade Side'] == 'SELL':
            direction = '🟢 BULLISH'
        elif row['Call/Put'] == 'C' and row['Trade Side'] == 'SELL':
            direction = '🟡 NEUTRAL'
        else:
            direction = '⚪ UNCLEAR'
        
        # Skip unclear directions
        if direction == '⚪ UNCLEAR':
            continue
        
        # Position activity
        if row['Vol/OI'] > 2.0:
            activity = '🔥 OPENING'
        elif row['Vol/OI'] < 0.5 and row['Open Interest'] > 0:
            activity = '❄️ CLOSING'
        else:
            activity = '📊 ACTIVE'
        
        # Build alert message
        alert = {
            'symbol': symbol,
            'type': 'CALL' if row['Call/Put'] == 'C' else 'PUT',
            'strike': row['Strike Price'],
            'expiration': row['Expiration'][:10],
            'dte': int(row['DTE']),
            'volume': int(row['Volume']),
            'oi': int(row['Open Interest']),
            'vol_oi': round(row['Vol/OI'], 2) if row['Open Interest'] > 0 else 0,
            'premium': row['Premium'],
            'trade_side': row['Trade Side'],
            'direction': direction,
            'activity': activity,
            'price': row['Last Price'],
            'timestamp': datetime.now()
        }
        
        alerts.append(alert)
    
    return alerts


def format_discord_message(alerts):
    """Format alerts for Discord (max 2000 chars per message)"""
    if not alerts:
        return []
    
    messages = []
    current_msg = "**🔔 POWER INFLOW ALERTS**\n\n"
    
    for alert in alerts:
        # Format single alert
        vol_oi_str = f"{alert['vol_oi']:.1f}" if alert['oi'] > 0 else "NEW"
        
        alert_line = (
            f"{alert['direction']} **${alert['symbol']}** {alert['type']} "
            f"${alert['strike']:.0f} {alert['expiration']} ({alert['dte']}d)\n"
            f"├ Volume: **{alert['volume']:,}** | OI: {alert['oi']:,} | V/OI: {vol_oi_str}\n"
            f"├ Premium: **${alert['premium']:,.0f}** | Side: {alert['trade_side']}\n"
            f"└ {alert['activity']}\n\n"
        )
        
        # Check if adding this alert exceeds Discord limit
        if len(current_msg) + len(alert_line) > 1900:
            messages.append(current_msg)
            current_msg = "**🔔 POWER INFLOW ALERTS (continued)**\n\n" + alert_line
        else:
            current_msg += alert_line
    
    if current_msg:
        messages.append(current_msg)
    
    return messages


def scan_for_power_inflows():
    """Main scanning function - returns list of Discord messages"""
    logger.info("Starting power inflow scan...")
    
    # Load state
    state = load_reported_flows()
    reported_flows = state['flows']
    
    # Fetch CBOE data
    df = fetch_cboe_data()
    if df is None:
        logger.error("Failed to fetch CBOE data")
        return []
    
    logger.info(f"Fetched {len(df)} rows from CBOE")
    
    # Scan each symbol
    all_alerts = []
    
    for symbol in WATCHLIST:
        logger.info(f"Scanning {symbol}...")
        
        # Fetch Schwab data for this symbol
        schwab_data = fetch_schwab_data(symbol)
        
        # Analyze
        alerts = analyze_stock_flow(df, symbol, schwab_data, reported_flows)
        
        if alerts:
            logger.info(f"Found {len(alerts)} new flows for {symbol}")
            all_alerts.extend(alerts)
    
    # Save updated state
    state['flows'] = reported_flows
    save_reported_flows(state)
    
    # Sort by premium (largest first)
    all_alerts.sort(key=lambda x: x['premium'], reverse=True)
    
    # Format for Discord
    messages = format_discord_message(all_alerts)
    
    logger.info(f"Scan complete. {len(all_alerts)} total alerts, {len(messages)} Discord messages")
    
    return messages


def get_summary_stats():
    """Get summary statistics for a quick status check"""
    state = load_reported_flows()
    
    last_run = state.get('last_run')
    if last_run:
        last_run_dt = datetime.fromisoformat(last_run)
        time_since = datetime.now() - last_run_dt
        last_run_str = f"{int(time_since.total_seconds() / 60)} minutes ago"
    else:
        last_run_str = "Never"
    
    flows_today = len([f for f, t in state['flows'].items() 
                       if datetime.fromisoformat(t).date() == datetime.now().date()])
    
    return {
        'last_run': last_run_str,
        'flows_today': flows_today,
        'symbols_watching': len(WATCHLIST)
    }


if __name__ == "__main__":
    # Test run
    messages = scan_for_power_inflows()
    
    print("\n" + "="*80)
    print("DISCORD MESSAGES OUTPUT:")
    print("="*80 + "\n")
    
    for i, msg in enumerate(messages, 1):
        print(f"--- Message {i} ---")
        print(msg)
        print()
    
    if not messages:
        print("No new significant flows detected.")
    
    # Print stats
    stats = get_summary_stats()
    print("\n" + "="*80)
    print("SCANNER STATS:")
    print("="*80)
    print(f"Last Run: {stats['last_run']}")
    print(f"Flows Today: {stats['flows_today']}")
    print(f"Watching: {stats['symbols_watching']} symbols")
