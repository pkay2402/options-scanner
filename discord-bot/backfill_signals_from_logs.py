#!/usr/bin/env python3
"""
Backfill signal storage database from Discord bot logs
Parses today's logs and extracts all signals that were sent
"""

import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.services.signal_storage import get_storage

def parse_whale_alerts(log_content):
    """Extract whale flow signals from logs"""
    signals = []
    
    # Pattern: Look for whale flow log entries
    # Example: "Sent whale flow alert with X new flows"
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Sent whale flow alert with (\d+) new flows'
    
    matches = re.finditer(pattern, log_content)
    for match in matches:
        timestamp = match.group(1)
        count = int(match.group(2))
        print(f"  Found whale flow alert at {timestamp}: {count} flows")
    
    return signals

def parse_zscore_alerts(log_content):
    """Extract z-score signals from logs"""
    signals = []
    
    # Pattern: Look for z-score alert log entries
    # Example: "Generated summary for TSLA (1 days)"
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Sending alert for (.+?) \(Z-Score: ([-\d.]+)\)'
    
    matches = re.finditer(pattern, log_content)
    for match in matches:
        timestamp_str = match.group(1)
        symbol = match.group(2)
        zscore = float(match.group(3))
        
        # Determine signal type and direction
        if zscore <= -2:
            subtype = 'BUY_SIGNAL'
            direction = 'BULLISH'
        elif zscore >= 2:
            subtype = 'SELL_SIGNAL'
            direction = 'BEARISH'
        elif -2 < zscore <= -1.5:
            subtype = 'RECOVERY'
            direction = 'BULLISH'
        else:
            continue
        
        signals.append({
            'timestamp': timestamp_str,
            'symbol': symbol,
            'signal_type': 'ZSCORE',
            'signal_subtype': subtype,
            'direction': direction,
            'zscore': zscore
        })
        print(f"  Found z-score alert: {symbol} ({zscore:.2f}) at {timestamp_str}")
    
    return signals

def parse_tos_alerts(log_content):
    """Extract TOS alert signals from logs"""
    signals = []
    
    # Pattern: Look for TOS alert log entries
    # Example: "Sent TOS alert: TSLA HG_30mins_L"
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Sent TOS alert: (\w+) (HG_30mins_[LS])'
    
    matches = re.finditer(pattern, log_content)
    for match in matches:
        timestamp_str = match.group(1)
        symbol = match.group(2)
        scan_name = match.group(3)
        
        is_long = '_L' in scan_name
        subtype = 'LONG' if is_long else 'SHORT'
        direction = 'BULLISH' if is_long else 'BEARISH'
        
        signals.append({
            'timestamp': timestamp_str,
            'symbol': symbol,
            'signal_type': 'TOS',
            'signal_subtype': subtype,
            'direction': direction,
            'scan_name': scan_name
        })
        print(f"  Found TOS alert: {symbol} {subtype} at {timestamp_str}")
    
    return signals

def parse_etf_momentum_alerts(log_content):
    """Extract ETF momentum signals from logs"""
    signals = []
    
    # Pattern: Look for ETF momentum alert log entries
    # Example: "Sent ETF Momentum alert with X ETFs"
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Sent ETF Momentum alert with (\d+) ETFs'
    
    matches = re.finditer(pattern, log_content)
    for match in matches:
        timestamp = match.group(1)
        count = int(match.group(2))
        print(f"  Found ETF momentum alert at {timestamp}: {count} ETFs")
    
    return signals

def backfill_from_logs(log_file_path, date_filter=None):
    """Parse logs and backfill signal database"""
    print(f"📖 Reading log file: {log_file_path}")
    
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        return
    
    # Filter by date if specified
    if date_filter:
        print(f"📅 Filtering logs for date: {date_filter}")
        lines = log_content.split('\n')
        filtered_lines = [line for line in lines if date_filter in line]
        log_content = '\n'.join(filtered_lines)
        print(f"   Found {len(filtered_lines)} log entries for this date")
    
    print("\n🔍 Parsing signals from logs...")
    
    # Parse different signal types
    print("\n1️⃣ Parsing whale flow alerts...")
    whale_signals = parse_whale_alerts(log_content)
    
    print("\n2️⃣ Parsing z-score alerts...")
    zscore_signals = parse_zscore_alerts(log_content)
    
    print("\n3️⃣ Parsing TOS alerts...")
    tos_signals = parse_tos_alerts(log_content)
    
    print("\n4️⃣ Parsing ETF momentum alerts...")
    etf_signals = parse_etf_momentum_alerts(log_content)
    
    # Combine all signals
    all_signals = zscore_signals + tos_signals
    
    if not all_signals:
        print("\n📭 No signals found in logs to backfill")
        return
    
    print(f"\n💾 Backfilling {len(all_signals)} signals to database...")
    
    # Store signals
    storage = get_storage()
    success_count = 0
    
    for signal in all_signals:
        try:
            # Convert timestamp to datetime
            timestamp = datetime.strptime(signal['timestamp'], '%Y-%m-%d %H:%M:%S')
            
            # Prepare data dict
            data = {k: v for k, v in signal.items() 
                   if k not in ['timestamp', 'symbol', 'signal_type', 'signal_subtype', 'direction']}
            
            # Store signal with original timestamp
            storage.store_signal(
                symbol=signal['symbol'],
                signal_type=signal['signal_type'],
                signal_subtype=signal['signal_subtype'],
                direction=signal['direction'],
                price=None,  # Not available in logs
                data=data
            )
            
            # Update the timestamp in database to match log timestamp
            # (by default it uses CURRENT_TIMESTAMP)
            cursor = storage.conn.cursor()
            cursor.execute(
                "UPDATE signals SET timestamp = ? WHERE id = (SELECT MAX(id) FROM signals)",
                (signal['timestamp'],)
            )
            storage.conn.commit()
            
            success_count += 1
            print(f"  ✓ Stored: {signal['symbol']} {signal['signal_type']} {signal['signal_subtype']}")
            
        except Exception as e:
            print(f"  ✗ Error storing {signal.get('symbol', 'unknown')}: {e}")
    
    print(f"\n✅ Backfill complete: {success_count}/{len(all_signals)} signals stored")
    
    # Show summary
    print("\n📊 Summary by signal type:")
    by_type = {}
    for signal in all_signals:
        sig_type = signal['signal_type']
        by_type[sig_type] = by_type.get(sig_type, 0) + 1
    
    for sig_type, count in by_type.items():
        print(f"  {sig_type}: {count} signals")

if __name__ == '__main__':
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("📊 Signal Storage Backfill Tool")
    print("=" * 60)
    print(f"\nBackfilling signals from: {today}")
    print("This will parse Discord bot logs and populate the database")
    print("with signals that were generated earlier today.\n")
    
    # Path to log file (adjust if needed)
    log_file = Path(__file__).parent.parent / 'logs' / 'discord_bot.log'
    
    if not log_file.exists():
        print(f"❌ Log file not found at: {log_file}")
        print("Please specify the correct path to discord_bot.log")
        sys.exit(1)
    
    backfill_from_logs(log_file, date_filter=today)
    
    print(f"\n🚀 Backfill complete! You can now use /summarize to query signals.")
