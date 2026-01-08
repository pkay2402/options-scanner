#!/usr/bin/env python3
"""
Test script to verify TOS email alerts fetching
Connects to Gmail and displays what alerts would be sent to Discord
"""

import imaplib
import email
import re
from datetime import datetime, date
from dateutil import parser
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
import os
import pytz

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Email configuration
EMAIL_ADDRESS = os.getenv('TOS_EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('TOS_EMAIL_PASSWORD')
SENDER_EMAIL = "alerts@thinkorswim.com"
KEYWORDS = ["HG_30mins_L", "HG_30mins_S"]

def parse_email_body(msg):
    """Parse email body with HTML handling"""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True).decode()
                    if part.get_content_type() == "text/html":
                        soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                        return soup.get_text(separator=' ', strip=True)
                    return body
        else:
            body = msg.get_payload(decode=True).decode()
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                return soup.get_text(separator=' ', strip=True)
            return body
    except Exception as e:
        print(f"Error parsing email body: {e}")
        return ""

def test_email_fetch():
    """Test fetching TOS alerts from email"""
    
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("❌ Email credentials not configured!")
        print("Set TOS_EMAIL_ADDRESS and TOS_EMAIL_PASSWORD in .env")
        return
    
    print(f"📧 Testing TOS Alerts Email Fetch")
    print(f"Email: {EMAIL_ADDRESS}")
    print("=" * 60)
    
    try:
        # Connect to Gmail
        print("\n🔌 Connecting to Gmail...")
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select('inbox')
        print("✅ Connected successfully!")
        
        # Get today's date in ET
        eastern = pytz.timezone('US/Eastern')
        today_et = datetime.now(eastern).date()
        date_since = today_et.strftime("%d-%b-%Y")
        
        print(f"\n📅 Searching for alerts since: {date_since} (ET)")
        
        alerts_found = []
        
        for keyword in KEYWORDS:
            print(f"\n🔍 Searching for: {keyword}")
            search_criteria = f'(FROM "{SENDER_EMAIL}" SUBJECT "{keyword}" SINCE "{date_since}")'
            _, data = mail.search(None, search_criteria)
            
            email_ids = data[0].split()
            print(f"   Found {len(email_ids)} emails")
            
            for num in email_ids:
                _, email_data = mail.fetch(num, '(RFC822)')
                msg = email.message_from_bytes(email_data[0][1])
                
                # Parse date
                email_datetime = parser.parse(msg['Date'])
                email_date = email_datetime.date()
                
                # Parse body for symbols
                body = parse_email_body(msg)
                symbols = re.findall(
                    r'New symbols:\s*([A-Z,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', 
                    body
                )
                
                if symbols:
                    for symbol_group in symbols:
                        extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                        signal_type = symbol_group[1]
                        
                        for ticker in extracted_symbols:
                            if ticker.isalpha():
                                alerts_found.append({
                                    'ticker': ticker,
                                    'signal': signal_type,
                                    'time': email_datetime,
                                    'is_long': '_L' in signal_type
                                })
        
        mail.close()
        mail.logout()
        
        # Display results
        print("\n" + "=" * 60)
        print(f"📊 RESULTS: Found {len(alerts_found)} alerts")
        print("=" * 60)
        
        if alerts_found:
            # Group by signal type
            long_alerts = [a for a in alerts_found if a['is_long']]
            short_alerts = [a for a in alerts_found if not a['is_long']]
            
            if long_alerts:
                print(f"\n🟢 LONG Signals ({len(long_alerts)}):")
                print("-" * 60)
                for alert in long_alerts:
                    emoji = "🟢"
                    print(f"{emoji} {alert['ticker']:<8} | {alert['signal']:<15} | {alert['time'].strftime('%I:%M %p ET')}")
                    print(f"   Would send Discord embed:")
                    print(f"   Title: 🟢 TOS LONG Alert: {alert['ticker']}")
                    print(f"   Signal: {alert['signal']}")
                    print(f"   Time: {alert['time'].strftime('%Y-%m-%d %I:%M:%S %p')}")
                    print()
            
            if short_alerts:
                print(f"\n🔴 SHORT Signals ({len(short_alerts)}):")
                print("-" * 60)
                for alert in short_alerts:
                    emoji = "🔴"
                    print(f"{emoji} {alert['ticker']:<8} | {alert['signal']:<15} | {alert['time'].strftime('%I:%M %p ET')}")
                    print(f"   Would send Discord embed:")
                    print(f"   Title: 🔴 TOS SHORT Alert: {alert['ticker']}")
                    print(f"   Signal: {alert['signal']}")
                    print(f"   Time: {alert['time'].strftime('%Y-%m-%d %I:%M:%S %p')}")
                    print()
        else:
            print(f"\n⚠️  No TOS alerts found for today ({today_et})")
            print(f"   Make sure TOS is sending alerts to {EMAIL_ADDRESS}")
            print(f"   Scans monitored: {', '.join(KEYWORDS)}")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_email_fetch()
