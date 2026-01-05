#!/usr/bin/env python3
"""
Power Inflow Detector - CBOE Edition
Analyzes options flow from CBOE data to identify significant institutional activity
Enhanced with Schwab API for real-time bid/ask prices
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.schwab_client import SchwabClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# CBOE CSV URLs (from options.py)
CBOE_URLS = [
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone',
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo',
    'https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt'
]


def fetch_schwab_options_data(symbol):
    """Fetch current bid/ask prices from Schwab API for a symbol"""
    try:
        client = SchwabClient()
        options_chain = client.get_options_chain(
            symbol,
            strike_count=50
        )
        
        if not options_chain:
            logger.warning(f"No options chain data for {symbol}")
            return None
        
        # Build lookup dictionary: (strike, expiration, call/put) -> (bid, ask, open_interest)
        bid_ask_map = {}
        
        # Process calls
        if 'callExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['callExpDateMap'].items():
                exp_key = exp_date.split(':')[0]  # Get just the date part
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if contracts:
                        contract = contracts[0]
                        bid = contract.get('bid', 0)
                        ask = contract.get('ask', 0)
                        oi = contract.get('openInterest', 0)
                        bid_ask_map[(strike, exp_key, 'C')] = (bid, ask, oi)
        
        # Process puts
        if 'putExpDateMap' in options_chain:
            for exp_date, strikes in options_chain['putExpDateMap'].items():
                exp_key = exp_date.split(':')[0]  # Get just the date part
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)
                    if contracts:
                        contract = contracts[0]
                        bid = contract.get('bid', 0)
                        ask = contract.get('ask', 0)
                        oi = contract.get('openInterest', 0)
                        bid_ask_map[(strike, exp_key, 'P')] = (bid, ask, oi)
        
        logger.info(f"✓ Fetched {len(bid_ask_map)} bid/ask quotes from Schwab for {symbol}")
        return bid_ask_map
        
    except Exception as e:
        logger.error(f"Error fetching Schwab data for {symbol}: {e}")
        return None


def fetch_cboe_data(date_str=None):
    """Fetch all options data from CBOE CSV files"""
    
    logger.info(f"Fetching CBOE data...")
    
    all_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url in CBOE_URLS:
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            
            if not df.empty:
                logger.info(f"✓ Got {len(df)} rows")
                all_data.append(df)
            else:
                logger.warning(f"Empty data")
                
        except Exception as e:
            logger.error(f"Failed: {e}")
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total rows: {len(combined)}")
        return combined
    else:
        logger.error("No data fetched")
        return None


def main():
    """Main execution"""
    
    print("\n🔍 POWER INFLOW DETECTOR - CBOE EDITION\n")
    
    # Fetch data
    df = fetch_cboe_data()
    
    if df is None:
        print("❌ Failed to fetch CBOE data")
        return
    
    print(f"\n✓ Got {len(df)} total rows")
    print(f"\nColumns: {list(df.columns)}")
    
    # Mag 7 stocks
    mag7_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    for symbol in mag7_symbols:
        analyze_symbol(df, symbol)


def analyze_symbol(df, symbol):
    """Analyze power inflows for a single symbol"""
    
    # Filter for symbol
    tsm = df[df['Symbol'].str.upper() == symbol].copy()
    print(f"\n{'='*80}")
    print(f"🎯 {symbol} POWER INFLOW ANALYSIS")
    print('='*80)
    print(f"✓ {symbol} rows: {len(tsm)} (before filtering)")
    
    if tsm.empty:
        print(f"❌ No {symbol} data found")
        return
    
    # Filter out expired contracts
    today = pd.Timestamp.now()
    tsm['Expiration_dt'] = pd.to_datetime(tsm['Expiration'], errors='coerce')
    tsm = tsm[tsm['Expiration_dt'] >= today].copy()
    expired_count = len(df[df['Symbol'].str.upper() == symbol]) - len(tsm)
    
    print(f"✓ Filtered out {expired_count} expired contracts")
    print(f"✓ Active {symbol} contracts: {len(tsm)}")
    
    if tsm.empty:
        print(f"❌ No active {symbol} contracts found")
        return
    
    # Fetch current bid/ask from Schwab API
    print(f"\n🔍 Fetching current bid/ask prices from Schwab API...")
    schwab_bid_ask = fetch_schwab_options_data(symbol)
    
    # Convert numeric columns
    tsm['Volume'] = pd.to_numeric(tsm['Volume'], errors='coerce')
    tsm['Strike Price'] = pd.to_numeric(tsm['Strike Price'], errors='coerce')
    tsm['Last Price'] = pd.to_numeric(tsm['Last Price'], errors='coerce')
    tsm['Bid Price'] = pd.to_numeric(tsm['Bid Price'], errors='coerce')
    tsm['Ask Price'] = pd.to_numeric(tsm['Ask Price'], errors='coerce')
    
    # Initialize Open Interest column
    tsm['Open Interest'] = 0
    
    # Update bid/ask prices and open interest from Schwab where available
    if schwab_bid_ask:
        def update_bid_ask_oi(row):
            try:
                strike = float(row['Strike Price'])
                exp_date = row['Expiration'][:10]  # YYYY-MM-DD format
                option_type = row['Call/Put']
                
                key = (strike, exp_date, option_type)
                if key in schwab_bid_ask:
                    bid, ask, oi = schwab_bid_ask[key]
                    return pd.Series({'Bid Price': bid, 'Ask Price': ask, 'Open Interest': oi})
            except:
                pass
            return pd.Series({'Bid Price': row['Bid Price'], 'Ask Price': row['Ask Price'], 'Open Interest': 0})
        
        tsm[['Bid Price', 'Ask Price', 'Open Interest']] = tsm.apply(update_bid_ask_oi, axis=1)
        
        # Count how many got updated
        updated = ((tsm['Bid Price'] > 0) | (tsm['Ask Price'] > 0)).sum()
        oi_updated = (tsm['Open Interest'] > 0).sum()
        print(f"✓ Updated {updated}/{len(tsm)} rows with Schwab bid/ask data")
        print(f"✓ Updated {oi_updated}/{len(tsm)} rows with Open Interest data")
    else:
        print("⚠️ Could not fetch Schwab data, using CBOE bid/ask only")
    
    # Calculate premium (Volume * Last Price * 100)
    tsm['Premium'] = tsm['Volume'] * tsm['Last Price'] * 100
    
    # Calculate Volume/OI ratio
    tsm['Vol/OI'] = 0.0
    tsm.loc[tsm['Open Interest'] > 0, 'Vol/OI'] = tsm['Volume'] / tsm['Open Interest']
    
    # Classify position activity based on Vol/OI ratio
    def classify_position_activity(row):
        vol_oi = row['Vol/OI']
        if row['Open Interest'] == 0:
            return 'NEW CONTRACT'
        elif vol_oi > 2.0:
            return 'OPENING'
        elif vol_oi > 1.0:
            return 'ACTIVE'
        elif vol_oi > 0.5:
            return 'NORMAL'
        else:
            return 'CLOSING'
    
    tsm['Position Activity'] = tsm.apply(classify_position_activity, axis=1)
    
    # Calculate mid price
    tsm['Mid Price'] = (tsm['Bid Price'] + tsm['Ask Price']) / 2
    tsm.loc[tsm['Mid Price'] == 0, 'Mid Price'] = tsm['Last Price']
    
    # Determine buy vs sell based on bid/ask
    def classify_trade_side(row):
        """Classify if trade was buy or sell side based on price vs bid/ask"""
        last = row['Last Price']
        bid = row['Bid Price']
        ask = row['Ask Price']
        
        if pd.isna(last) or last == 0:
            return 'UNKNOWN'
        
        if pd.isna(bid) or pd.isna(ask) or (bid == 0 and ask == 0):
            return 'UNKNOWN'
        
        # Calculate spread
        if ask > 0 and bid >= 0:
            spread = ask - bid
            mid = (bid + ask) / 2
            
            # Trade at or near ask = aggressive buying
            if last >= ask * 0.95:
                return 'BUY'
            # Trade at or near bid = aggressive selling
            elif last <= bid * 1.05:
                return 'SELL'
            # Trade near mid or unclear
            elif spread > 0 and abs(last - mid) / spread < 0.2:
                return 'MID'
            else:
                return 'UNKNOWN'
        
        return 'UNKNOWN'
    
    tsm['Trade Side'] = tsm.apply(classify_trade_side, axis=1)
    
    # Determine directional bias
    def determine_direction(row):
        """Determine if position is bullish or bearish"""
        option_type = row['Call/Put']
        trade_side = row['Trade Side']
        
        if trade_side == 'UNKNOWN':
            return 'UNCLEAR'
        
        # Call Buy = Bullish
        if option_type == 'C' and trade_side == 'BUY':
            return 'BULLISH'
        # Call Sell = Bearish (or neutral/covered)
        elif option_type == 'C' and trade_side == 'SELL':
            return 'BEARISH/NEUTRAL'
        # Put Buy = Bearish (or hedge)
        elif option_type == 'P' and trade_side == 'BUY':
            return 'BEARISH/HEDGE'
        # Put Sell = Bullish (willing to own at strike)
        elif option_type == 'P' and trade_side == 'SELL':
            return 'BULLISH'
        else:
            return 'UNCLEAR'
    
    tsm['Direction'] = tsm.apply(determine_direction, axis=1)
    
    print("\n" + "="*80)
    print(f"{symbol} POWER INFLOW ANALYSIS WITH BUY/SELL DETECTION")
    print("="*80)
    
    # Show trade side distribution
    print("\n📊 TRADE SIDE DISTRIBUTION:")
    print("-"*80)
    side_dist = tsm['Trade Side'].value_counts()
    for side, count in side_dist.items():
        pct = count / len(tsm) * 100
        print(f"  {side:10s}: {count:5d} contracts ({pct:5.1f}%)")
    
    # Show position activity distribution (Volume/OI analysis)
    print("\n📊 POSITION ACTIVITY (Volume/OI Analysis):")
    print("-"*80)
    activity_dist = tsm['Position Activity'].value_counts()
    for activity, count in activity_dist.items():
        pct = count / len(tsm) * 100
        print(f"  {activity:15s}: {count:5d} contracts ({pct:5.1f}%)")
    print("\n  Legend: OPENING (>2.0) = New positions | CLOSING (<0.5) = Unwinding")
    
    # Show top by volume with trade side
    print(f"\n\n📊 TOP 10 {symbol} BY VOLUME (with Position Activity):")
    print("-"*80)
    top_vol = tsm.nlargest(10, 'Volume')[['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Open Interest', 'Vol/OI', 'Trade Side', 'Position Activity', 'Direction', 'Premium']]
    print(top_vol.to_string(index=False))
    
    # Show top by premium with trade side
    print(f"\n\n💰 TOP 10 {symbol} BY PREMIUM (with Trade Side):")
    print("-"*80)
    top_prem = tsm.nlargest(10, 'Premium')[['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Last Price', 'Trade Side', 'Direction', 'Premium']]
    print(top_prem.to_string(index=False))
    
    # Power inflow definition
    print("\n\n" + "="*80)
    print("POWER INFLOW SCORING")
    print("="*80)
    
    # Calculate power score
    def calculate_power_score(row):
        score = 0
        
        # Volume score (0-30)
        vol = row['Volume']
        if vol >= 10000:
            score += 30
        elif vol >= 5000:
            score += 25
        elif vol >= 2000:
            score += 20
        elif vol >= 1000:
            score += 15
        elif vol >= 500:
            score += 10
        else:
            score += 5
        
        # Premium score (0-30)
        prem = row['Premium']
        if prem >= 5000000:
            score += 30
        elif prem >= 2000000:
            score += 25
        elif prem >= 1000000:
            score += 20
        elif prem >= 500000:
            score += 15
        elif prem >= 100000:
            score += 10
        else:
            score += 5
        
        # Expiration urgency (0-20)
        # Near-term = more urgent/aggressive
        try:
            exp_date = pd.to_datetime(row['Expiration'])
            dte = (exp_date - pd.Timestamp.now()).days
            if dte <= 7:
                score += 20
            elif dte <= 14:
                score += 15
            elif dte <= 30:
                score += 10
            else:
                score += 5
        except:
            score += 5
        
        # Call/Put positioning (0-20)
        # Calls = bullish, Puts = bearish
        if row['Call/Put'] == 'C':
            score += 10  # Calls = bullish
        else:
            score += 10  # Puts = bearish (equal weight)
        
        return score
    
    tsm['Power Score'] = tsm.apply(calculate_power_score, axis=1)
    
    # Show top power inflows
    print(f"\n🎯 TOP 10 {symbol} POWER INFLOWS (by score):")
    print("-"*80)
    top_power = tsm.nlargest(10, 'Power Score')[['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Vol/OI', 'Position Activity', 'Trade Side', 'Direction', 'Premium', 'Power Score']]
    print(top_power.to_string(index=False))
    
    # Separate BUY vs SELL flows
    print("\n\n" + "="*80)
    print(f"{symbol} SUMMARY WITH DIRECTIONAL BIAS")
    print("="*80)
    total_calls = tsm[tsm['Call/Put'] == 'C']['Volume'].sum()
    total_puts = tsm[tsm['Call/Put'] == 'P']['Volume'].sum()
    call_premium = tsm[tsm['Call/Put'] == 'C']['Premium'].sum()
    put_premium = tsm[tsm['Call/Put'] == 'P']['Premium'].sum()
    
    print(f"\nTotal Volume:")
    print(f"  Calls: {total_calls:,.0f}")
    print(f"  Puts:  {total_puts:,.0f}")
    print(f"  P/C Ratio: {total_puts/total_calls if total_calls > 0 else 0:.2f}")
    
    print(f"\nTotal Premium:")
    print(f"  Calls: ${call_premium:,.0f}")
    print(f"  Puts:  ${put_premium:,.0f}")
    print(f"  P/C Ratio: {put_premium/call_premium if call_premium > 0 else 0:.2f}")
    
    # Directional flows
    print(f"\nDirectional Breakdown:")
    bullish_premium = tsm[tsm['Direction'].str.contains('BULLISH', na=False)]['Premium'].sum()
    bearish_premium = tsm[tsm['Direction'].str.contains('BEARISH', na=False)]['Premium'].sum()
    
    print(f"  Bullish flows: ${bullish_premium:,.0f}")
    print(f"  Bearish flows: ${bearish_premium:,.0f}")
    
    if bullish_premium + bearish_premium > 0:
        bull_pct = bullish_premium / (bullish_premium + bearish_premium) * 100
        bear_pct = bearish_premium / (bullish_premium + bearish_premium) * 100
        print(f"  → {bull_pct:.1f}% Bullish vs {bear_pct:.1f}% Bearish")
    
    # Position flow analysis (Opening vs Closing)
    print(f"\nPosition Flow Analysis:")
    opening_premium = tsm[tsm['Position Activity'] == 'OPENING']['Premium'].sum()
    closing_premium = tsm[tsm['Position Activity'] == 'CLOSING']['Premium'].sum()
    opening_count = len(tsm[tsm['Position Activity'] == 'OPENING'])
    closing_count = len(tsm[tsm['Position Activity'] == 'CLOSING'])
    
    print(f"  Opening positions: {opening_count} contracts, ${opening_premium:,.0f}")
    print(f"  Closing positions: {closing_count} contracts, ${closing_premium:,.0f}")
    
    if opening_count + closing_count > 0:
        net_opening = opening_count - closing_count
        if net_opening > 0:
            print(f"  → NET OPENING: +{net_opening} contracts (bullish positioning)")
        elif net_opening < 0:
            print(f"  → NET CLOSING: {net_opening} contracts (unwinding positions)")
        else:
            print(f"  → BALANCED: Equal opening/closing activity")
    
    print(f"\nPower Inflows (Score >= 70):")
    print(f"\nPower Inflows (Score >= 70):")
    high_power = tsm[tsm['Power Score'] >= 70]
    print(f"  Count: {len(high_power)}")
    print(f"  Total Premium: ${high_power['Premium'].sum():,.0f}")
    
    if len(high_power) > 0:
        print(f"\n  Breakdown:")
        for _, row in high_power.iterrows():
            print(f"    • {row['Call/Put']} ${row['Strike Price']:.0f} {row['Expiration'][:10]} - {row['Trade Side']} - {row['Direction']} - ${row['Premium']:,.0f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

