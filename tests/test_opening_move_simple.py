#!/usr/bin/env python3
"""
Simplified test for Opening Move logic - tests the core analysis without Discord
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import json

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient


def load_watchlist():
    """Load watchlist from bot_watchlist.json (same as Discord bot)"""
    # Try bot watchlist file first
    try:
        watchlist_file = project_root / 'discord-bot' / 'bot_watchlist.json'
        if watchlist_file.exists():
            with open(watchlist_file, 'r') as f:
                data = json.load(f)
                symbols = data.get('symbols', [])
                # Create map from full data
                data_map = {item['symbol']: item for item in data.get('data', [])}
                print(f"   Source: bot_watchlist.json (updated: {data.get('last_updated', 'unknown')})")
                print(f"   Symbols: {len(symbols)}, Market data available: {len(data_map)}")
                return symbols, data_map
    except Exception as e:
        print(f"   bot_watchlist.json unavailable: {e}")
    
    # Fallback to user_preferences.json
    try:
        prefs_path = project_root / "user_preferences.json"
        if prefs_path.exists():
            with open(prefs_path, 'r') as f:
                prefs = json.load(f)
                print(f"   Source: user_preferences.json")
                return prefs.get('watchlist', []), {}
        return [], {}
    except Exception as e:
        print(f"Error loading watchlist: {e}")
        return [], {}


def analyze_volume_walls(options_data, underlying_price):
    """Analyze call/put walls and flip level"""
    try:
        import numpy as np
        
        # Collect data from all strikes
        call_volumes = {}
        put_volumes = {}
        call_oi = {}
        put_oi = {}
        call_gamma = {}
        put_gamma = {}
        
        # Process calls
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        strike = float(strike_str)
                        contract = contracts[0]
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        call_volumes[strike] = call_volumes.get(strike, 0) + volume
                        call_oi[strike] = call_oi.get(strike, 0) + oi
                        call_gamma[strike] = call_gamma.get(strike, 0) + gamma
        
        # Process puts
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        strike = float(strike_str)
                        contract = contracts[0]
                        volume = contract.get('totalVolume', 0)
                        oi = contract.get('openInterest', 0)
                        gamma = contract.get('gamma', 0)
                        
                        put_volumes[strike] = put_volumes.get(strike, 0) + volume
                        put_oi[strike] = put_oi.get(strike, 0) + oi
                        put_gamma[strike] = put_gamma.get(strike, 0) + gamma
        
        # Get all strikes
        all_strikes = sorted(set(call_volumes.keys()) | set(put_volumes.keys()))
        
        if not all_strikes:
            return None, None, None, None
        
        # Calculate net GEX for each strike
        gex_by_strike = {}
        for strike in all_strikes:
            call_gex = call_gamma.get(strike, 0) * call_oi.get(strike, 0) * 100 * underlying_price
            put_gex = put_gamma.get(strike, 0) * put_oi.get(strike, 0) * 100 * underlying_price * -1
            gex_by_strike[strike] = call_gex + put_gex
        
        # Find max GEX strike (flip level)
        max_gex_strike = max(gex_by_strike.items(), key=lambda x: abs(x[1]))[0] if gex_by_strike else None
        max_gex_value = gex_by_strike.get(max_gex_strike, 0) if max_gex_strike else 0
        
        # Find call wall (highest call OI above price)
        above_strikes = [s for s in all_strikes if s > underlying_price]
        call_wall = max([(s, call_oi.get(s, 0)) for s in above_strikes], 
                      key=lambda x: x[1])[0] if above_strikes else None
        
        # Find put wall (highest put OI below price)
        below_strikes = [s for s in all_strikes if s < underlying_price]
        put_wall = max([(s, put_oi.get(s, 0)) for s in below_strikes], 
                     key=lambda x: x[1])[0] if below_strikes else None
        
        return call_wall, put_wall, max_gex_strike, max_gex_value
        
    except Exception as e:
        print(f"Error analyzing walls: {e}")
        return None, None, None, None


def calculate_opportunity_score(symbol, quote_data, options_data, watchlist_data_map=None):
    """Calculate momentum and opportunity score for a symbol"""
    try:
        price = quote_data.get('lastPrice', 0)
        volume = quote_data.get('totalVolume', 0)
        prev_close = quote_data.get('closePrice', price)
        
        # If quote price is zero (market closed), get from options chain
        if price == 0 and options_data and 'underlyingPrice' in options_data:
            price = options_data.get('underlyingPrice', 0)
            # Use price as prev_close if we don't have close data
            if prev_close == 0:
                prev_close = price
        
        # Safety checks
        if price == 0:
            return None
        
        # Use Droplet API's daily_change_pct if available (more accurate)
        if watchlist_data_map and symbol in watchlist_data_map and 'daily_change_pct' in watchlist_data_map[symbol]:
            change_pct = watchlist_data_map[symbol]['daily_change_pct']
        else:
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0
        
        # Analyze options activity
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        
        if 'callExpDateMap' in options_data:
            for exp_date, strikes in options_data['callExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        total_call_volume += contracts[0].get('totalVolume', 0)
                        total_call_oi += contracts[0].get('openInterest', 0)
        
        if 'putExpDateMap' in options_data:
            for exp_date, strikes in options_data['putExpDateMap'].items():
                for strike_str, contracts in strikes.items():
                    if contracts:
                        total_put_volume += contracts[0].get('totalVolume', 0)
                        total_put_oi += contracts[0].get('openInterest', 0)
        
        # Use volume if available, otherwise fall back to OI (for after hours)
        call_metric = total_call_volume if total_call_volume > 0 else total_call_oi
        put_metric = total_put_volume if total_put_volume > 0 else total_put_oi
        
        # Calculate put/call ratio (handle zero volumes)
        if call_metric == 0 and put_metric == 0:
            # No options activity at all - skip this symbol
            return None
        pcr = put_metric / call_metric if call_metric > 0 else 999
        
        # Options volume vs stock volume ratio (use actual volume if available)
        options_dollar_volume = (total_call_volume + total_put_volume) * price * 100
        stock_dollar_volume = volume * price
        vol_ratio = options_dollar_volume / stock_dollar_volume if stock_dollar_volume > 0 else 0
        
        # Get walls
        call_wall, put_wall, max_gex, gex_value = analyze_volume_walls(options_data, price)
        
        # Calculate opportunity score (0-100)
        score = 0
        reasons = []
        
        # Factor 1: Price momentum (0-25 points)
        if abs(change_pct) > 2:
            momentum_score = min(abs(change_pct) * 5, 25)
            score += momentum_score
            reasons.append(f"Strong momentum: {change_pct:+.1f}%")
        
        # Factor 2: Options activity (0-25 points)
        if vol_ratio > 1:
            activity_score = min(vol_ratio * 10, 25)
            score += activity_score
            reasons.append(f"High options activity: {vol_ratio:.1f}x stock volume")
        
        # Factor 3: Put/Call imbalance (0-25 points)
        if pcr < 0.7:  # Bullish
            score += 20
            reasons.append(f"Bullish flow: PCR {pcr:.2f}")
        elif pcr > 1.3:  # Bearish
            score += 20
            reasons.append(f"Bearish flow: PCR {pcr:.2f}")
        
        # Factor 4: Near walls/flip level (0-25 points)
        if call_wall and price > 0 and abs(price - call_wall) / price < 0.03:
            score += 15
            reasons.append(f"Near call wall at ${call_wall:.2f}")
        if put_wall and price > 0 and abs(price - put_wall) / price < 0.03:
            score += 15
            reasons.append(f"Near put wall at ${put_wall:.2f}")
        
        # Determine direction based on price momentum AND options flow
        if abs(change_pct) > 1:  # Strong price movement
            # Price direction takes priority
            direction = "BULLISH" if change_pct > 0 else "BEARISH"
            # Add flow confirmation or divergence
            if (change_pct > 0 and pcr > 1.3):
                direction = "BULLISH (⚠️ hedging)"  # Up move but heavy puts
            elif (change_pct < 0 and pcr < 0.7):
                direction = "BEARISH (⚠️ bottom fishing)"  # Down move but heavy calls
        else:
            # No strong price movement, use options flow
            direction = "BULLISH" if pcr < 1 else "BEARISH"
        
        return {
            'symbol': symbol,
            'price': price,
            'change_pct': change_pct,
            'score': min(score, 100),
            'direction': direction,
            'pcr': pcr,
            'call_wall': call_wall,
            'put_wall': put_wall,
            'max_gex': max_gex,
            'gex_value': gex_value,
            'reasons': reasons,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'using_oi': total_call_volume == 0 and total_put_volume == 0
        }
        
    except Exception as e:
        # Silently skip symbols with errors during closed market
        return None


async def test_opening_move():
    """Test the opening move analysis"""
    print("\n" + "="*80)
    print("OPENING MOVE ALERT - STANDALONE TEST")
    print("="*80 + "\n")
    
    # Load watchlist
    watchlist, watchlist_data_map = load_watchlist()
    print(f"✅ Loaded watchlist: {len(watchlist)} symbols")
    
    # Smart selection for testing (same logic as bot)
    scan_count = 15  # Just 15 for testing
    if len(watchlist) > scan_count:
        # Test with top 10 gainers + 5 from middle
        symbols_to_test = watchlist[:10] + watchlist[len(watchlist)//2:len(watchlist)//2 + 5]
    else:
        symbols_to_test = watchlist[:scan_count]
    
    print(f"   Testing {len(symbols_to_test)} candidates from full list")
    print(f"   Top candidates: {', '.join(symbols_to_test[:10])}\n")
    
    # Initialize client
    print("Initializing Schwab client...")
    client = SchwabClient(interactive=False)
    
    # Scan opportunities
    print("Scanning for top opportunities...")
    print(f"(Smart scan: analyzing {len(symbols_to_test)} pre-selected candidates)\n")
    
    opportunities = []
    
    for i, symbol in enumerate(symbols_to_test, 1):
        try:
            print(f"  [{i}/{len(symbols_to_test)}] Analyzing {symbol}...", end=" ")
            
            # Get price from cached Droplet API data first
            price = 0
            if symbol in watchlist_data_map:
                cached_data = watchlist_data_map[symbol]
                price = cached_data.get('price', 0)
                # Create fake quote data structure from cached data
                quote_data = {
                    symbol: {
                        'lastPrice': price,
                        'mark': price,
                        'totalVolume': cached_data.get('volume', 0),
                        'change': cached_data.get('daily_change', 0),
                        'changePercent': cached_data.get('daily_change_pct', 0)
                    }
                }
            else:
                # Fallback: get fresh quote from Schwab
                quote_data = client.get_quote(symbol)
                if not quote_data or symbol not in quote_data:
                    print("❌ No quote data")
                    continue
                price = quote_data[symbol].get('lastPrice', 0)
            
            if not price:
                print("❌ No price data")
                continue
            
            # Get options data (still from Schwab - not in Droplet API)
            options_data = client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                strike_count=20
            )
            
            if not options_data:
                print("❌ No options data")
                continue
            
            # Quick check of what we got
            has_calls = 'callExpDateMap' in options_data and len(options_data['callExpDateMap']) > 0
            has_puts = 'putExpDateMap' in options_data and len(options_data['putExpDateMap']) > 0
            
            if not has_calls and not has_puts:
                print(f"❌ Empty options chain")
                continue
            
            # Calculate opportunity score
            analysis = calculate_opportunity_score(symbol, quote_data[symbol], options_data, watchlist_data_map)
            
            if analysis:
                if analysis['score'] > 30:
                    opportunities.append(analysis)
                    print(f"✅ Score: {analysis['score']:.0f}")
                else:
                    print(f"⚪ Score: {analysis['score']:.0f} (below threshold)")
            else:
                print(f"⚪ No valid data (likely market closed)")
            
            # Rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            continue
    
    # Sort and show top 5
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    top_5 = opportunities[:5]
    
    print(f"\n\n{'='*80}")
    print(f"TOP 5 TRADE OPPORTUNITIES (from {len(opportunities)} qualified)")
    print(f"{'='*80}\n")
    
    if not top_5:
        print("❌ No high-probability opportunities found")
        print("   (This is normal if market is closed or no strong setups exist)\n")
        return
    
    for i, opp in enumerate(top_5, 1):
        emoji = "🟢" if opp['direction'] == "BULLISH" else "🔴"
        print(f"\n{emoji} OPPORTUNITY #{i}: {opp['symbol']} - {opp['direction']}")
        print("─" * 60)
        print(f"Price:              ${opp['price']:.2f} ({opp['change_pct']:+.1f}%)")
        print(f"Opportunity Score:  {opp['score']:.0f}/100")
        print(f"Put/Call Ratio:     {opp['pcr']:.2f}")
        
        if opp.get('using_oi'):
            print(f"Call OI:            {opp['total_call_oi']:,} (using OI, volume=0)")
            print(f"Put OI:             {opp['total_put_oi']:,} (using OI, volume=0)")
        else:
            print(f"Call Volume:        {opp['total_call_volume']:,}")
            print(f"Put Volume:         {opp['total_put_volume']:,}")
        
        if opp['call_wall']:
            distance = ((opp['call_wall'] - opp['price']) / opp['price']) * 100
            print(f"📈 Call Wall:       ${opp['call_wall']:.2f} ({distance:+.1f}% away)")
        if opp['put_wall']:
            distance = ((opp['put_wall'] - opp['price']) / opp['price']) * 100
            print(f"📉 Put Wall:        ${opp['put_wall']:.2f} ({distance:+.1f}% away)")
        if opp['max_gex']:
            distance = ((opp['max_gex'] - opp['price']) / opp['price']) * 100
            print(f"⚡ Gamma Flip:      ${opp['max_gex']:.2f} ({distance:+.1f}% away)")
        
        print(f"\nWhy this is a TOP {i} play:")
        for reason in opp['reasons']:
            print(f"  • {reason}")
    
    print("\n\n" + "="*80)
    print("DEPLOYMENT INSTRUCTIONS")
    print("="*80)
    print("\n✅ Opening Move logic is working!\n")
    print("To deploy to Discord:")
    print("  1. cd discord-bot")
    print("  2. python run_bot.py")
    print("  3. In Discord, use: /setup_opening_move")
    print("  4. Test manually: /opening_move_now")
    print("  5. Start auto-scan: /start_opening_move")
    print("\nThe bot will send top 5 alerts every 15 minutes during market hours.\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_opening_move())
