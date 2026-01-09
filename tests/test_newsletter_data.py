#!/usr/bin/env python3
"""
Test script to see what output big_trades and market_dynamics generate
"""

import sys
import os
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_big_trades():
    """Test BigTradesDetector output"""
    print("\n" + "="*80)
    print("TESTING BIG TRADES DETECTOR")
    print("="*80 + "\n")
    
    try:
        from src.analysis.big_trades import BigTradesDetector
        
        # Initialize detector
        detector = BigTradesDetector()
        
        # Test with a few popular symbols
        test_symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
        
        print(f"Scanning for big trades in: {', '.join(test_symbols)}")
        print(f"Timestamp: {datetime.now()}\n")
        
        # Attempt to scan
        big_trades = detector.scan_for_big_trades(symbols=test_symbols, min_premium=50000)
        
        print(f"\nFound {len(big_trades)} big trades\n")
        
        if big_trades:
            # Show first 3 trades
            for i, trade in enumerate(big_trades[:3], 1):
                print(f"\nBig Trade #{i}:")
                print(f"  Symbol: {trade.symbol}")
                print(f"  Type: {trade.contract_type.upper()}")
                print(f"  Strike: ${trade.strike}")
                print(f"  Expiration: {trade.expiration}")
                print(f"  Volume: {trade.volume:,}")
                print(f"  Premium: ${trade.premium:,.2f}")
                print(f"  Notional Value: ${trade.notional_value:,.2f}")
                print(f"  Sentiment: {trade.sentiment}")
                print(f"  Size Score: {trade.size_score}/10")
                print(f"  Urgency Score: {trade.urgency_score}/10")
                print(f"  Confidence: {trade.confidence_level:.1%}")
                if trade.analysis_notes:
                    print(f"  Notes: {', '.join(trade.analysis_notes[:2])}")
        else:
            print("  No big trades detected (may need API credentials)")
            
        return big_trades
        
    except Exception as e:
        print(f"ERROR in BigTradesDetector: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def test_market_dynamics():
    """Test MarketDynamicsAnalyzer output"""
    print("\n\n" + "="*80)
    print("TESTING MARKET DYNAMICS ANALYZER")
    print("="*80 + "\n")
    
    try:
        from src.analysis.market_dynamics import MarketDynamicsAnalyzer
        
        # Initialize analyzer
        analyzer = MarketDynamicsAnalyzer()
        
        print("Running SHORT-TERM market dynamics analysis...")
        print(f"Timestamp: {datetime.now()}\n")
        
        # Test short-term analysis
        result = analyzer.analyze_short_term_dynamics(symbols=['SPY', 'QQQ', 'IWM'])
        
        print("\n--- ANALYSIS RESULTS ---\n")
        print(f"Analysis Type: {result.analysis_type}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Confidence Score: {result.confidence_score:.1%}\n")
        
        print("MARKET SENTIMENT:")
        print(f"  Put/Call Ratio: {result.sentiment.put_call_ratio:.2f}")
        print(f"  VIX Level: {result.sentiment.vix_level:.2f}")
        print(f"  Gamma Exposure: ${result.sentiment.gamma_exposure:,.0f}")
        print(f"  Dealer Positioning: {result.sentiment.dealer_positioning}")
        print(f"  Sentiment Score: {result.sentiment.sentiment_score:.2f}/10")
        print(f"  Confidence: {result.sentiment.confidence_level:.1%}\n")
        
        print("KEY LEVELS:")
        for symbol, level in result.key_levels.items():
            print(f"  {symbol}: ${level:,.2f}")
        
        print(f"\nUNUSUAL ACTIVITY: {len(result.unusual_activity)} detected")
        if result.unusual_activity:
            for i, activity in enumerate(result.unusual_activity[:3], 1):
                print(f"\n  Activity #{i}:")
                print(f"    Symbol: {activity.symbol}")
                print(f"    Type: {activity.contract_type}")
                print(f"    Volume: {activity.volume:,}")
                print(f"    IV: {activity.implied_volatility:.1%}")
        
        print(f"\nRECOMMENDATIONS: ({len(result.recommendations)})")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\nRISK FACTORS: ({len(result.risk_factors)})")
        for i, risk in enumerate(result.risk_factors, 1):
            print(f"  {i}. {risk}")
            
        return result
        
    except Exception as e:
        print(f"ERROR in MarketDynamicsAnalyzer: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_data_structure():
    """Test what data structures are available"""
    print("\n\n" + "="*80)
    print("AVAILABLE DATA STRUCTURES")
    print("="*80 + "\n")
    
    try:
        from src.analysis.big_trades import BigTrade, UnusualActivity
        from src.analysis.market_dynamics import MarketSentiment, OptionsFlow, MarketAnalysisResult
        
        print("BigTrade fields:")
        for field_name in BigTrade.__dataclass_fields__:
            print(f"  - {field_name}")
        
        print("\nMarketAnalysisResult fields:")
        for field_name in MarketAnalysisResult.__dataclass_fields__:
            print(f"  - {field_name}")
            
        print("\nMarketSentiment fields:")
        for field_name in MarketSentiment.__dataclass_fields__:
            print(f"  - {field_name}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# NEWSLETTER DATA GENERATION TEST")
    print("#"*80)
    
    # Test data structures
    test_data_structure()
    
    # Test big trades
    big_trades = test_big_trades()
    
    # Test market dynamics
    market_analysis = test_market_dynamics()
    
    print("\n\n" + "="*80)
    print("SUMMARY FOR NEWSLETTER GENERATION")
    print("="*80 + "\n")
    
    print("For 'Opening Move Report' newsletter, we can use:\n")
    print("1. MARKET PULSE:")
    if market_analysis:
        print(f"   ✓ Sentiment Score: {market_analysis.sentiment.sentiment_score:.1f}/10")
        print(f"   ✓ Put/Call Ratio: {market_analysis.sentiment.put_call_ratio:.2f}")
        print(f"   ✓ Dealer Positioning: {market_analysis.sentiment.dealer_positioning}")
    else:
        print("   ⚠ Need API credentials to fetch live data")
    
    print("\n2. BIG MONEY MOVES:")
    if big_trades:
        print(f"   ✓ {len(big_trades)} significant trades detected")
        print(f"   ✓ Notable: {big_trades[0].symbol} {big_trades[0].contract_type}")
    else:
        print("   ⚠ Need API credentials to fetch live big trades")
    
    print("\n3. TODAY'S SETUPS:")
    if market_analysis and market_analysis.recommendations:
        print(f"   ✓ {len(market_analysis.recommendations)} actionable recommendations")
    else:
        print("   ⚠ Generated from market analysis")
    
    print("\n4. KEY LEVELS:")
    if market_analysis and market_analysis.key_levels:
        print(f"   ✓ Support/Resistance for {len(market_analysis.key_levels)} symbols")
    else:
        print("   ⚠ Need options data")
    
    print("\n5. RISK ALERTS:")
    if market_analysis and market_analysis.risk_factors:
        print(f"   ✓ {len(market_analysis.risk_factors)} risk factors identified")
    else:
        print("   ⚠ Generated from market conditions")
    
    print("\n" + "="*80)
    print("Next step: Build newsletter formatter that combines this data")
    print("="*80 + "\n")
