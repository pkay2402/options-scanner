#!/usr/bin/env python3
"""
Test the Opening Move alert logic without Discord
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'discord-bot'))

async def test_opening_move():
    """Test the opening move analysis"""
    print("\n" + "="*80)
    print("TESTING OPENING MOVE ALERT")
    print("="*80 + "\n")
    
    try:
        # Import after paths are set
        from bot.commands.opening_move import OpeningMoveCommands
        from src.api.schwab_client import SchwabClient
        import json
        
        # Mock bot object
        class MockBot:
            pass
        
        bot = MockBot()
        
        # Initialize the command handler
        handler = OpeningMoveCommands(bot)
        
        print(f"Loaded watchlist: {len(handler.watchlist)} symbols")
        print(f"First 10: {', '.join(handler.watchlist[:10])}\n")
        
        # Check market hours
        is_open = handler.is_market_hours()
        print(f"Market currently open: {is_open}\n")
        
        # Test scanning top opportunities
        print("Scanning for top opportunities...")
        print("(This will take a minute, analyzing watchlist stocks...)\n")
        
        opportunities = await handler._scan_top_opportunities()
        
        print(f"\n✅ Found {len(opportunities)} high-probability opportunities\n")
        
        # Display results
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{'='*60}")
            print(f"OPPORTUNITY #{i}: {opp['symbol']} - {opp['direction']}")
            print(f"{'='*60}")
            print(f"Price:           ${opp['price']:.2f} ({opp['change_pct']:+.1f}%)")
            print(f"Opportunity Score: {opp['score']:.0f}/100")
            print(f"Put/Call Ratio:   {opp['pcr']:.2f}")
            
            if opp['call_wall']:
                print(f"Call Wall:        ${opp['call_wall']:.2f}")
            if opp['put_wall']:
                print(f"Put Wall:         ${opp['put_wall']:.2f}")
            if opp['max_gex']:
                print(f"Gamma Flip:       ${opp['max_gex']:.2f}")
            
            print(f"\nWhy this is a top play:")
            for reason in opp['reasons']:
                print(f"  • {reason}")
        
        # Test embed creation
        print("\n\n" + "="*80)
        print("TESTING DISCORD EMBED CREATION")
        print("="*80 + "\n")
        
        import pytz
        scan_time = datetime.now(pytz.UTC)
        embed = handler._create_opportunity_embed(opportunities, scan_time)
        
        print(f"Embed Title: {embed.title}")
        print(f"Embed Description: {embed.description}")
        print(f"Number of fields: {len(embed.fields)}")
        print(f"Embed color: {embed.color}")
        
        for field in embed.fields:
            print(f"\n--- Field: {field.name} ---")
            print(field.value[:200] + "..." if len(field.value) > 200 else field.value)
        
        print("\n\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Watchlist loaded: {len(handler.watchlist)} symbols")
        print(f"✅ Opportunities found: {len(opportunities)}")
        print(f"✅ Embed created successfully")
        print(f"✅ Ready for Discord deployment")
        print("\nNext steps:")
        print("  1. Run discord bot: cd discord-bot && python run_bot.py")
        print("  2. Use /setup_opening_move in your Discord channel")
        print("  3. Test with /opening_move_now")
        print("  4. Start auto-scanning with /start_opening_move")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_opening_move())
