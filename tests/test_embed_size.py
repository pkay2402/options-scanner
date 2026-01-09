#!/usr/bin/env python3
"""
Test Discord embed character count
"""

# Simulate the embed content
def calculate_embed_size(opportunities):
    """Calculate total character count of Discord embed"""
    
    # Title + Description
    title = "📊 Opening Move Report"
    description = f"Top 5 Trade Opportunities • 10:15 AM ET"
    
    title_desc_chars = len(title) + len(description)
    
    # Calculate each opportunity field
    total_field_chars = 0
    
    for i, opp in enumerate(opportunities, 1):
        # Field title
        emoji = "🟢" if "BULLISH" in opp['direction'] else "🔴"
        field_title = f"{emoji} #{i}: {opp['symbol']} - {opp['direction']}"
        
        # Field content
        lines = [
            f"**Price:** ${opp['price']:.2f} ({opp['change_pct']:+.1f}%)",
            f"**Score:** {opp['score']:.0f}/100",
            f"**Put/Call Ratio:** {opp['pcr']:.2f}",
            ""
        ]
        
        if opp.get('call_wall'):
            lines.append(f"📈 **Call Wall:** ${opp['call_wall']:.2f}")
        if opp.get('put_wall'):
            lines.append(f"📉 **Put Wall:** ${opp['put_wall']:.2f}")
        if opp.get('max_gex'):
            lines.append(f"⚡ **Flip Level:** ${opp['max_gex']:.2f}")
        
        lines.append("")
        
        if opp.get('reasons'):
            lines.append("**Why This Setup:**")
            for reason in opp['reasons'][:3]:
                lines.append(f"• {reason}")
        
        field_content = "\n".join(lines)
        
        field_chars = len(field_title) + len(field_content)
        total_field_chars += field_chars
        
        print(f"Field #{i} ({opp['symbol']}):")
        print(f"  Title: {len(field_title)} chars")
        print(f"  Content: {len(field_content)} chars")
        print(f"  Total: {field_chars} chars\n")
    
    # Footer
    footer = "Next scan in 15 minutes"
    footer_chars = len(footer)
    
    # Total
    total = title_desc_chars + total_field_chars + footer_chars
    
    print("="*60)
    print(f"Title + Description: {title_desc_chars} chars")
    print(f"All Fields: {total_field_chars} chars")
    print(f"Footer: {footer_chars} chars")
    print(f"="*60)
    print(f"TOTAL EMBED: {total} chars")
    print(f"="*60)
    
    return total

# Sample data - 5 opportunities
sample_opportunities = [
    {
        'symbol': 'MU',
        'price': 343.43,
        'change_pct': 10.4,
        'score': 60,
        'direction': 'BULLISH',
        'pcr': 0.42,
        'call_wall': 350.00,
        'put_wall': 250.00,
        'max_gex': 330.00,
        'reasons': [
            'Strong momentum: +10.4%',
            'Bullish flow: PCR 0.42',
            'Near call wall at $350.00'
        ]
    },
    {
        'symbol': 'DOCN',
        'price': 54.01,
        'change_pct': 8.1,
        'score': 55,
        'direction': 'BULLISH',
        'pcr': 0.04,
        'call_wall': 55.00,
        'put_wall': 50.00,
        'max_gex': 55.00,
        'reasons': [
            'Strong momentum: +8.1%',
            'Bullish flow: PCR 0.04',
            'Near call wall at $55.00'
        ]
    },
    {
        'symbol': 'VEEV',
        'price': 237.87,
        'change_pct': 7.5,
        'score': 60,
        'direction': 'BULLISH (⚠️ hedging)',
        'pcr': 2.16,
        'call_wall': 240.00,
        'put_wall': 220.00,
        'max_gex': 240.00,
        'reasons': [
            'Strong momentum: +7.5%',
            'Near call wall at $240.00'
        ]
    },
    {
        'symbol': 'AAPL',
        'price': 185.50,
        'change_pct': 2.3,
        'score': 45,
        'direction': 'BULLISH',
        'pcr': 0.85,
        'call_wall': 190.00,
        'put_wall': 180.00,
        'max_gex': 185.00,
        'reasons': [
            'Strong momentum: +2.3%',
            'Bullish flow: PCR 0.85'
        ]
    },
    {
        'symbol': 'TSLA',
        'price': 425.75,
        'change_pct': -1.8,
        'score': 40,
        'direction': 'BEARISH',
        'pcr': 1.45,
        'call_wall': 430.00,
        'put_wall': 420.00,
        'max_gex': 425.00,
        'reasons': [
            'Bearish flow: PCR 1.45',
            'Near put wall at $420.00'
        ]
    }
]

print("\n" + "="*60)
print("DISCORD EMBED CHARACTER COUNT TEST")
print("="*60 + "\n")

total_chars = calculate_embed_size(sample_opportunities)

print("\n" + "="*60)
print("DISCORD LIMITS")
print("="*60)
print("Regular message: 2000 chars")
print("Embed title: 256 chars")
print("Embed description: 4096 chars")
print("Field name: 256 chars")
print("Field value: 1024 chars")
print("Total embed: 6000 chars")
print("="*60)

if total_chars > 6000:
    print(f"\n⚠️  OVER LIMIT by {total_chars - 6000} chars!")
elif total_chars > 4000:
    print(f"\n⚠️  Getting close to limit ({total_chars}/6000)")
else:
    print(f"\n✅ WELL WITHIN LIMITS ({total_chars}/6000)")

print("="*60 + "\n")
