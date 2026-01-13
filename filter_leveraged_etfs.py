"""
Filter leveraged ETFs to keep only 1 long and 1 short per underlying stock
"""
import csv
from collections import defaultdict
import re

def extract_underlying_symbol(description):
    """Extract the underlying stock symbol from description"""
    # Look for patterns like "2X Long NVDA", "Bull 2X AAPL", etc.
    patterns = [
        r'Long (\w+) Daily',
        r'Short (\w+) Daily',
        r'(\w+) Bull',
        r'(\w+) Bear',
        r'2X Long (\w+)',
        r'3X Long (\w+)',
        r'2X Short (\w+)',
        r'3X Short (\w+)',
        r'Ultra (\w+)',
        r'UltraShort (\w+)',
        r'UltraPro (\w+)',
        r'2x Long (\w+)',
        r'2x Short (\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            return match.group(1).upper()
    
    return None

def is_inverse(description, fund_type):
    """Check if ETF is inverse/short"""
    inverse_keywords = ['Short', 'Bear', 'Inverse', 'UltraShort']
    return any(keyword in description or keyword in fund_type for keyword in inverse_keywords)

def get_priority_score(symbol, description):
    """Assign priority score - prefer Direxion, ProShares, then others"""
    if 'Direxion' in description:
        return 1
    elif 'ProShares' in description:
        return 2
    elif 'GraniteShares' in description or 'Leverage Shares' in description:
        return 3
    elif 'T-Rex' in description or 'T-REX' in description:
        return 4
    elif 'Defiance' in description:
        return 5
    else:
        return 6

# Read the CSV
etfs_by_underlying = defaultdict(lambda: {'long': [], 'short': []})
all_etfs = []

with open('/Users/piyushkhaitan/schwab/options/Results.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Remove quotes from keys
        row = {k.strip('"'): v.strip('"') if isinstance(v, str) else v for k, v in row.items()}
        all_etfs.append(row)
        underlying = extract_underlying_symbol(row['Description'])
        
        if underlying:
            direction = 'short' if is_inverse(row['Description'], row['Fund Type']) else 'long'
            priority = get_priority_score(row['Symbol'], row['Description'])
            etfs_by_underlying[underlying][direction].append((priority, row))

# Select best long and short for each underlying
filtered_etfs = []
processed_symbols = set()

for underlying, directions in sorted(etfs_by_underlying.items()):
    # Pick best long
    if directions['long']:
        directions['long'].sort(key=lambda x: x[0])  # Sort by priority
        best_long = directions['long'][0][1]
        filtered_etfs.append(best_long)
        processed_symbols.add(best_long['Symbol'])
        print(f"{underlying} Long: {best_long['Symbol']} - {best_long['Description']}")
    
    # Pick best short
    if directions['short']:
        directions['short'].sort(key=lambda x: x[0])  # Sort by priority
        best_short = directions['short'][0][1]
        filtered_etfs.append(best_short)
        processed_symbols.add(best_short['Symbol'])
        print(f"{underlying} Short: {best_short['Symbol']} - {best_short['Description']}")

# Add ETFs that couldn't be categorized (indices, commodities, etc.)
for etf in all_etfs:
    if etf['Symbol'] not in processed_symbols:
        underlying = extract_underlying_symbol(etf['Description'])
        if not underlying:
            filtered_etfs.append(etf)
            print(f"Other: {etf['Symbol']} - {etf['Description']}")

# Write filtered results
with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Symbol', 'Description', 'Fund Type', 'Leveraged ETP'])
    writer.writeheader()
    writer.writerows(filtered_etfs)

print(f"\nOriginal count: {len(all_etfs)}")
print(f"Filtered count: {len(filtered_etfs)}")
print(f"Removed: {len(all_etfs) - len(filtered_etfs)}")
