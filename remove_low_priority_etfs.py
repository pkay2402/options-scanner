"""
Remove low priority ETFs from Results_Filtered.csv
Categories to remove:
1. Niche/Low Volume Commodities
2. Income/Dividend ETFs
3. Complex/Exotic Strategy ETFs
4. Low Volume ETNs
"""
import csv

# Define symbols to remove
SYMBOLS_TO_REMOVE = set([
    # Niche/Low Volume Commodities
    'CORX', 'CXRN',  # Corn
    'WHTX', 'WXET',  # Wheat
    'TXXS',          # Sui crypto
    'CPXR',          # Copper
    
    # WeeklyPay ETFs (Income focused)
    'AAPW', 'AMDW', 'AMZW', 'ARMW', 'BABW', 'BRKW', 'COIW', 'COSW', 
    'GOOW', 'HOOW', 'METW', 'MSFW', 'MSTW', 'NFLW', 'PLTW', 'TSLW', 
    'UBEW', 'UNHW', 'TSYW', 'WPAY',
    
    # YieldBOOST ETFs (Income focused)
    'AMYY', 'AZYY', 'BBYY', 'COYY', 'FBYY', 'HMYY', 'HOYY', 'IOYY', 
    'MAAY', 'MTYY', 'NUGY', 'NVYY', 'PLYY', 'RGYY', 'RTYY', 'SMYY', 
    'TSYY', 'XBTY', 'SEMY',
    
    # Growth & Income ETFs
    'COII', 'CWII', 'HOII', 'LLII', 'MSII', 'NVII', 'PLTI', 'TSII', 'WMTI',
    
    # Complex/Exotic Strategy ETFs
    'FDRX',  # Founder-Led 2x
    'MVPL',  # Miller Value Partners
    'QXAS',  # Alt Season Crypto
    'WILD',  # Animal Spirits
    'TDAX', 'TSYX',  # Lift ETFs
    'QTAP', 'XTAP', 'XDQQ',  # Accelerated ETFs
    'YTSM',  # Bitcoin Treasury Harvester
    
    # ETRACS ETNs (Low volume)
    'BDCX', 'CEFD', 'HDLB', 'IWDL', 'IWFL', 'IWML', 'MLPR', 'MTUL', 
    'MVRL', 'PFFL', 'QULL', 'SCDL', 'SMHB', 'USML',
    
    # DB ETNs (Low volume)
    'AGATF', 'BDDXF', 'BOMMF', 'DAGXF', 'DEE', 'DGP', 'DYYXF', 'DZZ',
    
    # MicroSectors ETNs (Low volume)
    'JETD', 'JETU', 'FLYD', 'FLYU', 'GDXD', 'DULL', 'SHNY',
])

# Read the filtered CSV
filtered_etfs = []
removed_count = 0

with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Symbol'] not in SYMBOLS_TO_REMOVE:
            filtered_etfs.append(row)
        else:
            removed_count += 1
            print(f"Removing: {row['Symbol']} - {row['Description']}")

# Write updated file
with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'w', newline='', encoding='utf-8') as f:
    if filtered_etfs:
        fieldnames = filtered_etfs[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_etfs)

print(f"\n✅ Removed {removed_count} low-priority ETFs")
print(f"📊 Remaining ETFs: {len(filtered_etfs)}")

# Also update extracted_symbols.csv
print("\nUpdating extracted_symbols.csv...")
symbols_to_keep = {etf['Symbol'] for etf in filtered_etfs}

# Read current extracted_symbols.csv
extracted_rows = []
with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Symbol'] in symbols_to_keep:
            extracted_rows.append(row)

# Write updated extracted_symbols.csv
with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Symbol', 'Description'])
    writer.writeheader()
    writer.writerows(extracted_rows)

print(f"✅ Updated extracted_symbols.csv: {len(extracted_rows)} symbols")
