"""
Aggressive filtering to reduce to ~120 ETFs
Keep only the most liquid and relevant ETFs for momentum trading
"""
import csv

# Define HIGH PRIORITY symbols to KEEP (around 120)
SYMBOLS_TO_KEEP = set([
    # Major Indices (15)
    'TQQQ', 'SQQQ', 'QLD', 'QID',  # QQQ
    'SPXL', 'SPXS', 'SSO', 'SDS', 'UPRO', 'SPXU',  # SPY/S&P500
    'URTY', 'TZA', 'TNA', 'TWM',  # Russell 2000
    'QQQU',  # Mag 7
    
    # Major Sectors (20)
    'SOXL', 'SOXS', 'USD', 'SSG',  # Semiconductors
    'TECL', 'TECS',  # Technology
    'FAS', 'FAZ',  # Financials
    'ERX', 'ERY',  # Energy
    'LABU', 'LABD',  # Biotech
    'CURE',  # Healthcare
    'WEBL', 'WEBS',  # Internet
    'NAIL',  # Homebuilders
    'RETL',  # Retail
    'DFEN',  # Aerospace/Defense
    'DPST',  # Regional Banks
    'AIBU', 'AIBD',  # AI/Big Data
    
    # Mega Cap Tech - Most Liquid (20)
    'NVDU', 'NVD',  # NVIDIA
    'TSLL', 'TSDD',  # TESLA
    'AAPU',  # APPLE
    'MSFU',  # MICROSOFT
    'AMZU',  # AMAZON
    'AMUU', 'DAMD',  # AMD
    'METU',  # META
    'GGLL',  # GOOGL
    'AVL',  # AVGO
    'TSMX',  # TSM
    'ORCX',  # ORACLE
    'QCMU',  # QUALCOMM
    'ARMG',  # ARM
    'CRWL',  # CROWDSTRIKE
    'NOWL',  # SERVICENOW
    'PANW',  # PALO ALTO (keeping PALU)
    'ADBG',  # ADOBE
    
    # Hot Momentum Stocks (25)
    'PLTU', 'PLTZ',  # PALANTIR
    'CONX', 'CONI',  # COINBASE
    'MSTU', 'MSTZ',  # MICROSTRATEGY
    'HODU', 'HOOZ',  # ROBINHOOD
    'SMCL', 'SMCZ',  # SUPERMICRO
    'IONL', 'IONZ',  # IONQ
    'RKLX', 'RKLZ',  # ROCKET LAB
    'RGTX', 'RGTZ',  # RIGETTI
    'QBTX', 'QBTZ',  # D-WAVE QUANTUM
    'QUBX',  # QUANTUM COMPUTING
    'RDTL',  # REDDIT
    'RBLU',  # ROBLOX
    'RVNL',  # RIVIAN
    'UBER',  # UBER (keeping UBRL)
    'SNOW',  # SNOWFLAKE (keeping SNOU)
    
    # Crypto Proxies (10)
    'BTCL', 'BTCZ',  # Bitcoin
    'ETU', 'ETHD',  # Ether
    'MSTU', 'MSTZ',  # MSTR (duplicate but important)
    'CONX', 'CONI',  # COIN (duplicate but important)
    'BITX',  # Bitcoin Strategy
    'SLON',  # Solana
    'UXRP',  # XRP
    
    # Select Other High Volume Individual Stocks (15)
    'NFXL',  # NETFLIX
    'ELIL', 'LLYZ',  # ELI LILLY
    'SOFX',  # SOFI
    'CVNX',  # CARVANA
    'DLLL',  # DELL
    'GMEU',  # GAMESTOP
    'DKNG',  # DRAFTKINGS (keeping DKUP)
    'ABNB',  # AIRBNB (keeping ABNG)
    'LCDL',  # LUCID
    'RIVN',  # RIVIAN (keeping RVNL)
    'HOOD',  # HOOD (already in hot momentum)
    
    # Treasuries (4)
    'TMF', 'TMV',  # 20+ Year
    'TBT', 'UBT',  # 20+ Year alternatives
    
    # Volatility (2)
    'UVXY', 'SVXY',
    
    # Keep some sector diversity (10)
    'WANT',  # Consumer Discretionary
    'DUSL',  # Industrials
    'UTSL',  # Utilities
    'DRN', 'DRV',  # Real Estate
    'JNUG', 'DUST',  # Gold Miners
    'UGL', 'GLL',  # Gold
    'LMBO',  # Crypto Industry
])

# Add some that I mentioned by name
SYMBOLS_TO_KEEP.update(['PALU', 'UBRL', 'SNOU', 'DKUP', 'ABNG'])

print(f"Target symbols to keep: {len(SYMBOLS_TO_KEEP)}")

# Read the filtered CSV
kept_etfs = []
removed_etfs = []

with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Symbol'] in SYMBOLS_TO_KEEP:
            kept_etfs.append(row)
        else:
            removed_etfs.append(row)

# Sort by symbol for better organization
kept_etfs.sort(key=lambda x: x['Symbol'])

# Write updated file
with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'w', newline='', encoding='utf-8') as f:
    if kept_etfs:
        fieldnames = kept_etfs[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_etfs)

print(f"\n✅ Kept: {len(kept_etfs)} ETFs")
print(f"❌ Removed: {len(removed_etfs)} ETFs")

# Show what was removed (first 50)
print("\n📊 Sample of removed symbols (first 50):")
for i, etf in enumerate(removed_etfs[:50]):
    print(f"  {etf['Symbol']}")

# Update extracted_symbols.csv
print("\nUpdating extracted_symbols.csv...")
symbols_to_keep_extracted = {etf['Symbol'] for etf in kept_etfs}

extracted_rows = []
with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Symbol'] in symbols_to_keep_extracted:
            extracted_rows.append(row)

with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Symbol', 'Description'])
    writer.writeheader()
    writer.writerows(extracted_rows)

print(f"✅ Updated extracted_symbols.csv: {len(extracted_rows)} symbols")

# Show breakdown by category
print("\n📈 Final breakdown:")
print(f"Total: {len(kept_etfs)}")
