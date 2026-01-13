"""
Update extracted_symbols.csv to match filtered Results
"""
import csv

# Read filtered symbols
filtered_symbols = set()
with open('/Users/piyushkhaitan/schwab/options/Results_Filtered.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filtered_symbols.add(row['Symbol'])

print(f"Filtered ETFs: {len(filtered_symbols)}")

# Update extracted_symbols.csv
updated_rows = []
with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Symbol'] in filtered_symbols:
            updated_rows.append(row)

print(f"Matching symbols in extracted_symbols.csv: {len(updated_rows)}")

# Write updated file
with open('/Users/piyushkhaitan/schwab/options/extracted_symbols.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Symbol', 'Description'])
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"Updated extracted_symbols.csv with {len(updated_rows)} symbols")
