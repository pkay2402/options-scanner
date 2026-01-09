#!/usr/bin/env python3
"""Test SPY Z-score crossings around Dec 24-25"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fetch SPY data
symbol = 'SPY'
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

print(f'Fetching {symbol} data from {start_date.date()} to {end_date.date()}...')
df = yf.download(symbol, start=start_date, end=end_date, progress=False)

# Flatten multi-index columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Calculate Z-score (20-day rolling)
df['ma20'] = df['Close'].rolling(window=20).mean()
df['std20'] = df['Close'].rolling(window=20).std()
df['zscore'] = (df['Close'] - df['ma20']) / df['std20']

# Calculate RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Focus on Dec 20 - Jan 2
df_dec = df.loc['2024-12-20':'2026-01-02'].copy()

print('\n=== SPY Data Dec 20 - Jan 2 ===')
print(f"{'Date':<12} {'Close':<10} {'Z-score':<10} {'RSI':<10}")
print('-' * 45)
for idx, row in df_dec.iterrows():
    close_val = row['Close'] if 'Close' in df_dec.columns else row['close']
    print(f"{str(idx.date()):<12} ${close_val:<9.2f} {row['zscore']:<10.3f} {row['rsi']:<10.1f}")

print('\n=== Checking for +2σ crossings ===')
df_dec['z_prev'] = df_dec['zscore'].shift(1)

# Check for +2σ crossings (going ABOVE +2)
crossed_p2_up = (df_dec['z_prev'] <= 2) & (df_dec['zscore'] > 2)
crossed_p2_down = (df_dec['z_prev'] >= 2) & (df_dec['zscore'] < 2)

if crossed_p2_up.any():
    print('\n✅ Dates where Z-score crossed ABOVE +2σ (SELL signal):')
    for idx in df_dec[crossed_p2_up].index:
        row = df_dec.loc[idx]
        print(f"  {idx.date()}: {row['z_prev']:.3f} -> {row['zscore']:.3f} (Close=${row['Close']:.2f}, RSI={row['rsi']:.1f})")
else:
    print('\n❌ No crossings ABOVE +2σ found')

if crossed_p2_down.any():
    print('\n⚠️  Dates where Z-score crossed BELOW +2σ (reverting from overbought):')
    for idx in df_dec[crossed_p2_down].index:
        row = df_dec.loc[idx]
        print(f"  {idx.date()}: {row['z_prev']:.3f} -> {row['zscore']:.3f} (Close=${row['Close']:.2f}, RSI={row['rsi']:.1f})")
else:
    print('\n❌ No crossings BELOW +2σ found')

# Check Dec 24 specifically
print('\n=== Around Dec 24-25 (Christmas) ===')
dec24_range = df.loc['2024-12-23':'2024-12-27']
print(f"{'Date':<12} {'Close':<10} {'Z-score':<10} {'RSI':<10}")
print('-' * 45)
for idx, row in dec24_range.iterrows():
    marker = ' ← HOLIDAY' if idx.date().day in [24, 25] else ''
    print(f"{str(idx.date()):<12} ${row['Close']:<9.2f} {row['zscore']:<10.3f} {row['rsi']:<10.1f}{marker}")
