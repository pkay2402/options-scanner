# Scanner Memory Optimization Analysis

## Current Memory Usage
- **Discord Bot**: 199 MB (9.8%) - Highest
- **Market Sentiment Worker**: 156 MB (7.7%)
- **MACD Scanner**: 148 MB (7.3%)
- **TTM Squeeze Scanner**: 139 MB (6.9%)
- **VPB Scanner**: 133 MB (6.6%) - Had OOM kill on Jan 3

**Total Scanner Memory**: ~775 MB of 1.9 GB (41%)

## Identified Issues

### 1. **No Memory Cleanup**
- None of the scanners use `gc.collect()` or `del` to free memory
- DataFrames and price history accumulate in memory
- No explicit cleanup after each stock scan

### 2. **Large Price History Fetches**
- **MACD**: 3 months of daily data per symbol
- **TTM Squeeze**: 3 months of daily data  
- **VPB**: 2 months of daily data
- All 150 symbols × 60-90 days = ~9,000-13,500 candles in memory

### 3. **Long-Running Processes**
- MACD scanner: Running for 2 weeks
- TTM scanner: Running for 2 weeks
- No periodic restarts to clear accumulated memory

### 4. **Scan Intervals**
- All scanners: 1 hour (3600 seconds)
- Could be increased during off-peak hours

## Recommended Optimizations

### Quick Wins (No Functionality Loss)

#### 1. Add Memory Cleanup After Each Scan
```python
import gc

def scan_watchlist(watchlist):
    results = []
    for symbol in watchlist:
        result = scan_stock(client, symbol)
        if result:
            results.append(result)
        # Clear DataFrame memory after each symbol
        del result
    
    gc.collect()  # Force garbage collection
    return results
```

#### 2. Process Symbols in Batches
```python
BATCH_SIZE = 30  # Process 30 stocks at a time
for i in range(0, len(WATCHLIST), BATCH_SIZE):
    batch = WATCHLIST[i:i+BATCH_SIZE]
    process_batch(batch)
    gc.collect()  # Clean up after each batch
```

#### 3. Use Less Price History
- MACD only needs 26 days for calculation
- Current: 3 months (90 days) - **3x more than needed**
- Reduce to 1 month (30 days)

```python
price_history = client.get_price_history(
    symbol=symbol,
    period_type='month',
    period=1,  # Changed from 3 to 1
    frequency_type='daily',
    frequency=1
)
```

#### 4. Add Periodic Service Restarts
```systemd
[Service]
RuntimeMaxSec=21600  # Restart every 6 hours
```

#### 5. Optimize DataFrame Operations
```python
# Instead of keeping full DataFrame:
df = pd.DataFrame(price_history['candles'])

# Extract only needed columns:
df = pd.DataFrame(price_history['candles'])[['close', 'datetime', 'volume']]

# Or use dict comprehension (no pandas overhead):
closes = [c['close'] for c in price_history['candles']]
```

### Medium Optimizations

#### 6. Implement Result Caching
- Cache recent scan results for 15 minutes
- Avoid re-scanning same symbols
- Especially useful for API server requests

#### 7. Reduce Watchlist During Off-Hours
- Full 150 symbols: Market hours (9:30 AM - 4 PM)
- Reduced 50 symbols: After hours
- Saves ~66% memory

### Conservative Estimates

**Per Scanner Savings:**
- Memory cleanup: -15 MB (10%)
- Reduced history: -30 MB (20%)  
- Batch processing: -10 MB (7%)
- **Total: ~55 MB saved per scanner**

**Overall Savings: 165-220 MB (from 775 MB → 555-610 MB)**

## Implementation Priority

### Priority 1 (Immediate - No Risk)
1. Add `gc.collect()` after scans
2. Reduce price history from 3 months to 1 month
3. Add batch processing with cleanup

### Priority 2 (Low Risk)
4. Add RuntimeMaxSec to restart services every 6 hours
5. Optimize DataFrame column selection

### Priority 3 (Requires Testing)
6. Implement result caching
7. Dynamic watchlist sizing

## Implementation Plan

Create a patch file that:
1. Adds memory cleanup to all 3 scanners
2. Reduces price history periods
3. Implements batch processing
4. Updates systemd service files with RuntimeMaxSec

**Expected Outcome:**
- Reduce scanner memory from 775 MB to ~550-600 MB
- Prevent OOM kills
- Improve system stability
- No functionality loss
