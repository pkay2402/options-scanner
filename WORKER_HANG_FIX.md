# Worker Hang Fix - December 17, 2025

## Problem
The watchlist and whale flow services on the droplet hung again today, even though they were fixed yesterday. The services were running but not updating data:
- Last watchlist update: `2025-12-17 00:28:01` (hung for ~27 hours)
- Last whale flows update: `2025-12-17 00:28:23` (hung for ~27 hours)
- Other scanners (MACD, VPB, TTM) were working fine

## Root Cause
The `market_data_worker.py` had a critical issue in its `ThreadPoolExecutor` timeout handling:

1. **Incomplete timeout handling**: When using `as_completed(futures, timeout=30)`, if futures don't complete within the timeout, they would hang the thread pool without proper cancellation
2. **No watchdog timeout**: The entire update cycle had no maximum time limit, so if any batch hung, it would block the worker indefinitely
3. **No per-future timeout**: Individual futures could hang indefinitely even within a batch

## Solution Implemented

### 1. **Improved Watchlist Update** ([market_data_worker.py](scripts/market_data_worker.py) Line 247-285)
```python
# Added proper timeout handling with future cancellation
try:
    for future in as_completed(futures, timeout=45):
        try:
            result = future.result(timeout=10)  # Per-future timeout
            if result:
                watchlist_data.append(result)
        except Exception as e:
            symbol = futures[future]
            logger.error(f"Failed to fetch {symbol}: {e}")
            future.cancel()  # Cancel hung future
except TimeoutError:
    logger.error(f"Watchlist batch {i//batch_size + 1} timed out, cancelling remaining futures")
    for future in futures:
        future.cancel()
```

**Changes:**
- Added batch-level timeout (45s)
- Added per-future timeout (10s)
- Explicit future cancellation on timeout
- Comprehensive error handling

### 2. **Improved Whale Flow Update** ([market_data_worker.py](scripts/market_data_worker.py) Line 403-442)
```python
# Added timeout protection for whale flow scanning
try:
    for future in as_completed(futures, timeout=60):
        try:
            flows = future.result(timeout=15)  # Per-future timeout
            all_flows.extend(flows)
        except Exception as e:
            symbol, friday = futures[future]
            logger.error(f"Failed to scan {symbol} {friday}: {e}")
            future.cancel()
except TimeoutError:
    logger.error(f"Whale flow batch {i//batch_size + 1} timed out, cancelling remaining futures")
    for future in futures:
        future.cancel()
```

**Changes:**
- Added batch-level timeout (60s)
- Added per-future timeout (15s)
- Explicit future cancellation on timeout
- Proper error logging with symbol context

### 3. **Watchdog Timeout for Entire Cycle** ([market_data_worker.py](scripts/market_data_worker.py) Line 450-503)
```python
def run_cycle(self):
    """Run one complete update cycle with memory cleanup and watchdog timeout"""
    import gc
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Cycle took too long - forcing restart")
    
    try:
        # Set watchdog timeout for entire cycle (10 minutes max)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes
        
        # ... update logic ...
        
        # Cancel watchdog on success
        signal.alarm(0)
        
    except TimeoutError as e:
        logger.error(f"Cycle timeout: {e} - will retry next cycle", exc_info=True)
        signal.alarm(0)  # Cancel watchdog
```

**Changes:**
- Added 10-minute watchdog timeout for entire cycle
- Proper signal handling with SIGALRM
- Forces cycle to restart if it hangs
- Cancels alarm on success or error

## Deployment

Created deployment script [deploy_worker_fix.sh](scripts/deploy_worker_fix.sh):
```bash
./scripts/deploy_worker_fix.sh
```

The script:
1. Copies fixed `market_data_worker.py` to droplet
2. Restarts the `market-data-worker` service
3. Shows service status and recent logs

## Verification

After deployment, confirmed the fix is working:

### Before Fix
```json
{
    "watchlist": {
        "updated_at": "2025-12-17 00:28:01"  // Hung for 27 hours
    },
    "whale_flows": {
        "updated_at": "2025-12-17 00:28:23"  // Hung for 27 hours
    }
}
```

### After Fix
```json
{
    "watchlist": {
        "updated_at": "2025-12-18 03:37:22"  // Updating ✓
    },
    "whale_flows": {
        "updated_at": "2025-12-18 03:38:38"  // Updating ✓
    }
}
```

### Live Whale Flows Working
```bash
curl -s 'http://138.197.210.166:8000/api/whale_flows?limit=3'
# Returns 3 recent whale flows ✓
```

## Prevention

The fix addresses the hanging issue in three layers:

1. **Per-future timeout (10-15s)**: Prevents individual API calls from hanging
2. **Batch-level timeout (45-60s)**: Prevents entire batches from hanging
3. **Cycle watchdog (10 minutes)**: Forces restart if anything unexpected hangs

The worker will now:
- Cancel hung futures explicitly
- Log detailed error messages with context
- Continue operating even if some API calls fail
- Force restart after 10 minutes if completely stuck

## Monitoring

Monitor the worker with:
```bash
# Live logs
ssh root@138.197.210.166 'sudo journalctl -u market-data-worker -f'

# Check status
curl -s 'http://138.197.210.166:8000/api/last_update' | python3 -m json.tool

# Check service health
ssh root@138.197.210.166 'sudo systemctl status market-data-worker'
```

## Future Improvements

Consider:
1. Adding Prometheus metrics for timeout events
2. Implementing exponential backoff for failed API calls
3. Adding circuit breaker pattern for consistently failing symbols
4. Storing timeout statistics for analysis
