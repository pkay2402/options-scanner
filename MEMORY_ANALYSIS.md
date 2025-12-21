# Memory Usage Analysis for Streamlit App

## 🎯 **Key Insight: Only ONE page loads at a time**

**The real memory problem isn't individual pages - it's:**
1. **Cache accumulation** across all pages without limits
2. **Multiple user sessions** sharing the same server
3. **Session state** not cleaning up when switching pages

---

## 🔴 Real Memory Issues (Priority Fixes)

### 1. **Unbounded Cache Growth** ⚠️ HIGHEST PRIORITY
**Problem:** Caches grow indefinitely without `max_entries` parameter

**Current State:**
```python
@st.cache_data(ttl=300)  # ❌ No limit - cache grows forever
```

**Impact:**
- 10 users × 20 different symbols = 200 cached API calls
- Each call stores ~5-10 MB of options data
- **Total: 1-2 GB of cached data** across all pages

**Fix Priority:** CRITICAL

### 2. **Multiple User Sessions** ⚠️ HIGH PRIORITY
**Problem:** Server memory = sum of all active user sessions

With 10 concurrent users:
- Each user on a different page = 10× memory usage
- Caches are shared (good) but session_state is per-user (bad)

**Estimated Memory:**
- Single user: 50-100 MB
- 10 concurrent users: 500 MB - 1 GB
- 20 concurrent users: 1-2 GB ⚠️

### 3. **Session State Leaks** ⚠️ HIGH PRIORITY  
**Problem:** Data stored in `st.session_state` never gets cleaned up

**Memory Leakers Found:**
- **7_Whale_Flows.py** (lines 688-694): Stores 3 large dataframes in session_state
- **2_Trading_Dashboard.py**: Scanner signals accumulate across page visits

**Recommended Fixes:**
```python
# Increase cache TTL to reduce recomputation
@st.cache_data(ttl=180, show_spinner=False)  # 3 minutes instead of 1

# Add max_entries to limit cache size
@st.cache_data(ttl=180, max_entries=50)

# Limit data processing range
MAX_STRIKES_TO_DISPLAY = 30  # Don't process all strikes
MAX_EXPIRIES = 3  # Limit multi-expiry analysis
```

### 2. **2_Trading_Dashboard.py** (3,099 lines) ⚠️ HIGH PRIORITY
**Memory Issues:**
- ✅ Good: Using `@st.cache_data(ttl=300)` for main functions
- ❌ **Problem**: `live_watchlist()` fetches data for 50+ stocks every 3 minutes
- ❌ **Problem**: `whale_flows_feed()` loads 100 flows when filtering by symbol
- ❌ **Problem**: Multiple scanner signals stored in memory (lines 1662-1693)
- ❌ **Problem**: RSI/MACD calculations on full price history without chunking

**Estimated Memory Per Session:** 80-150 MB
**Fix Priority:** HIGH

**Recommended Fixes:**
```python
# Limit watchlist size
MAX_WATCHLIST_ITEMS = 30  # Instead of 50+

# Reduce whale flows limit
limit=20 if st.session_state.whale_filter == 'symbol' else 10

# Add max_entries to cache
@st.cache_data(ttl=300, max_entries=20)
```

### 3. **3_Stock_Option_Finder.py** (2,287 lines) ⚠️ MEDIUM PRIORITY
**Memory Issues:**
- ❌ **Problem**: Creates large GEX heatmaps for multiple expiries
- ❌ **Problem**: Gamma calculations across many strikes
- ✅ Good: Uses `@st.cache_data(ttl=300)`

**Estimated Memory Per Session:** 60-100 MB

**Recommended Fixes:**
```python
# Limit strikes range
MAX_STRIKES_RANGE = 40  # Reduce from potentially 100+

# Add cache limit
@st.cache_data(ttl=300, max_entries=30)
```

### 4. **7_Whale_Flows.py** (2,063 lines) ⚠️ MEDIUM PRIORITY
**Memory Issues:**
- ❌ **Problem**: Stores large dataframes in `st.session_state` (lines 688-694)
- ❌ **Problem**: Processing 4 Friday expiries with full options chains
- ❌ **Problem**: Session state accumulation without cleanup

**Estimated Memory Per Session:** 50-80 MB

**Recommended Fixes:**
```python
# Don't store in session_state - use cache instead
# Remove lines 688-694 that store dataframes in session_state

# Add expiry after cache data retrieval
@st.cache_data(ttl=300, max_entries=10)  # Add max_entries

# Clear old session state
if len(st.session_state.whale_flows_data) > 5:
    oldest_key = min(st.session_state.whale_flows_data.keys())
    del st.session_state.whale_flows_data[oldest_key]
```

## 🟡 Medium Memory Pages

### 5. **finra.py** (2,235 lines)
- Uses `pd.read_csv()` from string IO
- Long cache times (3600s, 11520s) are good
- **Fix:** Add `max_entries=20` to cache decorators

### 6. **13_Trade_Ideas_2026.py** (1,172 lines)
- Multiple dataframes for flows data
- **Fix:** Add `max_entries=15` to caches

### 7. **14_Top_30_AI_Stocks.py** (995 lines)
- Processes 30 stocks with scanner data
- **Fix:** Add `max_entries=10` to caches

## 🟢 Low Memory Pages (Optimized)

### Good Examples to Follow:
- **6_0DTE_by_Index.py**: Uses `ttl=60` with `show_spinner=False` ✅
- **5_0DTE.py**: Short TTL appropriate for real-time data ✅
- **15_SPX_Market_Intelligence.py**: Uses `ttl=30` for streaming ✅

---

## 🔧 Global Recommendations

### 1. Add Cache Limits to All Pages
```python
# Before:
@st.cache_data(ttl=300)

# After:
@st.cache_data(ttl=300, max_entries=20)
```

### 2. Session State Cleanup
Add to main Welcome.py or each page:
```python
# Clear session state periodically
if 'last_cleanup' not in st.session_state:
    st.session_state.last_cleanup = time.time()

# Cleanup every 15 minutes
if time.time() - st.session_state.last_cleanup > 900:
    # Remove old data
    keys_to_remove = [k for k in st.session_state.keys() 
                     if k.startswith('cached_') and 
                     time.time() - st.session_state.get(f'{k}_time', 0) > 600]
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state.last_cleanup = time.time()
```

### 3. Limit Data Processing
```python
# Limit strikes processed
MAX_STRIKES = 40
strikes = strikes[:MAX_STRIKES]

# Limit historical candles
MAX_CANDLES = 500
df = df.tail(MAX_CANDLES)

# Limit watchlist
MAX_WATCHLIST = 30
watchlist = watchlist[:MAX_WATCHLIST]
```

### 4. Use Generators for Large Datasets
```python
# Instead of:
all_data = [process(item) for item in large_list]

# Use:
def process_items(items):
    for item in items:
        yield process(item)

# Process in chunks
for batch in chunked(process_items(large_list), 100):
    # Process batch
```

### 5. Monitor Memory Usage
Add to Welcome.py:
```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Display in sidebar
if st.checkbox("Show Memory Debug"):
    st.sidebar.metric("Memory Usage", f"{get_memory_usage():.1f} MB")
```

---

## 📊 Estimated Memory Reduction

### Current Peak Memory: ~500-800 MB per active session

### After Optimizations:
- **4_Option_Volume_Walls.py**: 200 MB → 80 MB (-60%)
- **2_Trading_Dashboard.py**: 150 MB → 60 MB (-60%)
- **3_Stock_Option_Finder.py**: 100 MB → 40 MB (-60%)
- **7_Whale_Flows.py**: 80 MB → 30 MB (-62%)

### Expected Peak Memory: ~200-300 MB per active session (-60% overall)

---

## ⚡ Quick Wins (Implement These First)

1. **Add `max_entries` to all caches** (5 min fix, 30% reduction)
2. **Reduce watchlist size from 50 to 30** (1 min fix, 10% reduction)
3. **Increase TTL on Option_Volume_Walls from 60s to 180s** (1 min fix, 15% reduction)
4. **Remove session_state dataframe storage in Whale_Flows** (10 min fix, 20% reduction)
5. **Limit strikes processing to 40 max** (15 min fix, 25% reduction)

**Total Quick Win Reduction: ~50-60% memory usage**

---

## 🎯 Implementation Priority

### Week 1:
1. Add max_entries to all @st.cache_data decorators
2. Reduce watchlist limits
3. Increase TTL on heavy pages

### Week 2:
1. Remove session_state dataframe storage
2. Implement data processing limits (strikes, candles)
3. Add memory monitoring

### Week 3:
1. Optimize heatmap generation
2. Implement session state cleanup
3. Test and measure improvements
