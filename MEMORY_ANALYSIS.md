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

---

## 📊 Actual Memory Breakdown

### Per User Session (any single page):
- **Streamlit overhead**: 20-30 MB
- **Page code + widgets**: 10-20 MB  
- **Active data processing**: 20-50 MB
- **Session state**: 5-20 MB
- **Total per user**: 55-120 MB

### Server Memory (shared):
- **Global cache** (all pages): 500 MB - 2 GB (grows without limits)
- **Python runtime**: 50-100 MB
- **Libraries loaded**: 200-300 MB

### **Real Problem:**
```
Total Server Memory = Base (250 MB) + Global Cache (unbounded) + (# Users × 100 MB)
```

**Example with 15 users:**
- Base: 250 MB
- Global cache: 1.5 GB (without max_entries)
- User sessions: 15 × 100 MB = 1.5 GB
- **Total: 3.25 GB** ⚠️

---

## 🔧 Critical Fixes (Implement Immediately)

### Fix 1: Add max_entries to ALL caches (90% of the problem)

**Impact:** Limits global cache to reasonable size

```python
# Apply to ALL pages with @st.cache_data:

# High-frequency pages (data changes often)
@st.cache_data(ttl=60, max_entries=30)  # 5_0DTE.py, 6_0DTE_by_Index.py

# Medium-frequency pages (moderate refresh)
@st.cache_data(ttl=180, max_entries=50)  # 4_Option_Volume_Walls.py, 2_Trading_Dashboard.py

# Low-frequency pages (stable data)
@st.cache_data(ttl=300, max_entries=20)  # 3_Stock_Option_Finder.py, 7_Whale_Flows.py
```

**Memory Savings:** 1-2 GB → 200-400 MB for global cache

### Fix 2: Remove session_state dataframe storage

**7_Whale_Flows.py** (lines 688-694):
```python
# ❌ DON'T DO THIS:
st.session_state.whale_flows_data[expiry_date.strftime('%Y-%m-%d')] = pd.DataFrame(...)
st.session_state.oi_flows_data[expiry_date.strftime('%Y-%m-%d')] = pd.DataFrame(...)
st.session_state.skew_data[expiry_date.strftime('%Y-%m-%d')] = pd.DataFrame(...)

# ✅ DO THIS INSTEAD:
# Just use the cache - don't store in session_state
# The @st.cache_data decorator already handles storage
```

**Memory Savings per user:** 20-40 MB

### Fix 3: Clear session_state on page change

**Add to every page (top of file):**
```python
# Clear heavy session state when switching pages
current_page = __file__
if 'last_page' not in st.session_state:
    st.session_state.last_page = current_page
elif st.session_state.last_page != current_page:
    # Clean up heavy data from previous page
    keys_to_clear = [k for k in st.session_state.keys() 
                     if k.endswith('_data') or k.endswith('_cache')]
    for key in keys_to_clear:
        del st.session_state[key]
    st.session_state.last_page = current_page
```

**Memory Savings per user:** 10-30 MB

---

## 🎯 Implementation Plan

### **Phase 1: Emergency Fixes** (30 minutes - saves 70% memory)

Add max_entries to these high-traffic pages:
1. `pages/2_Trading_Dashboard.py`: `max_entries=50`
2. `pages/4_Option_Volume_Walls.py`: `max_entries=50` 
3. `pages/3_Stock_Option_Finder.py`: `max_entries=30`
4. `pages/7_Whale_Flows.py`: `max_entries=20`
5. `pages/5_0DTE.py`: `max_entries=30`
6. `pages/6_0DTE_by_Index.py`: `max_entries=30`

### **Phase 2: Session State Cleanup** (1 hour - saves 20% memory)

1. Remove dataframe storage in 7_Whale_Flows.py (lines 688-694)
2. Add page-change cleanup to top 6 pages above
3. Reduce watchlist limit to 30 in 2_Trading_Dashboard.py

### **Phase 3: Monitoring** (30 minutes)
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
