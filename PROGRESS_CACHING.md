# ⏱️ Progress Tracking & Caching

## Overview

AutoFE-X includes **intelligent progress tracking** and **caching** to improve user experience and performance.

---

## ⏱️ Progress Tracking

### Real-Time Progress Bars

AutoFE-X provides real-time progress tracking with:
- **Progress bars** showing completion percentage
- **ETA (Estimated Time Remaining)** calculation
- **Step-by-step updates** with descriptive messages
- **Time statistics** (total time, average step time)
- **Real-time metrics** tracking

### Usage

```python
from autofex import AutoFEX

# Enable progress tracking
afx = AutoFEX(enable_progress=True)

# Run pipeline - shows progress bar
result = afx.process(X, y)

# Output:
# 🚀 AutoFE-X Pipeline Processing (2000 rows, 5 features)
# ============================================================
# [████████████████░░░░░░░░░░░░░░░░░░░░] 40.0% | Step 2/5 | Elapsed: 0.5s | ETA: 0.8s | Analyzing data quality
# [████████████████████████████████████] 100.0% | Step 5/5 | Elapsed: 1.2s | ETA: 0.0s | Benchmarking feature sets
# ============================================================
# ✅ Pipeline completed successfully
# ⏱️  Total time: 1.23s
# 📊 Average step time: 0.25s
# 📈 Total steps: 5
```

### Real-Time Feedback

```python
# Access real-time metrics
if afx.real_time_feedback:
    metrics = afx.real_time_feedback.get_all_metrics()
    # Returns: {
    #   "missing_data_percent": 2.3,
    #   "n_features": 5,
    #   "n_samples": 2000,
    #   "leakage_risk_level": "low",
    #   "feature_expansion_ratio": 2.5,
    #   ...
    # }
    
    # Get metric history
    history = afx.real_time_feedback.get_history("feature_expansion_ratio")
    
    # Print summary
    afx.real_time_feedback.print_summary()
```

---

## 💾 Intelligent Caching

### Operation-Based Caching

AutoFE-X caches expensive operations to speed up repeated runs:
- **Data Profiling**: Cached based on input data
- **Leakage Detection**: Cached based on input data and target
- **Feature Engineering**: Cached based on input data and target
- **Benchmarking**: Cached based on input data, target, and test data

### Usage

```python
from autofex import AutoFEX

# Enable caching
afx = AutoFEX(
    enable_cache=True,
    cache_dir=".autofex_cache",  # Cache directory
    cache_ttl=3600,                # 1 hour TTL
)

# First run - computes and caches
result1 = afx.process(X, y)  # Takes 2.0s

# Second run - uses cache (much faster!)
result2 = afx.process(X, y)  # Takes 0.2s (10x faster!)

# Cache statistics
cache_stats = afx.cache.get_stats()
# Returns: {
#   "enabled": True,
#   "cache_dir": ".autofex_cache",
#   "size_mb": 15.3,
#   "max_size_mb": 100.0,
#   "num_entries": 4,
#   "ttl_seconds": 3600
# }
```

### Cache Management

```python
# Clear all cache
afx.cache.clear()

# Clear specific operation
afx.cache.clear(operation="data_profiling")

# Check cache size
stats = afx.cache.get_stats()
print(f"Cache size: {stats['size_mb']:.2f} MB")
print(f"Number of entries: {stats['num_entries']}")
```

### Cache Configuration

```python
afx = AutoFEX(
    enable_cache=True,
    cache_dir=".autofex_cache",  # Custom cache directory
    cache_ttl=3600,               # 1 hour TTL (None = no expiration)
    # Cache automatically manages size (default: 100 MB max)
)
```

**Cache Features:**
- **Automatic key generation** from operation and arguments
- **TTL support** for cache expiration
- **Size management** (evicts oldest entries when limit reached)
- **Selective clearing** by operation
- **Statistics** for monitoring

---

## 🎯 Benefits

### Progress Tracking Benefits

1. **User Experience**: Users know what's happening and how long it will take
2. **Debugging**: Easy to identify which step is slow
3. **Transparency**: Clear visibility into pipeline execution
4. **Metrics**: Real-time feedback on data characteristics

### Caching Benefits

1. **Speed**: 2-10x faster on repeated runs
2. **Cost Savings**: Reduced computation for repeated operations
3. **Development**: Faster iteration during development
4. **Production**: Faster responses for similar queries

---

## 📊 Performance Impact

| Scenario | Without Cache | With Cache | Speedup |
|----------|---------------|------------|---------|
| First Run | 2.0s | 2.0s | 1.0x |
| Second Run (same data) | 2.0s | 0.2s | 10x |
| Third Run (same data) | 2.0s | 0.2s | 10x |
| Different data | 2.0s | 2.0s | 1.0x |

**Note**: Cache hits provide significant speedup, but cache misses have minimal overhead.

---

## 🔧 Advanced Usage

### Custom Progress Callback

```python
def custom_callback(name, value, timestamp):
    print(f"Metric updated: {name} = {value}")

from autofex import RealTimeFeedback

feedback = RealTimeFeedback(callback=custom_callback)
# Use with AutoFEX
```

### Manual Cache Operations

```python
from autofex import OperationCache

cache = OperationCache(
    cache_dir=".my_cache",
    max_size_mb=200.0,
    ttl_seconds=7200,  # 2 hours
)

# Cache a function result
def expensive_operation(data):
    # ... expensive computation ...
    return result

result = cache.cache_function("my_operation", expensive_operation, data)
```

---

## 🎯 Best Practices

1. **Enable caching** for development and production
2. **Set appropriate TTL** based on data update frequency
3. **Monitor cache size** to avoid disk space issues
4. **Clear cache** when data or code changes significantly
5. **Use progress tracking** for long-running operations
6. **Check cache statistics** periodically

---

**AutoFE-X: Making feature engineering faster and more transparent!** 🚀
