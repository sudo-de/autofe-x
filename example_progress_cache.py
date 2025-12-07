"""
Progress Tracking and Caching Example

Demonstrates AutoFE-X progress tracking and caching capabilities.
"""

import pandas as pd
import numpy as np
from autofex import AutoFEX

# Generate sample data
np.random.seed(42)
n_samples = 2000

data = {
    "feature_1": np.random.normal(100, 15, n_samples),
    "feature_2": np.random.exponential(2, n_samples),
    "feature_3": np.random.uniform(0, 100, n_samples),
    "feature_4": np.random.gamma(2, 2, n_samples),
    "category": np.random.choice(["A", "B", "C"], n_samples),
}

df = pd.DataFrame(data)
df["target"] = (
    0.5 * df["feature_1"]
    + 0.3 * df["feature_2"]
    + np.random.normal(0, 10, n_samples)
)

print("🚀 AutoFE-X Progress Tracking & Caching Demo")
print("=" * 60)

# ============================================================
# 1. WITH PROGRESS TRACKING AND CACHING
# ============================================================
print("\n1️⃣ Running with Progress Tracking and Caching...")
print("-" * 60)

afx = AutoFEX(
    enable_progress=True,
    enable_cache=True,
    cache_dir=".autofex_cache",
    cache_ttl=3600,  # 1 hour TTL
)

# First run - will compute everything
print("\n📊 First Run (computing and caching):")
result1 = afx.process(df.drop("target", axis=1), df["target"])

print(f"\n✅ First run completed in {result1.processing_time:.2f}s")
print(f"📈 Engineered features: {result1.original_data.shape[1]} → {result1.engineered_features.shape[1]}")

# Second run - will use cache
print("\n📊 Second Run (using cache):")
result2 = afx.process(df.drop("target", axis=1), df["target"])

print(f"\n✅ Second run completed in {result2.processing_time:.2f}s")
print(f"⚡ Speedup: {result1.processing_time / result2.processing_time:.2f}x faster!")

# Cache statistics
if afx.cache:
    cache_stats = afx.cache.get_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"   • Cache size: {cache_stats.get('size_mb', 0):.2f} MB")
    print(f"   • Max size: {cache_stats.get('max_size_mb', 0):.2f} MB")
    print(f"   • Number of entries: {cache_stats.get('num_entries', 0)}")
    print(f"   • TTL: {cache_stats.get('ttl_seconds', 'None')} seconds")

# ============================================================
# 2. WITHOUT PROGRESS TRACKING (for comparison)
# ============================================================
print("\n2️⃣ Running without Progress Tracking (for comparison)...")
print("-" * 60)

afx_no_progress = AutoFEX(
    enable_progress=False,
    enable_cache=False,
)

result3 = afx_no_progress.process(df.drop("target", axis=1), df["target"])
print(f"✅ Completed in {result3.processing_time:.2f}s (no progress output)")

# ============================================================
# 3. CACHE MANAGEMENT
# ============================================================
print("\n3️⃣ Cache Management...")
print("-" * 60)

if afx.cache:
    # Clear specific operation cache
    print("🗑️  Clearing feature engineering cache...")
    afx.cache.clear(operation="feature_engineering")
    
    # Run again - will recompute feature engineering but use cached profiling
    print("📊 Running again (will recompute feature engineering)...")
    result4 = afx.process(df.drop("target", axis=1), df["target"])
    print(f"✅ Completed in {result4.processing_time:.2f}s")
    
    # Clear all cache
    print("\n🗑️  Clearing all cache...")
    afx.cache.clear()
    print("✅ Cache cleared")

# ============================================================
# 4. REAL-TIME FEEDBACK
# ============================================================
print("\n4️⃣ Real-Time Feedback Metrics...")
print("-" * 60)

if afx.real_time_feedback:
    metrics = afx.real_time_feedback.get_all_metrics()
    print("📊 Captured Metrics:")
    for name, value in metrics.items():
        print(f"   • {name}: {value}")
    
    history = afx.real_time_feedback.get_history()
    print(f"\n📈 Total metric updates: {len(history)}")

print("\n" + "=" * 60)
print("🎉 Progress Tracking & Caching Demo Complete!")
print("\n✨ Key Features Demonstrated:")
print("   • Real-time progress bars with ETA")
print("   • Step-by-step progress tracking")
print("   • Intelligent caching for repeated operations")
print("   • Cache statistics and management")
print("   • Real-time feedback metrics")
print("\n💡 Benefits:")
print("   • Faster repeated runs (cache hits)")
print("   • Better user experience (progress feedback)")
print("   • Reduced computation costs")
print("   • Transparent operation tracking")
