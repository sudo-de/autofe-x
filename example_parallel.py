"""
Parallel Processing Example

Demonstrates AutoFE-X parallel processing capabilities for faster feature engineering.
"""

import pandas as pd
import numpy as np
import time
from autofex import AutoFEX

# Generate large sample data
np.random.seed(42)
n_samples = 10000
n_features = 50

print("🚀 AutoFE-X Parallel Processing Demo")
print("=" * 60)

# Create dataset with many columns
data = {}
for i in range(n_features):
    if i % 3 == 0:
        data[f"numeric_{i}"] = np.random.normal(100, 15, n_samples)
    elif i % 3 == 1:
        data[f"categorical_{i}"] = np.random.choice(["A", "B", "C", "D", "E"], n_samples)
    else:
        data[f"numeric_{i}"] = np.random.exponential(2, n_samples)

df = pd.DataFrame(data)
df["target"] = (
    0.5 * df.select_dtypes(include=[np.number]).iloc[:, 0]
    + np.random.normal(0, 10, n_samples)
)

print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} features")
print(f"   • Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"   • Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")

# ============================================================
# 1. SEQUENTIAL PROCESSING (baseline)
# ============================================================
print("\n1️⃣ Sequential Processing (baseline)...")
print("-" * 60)

afx_sequential = AutoFEX(
    enable_progress=True,
    enable_cache=False,  # Disable cache for fair comparison
    n_jobs=1,  # Sequential
)

start_time = time.time()
result_seq = afx_sequential.process(df.drop("target", axis=1), df["target"])
sequential_time = time.time() - start_time

print(f"✅ Sequential time: {sequential_time:.2f}s")
print(f"📈 Engineered features: {result_seq.original_data.shape[1]} → {result_seq.engineered_features.shape[1]}")

# ============================================================
# 2. PARALLEL PROCESSING (threading)
# ============================================================
print("\n2️⃣ Parallel Processing (threading, 4 jobs)...")
print("-" * 60)

afx_parallel = AutoFEX(
    enable_progress=True,
    enable_cache=False,  # Disable cache for fair comparison
    n_jobs=4,  # 4 parallel jobs
    parallel_backend="threading",
)

start_time = time.time()
result_parallel = afx_parallel.process(df.drop("target", axis=1), df["target"])
parallel_time = time.time() - start_time

print(f"✅ Parallel time: {parallel_time:.2f}s")
print(f"📈 Engineered features: {result_parallel.original_data.shape[1]} → {result_parallel.engineered_features.shape[1]}")
print(f"⚡ Speedup: {sequential_time / parallel_time:.2f}x faster!")

# ============================================================
# 3. PARALLEL PROCESSING (all CPUs)
# ============================================================
print("\n3️⃣ Parallel Processing (all CPUs)...")
print("-" * 60)

afx_all_cpus = AutoFEX(
    enable_progress=True,
    enable_cache=False,  # Disable cache for fair comparison
    n_jobs=-1,  # All CPUs
    parallel_backend="threading",
)

start_time = time.time()
result_all = afx_all_cpus.process(df.drop("target", axis=1), df["target"])
all_cpus_time = time.time() - start_time

print(f"✅ All CPUs time: {all_cpus_time:.2f}s")
print(f"⚡ Speedup vs sequential: {sequential_time / all_cpus_time:.2f}x faster!")

# ============================================================
# 4. PARALLEL + CACHING (best performance)
# ============================================================
print("\n4️⃣ Parallel Processing + Caching (best performance)...")
print("-" * 60)

afx_best = AutoFEX(
    enable_progress=True,
    enable_cache=True,  # Enable caching
    cache_dir=".autofex_cache",
    n_jobs=-1,  # All CPUs
    parallel_backend="threading",
)

# First run - computes and caches
print("   📊 First run (computing and caching)...")
start_time = time.time()
result1 = afx_best.process(df.drop("target", axis=1), df["target"])
first_run_time = time.time() - start_time
print(f"   ✅ First run: {first_run_time:.2f}s")

# Second run - uses cache
print("   📊 Second run (using cache)...")
start_time = time.time()
result2 = afx_best.process(df.drop("target", axis=1), df["target"])
second_run_time = time.time() - start_time
print(f"   ✅ Second run: {second_run_time:.2f}s")
print(f"   ⚡ Cache speedup: {first_run_time / second_run_time:.2f}x faster!")

# ============================================================
# 5. PERFORMANCE SUMMARY
# ============================================================
print("\n5️⃣ Performance Summary")
print("-" * 60)
print(f"   Sequential:        {sequential_time:.2f}s (baseline)")
print(f"   Parallel (4 jobs): {parallel_time:.2f}s ({sequential_time / parallel_time:.2f}x speedup)")
print(f"   Parallel (all):    {all_cpus_time:.2f}s ({sequential_time / all_cpus_time:.2f}x speedup)")
print(f"   Parallel + Cache: {second_run_time:.2f}s ({sequential_time / second_run_time:.2f}x speedup)")

print("\n" + "=" * 60)
print("🎉 Parallel Processing Demo Complete!")
print("\n✨ Key Features Demonstrated:")
print("   • Parallel column processing")
print("   • Configurable number of jobs")
print("   • Threading backend (good for I/O-bound operations)")
print("   • Integration with progress tracking")
print("   • Integration with caching")
print("\n💡 Best Practices:")
print("   • Use n_jobs=-1 for all CPUs (best for large datasets)")
print("   • Use threading backend for I/O-bound operations")
print("   • Combine with caching for maximum performance")
print("   • Monitor progress to see parallel processing in action")
