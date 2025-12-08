"""
Complete Advanced Features Example

Demonstrates ALL advanced AutoFE-X capabilities using numpy, pandas, scipy, and scikit-learn.
"""

import pandas as pd
import numpy as np
from autofex import (
    AutoFEX,
    MathematicalModelingEngine,
    StatisticalTransforms,
    PandasOperations,
    NumpyOperations,
    ScipyOperations,
)

# Generate comprehensive sample data
np.random.seed(42)
n_samples = 2000

# Create diverse data types
data = {
    # Numeric features
    "price": np.random.normal(100, 15, n_samples),
    "volume": np.random.exponential(2, n_samples),
    "score": np.random.uniform(0, 100, n_samples),
    "rating": np.random.gamma(2, 2, n_samples),
    
    # Categorical features
    "category": np.random.choice(["A", "B", "C", "D"], n_samples),
    "region": np.random.choice(["North", "South", "East", "West"], n_samples),
    
    # String features
    "description": [f"Item_{i}_Category_{np.random.choice(['X', 'Y', 'Z'])}" for i in range(n_samples)],
    
    # Datetime features
    "date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
}

df = pd.DataFrame(data)
df["target"] = (
    0.5 * df["price"]
    + 0.3 * df["volume"]
    + np.random.normal(0, 10, n_samples)
)

print("🚀 AutoFE-X Complete Advanced Features Demo")
print("=" * 70)

# ============================================================
# 1. AUTOFEX PIPELINE
# ============================================================
print("\n1️⃣ AutoFEX Pipeline (with progress & caching)")
print("-" * 70)

afx = AutoFEX(
    enable_progress=True,
    enable_cache=True,
    n_jobs=-1,  # Parallel processing
)

result = afx.process(df.drop("target", axis=1), df["target"])
print(f"✅ AutoFEX: {result.original_data.shape[1]} → {result.engineered_features.shape[1]} features")

# ============================================================
# 2. MATHEMATICAL MODELING
# ============================================================
print("\n2️⃣ Mathematical Modeling Engine")
print("-" * 70)

math_engine = MathematicalModelingEngine({
    "polynomial_features": True,
    "spline_features": True,
    "pca_features": True,
    "cluster_features": True,
    "distribution_features": True,
})

math_features = math_engine.fit_transform(result.engineered_features)
print(f"✅ Mathematical: {math_features.shape[1]} features")
print(f"   • Polynomial, Spline, PCA, ICA, Clustering, Manifold")

# ============================================================
# 3. STATISTICAL TRANSFORMS
# ============================================================
print("\n3️⃣ Advanced Statistical Transforms")
print("-" * 70)

stat_transforms = StatisticalTransforms()
stat_features = stat_transforms.apply_all_transforms(result.engineered_features)
stat_summary = stat_transforms.create_statistical_features(result.engineered_features)

print(f"✅ Statistical Transforms: {stat_features.shape[1]} features")
print(f"✅ Statistical Summary: {stat_summary.shape[1]} features")
print(f"   • Box-Cox, Yeo-Johnson, Quantile, Power, Rank, Z-score")

# ============================================================
# 4. ADVANCED PANDAS OPERATIONS
# ============================================================
print("\n4️⃣ Advanced Pandas Operations")
print("-" * 70)

pandas_ops = PandasOperations({
    "window_features": True,
    "datetime_features": True,
    "string_features": True,
    "cumulative_features": True,
    "diff_features": True,
})

pandas_features = pandas_ops.fit_transform(df.drop("target", axis=1))
print(f"✅ Pandas Operations: {pandas_features.shape[1]} features")
print(f"   • Rolling windows, Datetime, String, Cumulative, Differences")

# ============================================================
# 5. ADVANCED NUMPY OPERATIONS
# ============================================================
print("\n5️⃣ Advanced Numpy Operations")
print("-" * 70)

numpy_ops = NumpyOperations({
    "array_features": True,
    "broadcasting_features": True,
    "matrix_features": True,
    "advanced_math_features": True,
    "aggregation_features": True,
})

numpy_features = numpy_ops.fit_transform(result.engineered_features)
print(f"✅ Numpy Operations: {numpy_features.shape[1]} features")
print(f"   • Array ops, Broadcasting, Matrix ops, Math functions, Aggregations")

# ============================================================
# 6. ADVANCED SCIPY OPERATIONS
# ============================================================
print("\n6️⃣ Advanced Scipy Operations")
print("-" * 70)

scipy_ops = ScipyOperations({
    "distance_features": True,
    "optimization_features": True,
    "signal_features": True,
})

scipy_features = scipy_ops.fit_transform(result.engineered_features)
print(f"✅ Scipy Operations: {scipy_features.shape[1]} features")
print(f"   • Distance metrics, Optimization, Signal processing, Special functions")

# ============================================================
# 7. COMBINE ALL FEATURES
# ============================================================
print("\n7️⃣ Combining All Features")
print("-" * 70)

all_features_list = [
    result.engineered_features,
    math_features,
    stat_features,
    stat_summary,
    pandas_features,
    numpy_features,
    scipy_features,
]

# Filter out empty DataFrames
all_features_list = [f for f in all_features_list if not f.empty]

combined_features = pd.concat(all_features_list, axis=1)
combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

print(f"✅ Combined Features: {combined_features.shape[1]} total")
print(f"   • Original: {result.original_data.shape[1]}")
print(f"   • AutoFEX: {result.engineered_features.shape[1]}")
print(f"   • Mathematical: {math_features.shape[1]}")
print(f"   • Statistical: {stat_features.shape[1] + stat_summary.shape[1]}")
print(f"   • Pandas: {pandas_features.shape[1]}")
print(f"   • Numpy: {numpy_features.shape[1]}")
print(f"   • Scipy: {scipy_features.shape[1]}")
print(f"   • Expansion: {combined_features.shape[1] / result.original_data.shape[1]:.1f}x")

# ============================================================
# 8. FEATURE SUMMARY BY LIBRARY
# ============================================================
print("\n8️⃣ Feature Summary by Library")
print("-" * 70)

library_features = {
    "AutoFEX Core": result.engineered_features.shape[1],
    "scikit-learn": math_features.shape[1],
    "scipy.stats": stat_features.shape[1] + stat_summary.shape[1],
    "pandas": pandas_features.shape[1],
    "numpy": numpy_features.shape[1],
    "scipy.advanced": scipy_features.shape[1],
}

print("📊 Features by Library:")
for library, count in sorted(library_features.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {library:20s}: {count:4d} features")

print("\n" + "=" * 70)
print("🎉 Complete Advanced Features Demo Complete!")
print("\n✨ All Libraries Leveraged:")
print("   • numpy: Array operations, mathematical functions, aggregations")
print("   • pandas: Window functions, groupby, datetime, string operations")
print("   • scipy: Statistical functions, signal processing, optimization, special functions")
print("   • scikit-learn: Polynomial, spline, PCA, ICA, clustering, manifold learning")
print("\n🚀 AutoFE-X: The most comprehensive feature engineering toolkit!")
