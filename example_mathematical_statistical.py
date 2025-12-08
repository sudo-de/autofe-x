"""
Mathematical Modeling & Advanced Statistical Transforms Example

Demonstrates AutoFE-X enhanced capabilities using numpy, pandas, scipy, and scikit-learn.
"""

import pandas as pd
import numpy as np
from autofex import (
    MathematicalModelingEngine,
    StatisticalTransforms,
    AutoFEX,
)

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    "feature_1": np.random.normal(100, 15, n_samples),
    "feature_2": np.random.exponential(2, n_samples),
    "feature_3": np.random.uniform(0, 100, n_samples),
    "feature_4": np.random.gamma(2, 2, n_samples),
    "feature_5": np.random.beta(2, 5, n_samples),
}

df = pd.DataFrame(data)
df["target"] = (
    0.5 * df["feature_1"]
    + 0.3 * df["feature_2"]
    + np.random.normal(0, 10, n_samples)
)

print("🚀 AutoFE-X Mathematical & Statistical Features Demo")
print("=" * 60)

# ============================================================
# 1. MATHEMATICAL MODELING ENGINE
# ============================================================
print("\n1️⃣ Mathematical Modeling Engine")
print("-" * 60)

math_engine = MathematicalModelingEngine({
    "polynomial_features": True,
    "spline_features": True,
    "pca_features": True,
    "cluster_features": True,
    "distribution_features": True,
    "n_components_pca": 3,
    "n_clusters": 5,
})

print("📊 Creating mathematical modeling features...")
math_features = math_engine.fit_transform(df.drop("target", axis=1), df["target"])

print(f"✅ Created {math_features.shape[1]} mathematical features")
print(f"   Original: {df.shape[1] - 1} features")
print(f"   Expanded: {math_features.shape[1]} features")
print(f"   Expansion ratio: {math_features.shape[1] / (df.shape[1] - 1):.2f}x")

# Show some feature types
print("\n   Feature types created:")
feature_types = {}
for col in math_features.columns:
    if "pca" in col:
        feature_types["PCA"] = feature_types.get("PCA", 0) + 1
    elif "spline" in col:
        feature_types["Spline"] = feature_types.get("Spline", 0) + 1
    elif "poly" in col or "^" in col:
        feature_types["Polynomial"] = feature_types.get("Polynomial", 0) + 1
    elif "cluster" in col:
        feature_types["Cluster"] = feature_types.get("Cluster", 0) + 1
    elif "dist" in col or "norm" in col or "gamma" in col:
        feature_types["Distribution"] = feature_types.get("Distribution", 0) + 1

for feat_type, count in feature_types.items():
    print(f"   • {feat_type}: {count} features")

# ============================================================
# 2. ADVANCED STATISTICAL TRANSFORMS
# ============================================================
print("\n2️⃣ Advanced Statistical Transforms")
print("-" * 60)

stat_transforms = StatisticalTransforms()

print("📊 Creating statistical transformation features...")
stat_features = stat_transforms.apply_all_transforms(df.drop("target", axis=1))

print(f"✅ Created {stat_features.shape[1]} statistical transform features")

# Show transformation types
print("\n   Transformation types:")
transform_types = {}
for col in stat_features.columns:
    if "boxcox" in col:
        transform_types["Box-Cox"] = transform_types.get("Box-Cox", 0) + 1
    elif "yeojohnson" in col:
        transform_types["Yeo-Johnson"] = transform_types.get("Yeo-Johnson", 0) + 1
    elif "quantile" in col:
        transform_types["Quantile"] = transform_types.get("Quantile", 0) + 1
    elif "rank" in col:
        transform_types["Rank"] = transform_types.get("Rank", 0) + 1
    elif "zscore" in col:
        transform_types["Z-score"] = transform_types.get("Z-score", 0) + 1
    elif "robust" in col:
        transform_types["Robust Scale"] = transform_types.get("Robust Scale", 0) + 1

for trans_type, count in transform_types.items():
    print(f"   • {trans_type}: {count} features")

# ============================================================
# 3. STATISTICAL FEATURE CREATION
# ============================================================
print("\n3️⃣ Statistical Feature Creation")
print("-" * 60)

stat_features_created = stat_transforms.create_statistical_features(df.drop("target", axis=1))
print(f"✅ Created {stat_features_created.shape[1]} statistical summary features")

print("\n   Statistical features include:")
print("   • Mean, std, median, MAD, IQR")
print("   • Percentiles (10, 25, 50, 75, 90, 95, 99)")
print("   • Skewness, kurtosis")
print("   • Range, coefficient of variation")
print("   • Normality test p-values")
print("   • Outlier counts")

# ============================================================
# 4. CORRELATION FEATURES
# ============================================================
print("\n4️⃣ Correlation-Based Features")
print("-" * 60)

corr_features = stat_transforms.create_correlation_features(df.drop("target", axis=1))
print(f"✅ Created {corr_features.shape[1]} correlation features")

print("\n   Correlation features include:")
print("   • Max correlation with other features")
print("   • Mean correlation with other features")
print("   • Min correlation with other features")

# ============================================================
# 5. INTEGRATED USAGE WITH AUTOFEX
# ============================================================
print("\n5️⃣ Integrated Usage with AutoFEX")
print("-" * 60)

print("📊 Running AutoFEX with mathematical and statistical features...")

# Create enhanced feature engineering config
enhanced_config = {
    "numeric_transforms": ["log", "sqrt", "standardize"],
    "categorical_transforms": ["frequency_encode"],
    "interaction_degree": 2,
}

afx = AutoFEX(
    feature_engineering_config=enhanced_config,
    enable_progress=True,
    enable_cache=True,
)

result = afx.process(df.drop("target", axis=1), df["target"])

print(f"✅ AutoFEX processed {result.original_data.shape[1]} → {result.engineered_features.shape[1]} features")
print(f"⏱️  Processing time: {result.processing_time:.2f}s")

# Add mathematical features
print("\n📊 Adding mathematical modeling features...")
math_features_aligned = math_engine.fit_transform(result.engineered_features)
combined_features = pd.concat([result.engineered_features, math_features_aligned], axis=1)

print(f"✅ Combined features: {combined_features.shape[1]} total")
print(f"   • AutoFEX features: {result.engineered_features.shape[1]}")
print(f"   • Mathematical features: {math_features_aligned.shape[1]}")

# ============================================================
# 6. FEATURE COMPARISON
# ============================================================
print("\n6️⃣ Feature Comparison")
print("-" * 60)

print("📊 Feature counts by type:")
print(f"   • Original features: {df.shape[1] - 1}")
print(f"   • AutoFEX engineered: {result.engineered_features.shape[1]}")
print(f"   • Mathematical modeling: {math_features.shape[1]}")
print(f"   • Statistical transforms: {stat_features.shape[1]}")
print(f"   • Statistical summaries: {stat_features_created.shape[1]}")
print(f"   • Correlation features: {corr_features.shape[1]}")

total_enhanced = (
    result.engineered_features.shape[1]
    + math_features.shape[1]
    + stat_features.shape[1]
    + stat_features_created.shape[1]
    + corr_features.shape[1]
)

print(f"\n   Total enhanced features: {total_enhanced}")
print(f"   Expansion ratio: {total_enhanced / (df.shape[1] - 1):.2f}x")

print("\n" + "=" * 60)
print("🎉 Mathematical & Statistical Features Demo Complete!")
print("\n✨ Key Features Demonstrated:")
print("   • Polynomial & Spline features (scikit-learn)")
print("   • PCA, ICA, Factor Analysis (dimensionality reduction)")
print("   • Cluster-based features (KMeans)")
print("   • Manifold learning (t-SNE, MDS, Isomap)")
print("   • Signal processing (FFT, spectral analysis)")
print("   • Distribution fitting (scipy.stats)")
print("   • Box-Cox & Yeo-Johnson transformations")
print("   • Quantile & Power transformations")
print("   • Statistical summary features")
print("   • Correlation-based features")
print("\n🚀 Leverages numpy, pandas, scipy, and scikit-learn!")
