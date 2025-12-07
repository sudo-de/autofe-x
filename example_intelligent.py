"""
Intelligent AutoFE-X Example

Demonstrates intelligent feature engineering with automatic recommendations,
quality scoring, and orchestration.
"""

import pandas as pd
import numpy as np
from autofex import (
    AutoFEX,
    IntelligentOrchestrator,
    FeatureQualityScorer,
    FeatureEngineeringRecommender,
)

# Generate sample data with various characteristics
np.random.seed(42)
n_samples = 2000

data = {
    # Skewed features
    "price": np.random.lognormal(mean=5, sigma=1, size=n_samples),
    "volume": np.random.exponential(2, n_samples),
    
    # Normal features
    "score": np.random.normal(50, 10, n_samples),
    "rating": np.random.normal(4, 0.5, n_samples),
    
    # Categorical
    "category": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
    "region": np.random.choice(["North", "South", "East", "West"], n_samples),
    
    # Datetime
    "date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
    
    # String
    "description": [f"Item_{i}" for i in range(n_samples)],
}

df = pd.DataFrame(data)
df["target"] = (
    0.5 * df["price"]
    + 0.3 * df["volume"]
    + np.random.normal(0, 10, n_samples)
)

print("🧠 AutoFE-X Intelligent Feature Engineering Demo")
print("=" * 70)

# ============================================================
# 1. INTELLIGENT RECOMMENDATIONS
# ============================================================
print("\n1️⃣ Intelligent Feature Engineering Recommendations")
print("-" * 70)

recommender = FeatureEngineeringRecommender()
recommendations = recommender.recommend_feature_engineering(df.drop("target", axis=1), df["target"])

print("📊 Transformation Recommendations:")
for rec in recommendations["transformations"][:5]:
    print(f"   • {rec['column']}: {rec['transformation']} ({rec['reason']})")

print("\n📊 Strategy Recommendations:")
for strategy in recommendations["strategies"]:
    print(f"   • {strategy['strategy']}: {strategy['reason']}")

if recommendations["warnings"]:
    print("\n⚠️  Warnings:")
    for warning in recommendations["warnings"]:
        print(f"   • {warning}")

if recommendations["opportunities"]:
    print("\n💡 Opportunities:")
    for opp in recommendations["opportunities"]:
        print(f"   • {opp}")

# Get auto-configuration
auto_config = recommender.get_auto_config(df.drop("target", axis=1), df["target"])
print(f"\n✅ Auto-generated configuration: {len(auto_config)} modules configured")

# ============================================================
# 2. INTELLIGENT ORCHESTRATION
# ============================================================
print("\n2️⃣ Intelligent Feature Engineering Orchestration")
print("-" * 70)

orchestrator = IntelligentOrchestrator({
    "auto_detect": True,
    "max_features": 500,
})

print("📊 Analyzing data characteristics...")
characteristics = orchestrator.analyze_data_characteristics(df.drop("target", axis=1), df["target"])

print(f"   • Samples: {characteristics['n_samples']}")
print(f"   • Features: {characteristics['n_features']}")
print(f"   • Numeric: {characteristics['numeric_count']}")
print(f"   • Categorical: {characteristics['categorical_count']}")
print(f"   • Datetime: {characteristics['datetime_count']}")
print(f"   • String: {characteristics['string_count']}")
print(f"   • Skewed features: {len(characteristics['skewed_features'])}")

print("\n📊 Intelligently engineering features...")
intelligent_features = orchestrator.intelligent_feature_engineering(
    df.drop("target", axis=1),
    df["target"]
)

print(f"✅ Intelligent engineering: {df.shape[1] - 1} → {intelligent_features.shape[1]} features")
print(f"   • Expansion ratio: {intelligent_features.shape[1] / (df.shape[1] - 1):.2f}x")

# ============================================================
# 3. FEATURE QUALITY SCORING
# ============================================================
print("\n3️⃣ Feature Quality Scoring")
print("-" * 70)

quality_scorer = FeatureQualityScorer()

print("📊 Scoring feature quality...")
quality_scores = quality_scorer.score_all_features(intelligent_features, df["target"], top_n=20)

print(f"✅ Scored {len(quality_scores)} features")
print("\n📊 Top 10 Features by Overall Quality:")
for idx, row in quality_scores.head(10).iterrows():
    print(f"   {idx+1:2d}. {row['feature']:30s} | Score: {row['overall_score']:.3f} | "
          f"Predictive: {row['predictive_power']:.3f} | Stable: {row['stability']:.3f}")

# Get feature rankings
rankings = quality_scorer.get_feature_rankings(intelligent_features, df["target"])

print("\n📊 Feature Rankings by Category:")
print(f"   • Top Predictive: {len(rankings['top_predictive'])} features")
print(f"   • Top Stable: {len(rankings['top_stable'])} features")
print(f"   • Top Unique: {len(rankings['top_unique'])} features")
print(f"   • Top Efficient: {len(rankings['top_efficient'])} features")

# Get top features
top_features = quality_scorer.get_top_features(intelligent_features, df["target"], n_features=50)
print(f"\n✅ Selected {len(top_features)} top-quality features")

# ============================================================
# 4. COMPLETE INTELLIGENT PIPELINE
# ============================================================
print("\n4️⃣ Complete Intelligent Pipeline")
print("-" * 70)

print("📊 Running AutoFEX with intelligent recommendations...")

# Use recommendations to configure AutoFEX
afx = AutoFEX(
    feature_engineering_config=auto_config.get("feature_engineering", {}),
    enable_progress=True,
    enable_cache=True,
    n_jobs=-1,
)

result = afx.process(df.drop("target", axis=1), df["target"])

print(f"✅ AutoFEX: {result.original_data.shape[1]} → {result.engineered_features.shape[1]} features")

# Score AutoFEX features
afx_quality = quality_scorer.score_all_features(result.engineered_features, df["target"], top_n=10)
print(f"\n📊 Top AutoFEX Features by Quality:")
for idx, row in afx_quality.head(5).iterrows():
    print(f"   {idx+1}. {row['feature']:30s} | Score: {row['overall_score']:.3f}")

# ============================================================
# 5. COMPARISON
# ============================================================
print("\n5️⃣ Comparison: Manual vs Intelligent")
print("-" * 70)

print("📊 Manual Feature Engineering:")
manual_features = result.engineered_features
print(f"   • Features: {manual_features.shape[1]}")

print("\n📊 Intelligent Feature Engineering:")
intelligent_features_final = orchestrator.intelligent_feature_engineering(
    result.engineered_features,
    df["target"]
)
print(f"   • Features: {intelligent_features_final.shape[1]}")

# Score both
manual_scores = quality_scorer.score_all_features(manual_features, df["target"])
intelligent_scores = quality_scorer.score_all_features(intelligent_features_final, df["target"])

print(f"\n📊 Quality Comparison:")
print(f"   • Manual avg score: {manual_scores['overall_score'].mean():.3f}")
print(f"   • Intelligent avg score: {intelligent_scores['overall_score'].mean():.3f}")
print(f"   • Improvement: {((intelligent_scores['overall_score'].mean() / manual_scores['overall_score'].mean()) - 1) * 100:.1f}%")

print("\n" + "=" * 70)
print("🎉 Intelligent Feature Engineering Demo Complete!")
print("\n✨ Key Features Demonstrated:")
print("   • Automatic feature engineering recommendations")
print("   • Intelligent orchestration based on data characteristics")
print("   • Comprehensive feature quality scoring")
print("   • Feature rankings by multiple dimensions")
print("   • Auto-configuration generation")
print("\n🧠 AutoFE-X: Now with Intelligence!")
