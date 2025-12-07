"""
Advanced AutoFE-X Example

Demonstrates capabilities that go beyond basic Scipy, Matplotlib, and Plotly:
- Interactive dashboards with integrated analysis
- Advanced statistical testing with automated interpretations
- Actionable insights and recommendations
"""

import pandas as pd
import numpy as np
from autofex import (
    AutoFEX,
    InteractiveDashboard,
    AdvancedStatisticalAnalyzer,
    FeatureVisualizer,
)

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    "feature_1": np.random.normal(100, 15, n_samples),
    "feature_2": np.random.exponential(2, n_samples),
    "feature_3": np.random.uniform(0, 100, n_samples),
    "feature_4": np.random.gamma(2, 2, n_samples),
    "category": np.random.choice(["A", "B", "C"], n_samples),
}

df = pd.DataFrame(data)

# Create target with some relationship
df["target"] = (
    0.5 * df["feature_1"]
    + 0.3 * df["feature_2"]
    + np.random.normal(0, 10, n_samples)
)

# Add some missing values
df.loc[df.sample(frac=0.1).index, "feature_1"] = np.nan

print("🚀 AutoFE-X Advanced Analysis")
print("=" * 50)

# 1. Run AutoFEX pipeline
print("\n1️⃣ Running AutoFEX pipeline...")
afx = AutoFEX()
result = afx.process(df.drop("target", axis=1), df["target"])

print(f"✅ Processed {result.original_data.shape[1]} → {result.engineered_features.shape[1]} features")
print(f"⏱️  Processing time: {result.processing_time:.2f}s")

# 2. Create Interactive Dashboard
print("\n2️⃣ Creating interactive dashboard...")
dashboard = InteractiveDashboard(backend="plotly")
fig = dashboard.create_comprehensive_dashboard(
    result,
    save_path="dashboard.html",
    title="AutoFE-X Comprehensive Analysis Dashboard"
)
print("✅ Dashboard saved to 'dashboard.html'")
print("   📊 Includes: Feature importance, data quality, leakage risk, correlations")

# 3. Generate Insights Report
print("\n3️⃣ Generating actionable insights...")
insights = dashboard.create_insights_report(result, save_path="insights_report.html")
print("✅ Insights report saved to 'insights_report.html'")
print(f"\n📋 Summary:")
for key, value in insights["summary"].items():
    print(f"   • {key}: {value}")

if insights["warnings"]:
    print(f"\n⚠️  Warnings ({len(insights['warnings'])}):")
    for warning in insights["warnings"][:3]:
        print(f"   • {warning}")

if insights["recommendations"]:
    print(f"\n💡 Recommendations ({len(insights['recommendations'])}):")
    for rec in insights["recommendations"][:3]:
        print(f"   • {rec}")

# 4. Advanced Statistical Analysis
print("\n4️⃣ Running advanced statistical analysis...")
analyzer = AdvancedStatisticalAnalyzer(alpha=0.05)

# Normality testing
print("\n   📊 Normality Analysis:")
norm_result = analyzer.comprehensive_normality_test(df["feature_1"])
print(f"   • Interpretation: {norm_result['interpretation']}")
print(f"   • Recommendation: {norm_result['recommendation']}")
if "tests" in norm_result:
    for test_name, test_result in norm_result["tests"].items():
        if isinstance(test_result, dict) and "p_value" in test_result:
            sig = "✓" if test_result["significant"] else "✗"
            print(f"   • {test_name}: p={test_result['p_value']:.4f} {sig}")

# Group comparison
print("\n   📊 Group Comparison Analysis:")
group_a = df[df["category"] == "A"]["feature_1"].dropna()
group_b = df[df["category"] == "B"]["feature_1"].dropna()
comp_result = analyzer.comprehensive_comparison_test(group_a, group_b)
print(f"   • Interpretation: {comp_result['interpretation']}")
print(f"   • Recommendation: {comp_result['recommendation']}")
if "effect_sizes" in comp_result:
    for effect_name, effect_value in comp_result["effect_sizes"].items():
        if isinstance(effect_value, (int, float)):
            print(f"   • {effect_name}: {effect_value:.4f}")

# Correlation analysis
print("\n   📊 Advanced Correlation Analysis:")
corr_result = analyzer.correlation_analysis_advanced(
    df.drop("target", axis=1), df["target"]
)
if "target_correlations" in corr_result and "top_features" in corr_result["target_correlations"]:
    print("   • Top correlated features:")
    for feat in corr_result["target_correlations"]["top_features"][:5]:
        print(f"     - {feat['feature']}: {feat['correlation']:.3f} (p={feat['p_value']:.4f})")

# 5. Automated Insights
print("\n5️⃣ Generating automated insights...")
auto_insights = analyzer.automated_insights(df.drop("target", axis=1), df["target"])
print(f"✅ Generated {len(auto_insights.get('recommendations', []))} recommendations")
print(f"⚠️  Found {len(auto_insights.get('warnings', []))} warnings")

if auto_insights["recommendations"]:
    print("\n   💡 Top Recommendations:")
    for rec in auto_insights["recommendations"][:3]:
        print(f"   • {rec}")

# 6. Feature Visualization
print("\n6️⃣ Creating feature visualizations...")
viz = FeatureVisualizer()

# Feature importance plot
importance = result.benchmark_results.get("feature_importance")
if importance is not None and len(importance) > 0:
    viz.plot_feature_importance(importance, top_n=15, save_path="feature_importance.png")
    print("✅ Feature importance plot saved to 'feature_importance.png'")

# Data quality summary
viz.plot_data_quality_summary(
    result.data_quality_report, save_path="data_quality.png"
)
print("✅ Data quality dashboard saved to 'data_quality.png'")

print("\n" + "=" * 50)
print("🎉 Advanced Analysis Complete!")
print("\n📁 Generated Files:")
print("   • dashboard.html - Interactive dashboard")
print("   • insights_report.html - Actionable insights")
print("   • feature_importance.png - Feature importance plot")
print("   • data_quality.png - Data quality dashboard")
print("\n✨ AutoFE-X goes beyond basic libraries with:")
print("   • Intelligent test selection")
print("   • Effect size calculations")
print("   • Automated interpretations")
print("   • Integrated dashboards")
print("   • Actionable recommendations")
