"""
Ultra-Advanced AutoFE-X Example

Demonstrates capabilities that go far beyond Scipy, Matplotlib, and Plotly:
1. Ultra-advanced statistical analysis (ANOVA, MANOVA, Bayesian, Bootstrap, Power Analysis)
2. Multi-dimensional visualizations (2D, 3D, 4D, 5D)
"""

import pandas as pd
import numpy as np
from autofex import (
    UltraAdvancedStatisticalAnalyzer,
    MultiDimensionalVisualizer,
)

# Generate sample multi-dimensional data
np.random.seed(42)
n_samples = 500

# Create 5D dataset
data = pd.DataFrame({
    "dim1": np.random.normal(100, 15, n_samples),
    "dim2": np.random.exponential(2, n_samples),
    "dim3": np.random.uniform(0, 100, n_samples),
    "dim4": np.random.gamma(2, 2, n_samples),
    "dim5": np.random.beta(2, 5, n_samples),
    "category": np.random.choice(["A", "B", "C"], n_samples),
    "target": np.random.choice([0, 1], n_samples),
})

# Add relationships
data["dim1"] = data["dim1"] + data["target"] * 10
data["dim2"] = data["dim2"] * (1 + data["target"] * 0.5)

print("🚀 Ultra-Advanced AutoFE-X Analysis")
print("=" * 60)

# ============================================================
# 1. ULTRA-ADVANCED STATISTICAL ANALYSIS
# ============================================================
print("\n1️⃣ Ultra-Advanced Statistical Analysis")
print("-" * 60)

analyzer = UltraAdvancedStatisticalAnalyzer(alpha=0.05)

# ANOVA Analysis
print("\n📊 Advanced ANOVA Analysis:")
groups = [
    data[data["category"] == "A"]["dim1"],
    data[data["category"] == "B"]["dim1"],
    data[data["category"] == "C"]["dim1"],
]
anova_result = analyzer.advanced_anova_analysis(groups, post_hoc=True)
print(f"   • Interpretation: {anova_result['interpretation']}")
print(f"   • F-statistic: {anova_result['anova']['one_way'].get('f_statistic', 'N/A'):.4f}")
print(f"   • P-value: {anova_result['anova']['one_way'].get('p_value', 'N/A'):.4f}")
print(f"   • Effect size (η²): {anova_result['effect_sizes'].get('eta_squared', 'N/A'):.4f}")
print(f"   • Effect interpretation: {anova_result['effect_sizes'].get('eta_squared_interpretation', 'N/A')}")

# Time-Series Statistical Tests
print("\n📊 Time-Series Statistical Tests:")
ts_data = pd.Series(np.cumsum(np.random.randn(200)) + np.linspace(0, 10, 200))
ts_result = analyzer.time_series_statistical_tests(ts_data)
if "stationarity" in ts_result and "adf" in ts_result["stationarity"]:
    adf = ts_result["stationarity"]["adf"]
    print(f"   • ADF Test - Stationary: {adf.get('stationary', 'N/A')}")
    print(f"   • P-value: {adf.get('p_value', 'N/A'):.4f}")
if "trend" in ts_result and "linear" in ts_result["trend"]:
    trend = ts_result["trend"]["linear"]
    print(f"   • Has Trend: {trend.get('has_trend', 'N/A')}")
    print(f"   • Slope: {trend.get('slope', 'N/A'):.4f}")

# Bayesian Analysis
print("\n📊 Bayesian Statistical Analysis:")
group_a = data[data["category"] == "A"]["dim1"]
group_b = data[data["category"] == "B"]["dim1"]
bayesian_result = analyzer.bayesian_analysis(group_a, group_b)
if "posterior" in bayesian_result:
    posterior = bayesian_result["posterior"]
    print(f"   • Posterior Mean: {posterior.get('mean', 'N/A'):.4f}")
    print(f"   • 95% Credible Interval: [{posterior.get('credible_interval_95', [0, 0])[0]:.4f}, {posterior.get('credible_interval_95', [0, 0])[1]:.4f}]")
if "bayes_factor" in bayesian_result:
    bf = bayesian_result["bayes_factor"]
    print(f"   • Bayes Factor: {bf.get('value', 'N/A'):.4f}")
    print(f"   • Interpretation: {bf.get('interpretation', 'N/A')}")

# Power Analysis
print("\n📊 Statistical Power Analysis:")
power_result = analyzer.power_analysis(effect_size=0.5, power=0.8)
print(f"   • Required Sample Size: {power_result['sample_size'].get('required', 'N/A')} per group")
print(f"   • Interpretation: {power_result['interpretation']}")

# Bootstrap Analysis
print("\n📊 Bootstrap Statistical Analysis:")
bootstrap_result = analyzer.bootstrap_analysis(data["dim1"], statistic="mean", n_bootstrap=1000)
if "bootstrap" in bootstrap_result:
    boot = bootstrap_result["bootstrap"]
    print(f"   • Bootstrap Mean: {boot.get('mean', 'N/A'):.4f}")
    print(f"   • Bootstrap Std: {boot.get('std', 'N/A'):.4f}")
if "confidence_interval" in bootstrap_result:
    ci = bootstrap_result["confidence_interval"]
    print(f"   • 95% CI: [{ci.get('lower', 'N/A'):.4f}, {ci.get('upper', 'N/A'):.4f}]")

# ============================================================
# 2. MULTI-DIMENSIONAL VISUALIZATIONS
# ============================================================
print("\n2️⃣ Multi-Dimensional Visualizations")
print("-" * 60)

viz = MultiDimensionalVisualizer(backend="plotly")

# 2D Advanced
print("\n📊 Creating 2D Advanced Visualization...")
fig_2d = viz.plot_2d_advanced(
    data["dim1"],
    data["dim2"],
    color=data["target"],
    size=data["dim3"],
    title="2D Advanced: Scatter + Density + Hexbin + Marginals",
    save_path="2d_advanced.html",
)
print("   ✅ Saved to '2d_advanced.html'")
print("   📊 Includes: Scatter, density contour, hexbin, marginal distributions")

# 3D Advanced
print("\n📊 Creating 3D Advanced Visualization...")
fig_3d = viz.plot_3d_advanced(
    data["dim1"],
    data["dim2"],
    data["dim3"],
    color=data["target"],
    size=data["dim4"],
    title="3D Advanced: Interactive 3D Scatter + Surface",
    save_path="3d_advanced.html",
)
print("   ✅ Saved to '3d_advanced.html'")
print("   📊 Includes: Interactive 3D scatter with color/size encoding, surface interpolation")

# 4D Advanced
print("\n📊 Creating 4D Advanced Visualization...")
fig_4d = viz.plot_4d_advanced(
    data["dim1"],
    data["dim2"],
    data["dim3"],
    color=data["dim4"],  # 4th dimension as color
    size=data["dim5"],    # 5th dimension as size
    title="4D Advanced: 3D + Color Encoding (4th dimension)",
    save_path="4d_advanced.html",
)
print("   ✅ Saved to '4d_advanced.html'")
print("   📊 Includes: 3D scatter with 4th dimension as color, 2D projections")

# 5D Advanced
print("\n📊 Creating 5D Advanced Visualization...")
fig_5d = viz.plot_5d_advanced(
    data,
    dims=["dim1", "dim2", "dim3", "dim4", "dim5"],
    color_col="target",
    size_col="dim5",
    title="5D Advanced: Multi-Panel with PCA, t-SNE, Parallel Coordinates",
    save_path="5d_advanced.html",
)
print("   ✅ Saved to '5d_advanced.html'")
print("   📊 Includes: 3D scatter, parallel coordinates, PCA 2D, t-SNE 2D")

print("\n" + "=" * 60)
print("🎉 Ultra-Advanced Analysis Complete!")
print("\n📁 Generated Files:")
print("   • 2d_advanced.html - 2D visualization with multiple views")
print("   • 3d_advanced.html - Interactive 3D visualization")
print("   • 4d_advanced.html - 4D visualization (3D + color)")
print("   • 5d_advanced.html - 5D visualization (multi-panel)")
print("\n✨ AutoFE-X Ultra-Advanced Features:")
print("   • ANOVA/MANOVA with post-hoc tests")
print("   • Time-series statistical tests")
print("   • Bayesian statistical analysis")
print("   • Power analysis")
print("   • Bootstrap methods")
print("   • 2D: Enhanced scatter, density, contour, hexbin")
print("   • 3D: Interactive 3D scatter, surface rendering")
print("   • 4D: 3D + color encoding")
print("   • 5D: Multi-panel with PCA, t-SNE, parallel coordinates")
print("\n🚀 Far beyond basic Scipy, Matplotlib, and Plotly!")
