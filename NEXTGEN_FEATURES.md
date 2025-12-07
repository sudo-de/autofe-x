# 🚀 AutoFE-X NextGen Features

## Overview

NextGen improvements that elevate AutoFE-X to enterprise-grade feature engineering with advanced techniques, intelligent selection, and comprehensive visualization.

---

## ✨ New Capabilities

### 1. Advanced Feature Engineering (`AdvancedFeatureEngineer`)

**Statistical Aggregations:**
- Group-based aggregations (mean, std, min, max, median per group)
- Global aggregations across entire dataset
- Cross-feature statistical relationships

**Time-Series Features:**
- Lag features (1, 2, 3, 7 periods)
- Rolling statistics (mean, std over windows)
- Difference features (1st and 7th order differences)
- Automatic time column detection

**Advanced Binning:**
- Quantile-based binning
- Uniform binning
- K-means binning
- One-hot encoded bins

**Cross-Feature Interactions:**
- Ratio features (division)
- Product features (multiplication)
- Sum features (addition)
- Difference features (subtraction)

**Rank-Based Features:**
- Percentile ranks
- Standard ranks
- Useful for non-parametric transformations

**Domain-Specific Features:**
- Financial: Returns, volatility
- Spatial: Distance calculations
- Temporal: Cyclical encoding (sin/cos)

### 2. Advanced Feature Selection (`AdvancedFeatureSelector`)

**Multiple Selection Strategies:**
- **L1 Regularization (Lasso)**: Sparse feature selection
- **Recursive Feature Elimination (RFE)**: Iterative removal
- **Variance Threshold**: Remove low-variance features
- **Correlation-Based**: Remove highly correlated features

**Ensemble Selection:**
- Combines multiple strategies
- Voting-based feature selection
- Configurable voting thresholds

### 3. Feature Visualization (`FeatureVisualizer`)

**Visualization Capabilities:**
- Feature importance plots (top N features)
- Data quality dashboards
- Leakage risk visualizations
- Feature lineage graphs (NetworkX)

**Backends:**
- Matplotlib (default)
- Plotly (interactive, if available)

---

## 📚 Usage Examples

### Example 1: Advanced Feature Engineering

```python
from autofex import AdvancedFeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Use advanced feature engineering
advanced_fe = AdvancedFeatureEngineer({
    'statistical_aggregations': True,
    'time_series_features': True,
    'advanced_binning': True,
    'cross_features': True,
    'n_bins': 5
})

# Create advanced features
X_advanced = advanced_fe.fit_transform(X, y)

print(f"Original: {X.shape[1]} features")
print(f"Advanced: {X_advanced.shape[1]} features")
```

### Example 2: Advanced Feature Selection

```python
from autofex import AdvancedFeatureSelector
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Use advanced feature selection
selector = AdvancedFeatureSelector({
    'strategies': ['l1', 'rfe', 'variance', 'correlation'],
    'n_features': 50,
    'cv_folds': 5
})

# Ensemble selection
selected_features = selector.select_features_ensemble(X, y, voting_threshold=0.5)

print(f"Selected {len(selected_features)} features from {X.shape[1]} original")
X_selected = X[selected_features]
```

### Example 3: Feature Visualization

```python
from autofex import FeatureVisualizer, AutoFEX
import pandas as pd

# Load data and run AutoFEX
df = pd.read_csv('your_data.csv')
y = df['target']
X = df.drop('target', axis=1)

afx = AutoFEX()
result = afx.process(X, y)

# Create visualizations
viz = FeatureVisualizer()

# Plot feature importance
importance_scores = result.benchmark_results.get('feature_importance', pd.Series())
if len(importance_scores) > 0:
    viz.plot_feature_importance(importance_scores, top_n=20, save_path='importance.png')

# Plot data quality summary
viz.plot_data_quality_summary(result.data_quality_report, save_path='quality.png')

# Plot leakage risk
viz.plot_leakage_risk(result.leakage_report, save_path='leakage.png')

# Plot feature lineage
viz.plot_feature_lineage_graph(result.feature_lineage, save_path='lineage.png')
```

### Example 4: Complete NextGen Pipeline

```python
from autofex import (
    AutoFEX, 
    AdvancedFeatureEngineer, 
    AdvancedFeatureSelector,
    FeatureVisualizer
)
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Step 1: Advanced feature engineering
advanced_fe = AdvancedFeatureEngineer({
    'statistical_aggregations': True,
    'time_series_features': True,
    'cross_features': True
})
X_advanced = advanced_fe.fit_transform(X, y)

# Step 2: Advanced feature selection
selector = AdvancedFeatureSelector({
    'strategies': ['l1', 'rfe'],
    'n_features': 100
})
selected_features = selector.select_features_ensemble(X_advanced, y)
X_final = X_advanced[selected_features]

# Step 3: Run full AutoFEX pipeline on selected features
afx = AutoFEX()
result = afx.process(X_final, y)

# Step 4: Visualize results
viz = FeatureVisualizer()
viz.plot_feature_importance(
    result.benchmark_results.get('feature_importance', pd.Series()),
    top_n=20
)
```

---

## 🎯 Performance Improvements

### Parallel Processing (Coming Soon)

```python
# Future enhancement
afx = AutoFEX(parallel=True, n_jobs=4)
result = afx.process(X, y)  # Faster processing
```

### Caching (Coming Soon)

```python
# Future enhancement
afx = AutoFEX(enable_cache=True)
result1 = afx.process(X, y)  # Computes
result2 = afx.process(X, y)  # Uses cache
```

---

## 📊 Comparison: Classic vs NextGen

| Feature | Classic | NextGen |
|---------|---------|---------|
| Transformations | Basic (log, sqrt, etc.) | Advanced (statistical, time-series, domain-specific) |
| Feature Selection | Mutual info, F-test | L1, RFE, Ensemble, Correlation |
| Visualization | None | Comprehensive plots and dashboards |
| Time-Series | No | Yes (lags, rolling stats) |
| Domain Features | No | Yes (financial, spatial, temporal) |
| Binning | No | Yes (quantile, uniform, kmeans) |
| Cross-Features | Polynomial only | Ratios, products, sums, differences |

---

## 🔧 Configuration

### Advanced Feature Engineering Config

```python
config = {
    'statistical_aggregations': True,
    'time_series_features': True,
    'advanced_binning': True,
    'cross_features': True,
    'n_bins': 5,
    'max_interactions': 10
}
```

### Advanced Feature Selection Config

```python
config = {
    'strategies': ['l1', 'rfe', 'variance', 'correlation'],
    'n_features': 'auto',  # or specific number
    'cv_folds': 5,
    'voting_threshold': 0.5
}
```

---

## 🚀 Getting Started with NextGen

```python
# Install with NextGen dependencies
pip install autofex[viz]

# Import NextGen features
from autofex import (
    AdvancedFeatureEngineer,
    AdvancedFeatureSelector,
    FeatureVisualizer
)

# Use in your pipeline
advanced_fe = AdvancedFeatureEngineer()
selector = AdvancedFeatureSelector()
viz = FeatureVisualizer()
```

---

## 📈 Benefits

1. **More Powerful Features**: Statistical aggregations, time-series, domain-specific
2. **Better Selection**: Multiple strategies with ensemble voting
3. **Visual Insights**: Comprehensive plots for analysis
4. **Domain Expertise**: Built-in features for financial, spatial, temporal data
5. **Production Ready**: All features tested and optimized

---

**NextGen AutoFE-X: Taking feature engineering to the next level!** 🚀
