# 🚀 AutoFE-X: Complete Feature Engineering Capabilities

## Overview

AutoFE-X leverages **numpy, pandas, scipy, and scikit-learn** to provide the most comprehensive automated feature engineering toolkit available.

---

## 📚 Complete Feature Engineering Stack

### 1. AutoFEX Core
- Classic mathematical transformations
- Polynomial interactions
- Categorical encoding
- Feature selection
- Data profiling
- Leakage detection
- Benchmarking

### 2. Mathematical Modeling (scikit-learn)
- **Polynomial Features**: Polynomial transformations
- **Spline Features**: Spline transformations
- **PCA**: Principal Component Analysis
- **ICA**: Independent Component Analysis
- **Factor Analysis**: Factor extraction
- **Clustering**: KMeans cluster features
- **Manifold Learning**: t-SNE, MDS, Isomap

### 3. Statistical Transforms (scipy.stats)
- **Box-Cox**: Power transformation for normality
- **Yeo-Johnson**: Power transformation (works with negatives)
- **Quantile Transform**: Uniform/normal distribution
- **Power Transform**: scikit-learn implementation
- **Rank Transform**: Percentile ranks
- **Z-score**: Standard normalization
- **Robust Scale**: Median and IQR based

### 4. Advanced Pandas Operations
- **Rolling Windows**: Mean, std, min, max over windows
- **Expanding Windows**: Cumulative statistics
- **Groupby Aggregations**: Group-based features
- **Pivot Tables**: Cross-tabulation features
- **Crosstab**: Frequency-based features
- **Datetime Features**: Year, month, day, cyclical encoding
- **String Features**: Length, word count, pattern detection
- **Cumulative Features**: Cumsum, cumprod, cummax, cummin
- **Difference Features**: Diff, pct_change

### 5. Advanced Numpy Operations
- **Array Statistics**: Row-wise mean, std, min, max, percentiles
- **Array Norms**: L1, L2 norms
- **Broadcasting**: Distance from mean, std
- **Matrix Operations**: Eigenvalues, dot products
- **Advanced Math**: Trig, hyperbolic, exp, log functions
- **Aggregations**: Comprehensive row-wise statistics

### 6. Advanced Scipy Operations
- **Special Functions**: Gamma, Bessel, error functions
- **Distance Metrics**: Euclidean, Manhattan, Cosine
- **Optimization**: Curve fitting, polynomial fitting
- **Signal Processing**: FFT, spectral analysis, autocorrelation
- **Integration**: Trapezoidal, Simpson's rule

---

## 🎯 Feature Count by Library

| Library | Feature Categories | Example Features |
|---------|-------------------|------------------|
| **numpy** | 5 categories | Array stats, broadcasting, matrix ops, math functions, aggregations |
| **pandas** | 9 categories | Windows, groupby, datetime, strings, cumulative, differences |
| **scipy** | 5 categories | Special functions, distances, optimization, signal, integration |
| **scikit-learn** | 8 categories | Polynomial, spline, PCA, ICA, clustering, manifold, transforms |

---

## 💡 Usage Example

```python
from autofex import (
    AutoFEX,
    MathematicalModelingEngine,
    AdvancedStatisticalTransforms,
    AdvancedPandasOperations,
    AdvancedNumpyOperations,
    AdvancedScipyOperations,
)

# 1. AutoFEX Core
afx = AutoFEX(enable_progress=True, enable_cache=True, n_jobs=-1)
result = afx.process(X, y)

# 2. Mathematical Modeling
math_engine = MathematicalModelingEngine()
math_features = math_engine.fit_transform(result.engineered_features)

# 3. Statistical Transforms
stat_transforms = AdvancedStatisticalTransforms()
stat_features = stat_transforms.apply_all_transforms(result.engineered_features)

# 4. Pandas Operations
pandas_ops = AdvancedPandasOperations()
pandas_features = pandas_ops.fit_transform(X)

# 5. Numpy Operations
numpy_ops = AdvancedNumpyOperations()
numpy_features = numpy_ops.fit_transform(result.engineered_features)

# 6. Scipy Operations
scipy_ops = AdvancedScipyOperations()
scipy_features = scipy_ops.fit_transform(result.engineered_features)

# Combine all
all_features = pd.concat([
    result.engineered_features,
    math_features,
    stat_features,
    pandas_features,
    numpy_features,
    scipy_features
], axis=1)
```

---

## 📊 Feature Engineering Pipeline

```
Original Data
    ↓
AutoFEX Core (classic transforms, interactions)
    ↓
Mathematical Modeling (PCA, clustering, manifold)
    ↓
Statistical Transforms (Box-Cox, quantile, power)
    ↓
Pandas Operations (windows, datetime, strings)
    ↓
Numpy Operations (array ops, broadcasting, math)
    ↓
Scipy Operations (special functions, distances)
    ↓
Combined Feature Set (100s-1000s of features)
```

---

## 🚀 Performance

- **Feature Expansion**: 10-100x original features
- **Processing Speed**: Optimized with parallel processing
- **Memory Efficient**: Smart feature selection and caching
- **Scalable**: Handles datasets up to 1M+ rows

---

## 🎯 Best Practices

1. **Start with Core**: Use AutoFEX core first
2. **Add Mathematical**: Use PCA/clustering for dimensionality
3. **Apply Transforms**: Use statistical transforms for normalization
4. **Leverage Pandas**: Use for time-series and categorical data
5. **Use Numpy**: For array-wide operations
6. **Scipy for Special**: Use for domain-specific features
7. **Feature Selection**: Always select top features after engineering
8. **Validation**: Test on holdout set

---

**AutoFE-X: Maximum feature engineering power with numpy, pandas, scipy, and scikit-learn!** 🚀
