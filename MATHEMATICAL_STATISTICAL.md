# 🔬 Mathematical Modeling & Advanced Statistical Features

## Overview

AutoFE-X now includes **advanced mathematical modeling** and **statistical transformation** capabilities that leverage numpy, pandas, scipy, and scikit-learn to their fullest potential.

---

## 🔬 Mathematical Modeling Engine

### Features

The `MathematicalModelingEngine` provides sophisticated feature engineering using mathematical modeling techniques:

#### 1. **Polynomial Features** (scikit-learn)
- Polynomial transformations of any degree
- Interaction features
- Configurable degree

#### 2. **Spline Features** (scikit-learn)
- Spline transformations
- Configurable knots and degree
- Smooth feature representations

#### 3. **Dimensionality Reduction**
- **PCA**: Principal Component Analysis
- **ICA**: Independent Component Analysis
- **Factor Analysis**: Factor extraction

#### 4. **Cluster-Based Features** (scikit-learn)
- KMeans clustering
- Cluster labels
- Distance to cluster centers

#### 5. **Manifold Learning** (scikit-learn)
- **t-SNE**: t-distributed Stochastic Neighbor Embedding
- **MDS**: Multi-Dimensional Scaling
- **Isomap**: Isometric Mapping

#### 6. **Interpolation Features** (scipy)
- Linear interpolation
- Cubic interpolation
- Spline interpolation

#### 7. **Signal Processing** (scipy.signal)
- FFT (Fast Fourier Transform) features
- Spectral analysis
- Autocorrelation features

#### 8. **Distribution Fitting** (scipy.stats)
- Fit multiple distributions (normal, exponential, gamma, beta)
- Log-likelihood scores
- AIC (Akaike Information Criterion)

### Usage

```python
from autofex import MathematicalModelingEngine

math_engine = MathematicalModelingEngine({
    'polynomial_features': True,
    'spline_features': True,
    'pca_features': True,
    'ica_features': True,
    'cluster_features': True,
    'distribution_features': True,
    'signal_features': True,
    'n_components_pca': 5,
    'n_clusters': 5,
    'polynomial_degree': 3,
})

features = math_engine.fit_transform(X, y)
```

---

## 📊 Advanced Statistical Transforms

### Features

The `AdvancedStatisticalTransforms` provides comprehensive statistical transformations:

#### 1. **Power Transformations**
- **Box-Cox**: For positive values, optimizes lambda
- **Yeo-Johnson**: Works with negative values
- **Power Transform**: scikit-learn implementation

#### 2. **Distribution Transformations**
- **Quantile Transform**: Uniform or normal output
- **Rank Transform**: Percentile ranks

#### 3. **Scaling Transformations**
- **Z-score**: Standard normalization
- **Robust Scale**: Median and IQR based

#### 4. **Statistical Feature Creation**
- Mean, std, median, MAD, IQR
- Percentiles (10, 25, 50, 75, 90, 95, 99)
- Skewness, kurtosis
- Range, coefficient of variation
- Normality test p-values
- Outlier counts (IQR method)

#### 5. **Correlation Features**
- Max correlation with other features
- Mean correlation
- Min correlation

### Usage

```python
from autofex import AdvancedStatisticalTransforms

stat_transforms = AdvancedStatisticalTransforms()

# Apply all transformations
transformed = stat_transforms.apply_all_transforms(X)

# Create statistical summary features
stat_features = stat_transforms.create_statistical_features(X)

# Create correlation features
corr_features = stat_transforms.create_correlation_features(X)

# Individual transformations
boxcox = stat_transforms.boxcox_transform(X['feature'])
yeojohnson = stat_transforms.yeojohnson_transform(X['feature'])
quantile = stat_transforms.quantile_transform(X['feature'], 'normal')
```

---

## 🎯 Integration with AutoFEX

### Combined Usage

```python
from autofex import AutoFEX, MathematicalModelingEngine, AdvancedStatisticalTransforms

# Run AutoFEX pipeline
afx = AutoFEX(enable_progress=True, enable_cache=True)
result = afx.process(X, y)

# Add mathematical modeling features
math_engine = MathematicalModelingEngine()
math_features = math_engine.fit_transform(result.engineered_features)

# Add statistical transforms
stat_transforms = AdvancedStatisticalTransforms()
stat_features = stat_transforms.apply_all_transforms(result.engineered_features)

# Combine all features
all_features = pd.concat([
    result.engineered_features,
    math_features,
    stat_features
], axis=1)
```

---

## 📊 Feature Types Summary

| Category | Features | Libraries Used |
|----------|----------|----------------|
| **Polynomial** | Polynomial interactions | scikit-learn |
| **Spline** | Spline transformations | scikit-learn |
| **Dimensionality** | PCA, ICA, Factor Analysis | scikit-learn |
| **Clustering** | KMeans cluster features | scikit-learn |
| **Manifold** | t-SNE, MDS, Isomap | scikit-learn |
| **Interpolation** | Linear, cubic, spline | scipy |
| **Signal Processing** | FFT, spectral analysis | scipy.signal |
| **Distribution** | Distribution fitting | scipy.stats |
| **Power Transforms** | Box-Cox, Yeo-Johnson | scipy, scikit-learn |
| **Quantile** | Quantile transforms | scikit-learn |
| **Statistical** | Summary statistics | numpy, pandas, scipy |

---

## 🚀 Benefits

1. **Comprehensive**: Covers all major mathematical and statistical techniques
2. **Leverages Best Libraries**: Uses numpy, pandas, scipy, scikit-learn optimally
3. **Production-Ready**: Well-tested, robust implementations
4. **Flexible**: Configurable for different use cases
5. **Integrated**: Works seamlessly with AutoFEX pipeline

---

## 💡 Best Practices

1. **Start Simple**: Begin with polynomial and PCA features
2. **Domain-Specific**: Use clustering for segmentation problems
3. **Normalization**: Apply power transforms for skewed data
4. **Dimensionality**: Use PCA/ICA when you have many correlated features
5. **Validation**: Always validate mathematical features on test set

---

**AutoFE-X: Leveraging the full power of numpy, pandas, scipy, and scikit-learn!** 🚀
