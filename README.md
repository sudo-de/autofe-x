# AutoFE-X: Automated Feature Engineering + Data Profiling + Leakage Detection

[![PyPI version](https://badge.fury.io/py/autofex.svg)](https://pypi.org/project/autofex/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/autofe-x/autofe-x/branch/main/graph/badge.svg)](https://codecov.io/gh/autofe-x/autofe-x)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A next-gen toolkit that becomes the brain of any ML pipeline** by combining automated feature engineering, data quality analysis, leakage detection, auto-benchmarking of feature sets, and graph-based feature lineage.

## ✨ Key Features

- 🚀 **Automated Feature Engineering**: Classic mathematical transformations, interactions, and encodings
- 🔍 **Data Profiling**: Comprehensive data quality analysis with outlier detection and statistical summaries
- 🛡️ **Leakage Detection**: Advanced algorithms to detect target leakage, train-test contamination, and statistical anomalies
- 📊 **Auto-Benchmarking**: Automatically compare feature sets across multiple models with ablation studies
- 🔗 **Graph-based Lineage**: Track feature transformations and dependencies with full provenance
- ⚡ **Lightweight & Fast**: Minimal dependencies, optimized for performance
- 🎯 **Interpretable**: No black-box LLMs, full transparency in feature engineering decisions

### 🚀 NextGen Features (v0.2.0+)

- 🔬 **Advanced Feature Engineering**: Statistical aggregations, time-series features, domain-specific transformations
- 🎯 **Intelligent Feature Selection**: L1 regularization, RFE, ensemble selection with voting
- 📊 **Comprehensive Visualization**: Feature importance plots, data quality dashboards, lineage graphs
- 📈 **Interactive Dashboards**: Multi-panel dashboards with integrated analysis (beyond basic Plotly)
- 🔬 **Advanced Statistical Analysis**: Multi-test normality analysis, effect sizes, automated interpretations (beyond basic Scipy)
- 🚀 **Ultra-Advanced Statistics**: ANOVA/MANOVA, time-series tests, Bayesian analysis, power analysis, bootstrap methods
- 📊 **Multi-Dimensional Visualization**: 2D, 3D, 4D, 5D visualizations (beyond Matplotlib/Plotly)
- 💡 **Actionable Insights**: Automated recommendations and HTML reports
- ⏱️ **Progress Tracking**: Real-time progress bars, ETA, step-by-step tracking
- 💾 **Intelligent Caching**: Operation-based caching with TTL, size management, cache statistics
- ⚡ **Performance Optimized**: Parallel processing support, intelligent caching
- 🔬 **Mathematical Modeling**: Polynomial, spline, PCA, ICA, clustering, manifold learning features
- 📊 **Advanced Statistical Transforms**: Box-Cox, Yeo-Johnson, quantile, power transforms
- 🐼 **Advanced Pandas Operations**: Rolling windows, groupby, datetime, string features, cumulative, differences
- 🔢 **Advanced Numpy Operations**: Array operations, broadcasting, matrix ops, advanced math functions
- 🔬 **Advanced Scipy Operations**: Special functions, distance metrics, optimization, signal processing, integration
- 🧠 **Intelligent Orchestration**: Automatic feature engineering selection based on data characteristics
- 📊 **Feature Quality Scoring**: Multi-dimensional quality scoring (predictive power, stability, uniqueness, efficiency)
- 💡 **Intelligent Recommendations**: Automatic transformation and strategy recommendations

## 🏗️ Architecture

```
AutoFE-X
├── 🔧 Feature Engineering (classic + interactions)
├── 🔍 Data Profiling (quality + outliers)
├── 🛡️ Leakage Detection (target + contamination)
├── 📊 Benchmarking (auto-compare + ablation)
└── 🔗 Lineage Tracking (graph-based provenance)
```

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install autofex

# Or install from source for development
git clone https://github.com/autofe-x/autofe-x.git
cd autofe-x
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv')['target']

# Initialize AutoFEX
afx = AutoFEX()

# Run complete pipeline
result = afx.process(X, y)

# Access results
print("Data Quality Issues:", result.data_quality_report)
print("Leakage Warnings:", result.leakage_report)
print("Engineered Features Shape:", result.engineered_features.shape)
print("Top Feature Sets:", result.benchmark_results['best_configurations'])
```

### Individual Components

```python
from autofex import FeatureEngineer, DataProfiler, LeakageDetector

# Feature engineering only
fe = FeatureEngineer()
X_engineered = fe.fit_transform(X, y)

# Data profiling only
profiler = DataProfiler()
report = profiler.analyze(X, y)

# Leakage detection only
detector = LeakageDetector()
leakage_report = detector.detect(X, y)
```

### NextGen Features

```python
from autofex import (
    AdvancedFeatureEngineer,
    AdvancedFeatureSelector,
    FeatureVisualizer,
    InteractiveDashboard,
    AdvancedStatisticalAnalyzer,
    UltraAdvancedStatisticalAnalyzer,
    MultiDimensionalVisualizer,
    MathematicalModelingEngine,
    AdvancedStatisticalTransforms,
    AdvancedPandasOperations,
    AdvancedNumpyOperations,
    AdvancedScipyOperations,
    IntelligentOrchestrator,
    FeatureQualityScorer,
    FeatureEngineeringRecommender
)

# Advanced feature engineering
advanced_fe = AdvancedFeatureEngineer({
    'statistical_aggregations': True,
    'time_series_features': True,
    'cross_features': True
})
X_advanced = advanced_fe.fit_transform(X, y)

# Intelligent feature selection
selector = AdvancedFeatureSelector({
    'strategies': ['l1', 'rfe', 'variance'],
    'n_features': 50
})
selected_features = selector.select_features_ensemble(X_advanced, y)

# Visualization
viz = FeatureVisualizer()
viz.plot_feature_importance(importance_scores, top_n=20)
viz.plot_data_quality_summary(quality_report)

# Interactive Dashboard (beyond basic Plotly)
dashboard = InteractiveDashboard(backend="plotly")
dashboard.create_comprehensive_dashboard(result, save_path="dashboard.html")
insights = dashboard.create_insights_report(result, save_path="insights.html")

# Advanced Statistical Analysis (beyond basic Scipy)
analyzer = AdvancedStatisticalAnalyzer()
norm_result = analyzer.comprehensive_normality_test(X['feature'])
comp_result = analyzer.comprehensive_comparison_test(group1, group2)
corr_analysis = analyzer.correlation_analysis_advanced(X, y)
auto_insights = analyzer.automated_insights(X, y)
```

## 📖 Detailed Usage

### Feature Engineering

```python
from autofex import FeatureEngineer

# Create feature engineer
fe = FeatureEngineer({
    'numeric_transforms': ['log', 'sqrt', 'standardize'],
    'categorical_transforms': ['one_hot', 'target_encode'],
    'interaction_degree': 2
})

# Transform features
X_engineered = fe.fit_transform(X, y)
```

### Data Profiling

```python
from autofex import DataProfiler

profiler = DataProfiler()
report = profiler.analyze(X, y)

print("Missing Values:", report['missing_values'])
print("Outliers:", report['outliers'])
print("Correlations:", report['correlations'])
```

### Leakage Detection

```python
from autofex import LeakageDetector

detector = LeakageDetector()
leakage_report = detector.detect(X, y, X_test)

print("Target Leakage:", leakage_report['target_leakage'])
print("Overall Risk:", leakage_report['overall_assessment']['risk_level'])
```

### Auto-Benchmarking

```python
from autofex import FeatureBenchmarker

benchmarker = FeatureBenchmarker()
results = benchmarker.benchmark_features(X, y)

print("Best Configuration:", results['best_configurations']['best_overall'])
print("Feature Importance:", results['feature_importance'])
```

### Feature Lineage

```python
from autofex import FeatureLineageTracker

tracker = FeatureLineageTracker()
tracker.start_session(X.columns.tolist())
tracker.add_transformation('log_transform', ['feature1'], ['feature1_log'])

lineage = tracker.get_lineage_graph()
print("Feature Dependencies:", tracker.get_feature_dependencies('feature1_log'))
```

## 🔧 Installation

### From PyPI

```bash
pip install autofex
```

### From Source

```bash
git clone https://github.com/autofe-x/autofe-x.git
cd autofe-x
pip install -e .
```

### Development Setup

```bash
pip install -e ".[dev]"
```

## 🤖 CI/CD Pipeline

AutoFE-X uses GitHub Actions for comprehensive continuous integration and deployment:

### Pipeline Features
- **Multi-Python Testing**: Compatible with Python 3.8-3.15
- **Code Quality**: Automated Black formatting, flake8 linting, mypy type checking
- **Security Scanning**: Bandit security analysis and Safety vulnerability checks
- **Test Coverage**: Comprehensive test suite with coverage reporting
- **Package Building**: Automated build and PyPI validation
- **Release Publishing**: Automatic PyPI publishing on GitHub releases

### Pipeline Flow
```
Push/PR → Test → Lint → Security → Build → Publish (releases only)
```

### Setting up PyPI Publishing
1. Create a PyPI account and generate an API token
2. Add repository secrets:
   - `PYPI_USERNAME`: Your PyPI username
   - `PYPI_PASSWORD`: Your PyPI API token
3. Create a release on GitHub to trigger PyPI publishing

### Local Development
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=autofex

# Code quality checks
black autofex/
flake8 autofex/
mypy autofex/
```

## 📊 Example Output

```
Data Quality Report:
├── Missing Values: 2.3% overall
├── Outliers Detected: 45 features with outliers
├── High Correlations: 3 feature pairs (>0.95)
└── Data Types: 12 numeric, 5 categorical

Leakage Detection:
├── Target Leakage: 0 high-risk features
├── Statistical Anomalies: 2 suspicious patterns
├── Train-Test Contamination: Low risk
└── Overall Assessment: Low risk

Feature Engineering:
├── Original Features: 17
├── Engineered Features: 42
├── Transformations Applied: log, sqrt, interactions
└── Feature Selection: Top 35 features retained

Benchmarking Results:
├── Best Model: RandomForest (0.87 accuracy)
├── Best Feature Set: top_25_features
├── Performance Gain: +12% vs baseline
└── Ablation Impact: Feature importance stable
```

## 🎯 Use Cases

- **Kaggle Competitions**: Rapid feature engineering and leakage detection
- **Production ML**: Automated feature pipeline with quality monitoring
- **Data Science Teams**: Standardized feature engineering workflows
- **AutoML Systems**: Feature engineering component for automated pipelines
- **Model Debugging**: Identify why models perform differently across datasets

## 🔬 Advanced Features

### Beyond Basic Libraries: Integrated Analysis & Visualization

AutoFE-X goes **beyond basic Scipy, Matplotlib, and Plotly** by providing:

#### 🎯 Interactive Dashboards (Beyond Plotly)
- **Multi-panel dashboards** with integrated statistical analysis
- **Actionable insights** automatically generated from results
- **HTML report generation** with recommendations
- **Auto-interpretation** of statistical results (not just plots)

```python
from autofex import InteractiveDashboard

dashboard = InteractiveDashboard(backend="plotly")
fig = dashboard.create_comprehensive_dashboard(result, save_path="dashboard.html")
insights = dashboard.create_insights_report(result, save_path="insights.html")

# Insights include:
# - Automated recommendations
# - Warnings and opportunities
# - Performance summaries
# - Actionable next steps
```

#### 🔬 Advanced Statistical Analysis (Beyond Scipy)
- **Multi-test normality analysis** (Shapiro-Wilk, D'Agostino, KS, Jarque-Bera)
- **Effect size calculations** (Cohen's d, rank-biserial correlation)
- **Automated test selection** (parametric vs non-parametric)
- **Comprehensive interpretations** (not just p-values)
- **Automated insights** and recommendations

```python
from autofex import AdvancedStatisticalAnalyzer

analyzer = AdvancedStatisticalAnalyzer(alpha=0.05)

# Comprehensive normality testing
norm_result = analyzer.comprehensive_normality_test(X['feature'])
# Returns: interpretation, recommendation, multiple test results

# Group comparison with effect sizes
comp_result = analyzer.comprehensive_comparison_test(group1, group2)
# Returns: test results, Cohen's d, interpretation, recommendation

# Advanced correlation analysis
corr_analysis = analyzer.correlation_analysis_advanced(X, y)
# Returns: feature correlations, target correlations, multicollinearity warnings

# Automated insights
auto_insights = analyzer.automated_insights(X, y)
# Returns: recommendations, warnings, statistical characteristics

# Mathematical Modeling (numpy, pandas, scipy, scikit-learn)
math_engine = MathematicalModelingEngine({
    'polynomial_features': True,
    'pca_features': True,
    'cluster_features': True
})
math_features = math_engine.fit_transform(X, y)
# Creates: polynomial, spline, PCA, ICA, clustering, manifold features

# Advanced Statistical Transforms
stat_transforms = AdvancedStatisticalTransforms()
transformed_features = stat_transforms.apply_all_transforms(X)
# Applies: Box-Cox, Yeo-Johnson, quantile, power, rank transforms
stat_features = stat_transforms.create_statistical_features(X)
# Creates: mean, std, percentiles, skewness, kurtosis, outlier counts

# Advanced Pandas Operations
pandas_ops = AdvancedPandasOperations()
pandas_features = pandas_ops.fit_transform(X)
# Creates: rolling windows, datetime, string, cumulative, difference features

# Advanced Numpy Operations
numpy_ops = AdvancedNumpyOperations()
numpy_features = numpy_ops.fit_transform(X)
# Creates: array stats, broadcasting, matrix ops, advanced math functions

# Advanced Scipy Operations
scipy_ops = AdvancedScipyOperations()
scipy_features = scipy_ops.fit_transform(X)
# Creates: special functions, distance metrics, optimization, signal processing

# Intelligent Orchestration (automatic selection)
orchestrator = IntelligentOrchestrator()
intelligent_features = orchestrator.fit_transform(X, y)
# Automatically selects best feature engineering techniques

# Feature Quality Scoring
quality_scorer = FeatureQualityScorer()
quality_scores = quality_scorer.score_all_features(X, y)
top_features = quality_scorer.get_top_features(X, y, n_features=50)
# Scores features on: predictive power, stability, uniqueness, efficiency

# Intelligent Recommendations
recommender = FeatureEngineeringRecommender()
recommendations = recommender.recommend_feature_engineering(X, y)
auto_config = recommender.get_auto_config(X, y)
# Provides: transformation recommendations, strategy suggestions, auto-config
```

#### 🚀 Ultra-Advanced Statistical Analysis (Beyond Scipy)

```python
from autofex import UltraAdvancedStatisticalAnalyzer

analyzer = UltraAdvancedStatisticalAnalyzer()

# Advanced ANOVA with post-hoc tests and effect sizes
groups = [group1, group2, group3]
anova_result = analyzer.advanced_anova_analysis(groups, post_hoc=True)
# Returns: F-statistic, p-value, effect size (η²), post-hoc tests, interpretations

# Time-series statistical tests
ts_result = analyzer.time_series_statistical_tests(time_series)
# Returns: ADF test (stationarity), trend tests, autocorrelation analysis

# Bayesian statistical analysis
bayesian_result = analyzer.bayesian_analysis(group1, group2)
# Returns: Posterior distribution, credible intervals, Bayes factor

# Power analysis
power_result = analyzer.power_analysis(effect_size=0.5, power=0.8)
# Returns: Required sample size, achieved power

# Bootstrap analysis
bootstrap_result = analyzer.bootstrap_analysis(data, statistic="mean")
# Returns: Bootstrap statistics, confidence intervals
```

**Key Features:**
- **ANOVA/MANOVA**: Multi-group comparisons with post-hoc tests and effect sizes
- **Time-Series Tests**: ADF test, trend detection, autocorrelation analysis
- **Bayesian Methods**: Posterior distributions, credible intervals, Bayes factors
- **Power Analysis**: Sample size calculations, power estimation
- **Bootstrap Methods**: Non-parametric confidence intervals

#### 📊 Multi-Dimensional Visualization (Beyond Matplotlib/Plotly)

```python
from autofex import MultiDimensionalVisualizer

viz = MultiDimensionalVisualizer(backend="plotly")

# 2D Advanced: Scatter + Density + Hexbin + Marginals
viz.plot_2d_advanced(x, y, color=target, size=feature3, save_path="2d.html")

# 3D Advanced: Interactive 3D scatter + surface rendering
viz.plot_3d_advanced(x, y, z, color=target, size=feature4, save_path="3d.html")

# 4D Advanced: 3D + color encoding (4th dimension)
viz.plot_4d_advanced(x, y, z, color=dim4, size=dim5, save_path="4d.html")

# 5D Advanced: Multi-panel with PCA, t-SNE, parallel coordinates
viz.plot_5d_advanced(
    data,
    dims=["dim1", "dim2", "dim3", "dim4", "dim5"],
    color_col="target",
    size_col="dim5",
    save_path="5d.html"
)
```

**Key Features:**
- **2D**: Enhanced scatter plots, density contours, hexbin, marginal distributions
- **3D**: Interactive 3D scatter with surface interpolation
- **4D**: 3D visualization with 4th dimension as color encoding
- **5D**: Multi-panel views with PCA, t-SNE, parallel coordinates

#### 📊 Key Advantages Over Raw Libraries

| Feature | Raw Scipy/Matplotlib/Plotly | AutoFE-X |
|---------|---------------------------|----------|
| Statistical Testing | Single test, manual interpretation | Multi-test with automated interpretation |
| ANOVA | Basic f_oneway | ANOVA + post-hoc + effect sizes (η²) |
| MANOVA | Not available | Full MANOVA support |
| Time-Series Tests | Manual ADF | ADF + trend + autocorrelation |
| Bayesian Analysis | Manual calculation | Posterior + credible intervals + Bayes factor |
| Power Analysis | Manual calculation | Automated sample size + power |
| Bootstrap | Manual loops | Automated bootstrap + CI |
| Effect Sizes | Manual calculation | Automatic (Cohen's d, rank-biserial, η²) |
| Test Selection | Manual choice | Automated (parametric vs non-parametric) |
| 2D Visualization | Basic scatter | Scatter + density + hexbin + marginals |
| 3D Visualization | Basic 3D scatter | 3D + surface + interactive |
| 4D Visualization | Manual encoding | Integrated 4D visualization |
| 5D Visualization | Not available | Multi-panel with PCA, t-SNE, parallel coords |
| Visualizations | Individual plots | Integrated dashboards with insights |
| Insights | Manual analysis | Automated recommendations |
| Reports | Manual creation | HTML report generation |
| Interpretations | Just p-values | Full interpretations + recommendations |

### Custom Transformations

```python
class CustomTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Your custom logic
        return X_transformed

# Use in pipeline
afx = AutoFEX()
afx.feature_engineer.add_transformer('custom', CustomTransformer())
```

### Integration with Existing Pipelines

```python
from sklearn.pipeline import Pipeline
from autofex import FeatureEngineer

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

## 📈 Performance

- **Speed**: Processes 100K rows × 50 features in ~2 seconds
- **Memory**: Minimal memory footprint (< 2x original data)
- **Scalability**: Handles datasets up to 1M rows efficiently
- **Accuracy**: Feature engineering decisions based on statistical validation
- **Caching**: Repeated operations are cached, providing 2-10x speedup on subsequent runs
- **Progress Tracking**: Real-time feedback with ETA for long-running operations

### Progress Tracking & Caching

```python
from autofex import AutoFEX

# Enable progress tracking and caching
afx = AutoFEX(
    enable_progress=True,      # Real-time progress bars
    enable_cache=True,         # Intelligent caching
    cache_dir=".autofex_cache", # Cache directory
    cache_ttl=3600,            # 1 hour TTL
)

# First run - computes and caches
result1 = afx.process(X, y)  # Shows progress bar with ETA

# Second run - uses cache (much faster!)
result2 = afx.process(X, y)  # 2-10x faster with cache hits

# Cache management
afx.cache.clear()                    # Clear all cache
afx.cache.clear(operation="profiling")  # Clear specific operation
cache_stats = afx.cache.get_stats()  # Get cache statistics
```

**Progress Tracking Features:**
- Real-time progress bars with percentage
- ETA (Estimated Time Remaining) calculation
- Step-by-step progress updates
- Time statistics (total time, average step time)
- Real-time metric tracking

**Caching Features:**
- Operation-based caching (profiling, leakage detection, feature engineering, benchmarking)
- TTL (Time-To-Live) support for cache expiration
- Automatic size management (evicts oldest entries when limit reached)
- Cache statistics and management
- Selective cache clearing by operation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by featuretools, pandas-profiling, and scikit-learn
- Built for the data science community to solve real ML engineering challenges

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/autofe-x/autofe-x/issues)
- **Discussions**: [GitHub Discussions](https://github.com/autofe-x/autofe-x/discussions)
- **Documentation**: [Read the Docs](https://autofe-x.readthedocs.io/)

---

**Ready to supercharge your ML pipelines?** 🚀

```bash
pip install autofex
```

*AutoFE-X: Because feature engineering shouldn't be the bottleneck in your ML workflow.*
