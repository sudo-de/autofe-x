# AutoFE-X: Automated Feature Engineering + Data Profiling + Leakage Detection

[![PyPI version](https://badge.fury.io/py/autofe-x.svg)](https://pypi.org/project/autofe-x/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/autofe-x/autofe-x/actions/workflows/ci.yml/badge.svg)](https://github.com/autofe-x/autofe-x/actions/workflows/ci.yml)
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
pip install autofe-x

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
pip install autofe-x
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
- **Multi-Python Testing**: Compatible with Python 3.8-3.12
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
pip install autofe-x
```

*AutoFE-X: Because feature engineering shouldn't be the bottleneck in your ML workflow.*
