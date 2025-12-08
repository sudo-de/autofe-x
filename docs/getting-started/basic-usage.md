# Basic Usage

This guide covers the core concepts and common usage patterns of AutoFE-X.

## Core Concepts

### AutoFEX Pipeline

AutoFE-X follows a simple pipeline:

1. **Feature Engineering** - Automatically create new features
2. **Data Profiling** - Analyze data quality
3. **Leakage Detection** - Identify potential data leakage
4. **Benchmarking** - Compare feature sets across models
5. **Lineage Tracking** - Track feature transformations

### Basic Workflow

```python
from autofex import AutoFEX
import pandas as pd

# 1. Prepare data
X = pd.DataFrame(...)  # Features
y = pd.Series(...)     # Target

# 2. Initialize
afx = AutoFEX()

# 3. Process
result = afx.process(X, y)

# 4. Use results
features = result.engineered_features
```

## Common Use Cases

### Feature Engineering Only

```python
from autofex import FeatureEngineer

engineer = FeatureEngineer()
engineered_X = engineer.fit_transform(X, y)
```

### Data Profiling Only

```python
from autofex import DataProfiler

profiler = DataProfiler()
report = profiler.profile(X)
print(report.summary)
```

### Leakage Detection Only

```python
from autofex import LeakageDetector

detector = LeakageDetector()
leakage_report = detector.detect(X, y)
print(leakage_report.summary)
```

### Benchmarking Feature Sets

```python
from autofex import FeatureBenchmarker

benchmarker = FeatureBenchmarker()
results = benchmarker.benchmark(X, y, feature_sets=['original', 'engineered'])
print(results.comparison)
```

## Configuration

### Customizing AutoFEX

```python
afx = AutoFEX(
    feature_engineering=True,      # Enable feature engineering
    data_profiling=True,           # Enable data profiling
    leakage_detection=True,        # Enable leakage detection
    benchmarking=True,             # Enable benchmarking
    lineage_tracking=True          # Enable lineage tracking
)
```

### Feature Engineering Options

```python
from autofex import FeatureEngineer

engineer = FeatureEngineer(
    include_interactions=True,     # Create interaction features
    include_polynomials=True,      # Create polynomial features
    include_encodings=True,        # Create encoding features
    max_features=100               # Maximum number of features
)
```

### Data Profiling Options

```python
from autofex import DataProfiler

profiler = DataProfiler(
    detect_outliers=True,          # Detect outliers
    check_missing=True,            # Check for missing values
    statistical_summary=True,      # Generate statistical summary
    correlation_analysis=True      # Analyze correlations
)
```

## Working with Results

### Accessing Engineered Features

```python
result = afx.process(X, y)

# Get all engineered features
features = result.engineered_features

# Get feature names
feature_names = result.engineered_features.columns.tolist()

# Get feature importance
importance = result.feature_importance
```

### Understanding Reports

```python
# Profiling report
profiling = result.profiling_report
print(profiling.summary)
print(profiling.outliers)
print(profiling.missing_values)

# Leakage report
leakage = result.leakage_report
print(leakage.summary)
print(leakage.suspicious_features)

# Benchmark results
benchmark = result.benchmark_results
print(benchmark.comparison)
print(benchmark.best_model)
```

## Best Practices

1. **Always check data profiling first** - Understand your data before engineering features
2. **Run leakage detection** - Ensure your features are valid
3. **Use benchmarking** - Compare different feature sets objectively
4. **Review lineage** - Understand how features were created
5. **Start simple** - Begin with default settings, then customize

## Next Steps

- Explore [Feature Engineering Guide](guides/feature-engineering.md)
- Learn about [Advanced Features](advanced/advanced-capabilities.md)
- Check out [Examples](examples/basic-examples.md)

