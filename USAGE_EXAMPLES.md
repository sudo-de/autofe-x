# 📚 AutoFE-X Usage Examples

## Common Errors & Solutions

### ❌ Error 1: Wrong Class Name

```python
from autofex import AutoFEX

fe = AutoFE()  # ❌ NameError: name 'AutoFE' is not defined
```

**✅ Solution:**
```python
from autofex import AutoFEX

fe = AutoFEX()  # ✅ Correct - capital FEX
```

### ❌ Error 2: Wrong Method Name

```python
from autofex import AutoFEX

fe = AutoFEX()
report = fe.fit_transform(df, target="label")  # ❌ AttributeError
```

**✅ Solution - Option 1: Use AutoFEX.process()**
```python
from autofex import AutoFEX
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
y = df['label']  # Extract target
X = df.drop('label', axis=1)  # Features

# Use AutoFEX
fe = AutoFEX()
result = fe.process(X, y)  # ✅ Correct method

# Access engineered features
engineered_df = result.engineered_features
```

**✅ Solution - Option 2: Use FeatureEngineer directly**
```python
from autofex import FeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
y = df['label']
X = df.drop('label', axis=1)

# Use FeatureEngineer directly
fe = FeatureEngineer()
engineered_df = fe.fit_transform(X, y)  # ✅ This works!
```

## Complete Usage Examples

### Example 1: Full AutoFEX Pipeline

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
df = pd.read_csv('your_data.csv')
y = df['target_column']  # Target variable
X = df.drop('target_column', axis=1)  # Features

# Initialize AutoFEX
afx = AutoFEX()

# Run complete pipeline
result = afx.process(X, y)

# Access results
print("Original features:", X.shape[1])
print("Engineered features:", result.engineered_features.shape[1])
print("Data quality report:", result.data_quality_report)
print("Leakage risk:", result.leakage_report['overall_assessment']['risk_level'])
print("Best model:", result.benchmark_results['best_configurations'])

# Use engineered features
X_engineered = result.engineered_features
```

### Example 2: Feature Engineering Only

```python
from autofex import FeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
y = df['label']
X = df.drop('label', axis=1)

# Feature engineering only
fe = FeatureEngineer()
X_engineered = fe.fit_transform(X, y)

print(f"Original: {X.shape[1]} features")
print(f"Engineered: {X_engineered.shape[1]} features")
```

### Example 3: Data Profiling Only

```python
from autofex import DataProfiler
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
y = df['label']
X = df.drop('label', axis=1)

# Data profiling only
profiler = DataProfiler()
report = profiler.analyze(X, y)

print("Missing values:", report['missing_values'])
print("Outliers:", report['outliers'])
print("Correlations:", report['correlations'])
```

### Example 4: Leakage Detection Only

```python
from autofex import LeakageDetector
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
y = df['label']
X = df.drop('label', axis=1)

# Leakage detection only
detector = LeakageDetector()
leakage_report = detector.detect(X, y)

print("Risk level:", leakage_report['overall_assessment']['risk_level'])
print("Leakage candidates:", leakage_report['target_leakage']['leakage_candidates'])
```

## API Reference

### AutoFEX Class

```python
from autofex import AutoFEX

# Initialize
afx = AutoFEX()

# Main method
result = afx.process(X, y, X_test=None)

# Result object contains:
# - result.original_data: Original DataFrame
# - result.engineered_features: Engineered features DataFrame
# - result.data_quality_report: Quality analysis dict
# - result.leakage_report: Leakage detection dict
# - result.benchmark_results: Benchmarking results dict
# - result.feature_lineage: Lineage graph dict
# - result.processing_time: Time taken in seconds
```

### FeatureEngineer Class

```python
from autofex import FeatureEngineer

# Initialize
fe = FeatureEngineer()

# Fit and transform
X_engineered = fe.fit_transform(X, y)

# Or separately
fe.fit(X, y)
X_engineered = fe.transform(X)
```

## Quick Reference

| Task | Class | Method |
|------|-------|--------|
| Full pipeline | `AutoFEX` | `process()` |
| Feature engineering | `FeatureEngineer` | `fit_transform()` |
| Data profiling | `DataProfiler` | `analyze()` |
| Leakage detection | `LeakageDetector` | `detect()` |
| Benchmarking | `FeatureBenchmarker` | `benchmark_features()` |
| Lineage tracking | `FeatureLineageTracker` | `start_session()`, `add_transformation()` |
