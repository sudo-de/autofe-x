# Basic Examples

Simple examples to get you started with AutoFE-X.

## Complete Pipeline Example

```python
import pandas as pd
import numpy as np
from autofex import AutoFEX

# Generate sample data
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
})
y = pd.Series(X['feature1'] * 2 + X['feature2'] + np.random.randn(1000) * 0.1)

# Initialize AutoFEX
afx = AutoFEX()

# Process data
result = afx.process(X, y)

# View results
print(f"Original features: {X.shape[1]}")
print(f"Engineered features: {result.engineered_features.shape[1]}")
print(f"\nTop features:\n{result.feature_importance.head(10)}")
```

## Feature Engineering Only

```python
from autofex import FeatureEngineer

# Initialize
engineer = FeatureEngineer()

# Fit and transform
engineered_X = engineer.fit_transform(X, y)

print(f"Original: {X.shape}")
print(f"Engineered: {engineered_X.shape}")
```

## Data Profiling Only

```python
from autofex import DataProfiler

# Initialize
profiler = DataProfiler()

# Profile data
report = profiler.profile(X)

# View summary
print(report.summary)
```

## Leakage Detection Only

```python
from autofex import LeakageDetector

# Initialize
detector = LeakageDetector()

# Detect leakage
report = detector.detect(X, y)

# Check results
if report.has_leakage:
    print("Leakage detected!")
    print(report.suspicious_features)
else:
    print("No leakage detected")
```

## Benchmarking Only

```python
from autofex import FeatureBenchmarker

# Initialize
benchmarker = FeatureBenchmarker()

# Benchmark feature sets
results = benchmarker.benchmark(
    X, y,
    feature_sets=['original', 'engineered']
)

# View comparison
print(results.comparison)
```

## Next Steps

- See [Advanced Examples](advanced-examples.md)
- Check [Use Cases](use-cases.md)
- Explore [Guides](../guides/feature-engineering.md)

