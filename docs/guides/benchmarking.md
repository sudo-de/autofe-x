# Benchmarking Guide

Compare feature sets across multiple models to find the best combination.

## Overview

Benchmarking helps you objectively compare different feature sets and select the best one for your problem.

## Basic Usage

```python
from autofex import FeatureBenchmarker
import pandas as pd

# Initialize benchmarker
benchmarker = FeatureBenchmarker()

# Benchmark feature sets
results = benchmarker.benchmark(
    X, y,
    feature_sets=['original', 'engineered']
)

# View results
print(results.comparison)
print(results.best_model)
```

## Comparing Feature Sets

```python
# Compare multiple feature sets
feature_sets = {
    'original': X_original,
    'engineered': X_engineered,
    'selected': X_selected
}

results = benchmarker.benchmark(X, y, feature_sets)
```

## Model Selection

```python
# Specify models to use
benchmarker = FeatureBenchmarker(
    models=['logistic_regression', 'random_forest', 'xgboost']
)
```

## Cross-Validation

```python
# Configure cross-validation
benchmarker = FeatureBenchmarker(
    cv_folds=5,
    cv_strategy='stratified'
)
```

## Understanding Results

### Comparison Table

```python
# Get comparison results
comparison = results.comparison
print(comparison)
```

### Best Model

```python
# Get best performing model
best = results.best_model
print(f"Best model: {best.name}")
print(f"Best score: {best.score}")
```

### Feature Importance

```python
# Get feature importance
importance = results.feature_importance
print(importance.head())
```

## Ablation Studies

```python
# Run ablation study
ablation = benchmarker.ablation_study(X, y)
print(ablation.results)
```

## Best Practices

1. **Compare systematically** - Test feature sets consistently
2. **Use cross-validation** - Get reliable estimates
3. **Track multiple metrics** - Don't rely on one metric
4. **Consider complexity** - Balance performance and complexity
5. **Document results** - Keep track of what works

## Next Steps

- Learn about [Feature Engineering](feature-engineering.md)
- Explore [Lineage Tracking](lineage-tracking.md)
- Check [API Reference](../api/benchmarking.md)

