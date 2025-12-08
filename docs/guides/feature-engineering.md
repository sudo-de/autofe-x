# Feature Engineering Guide

AutoFE-X provides powerful automated feature engineering capabilities.

## Overview

Feature engineering is the process of creating new features from existing data to improve model performance. AutoFE-X automates this process with:

- Mathematical transformations
- Feature interactions
- Polynomial features
- Encoding transformations
- Domain-specific features

## Basic Usage

```python
from autofex import FeatureEngineer
import pandas as pd

# Initialize
engineer = FeatureEngineer()

# Fit and transform
engineered_X = engineer.fit_transform(X, y)

# Or separate steps
engineer.fit(X, y)
engineered_X = engineer.transform(X)
```

## Feature Types

### Mathematical Transformations

```python
engineer = FeatureEngineer(
    include_math_transforms=True
)

# Creates features like:
# - log(x), sqrt(x), exp(x)
# - x^2, x^3
# - abs(x), sign(x)
```

### Feature Interactions

```python
engineer = FeatureEngineer(
    include_interactions=True,
    interaction_depth=2  # Pairwise interactions
)

# Creates features like:
# - feature1 * feature2
# - feature1 / feature2
# - feature1 + feature2
```

### Polynomial Features

```python
engineer = FeatureEngineer(
    include_polynomials=True,
    polynomial_degree=2
)

# Creates features like:
# - x^2, x^3, ...
# - x1 * x2, x1^2 * x2, ...
```

### Encoding Features

```python
engineer = FeatureEngineer(
    include_encodings=True
)

# Creates features like:
# - One-hot encodings
# - Target encodings
# - Frequency encodings
```

## Advanced Feature Engineering

For more advanced features, use `FeatureEngineer`:

```python
from autofex import FeatureEngineer

advanced_engineer = FeatureEngineer(
    include_statistical=True,      # Statistical aggregations
    include_time_series=True,      # Time-series features
    include_domain_specific=True   # Domain-specific features
)
```

### Statistical Aggregations

```python
# Creates features like:
# - mean, median, std per group
# - min, max per group
# - quantiles per group
```

### Time-Series Features

```python
# Creates features like:
# - Lag features
# - Rolling statistics
# - Time-based encodings
```

## Feature Selection

After engineering features, select the best ones:

```python
from autofex import FeatureSelector

selector = FeatureSelector(
    method='l1',           # L1 regularization
    n_features=50          # Select top 50 features
)

selected_features = selector.fit_transform(engineered_X, y)
```

## Best Practices

1. **Start with defaults** - AutoFE-X has sensible defaults
2. **Monitor feature count** - Use `max_features` to limit explosion
3. **Use feature selection** - Remove redundant features
4. **Check feature importance** - Understand which features matter
5. **Validate features** - Use leakage detection to ensure validity

## Examples

See [Feature Engineering Examples](examples/feature-engineering-examples.md) for detailed examples.

## Next Steps

- Learn about [Data Profiling](data-profiling.md)
- Explore [Advanced Features](advanced/advanced-capabilities.md)
- Check [API Reference](api/feature-engineering.md)

