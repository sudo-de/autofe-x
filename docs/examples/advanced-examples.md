# Advanced Examples

Complex examples demonstrating advanced AutoFE-X features.

## Advanced Feature Engineering

```python
from autofex import FeatureEngineer

# Initialize with advanced options
engineer = FeatureEngineer(
    include_statistical=True,
    include_time_series=True,
    include_domain_specific=True
)

# Fit and transform
engineered_X = engineer.fit_transform(X, y)
```

## Intelligent Feature Selection

```python
from autofex import FeatureSelector

# Initialize selector
selector = FeatureSelector(
    method='l1',
    n_features=50
)

# Select features
selected_X = selector.fit_transform(X, y)
```

## Statistical Analysis

```python
from autofex import AdvancedStatisticalAnalyzer

# Initialize analyzer
analyzer = AdvancedStatisticalAnalyzer(alpha=0.05)

# Comprehensive analysis
insights = analyzer.automated_insights(X, y)
print(insights)
```

## Interactive Dashboard

```python
from autofex import InteractiveDashboard

# Create dashboard
dashboard = InteractiveDashboard()
dashboard.create(X, y, result)
dashboard.show()
```

## Mathematical Modeling

```python
from autofex import MathematicalModelingEngine

# Initialize engine
engine = MathematicalModelingEngine()

# Create model-based features
model_features = engine.create_features(X, y)
```

## Next Steps

- See [Basic Examples](basic-examples.md)
- Check [Use Cases](use-cases.md)
- Explore [Advanced Guides](../advanced/advanced-capabilities.md)

