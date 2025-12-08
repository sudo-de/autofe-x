# Real-World Use Cases

Production examples and real-world scenarios.

## E-Commerce Recommendation System

```python
from autofex import AutoFEX

# Load e-commerce data
X = load_ecommerce_features()
y = load_purchase_target()

# Process with AutoFE-X
afx = AutoFEX()
result = afx.process(X, y)

# Use engineered features for recommendation
recommendations = train_recommender(result.engineered_features, y)
```

## Financial Fraud Detection

```python
from autofex import AutoFEX, LeakageDetector

# Load transaction data
X = load_transaction_features()
y = load_fraud_labels()

# Detect leakage first
detector = LeakageDetector()
leakage_report = detector.detect(X, y)

# Process if no leakage
if not leakage_report.has_leakage:
    afx = AutoFEX()
    result = afx.process(X, y)
```

## Healthcare Predictive Modeling

```python
from autofex import AutoFEX, DataProfiler

# Load patient data
X = load_patient_features()
y = load_outcome_target()

# Profile data quality
profiler = DataProfiler()
profile = profiler.profile(X)

# Process with quality checks
afx = AutoFEX()
result = afx.process(X, y)
```

## Time-Series Forecasting

```python
from autofex import AdvancedFeatureEngineer

# Load time-series data
X = load_time_series_features()
y = load_target_series()

# Engineer time-series features
engineer = AdvancedFeatureEngineer(
    include_time_series=True
)
engineered_X = engineer.fit_transform(X, y)
```

## Next Steps

- See [Basic Examples](basic-examples.md)
- Check [Advanced Examples](advanced-examples.md)
- Explore [Guides](../guides/feature-engineering.md)

