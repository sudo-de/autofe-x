# Leakage Detection Guide

Detect and prevent data leakage in your machine learning pipeline.

## Overview

Data leakage occurs when information from the target variable leaks into features, leading to overly optimistic performance estimates.

## Basic Usage

```python
from autofex import LeakageDetector
import pandas as pd

# Initialize detector
detector = LeakageDetector()

# Detect leakage
report = detector.detect(X, y)

# Check results
print(report.summary)
print(report.suspicious_features)
```

## Detection Methods

### Correlation-Based Detection

```python
detector = LeakageDetector(
    methods=['correlation'],
    alpha=0.05
)
```

### Information-Theoretic Detection

```python
detector = LeakageDetector(
    methods=['mutual_information'],
    alpha=0.05
)
```

### Statistical Anomaly Detection

```python
detector = LeakageDetector(
    methods=['statistical'],
    alpha=0.05
)
```

## Understanding Results

### Suspicious Features

```python
# Get suspicious features
suspicious = report.suspicious_features
for feature, score in suspicious.items():
    print(f"{feature}: {score}")
```

### Leakage Scores

```python
# Get leakage scores
scores = report.leakage_scores
print(scores.head())
```

## Best Practices

1. **Run before modeling** - Detect leakage early
2. **Review suspicious features** - Investigate flagged features
3. **Remove leaking features** - Don't use features with leakage
4. **Check train-test split** - Ensure proper data splitting
5. **Validate detection** - Manually verify suspicious features

## Common Leakage Sources

- Target encoding leakage
- Future information leakage
- Data collection issues
- Preprocessing mistakes
- Feature engineering errors

## Next Steps

- Learn about [Feature Engineering](feature-engineering.md)
- Explore [Benchmarking](benchmarking.md)
- Check [API Reference](../api/leakage-detection.md)

