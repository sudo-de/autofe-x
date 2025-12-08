# Data Profiling Guide

Learn how to use AutoFE-X for comprehensive data quality analysis.

## Overview

Data profiling helps you understand your data quality, identify issues, and make informed decisions about feature engineering.

## Basic Usage

```python
from autofex import DataProfiler
import pandas as pd

# Initialize profiler
profiler = DataProfiler()

# Profile your data
report = profiler.profile(X)

# Access results
print(report.summary)
print(report.outliers)
print(report.missing_values)
```

## Profiling Features

### Statistical Summary

```python
# Get comprehensive statistical summary
summary = report.statistical_summary
print(summary.describe())
```

### Outlier Detection

```python
# Detect outliers
outliers = report.outliers
print(f"Found {len(outliers)} outliers")
```

### Missing Value Analysis

```python
# Analyze missing values
missing = report.missing_values
print(missing.summary)
print(missing.patterns)
```

### Correlation Analysis

```python
# Analyze correlations
correlations = report.correlations
print(correlations.matrix)
print(correlations.high_correlations)
```

## Advanced Profiling

For more advanced profiling, use additional options:

```python
profiler = DataProfiler(
    detect_outliers=True,
    check_missing=True,
    statistical_summary=True,
    correlation_analysis=True,
    distribution_analysis=True
)
```

## Best Practices

1. **Profile before engineering** - Understand data quality first
2. **Check for outliers** - Handle outliers appropriately
3. **Analyze missing values** - Understand missing patterns
4. **Review correlations** - Identify redundant features
5. **Use profiling reports** - Make data-driven decisions

## Next Steps

- Learn about [Feature Engineering](feature-engineering.md)
- Explore [Leakage Detection](leakage-detection.md)
- Check [API Reference](../api/data-profiling.md)

