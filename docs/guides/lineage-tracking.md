# Lineage Tracking Guide

Track feature transformations and understand feature provenance.

## Overview

Lineage tracking helps you understand how features were created, their dependencies, and transformation history.

## Basic Usage

```python
from autofex import FeatureLineageTracker

# Initialize tracker
tracker = FeatureLineageTracker()

# Track a transformation
tracker.track_transformation(
    feature_name='feature_log_x',
    transformation='log',
    parent_features=['x']
)

# Get lineage
lineage = tracker.get_lineage('feature_log_x')
print(lineage)
```

## Using with AutoFEX

```python
from autofex import AutoFEX

# Enable lineage tracking
afx = AutoFEX(lineage_tracking=True)

# Process data
result = afx.process(X, y)

# Access lineage graph
graph = result.lineage_graph
```

## Exploring Lineage

### Get Feature History

```python
# Get transformation history
history = tracker.get_history('feature_name')
print(history)
```

### Find Dependencies

```python
# Get feature dependencies
dependencies = tracker.get_dependencies('feature_name')
print(dependencies)
```

### Visualize Lineage

```python
# Visualize lineage graph
tracker.visualize('feature_name')
```

## Best Practices

1. **Enable tracking** - Always track transformations
2. **Use descriptive names** - Name features clearly
3. **Document transformations** - Record what was done
4. **Review lineage** - Understand feature creation
5. **Share lineage** - Help others understand features

## Next Steps

- Learn about [Feature Engineering](feature-engineering.md)
- Explore [Visualization](../advanced/advanced-capabilities.md)
- Check [API Reference](../api/lineage.md)

