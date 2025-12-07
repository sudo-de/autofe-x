# 🚀 AutoFE-X Quick Start

## Installation

```bash
pip install autofex
```

```python
from autofex import AutoFEX  # ✅ Use underscore, correct class name
```

## Complete Example

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv')['target']

# Initialize AutoFEX (note: capital FEX)
afx = AutoFEX()

# Run complete pipeline
result = afx.process(X, y)

# Access results
print("Engineered Features:", result.engineered_features.shape)
print("Leakage Risk:", result.leakage_report['overall_assessment']['risk_level'])
```

## Available Imports

```python
# Main class
from autofex import AutoFEX

# Individual components
from autofex import (
    FeatureEngineer,
    DataProfiler,
    LeakageDetector,
    FeatureBenchmarker,
    FeatureLineageTracker
)

# Or import all at once
from autofex import AutoFEX, FeatureEngineer, DataProfiler, LeakageDetector
```

## Common Mistakes

1. **Using hyphen in import:** `from autofe-x import ...` ❌
   - **Fix:** Use underscore: `from autofex import ...` ✅

2. **Wrong class name:** `AutoFE` ❌
   - **Fix:** Use correct name: `AutoFEX` ✅ (capital FEX)

3. **Not installed:** `ModuleNotFoundError: No module named 'autofex'`
   - **Fix:** Run `pip install autofex` first

## Verification

```python
import autofex
print(f"AutoFE-X version: {autofex.__version__}")

from autofex import AutoFEX
print("✅ Import successful!")
```
