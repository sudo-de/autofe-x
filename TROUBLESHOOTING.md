# 🔧 AutoFE-X Troubleshooting Guide

## Common Errors & Solutions

### Error: KeyError: 'label' (or any column name)

**Problem:** The column name you're trying to access doesn't exist in your DataFrame.

**Solution 1: Check available columns first**

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
df = pd.read_csv('your_data.csv')

# Check what columns exist
print("Available columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Find the target column (adjust name as needed)
# Common names: 'target', 'label', 'y', 'class', 'outcome', etc.
target_column = 'target'  # Change this to match your data

# Verify column exists
if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
else:
    # Extract target and features
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Use AutoFEX
    afx = AutoFEX()
    result = afx.process(X, y)
    
    print("✅ Success!")
```

**Solution 2: Auto-detect target column**

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
df = pd.read_csv('your_data.csv')

# Common target column names to try
possible_targets = ['target', 'label', 'y', 'class', 'outcome', 'target_variable']

# Find the target column
target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # If none found, use the last column as target
    target_column = df.columns[-1]
    print(f"Using last column '{target_column}' as target")

# Extract target and features
y = df[target_column]
X = df.drop(target_column, axis=1)

# Use AutoFEX
afx = AutoFEX()
result = afx.process(X, y)
```

**Solution 3: Specify target column explicitly**

```python
import pandas as pd
from autofex import AutoFEX

# Load your data
df = pd.read_csv('your_data.csv')

# Method 1: If target is a column name
target_column_name = 'your_target_column_name'  # Replace with actual name
y = df[target_column_name]
X = df.drop(target_column_name, axis=1)

# Method 2: If target is a separate Series/array
# y = pd.Series([...])  # Your target values
# X = df  # Your features DataFrame

# Use AutoFEX
afx = AutoFEX()
result = afx.process(X, y)
```

## Complete Working Example

```python
import pandas as pd
import numpy as np
from autofex import AutoFEX

# Example 1: Load from CSV
df = pd.read_csv('your_data.csv')

# Check the data structure
print("DataFrame shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Identify target column (adjust based on your data)
# Option A: If you know the column name
target_col = 'target'  # Change to your actual column name
if target_col in df.columns:
    y = df[target_col]
    X = df.drop(target_col, axis=1)
else:
    print(f"Column '{target_col}' not found!")
    print("Available columns:", df.columns.tolist())
    # Use the last column as fallback
    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    print(f"Using '{target_col}' as target")

# Use AutoFEX
afx = AutoFEX()
result = afx.process(X, y)

print(f"\n✅ Success!")
print(f"Original features: {X.shape[1]}")
print(f"Engineered features: {result.engineered_features.shape[1]}")
```

## Example 2: Using FeatureEngineer Only

```python
from autofex import FeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Check columns
print("Columns:", df.columns.tolist())

# Extract target (adjust column name)
target_col = 'target'  # Change to match your data
if target_col in df.columns:
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    # Feature engineering only
    fe = FeatureEngineer()
    X_engineered = fe.fit_transform(X, y)
    
    print(f"✅ Engineered {X.shape[1]} → {X_engineered.shape[1]} features")
else:
    print(f"Error: Column '{target_col}' not found!")
    print("Available columns:", df.columns.tolist())
```

## Debugging Tips

### 1. Inspect Your Data

```python
import pandas as pd

df = pd.read_csv('your_data.csv')

# Basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:", df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
```

### 2. Check for Missing Columns

```python
# Check if specific column exists
if 'label' in df.columns:
    print("✅ 'label' column found")
else:
    print("❌ 'label' column NOT found")
    print("Available columns:", df.columns.tolist())
```

### 3. Handle Different Column Names

```python
# Map common variations
column_mapping = {
    'target': ['target', 'label', 'y', 'class', 'outcome'],
    'id': ['id', 'ID', 'index', 'Index']
}

# Find target column
target_col = None
for col in df.columns:
    if col.lower() in column_mapping['target']:
        target_col = col
        break

if target_col:
    y = df[target_col]
    X = df.drop(target_col, axis=1)
else:
    print("Could not find target column")
```

## Quick Fix Template

```python
import pandas as pd
from autofex import AutoFEX

# Load data
df = pd.read_csv('your_data.csv')

# REPLACE 'label' with your actual target column name
TARGET_COLUMN = 'label'  # ⬅️ CHANGE THIS

# Check and extract
if TARGET_COLUMN in df.columns:
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    
    # Run AutoFEX
    afx = AutoFEX()
    result = afx.process(X, y)
    
    print("✅ Success!")
else:
    print(f"❌ Column '{TARGET_COLUMN}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease update TARGET_COLUMN variable with the correct column name.")
```
