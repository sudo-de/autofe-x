"""
AutoFE-X Example: Housing Dataset
This example shows how to use AutoFE-X with a housing price prediction dataset.
"""

import pandas as pd
from autofex import AutoFEX

# Load your housing data
df = pd.read_csv('your_housing_data.csv')  # Replace with your actual file path

# Check available columns
print("Available columns:", df.columns.tolist())
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# For housing data, 'median_house_value' is typically the target
target_column = 'median_house_value'

# Verify column exists and extract
if target_column in df.columns:
    print(f"\n✅ Using '{target_column}' as target variable")
    
    # Extract target and features
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target: {y.name}")
    
    # Use AutoFEX
    print("\n🚀 Running AutoFE-X pipeline...")
    afx = AutoFEX()
    result = afx.process(X, y)
    
    # Access results
    print("\n✅ Pipeline completed successfully!")
    print(f"Original features: {result.original_data.shape[1]}")
    print(f"Engineered features: {result.engineered_features.shape[1]}")
    print(f"New features created: {result.engineered_features.shape[1] - result.original_data.shape[1]}")
    
    # Get engineered features DataFrame
    engineered_df = result.engineered_features
    
    print(f"\nEngineered DataFrame shape: {engineered_df.shape}")
    print(f"Engineered features: {engineered_df.shape[1]} features")
    
    # Access other results
    print(f"\nData Quality Report:")
    print(f"  - Missing values: {result.data_quality_report['missing_values']['total_missing_cells']}")
    print(f"  - Outlier features: {len(result.data_quality_report['outliers'])}")
    
    print(f"\nLeakage Detection:")
    print(f"  - Risk level: {result.leakage_report['overall_assessment']['risk_level']}")
    
    # Use engineered features for your model
    # X_engineered = result.engineered_features
    
else:
    print(f"\n❌ Column '{target_column}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease update 'target_column' variable with one of the column names above.")
