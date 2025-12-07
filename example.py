#!/usr/bin/env python3
"""
Example usage of AutoFE-X package.

This script demonstrates the main features of AutoFE-X on a sample dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from autofex import AutoFEX

def main():
    print("AutoFE-X Demo")
    print("=" * 50)

    # Generate sample data
    print("Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')

    # Add some missing values and outliers
    X.loc[np.random.choice(X.index, 50, replace=False), 'feature_0'] = np.nan
    X.loc[X.index[:10], 'feature_1'] = X.loc[X.index[:10], 'feature_1'] * 10  # Outliers

    # Add a categorical feature (commented out to avoid issues)
    # X['category'] = np.random.choice(['A', 'B', 'C'], size=len(X))

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Initialize AutoFEX
    print("\nInitializing AutoFEX...")
    afx = AutoFEX()

    # Process the data
    print("⚡ Running AutoFEX pipeline...")
    result = afx.process(X_train, y_train, X_test)

    # Display results
    print("\nResults Summary")
    print("=" * 30)

    # Data Quality Report
    print("Data Quality:")
    quality = result.data_quality_report
    print(f"  - Total features: {quality['overview']['n_features']}")
    print(f"  - Missing values: {quality['missing_values']['total_missing_cells']}")
    print(".2f")
    print(f"  - Outlier features: {len(quality['outliers'])}")

    # Leakage Report
    print("\nLeakage Detection:")
    leakage = result.leakage_report
    print(f"  - Target leakage candidates: {leakage['target_leakage']['total_candidates']}")
    print(f"  - Perfect features: {leakage['feature_perfection']['total_perfect_features']}")
    print(f"  - Overall risk: {leakage['overall_assessment']['risk_level']}")

    # Feature Engineering
    print("\n🔧 Feature Engineering:")
    print(f"  - Original features: {result.original_data.shape[1]}")
    print(f"  - Engineered features: {result.engineered_features.shape[1]}")
    print(f"  - New features created: {result.engineered_features.shape[1] - result.original_data.shape[1]}")

    # Benchmarking
    print("\nBenchmarking Results:")
    bench = result.benchmark_results
    if bench.get('best_configurations'):
        best = bench['best_configurations'].get('best_overall')
        if best:
            print(f"  - Best model: {best['model']}")
            print(".3f")
            print(f"  - Features used: {best['n_features']}")

    # Feature Lineage
    print("\nFeature Lineage:")
    lineage = result.feature_lineage
    print(f"  - Total transformations tracked: {lineage['metadata']['total_transformations']}")
    print(f"  - Feature dependency graph created: {lineage['metadata']['total_features']} nodes")

    # Performance
    print(".2f")

    # Show top engineered features
    print("\nTop Features by Importance:")
    if bench.get('feature_importance') is not None:
        importance = bench['feature_importance'].head(5)
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(".4f")

    # Show best configuration details
    if bench.get('best_configurations'):
        best = bench['best_configurations'].get('best_overall')
        if best:
            print(f"\n🏆 Best Overall Configuration:")
            print(f"  - Model: {best.get('model', 'N/A')}")
            print(".3f")
            print(f"  - Feature Set: {best.get('feature_set', 'N/A')}")

    print("\nAutoFEX pipeline completed successfully!")
    print("\nUse result.engineered_features for your ML models!")

if __name__ == "__main__":
    main()
