"""
AutoFE-X NextGen Features Example

Demonstrates advanced feature engineering, intelligent selection, and visualization.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from autofex import (
    AutoFEX,
    FeatureEngineer,
    FeatureSelector,
    FeatureVisualizer,
)


def main():
    print("🚀 AutoFE-X NextGen Features Demo")
    print("=" * 50)

    # Generate sample data
    print("\n📊 Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    # Add some time-series like data
    X["timestamp"] = pd.date_range("2023-01-01", periods=len(X), freq="D")
    X["value_series"] = np.cumsum(np.random.randn(len(X)))

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # ============================================
    # NextGen Feature 1: Advanced Feature Engineering
    # ============================================
    print("\n" + "=" * 50)
    print("🔧 Advanced Feature Engineering")
    print("=" * 50)

    advanced_fe = FeatureEngineer(
        {
            "statistical_aggregations": True,
            "time_series_features": True,
            "advanced_binning": True,
            "cross_features": True,
            "n_bins": 5,
        }
    )

    X_advanced = advanced_fe.fit_transform(X, y)

    print(f"Original features: {X.shape[1]}")
    print(f"Advanced engineered features: {X_advanced.shape[1]}")
    print(f"New features created: {X_advanced.shape[1] - X.shape[1]}")

    # ============================================
    # NextGen Feature 2: Advanced Feature Selection
    # ============================================
    print("\n" + "=" * 50)
    print("🎯 Advanced Feature Selection")
    print("=" * 50)

    selector = FeatureSelector(
        {
            "strategies": ["l1", "rfe", "variance", "correlation"],
            "n_features": 50,
            "cv_folds": 5,
        }
    )

    # Ensemble selection
    selected_features = selector.select_features_ensemble(X_advanced, y, voting_threshold=0.5)

    print(f"Selected {len(selected_features)} features from {X_advanced.shape[1]} advanced features")
    print(f"Reduction: {X_advanced.shape[1] - len(selected_features)} features removed")

    X_selected = X_advanced[selected_features]

    # ============================================
    # NextGen Feature 3: Full AutoFEX Pipeline
    # ============================================
    print("\n" + "=" * 50)
    print("⚡ Full AutoFEX Pipeline on Selected Features")
    print("=" * 50)

    afx = AutoFEX()
    result = afx.process(X_selected, y)

    print(f"✅ Pipeline completed!")
    print(f"Final engineered features: {result.engineered_features.shape[1]}")
    print(f"Processing time: {result.processing_time:.2f} seconds")

    # ============================================
    # NextGen Feature 4: Visualization
    # ============================================
    print("\n" + "=" * 50)
    print("📊 Feature Visualization")
    print("=" * 50)

    viz = FeatureVisualizer()

    # Feature importance
    importance_scores = result.benchmark_results.get("feature_importance")
    if importance_scores is not None and len(importance_scores) > 0:
        print("Creating feature importance plot...")
        try:
            viz.plot_feature_importance(importance_scores, top_n=15)
            print("✅ Feature importance plot created")
        except Exception as e:
            print(f"⚠️ Visualization skipped: {e}")

    # Data quality summary
    print("Creating data quality dashboard...")
    try:
        viz.plot_data_quality_summary(result.data_quality_report)
        print("✅ Data quality dashboard created")
    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")

    # Leakage risk
    print("Creating leakage risk visualization...")
    try:
        viz.plot_leakage_risk(result.leakage_report)
        print("✅ Leakage risk visualization created")
    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 50)
    print("📋 NextGen Pipeline Summary")
    print("=" * 50)
    print(f"Original features: {X.shape[1]}")
    print(f"After advanced engineering: {X_advanced.shape[1]}")
    print(f"After intelligent selection: {X_selected.shape[1]}")
    print(f"Final engineered features: {result.engineered_features.shape[1]}")
    print(f"Total feature expansion: {result.engineered_features.shape[1] / X.shape[1]:.1f}x")
    print(f"\n✅ NextGen AutoFE-X pipeline completed successfully!")


if __name__ == "__main__":
    main()
