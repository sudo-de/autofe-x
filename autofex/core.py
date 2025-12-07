"""
Core AutoFEX class that orchestrates all feature engineering and analysis components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from .feature_engineering import FeatureEngineer
from .data_profiling import DataProfiler
from .leakage_detection import LeakageDetector
from .benchmarking import FeatureBenchmarker
from .lineage import FeatureLineageTracker


@dataclass
class AutoFEXResult:
    """Container for AutoFEX pipeline results."""
    original_data: pd.DataFrame
    engineered_features: pd.DataFrame
    data_quality_report: Dict[str, Any]
    leakage_report: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    feature_lineage: Dict[str, Any]
    processing_time: float


class AutoFEX:
    """
    Main AutoFEX class that orchestrates automated feature engineering,
    data profiling, leakage detection, and benchmarking.

    Example:
        >>> afx = AutoFEX()
        >>> result = afx.process(X_train, y_train, X_test)
        >>> print(result.data_quality_report)
        >>> engineered_train = result.engineered_features
    """

    def __init__(
        self,
        feature_engineering_config: Optional[Dict[str, Any]] = None,
        profiling_config: Optional[Dict[str, Any]] = None,
        leakage_config: Optional[Dict[str, Any]] = None,
        benchmarking_config: Optional[Dict[str, Any]] = None,
        lineage_config: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize AutoFEX pipeline.

        Args:
            feature_engineering_config: Configuration for feature engineering
            profiling_config: Configuration for data profiling
            leakage_config: Configuration for leakage detection
            benchmarking_config: Configuration for benchmarking
            lineage_config: Configuration for lineage tracking
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

        # Initialize components
        self.feature_engineer = FeatureEngineer(
            config=feature_engineering_config or {}
        )
        self.data_profiler = DataProfiler(
            config=profiling_config or {}
        )
        self.leakage_detector = LeakageDetector(
            config=leakage_config or {}
        )
        self.benchmarker = FeatureBenchmarker(
            config=benchmarking_config or {}
        )
        self.lineage_tracker = FeatureLineageTracker(
            config=lineage_config or {}
        )

    def process(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        X_test: Optional[pd.DataFrame] = None,
        target_name: str = "target"
    ) -> AutoFEXResult:
        """
        Run the complete AutoFEX pipeline.

        Args:
            X: Training features
            y: Training target (optional, required for supervised operations)
            X_test: Test features (optional)
            target_name: Name for target variable in lineage tracking

        Returns:
            AutoFEXResult containing all pipeline outputs
        """
        import time
        start_time = time.time()

        # Initialize lineage tracking
        self.lineage_tracker.start_session(X.columns.tolist())

        # Step 1: Data profiling
        quality_report = self.data_profiler.analyze(X, y)

        # Step 2: Leakage detection (if target available)
        leakage_report = {}
        if y is not None:
            leakage_report = self.leakage_detector.detect(X, y)

        # Step 3: Feature engineering
        engineered_features = self.feature_engineer.fit_transform(X, y)

        # Step 4: Update lineage with engineered features
        self.lineage_tracker.add_transformation(
            "feature_engineering",
            X.columns.tolist(),
            engineered_features.columns.tolist()
        )

        # Step 5: Benchmarking (if target available)
        benchmark_results = {}
        if y is not None:
            # Use original features for benchmarking (engineered features would be inconsistent with X_test)
            benchmark_results = self.benchmarker.benchmark_features(
                X, y, X_test
            )

        processing_time = time.time() - start_time

        return AutoFEXResult(
            original_data=X,
            engineered_features=engineered_features,
            data_quality_report=quality_report,
            leakage_report=leakage_report,
            benchmark_results=benchmark_results,
            feature_lineage=self.lineage_tracker.get_lineage_graph(),
            processing_time=processing_time
        )

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info"
    ) -> pd.Series:
        """
        Get feature importance scores.

        Args:
            X: Features
            y: Target
            method: Importance method ("mutual_info", "f_test", "chi2")

        Returns:
            Series with feature importance scores
        """
        return self.benchmarker.get_feature_importance(X, y, method)

    def suggest_feature_transformations(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest feature transformations based on data analysis.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            List of suggested transformations
        """
        suggestions = []

        # Analyze data types and suggest transformations
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # Numeric feature suggestions
        for col in numeric_cols:
            if X[col].skew() > 1:
                suggestions.append({
                    "column": col,
                    "transformation": "log_transform",
                    "reason": f"High skewness ({X[col].skew():.2f})",
                    "expected_impact": "Normalize distribution"
                })

        # Categorical feature suggestions
        for col in categorical_cols:
            if X[col].nunique() > 10:
                suggestions.append({
                    "column": col,
                    "transformation": "target_encoding",
                    "reason": f"High cardinality ({X[col].nunique()} unique values)",
                    "expected_impact": "Reduce dimensionality"
                })

        return suggestions
