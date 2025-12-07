"""
Feature Engineering Engine

Implements automated feature engineering with classic and deep techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings


class FeatureEngineer:
    """
    Automated feature engineering engine that creates new features
    through mathematical transformations, interactions, and aggregations.

    Supports both supervised and unsupervised feature creation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary with feature engineering options
        """
        self.config = config or {}
        self.numeric_transforms = self.config.get('numeric_transforms', [
            'log', 'sqrt', 'square', 'cube', 'reciprocal', 'standardize'
        ])
        self.categorical_transforms = self.config.get('categorical_transforms', [
            'one_hot', 'label_encode', 'frequency_encode', 'target_encode'
        ])
        self.interaction_degree = self.config.get('interaction_degree', 2)
        self.max_features = self.config.get('max_features', 1000)
        self.supervised = self.config.get('supervised', True)

        self.fitted_encoders = {}
        self.feature_names = []
        self.column_scalers = {}

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the feature engineer on training data.

        Args:
            X: Training features
            y: Training target (optional)
        """
        self.feature_names = list(X.columns)

        # Fit encoders for categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if 'target_encode' in self.categorical_transforms and y is not None:
                self._fit_target_encoder(X[col], y, col)

        # Fit scalers for numeric features (per column)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.column_scalers = {}
        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(X[[col]])  # Fit on single column
            self.column_scalers[col] = scaler

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted transformations.

        Args:
            X: Input features

        Returns:
            Transformed features DataFrame
        """
        if not self.feature_names:
            raise ValueError("FeatureEngineer must be fitted before transform")

        X_transformed = X.copy()
        new_features = []

        # Numeric transformations
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            numeric_features = self._create_numeric_features(X[col], col)
            new_features.extend(numeric_features)

        # Categorical transformations
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            cat_features = self._create_categorical_features(X[col], col)
            new_features.extend(cat_features)

        # Interaction features
        if self.interaction_degree > 1:
            interaction_features = self._create_interaction_features(X_transformed)
            new_features.extend(interaction_features)

        # Combine all features
        if new_features:
            feature_df = pd.concat([X_transformed] + new_features, axis=1)
        else:
            feature_df = X_transformed

        # Limit features if needed
        if len(feature_df.columns) > self.max_features:
            feature_df = self._select_top_features(feature_df, y=None)

        return feature_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """
        Fit and transform features in one step.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Transformed features DataFrame
        """
        return self.fit(X, y).transform(X)

    def _create_numeric_features(self, series: pd.Series, col_name: str) -> List[pd.DataFrame]:
        """
        Create numeric transformations for a single column.

        Args:
            series: Numeric column
            col_name: Column name

        Returns:
            List of DataFrames with new features
        """
        features = []

        # Handle zeros and negatives for log transform
        if 'log' in self.numeric_transforms:
            log_vals = np.log1p(series.clip(lower=0))
            features.append(pd.DataFrame({f'{col_name}_log': log_vals}))

        if 'sqrt' in self.numeric_transforms:
            sqrt_vals = np.sqrt(series.clip(lower=0))
            features.append(pd.DataFrame({f'{col_name}_sqrt': sqrt_vals}))

        if 'square' in self.numeric_transforms:
            square_vals = series ** 2
            features.append(pd.DataFrame({f'{col_name}_square': square_vals}))

        if 'cube' in self.numeric_transforms:
            cube_vals = series ** 3
            features.append(pd.DataFrame({f'{col_name}_cube': cube_vals}))

        if 'reciprocal' in self.numeric_transforms:
            # Avoid division by zero
            reciprocal_vals = 1 / (series + 1e-8)
            features.append(pd.DataFrame({f'{col_name}_reciprocal': reciprocal_vals}))

        if 'standardize' in self.numeric_transforms and col_name in self.column_scalers:
            standardized = self.column_scalers[col_name].transform(series.values.reshape(-1, 1)).flatten()
            features.append(pd.DataFrame({f'{col_name}_standardized': standardized}))

        return features

    def _create_categorical_features(self, series: pd.Series, col_name: str) -> List[pd.DataFrame]:
        """
        Create categorical transformations for a single column.

        Args:
            series: Categorical column
            col_name: Column name

        Returns:
            List of DataFrames with new features
        """
        features = []

        if 'frequency_encode' in self.categorical_transforms:
            freq_map = series.value_counts(normalize=True)
            freq_encoded = series.map(freq_map)
            features.append(pd.DataFrame({f'{col_name}_freq': freq_encoded}))

        if 'label_encode' in self.categorical_transforms:
            label_map = {val: i for i, val in enumerate(series.unique())}
            label_encoded = series.map(label_map)
            features.append(pd.DataFrame({f'{col_name}_label': label_encoded}))

        if 'target_encode' in self.categorical_transforms and col_name in self.fitted_encoders:
            target_encoded = series.map(self.fitted_encoders[col_name])
            target_encoded = target_encoded.fillna(target_encoded.mean())
            features.append(pd.DataFrame({f'{col_name}_target': target_encoded}))

        return features

    def _create_interaction_features(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create polynomial interaction features.

        Args:
            X: Input features DataFrame

        Returns:
            List of DataFrames with interaction features
        """
        features = []

        # Use only numeric columns for polynomial features
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            # Fill NaN values for polynomial features (they don't handle NaN)
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())

            poly = PolynomialFeatures(
                degree=self.interaction_degree,
                interaction_only=True,
                include_bias=False
            )

            poly_features = poly.fit_transform(X_numeric)
            feature_names = poly.get_feature_names_out(numeric_cols)

            # Create DataFrame with interaction features only (exclude originals)
            interaction_df = pd.DataFrame(
                poly_features[:, len(numeric_cols):],
                columns=feature_names[len(numeric_cols):],
                index=X.index
            )

            if not interaction_df.empty:
                features.append(interaction_df)

        return features

    def _fit_target_encoder(self, series: pd.Series, y: Union[pd.Series, np.ndarray], col_name: str):
        """
        Fit target encoder for categorical column.

        Args:
            series: Categorical column
            y: Target variable
            col_name: Column name
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Calculate mean target for each category
        target_means = y.groupby(series).mean()
        global_mean = y.mean()

        # Store encoder (with smoothing)
        self.fitted_encoders[col_name] = target_means.reindex(
            series.unique()
        ).fillna(global_mean)

    def _select_top_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, k: int = None) -> pd.DataFrame:
        """
        Select top k features based on importance.

        Args:
            X: Features DataFrame
            y: Target variable
            k: Number of features to keep

        Returns:
            Selected features DataFrame
        """
        if k is None:
            k = self.max_features

        if y is None or not self.supervised:
            # Unsupervised selection: keep original + most variant new features
            variances = X.var()
            top_features = variances.nlargest(k).index
        else:
            # Supervised selection: use mutual information
            try:
                if y.dtype in ['object', 'category'] or len(y.unique()) < 20:
                    # Classification
                    mi_scores = mutual_info_classif(X.fillna(X.mean()), y)
                else:
                    # Regression
                    mi_scores = mutual_info_regression(X.fillna(X.mean()), y)

                feature_scores = pd.Series(mi_scores, index=X.columns)
                top_features = feature_scores.nlargest(k).index
            except Exception as e:
                warnings.warn(f"Feature selection failed: {e}. Keeping all features.")
                return X

        return X[top_features]
