"""
Pytest configuration and fixtures for AutoFE-X tests.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')
    return X, y


@pytest.fixture
def sample_data_with_missing(sample_data):
    """Create sample data with missing values."""
    X, y = sample_data
    # Add some missing values
    X_missing = X.copy()
    np.random.seed(42)
    mask = np.random.random(X.shape) < 0.1  # 10% missing
    X_missing[mask] = np.nan
    return X_missing, y


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 8), columns=[f'feature_{i}' for i in range(8)])
    y = pd.Series(X.sum(axis=1) + np.random.randn(100) * 0.1, name='target')
    return X, y
