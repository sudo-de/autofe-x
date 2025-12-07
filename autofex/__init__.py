"""
AutoFE-X: Automated Feature Engineering + Data Profiling + Leakage Detection

A next-gen toolkit that becomes the brain of any ML pipeline by combining:
- Automatic feature engineering (classic + deep)
- Data quality analysis
- Leakage detection
- Auto-benchmarking of feature sets
- Graph-based feature lineage

Lightweight, fast, and interpretable. No LLMs.
"""

from .core import AutoFEX
from .feature_engineering import FeatureEngineer
from .data_profiling import DataProfiler
from .leakage_detection import LeakageDetector
from .benchmarking import FeatureBenchmarker
from .lineage import FeatureLineageTracker

__version__ = "0.1.0"
__all__ = [
    "AutoFEX",
    "FeatureEngineer",
    "DataProfiler",
    "LeakageDetector",
    "FeatureBenchmarker",
    "FeatureLineageTracker",
]
