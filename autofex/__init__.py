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

# NextGen improvements
try:
    from .feature_engineering.advanced import AdvancedFeatureEngineer
    from .feature_selection.selector import AdvancedFeatureSelector
    from .visualization import FeatureVisualizer
    from .analysis.statistical import AdvancedStatisticalAnalyzer

    _NEXTGEN_AVAILABLE = True
except ImportError:
    _NEXTGEN_AVAILABLE = False

try:
    from .visualization.dashboard import InteractiveDashboard

    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    InteractiveDashboard = None

try:
    from .visualization.multidimensional import MultiDimensionalVisualizer

    _MULTIDIM_AVAILABLE = True
except ImportError:
    _MULTIDIM_AVAILABLE = False
    MultiDimensionalVisualizer = None

try:
    from .analysis.advanced_stats import UltraAdvancedStatisticalAnalyzer

    _ULTRA_STATS_AVAILABLE = True
except ImportError:
    _ULTRA_STATS_AVAILABLE = False
    UltraAdvancedStatisticalAnalyzer = None

try:
    from .mathematical.modeling import MathematicalModelingEngine

    _MATH_MODELING_AVAILABLE = True
except ImportError:
    _MATH_MODELING_AVAILABLE = False
    MathematicalModelingEngine = None

try:
    from .statistical.advanced_transforms import AdvancedStatisticalTransforms

    _STAT_TRANSFORMS_AVAILABLE = True
except ImportError:
    _STAT_TRANSFORMS_AVAILABLE = False
    AdvancedStatisticalTransforms = None

try:
    from .pandas_advanced.operations import AdvancedPandasOperations

    _PANDAS_ADVANCED_AVAILABLE = True
except ImportError:
    _PANDAS_ADVANCED_AVAILABLE = False
    AdvancedPandasOperations = None

try:
    from .numpy_advanced.operations import AdvancedNumpyOperations

    _NUMPY_ADVANCED_AVAILABLE = True
except ImportError:
    _NUMPY_ADVANCED_AVAILABLE = False
    AdvancedNumpyOperations = None

try:
    from .scipy_advanced.operations import AdvancedScipyOperations

    _SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    _SCIPY_ADVANCED_AVAILABLE = False
    AdvancedScipyOperations = None

try:
    from .intelligence.orchestrator import IntelligentOrchestrator
    from .intelligence.quality_scorer import FeatureQualityScorer
    from .intelligence.recommender import FeatureEngineeringRecommender

    _INTELLIGENCE_AVAILABLE = True
except ImportError:
    _INTELLIGENCE_AVAILABLE = False
    IntelligentOrchestrator = None
    FeatureQualityScorer = None
    FeatureEngineeringRecommender = None

__version__ = "0.1.0"
__all__ = [
    "AutoFEX",
    "FeatureEngineer",
    "DataProfiler",
    "LeakageDetector",
    "FeatureBenchmarker",
    "FeatureLineageTracker",
]

if _NEXTGEN_AVAILABLE:
    __all__.extend(
        [
            "AdvancedFeatureEngineer",
            "AdvancedFeatureSelector",
            "FeatureVisualizer",
            "AdvancedStatisticalAnalyzer",
        ]
    )

if _DASHBOARD_AVAILABLE:
    __all__.append("InteractiveDashboard")

if _MULTIDIM_AVAILABLE:
    __all__.append("MultiDimensionalVisualizer")

if _ULTRA_STATS_AVAILABLE:
    __all__.append("UltraAdvancedStatisticalAnalyzer")

if _MATH_MODELING_AVAILABLE:
    __all__.append("MathematicalModelingEngine")

if _STAT_TRANSFORMS_AVAILABLE:
    __all__.append("AdvancedStatisticalTransforms")

if _PANDAS_ADVANCED_AVAILABLE:
    __all__.append("AdvancedPandasOperations")

if _NUMPY_ADVANCED_AVAILABLE:
    __all__.append("AdvancedNumpyOperations")

if _SCIPY_ADVANCED_AVAILABLE:
    __all__.append("AdvancedScipyOperations")

if _INTELLIGENCE_AVAILABLE:
    __all__.extend(
        [
            "IntelligentOrchestrator",
            "FeatureQualityScorer",
            "FeatureEngineeringRecommender",
        ]
    )
