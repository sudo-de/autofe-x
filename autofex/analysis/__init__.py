"""
Advanced Statistical Analysis Module

Goes beyond basic scipy to provide integrated statistical insights.
"""

from .statistical import AdvancedStatisticalAnalyzer

try:
    from .advanced_stats import UltraAdvancedStatisticalAnalyzer
    _ULTRA_ADVANCED_AVAILABLE = True
except ImportError:
    _ULTRA_ADVANCED_AVAILABLE = False
    UltraAdvancedStatisticalAnalyzer = None

__all__ = ["AdvancedStatisticalAnalyzer"]

if _ULTRA_ADVANCED_AVAILABLE:
    __all__.append("UltraAdvancedStatisticalAnalyzer")
