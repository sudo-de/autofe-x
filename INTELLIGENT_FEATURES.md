# 🧠 AutoFE-X: Intelligent Feature Engineering

## Overview

AutoFE-X now includes **intelligent automation** that analyzes your data and automatically selects the best feature engineering strategies, scores feature quality, and provides actionable recommendations.

---

## 🧠 Intelligent Features

### 1. Intelligent Orchestrator

**Automatically selects and combines the best feature engineering techniques** based on data characteristics.

**Features:**
- Automatic data analysis (numeric, categorical, datetime, string detection)
- Smart feature engineering selection
- Automatic module combination
- Data-driven recommendations

**Usage:**
```python
from autofex import IntelligentOrchestrator

orchestrator = IntelligentOrchestrator()
intelligent_features = orchestrator.fit_transform(X, y)

# Automatically:
# - Detects skewed features → applies power transforms
# - Detects datetime → creates datetime features
# - Detects strings → creates string features
# - High dimensionality → applies PCA
# - Combines all intelligently
```

### 2. Feature Quality Scorer

**Comprehensive multi-dimensional quality scoring** for all features.

**Quality Dimensions:**
- **Predictive Power**: How well the feature predicts the target (mutual info, F-test)
- **Stability**: Consistency across samples (coefficient of variation)
- **Uniqueness**: Non-redundancy (correlation with other features)
- **Efficiency**: Data quality (missing data, outliers)
- **Overall Score**: Weighted combination of all dimensions

**Usage:**
```python
from autofex import FeatureQualityScorer

scorer = FeatureQualityScorer()

# Score all features
quality_scores = scorer.score_all_features(X, y, top_n=20)

# Get top features
top_features = scorer.get_top_features(X, y, n_features=50, min_score=0.1)

# Get rankings
rankings = scorer.get_feature_rankings(X, y)
# Returns: top_predictive, top_stable, top_unique, top_efficient, top_overall
```

### 3. Feature Engineering Recommender

**Automatic recommendations** for feature engineering strategies.

**Recommendations Include:**
- **Transformations**: Which transformations to apply (Box-Cox, log, etc.)
- **Strategies**: Overall strategies (PCA, target encoding, etc.)
- **Warnings**: Data quality issues
- **Opportunities**: Potential improvements
- **Auto-Config**: Ready-to-use configuration

**Usage:**
```python
from autofex import FeatureEngineeringRecommender

recommender = FeatureEngineeringRecommender()

# Get recommendations
recommendations = recommender.recommend_feature_engineering(X, y)

# Transformations
for rec in recommendations["transformations"]:
    print(f"{rec['column']}: {rec['transformation']} - {rec['reason']}")

# Strategies
for strategy in recommendations["strategies"]:
    print(f"{strategy['strategy']}: {strategy['reason']}")

# Auto-configuration
auto_config = recommender.get_auto_config(X, y)
# Use with AutoFEX
afx = AutoFEX(feature_engineering_config=auto_config["feature_engineering"])
```

---

## 🎯 Complete Intelligent Pipeline

```
1. Analyze Data
   ↓
2. Generate Recommendations
   ↓
3. Intelligent Feature Engineering
   ↓
4. Quality Scoring
   ↓
5. Feature Selection
   ↓
6. Final Feature Set
```

### Example

```python
from autofex import (
    IntelligentOrchestrator,
    FeatureQualityScorer,
    FeatureEngineeringRecommender,
)

# 1. Get recommendations
recommender = FeatureEngineeringRecommender()
recommendations = recommender.recommend_feature_engineering(X, y)

# 2. Intelligent engineering
orchestrator = IntelligentOrchestrator()
intelligent_features = orchestrator.fit_transform(X, y)

# 3. Quality scoring
scorer = FeatureQualityScorer()
quality_scores = scorer.score_all_features(intelligent_features, y)

# 4. Select top features
top_features = scorer.get_top_features(intelligent_features, y, n_features=50)
final_features = intelligent_features[top_features]
```

---

## 📊 Quality Scoring Details

### Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Predictive Power** | 40% | Mutual information, F-test scores |
| **Stability** | 20% | Coefficient of variation |
| **Uniqueness** | 20% | Correlation with other features |
| **Efficiency** | 20% | Missing data, outlier percentage |
| **Overall Score** | - | Weighted average |

### Score Interpretation

- **0.8-1.0**: Excellent feature
- **0.6-0.8**: Good feature
- **0.4-0.6**: Acceptable feature
- **0.2-0.4**: Poor feature
- **0.0-0.2**: Very poor feature

---

## 🎯 Benefits

1. **Automation**: No manual configuration needed
2. **Intelligence**: Data-driven decisions
3. **Quality**: Multi-dimensional scoring
4. **Efficiency**: Automatic feature selection
5. **Transparency**: Clear recommendations and reasons

---

## 💡 Best Practices

1. **Start with Recommendations**: Use recommender to understand your data
2. **Use Orchestrator**: Let it automatically select best techniques
3. **Score Features**: Always score features before selection
4. **Filter by Quality**: Use quality scores to filter features
5. **Validate**: Test on holdout set

---

**AutoFE-X: Intelligent feature engineering that thinks for you!** 🧠🚀
