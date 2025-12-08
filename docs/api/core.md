# Core API Reference

## AutoFEX

Main class that orchestrates the AutoFE-X pipeline.

### Class Definition

```python
class AutoFEX:
    def __init__(
        self,
        feature_engineering: bool = True,
        data_profiling: bool = True,
        leakage_detection: bool = True,
        benchmarking: bool = True,
        lineage_tracking: bool = True,
        **kwargs
    ):
        ...
```

### Parameters

- `feature_engineering` (bool): Enable feature engineering
- `data_profiling` (bool): Enable data profiling
- `leakage_detection` (bool): Enable leakage detection
- `benchmarking` (bool): Enable benchmarking
- `lineage_tracking` (bool): Enable lineage tracking
- `**kwargs`: Additional arguments passed to components

### Methods

#### `process(X, y)`

Process data through the complete pipeline.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame
- `y` (pd.Series): Target variable

**Returns:**
- `AutoFEXResult`: Result object containing all outputs

**Example:**
```python
afx = AutoFEX()
result = afx.process(X, y)
```

### Result Object

The `process()` method returns an `AutoFEXResult` object with:

- `engineered_features` (pd.DataFrame): Engineered features
- `profiling_report` (DataProfilingReport): Data profiling results
- `leakage_report` (LeakageReport): Leakage detection results
- `benchmark_results` (BenchmarkResults): Benchmarking results
- `feature_importance` (pd.DataFrame): Feature importance scores
- `lineage_graph` (networkx.DiGraph): Feature lineage graph

## FeatureEngineer

Automated feature engineering.

### Class Definition

```python
class FeatureEngineer:
    def __init__(
        self,
        include_interactions: bool = True,
        include_polynomials: bool = True,
        include_encodings: bool = True,
        max_features: int = None,
        **kwargs
    ):
        ...
```

### Methods

#### `fit(X, y)`

Fit the feature engineer on training data.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame
- `y` (pd.Series): Target variable

#### `transform(X)`

Transform data using fitted feature engineer.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame

**Returns:**
- `pd.DataFrame`: Engineered features

#### `fit_transform(X, y)`

Fit and transform in one step.

## DataProfiler

Data quality analysis and profiling.

### Class Definition

```python
class DataProfiler:
    def __init__(
        self,
        detect_outliers: bool = True,
        check_missing: bool = True,
        statistical_summary: bool = True,
        **kwargs
    ):
        ...
```

### Methods

#### `profile(X)`

Profile the data.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame

**Returns:**
- `DataProfilingReport`: Profiling report

## LeakageDetector

Detect data leakage and contamination.

### Class Definition

```python
class LeakageDetector:
    def __init__(
        self,
        alpha: float = 0.05,
        methods: List[str] = None,
        **kwargs
    ):
        ...
```

### Methods

#### `detect(X, y)`

Detect leakage in data.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame
- `y` (pd.Series): Target variable

**Returns:**
- `LeakageReport`: Leakage detection report

## FeatureBenchmarker

Compare feature sets across models.

### Class Definition

```python
class FeatureBenchmarker:
    def __init__(
        self,
        models: List[str] = None,
        cv_folds: int = 5,
        **kwargs
    ):
        ...
```

### Methods

#### `benchmark(X, y, feature_sets)`

Benchmark multiple feature sets.

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame
- `y` (pd.Series): Target variable
- `feature_sets` (List[str]): List of feature set names

**Returns:**
- `BenchmarkResults`: Benchmarking results

## FeatureLineageTracker

Track feature transformations and dependencies.

### Class Definition

```python
class FeatureLineageTracker:
    def __init__(self):
        ...
```

### Methods

#### `track_transformation(feature_name, transformation, parent_features)`

Track a feature transformation.

**Parameters:**
- `feature_name` (str): Name of new feature
- `transformation` (str): Transformation applied
- `parent_features` (List[str]): Parent feature names

#### `get_lineage(feature_name)`

Get lineage for a feature.

**Parameters:**
- `feature_name` (str): Feature name

**Returns:**
- `List[str]`: Lineage path

## Next Steps

- See individual API docs for each component
- Check [Examples](examples/basic-examples.md) for usage
- Explore [Advanced Features](advanced/advanced-capabilities.md)

