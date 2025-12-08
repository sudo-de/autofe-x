# 🚀 AutoFE-X v0.1.0 - Initial Release

**Automated Feature Engineering + Data Profiling + Leakage Detection**

The first production-ready release of AutoFE-X, a next-generation toolkit that becomes the brain of any ML pipeline.

## ✨ Key Features

- 🔧 **Automated Feature Engineering**: Mathematical transforms, interactions, encodings
- 🔍 **Data Profiling**: Missing values, outliers, distributions, correlations
- 🛡️ **Leakage Detection**: Target leakage, contamination, statistical anomalies
- 📊 **Auto-Benchmarking**: Model comparison, feature importance, ablation studies
- 🔗 **Feature Lineage**: Graph-based transformation tracking and provenance

## 📦 Installation

```bash
pip install autofex
```

## 🚀 Quick Start

```python
from autofex import AutoFEX

afx = AutoFEX()
result = afx.process(X, y)

print("Engineered Features:", result.engineered_features.shape)
print("Leakage Risk:", result.leakage_report['overall_assessment']['risk_level'])
```

## 🎯 Perfect For

- Kaggle competitions
- Production ML pipelines
- Data science teams
- AutoML systems
- Model debugging

## ⚡ Performance

- Processes 100K rows × 50 features in ~2 seconds
- Minimal memory footprint
- Handles datasets up to 1M rows efficiently

## 🔒 Quality

- ✅ 34 comprehensive tests
- ✅ 100% MyPy type checking
- ✅ Security scanning enabled
- ✅ Multi-Python support (3.8-3.12)

---

**Ready to supercharge your ML pipelines?** 🚀

```bash
pip install autofex
```
