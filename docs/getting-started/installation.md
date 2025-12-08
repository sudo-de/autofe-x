# Installation Guide

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### Standard Installation

Install AutoFE-X from PyPI:

```bash
pip install autofex
```

### Development Installation

To install AutoFE-X in development mode:

```bash
git clone https://github.com/autofe-x/autofe-x.git
cd autofe-x
pip install -e .
```

### Install with Optional Dependencies

For full functionality including advanced visualization and statistical features:

```bash
pip install autofex[all]
```

Or install specific optional dependencies:

```bash
# For advanced visualization
pip install autofex[visualization]

# For advanced statistical analysis
pip install autofex[stats]

# For parallel processing
pip install autofex[parallel]
```

## Verify Installation

After installation, verify that AutoFE-X is correctly installed:

```python
import autofex
print(autofex.__version__)
```

You should see the version number printed (e.g., `0.1.0`).

## Dependencies

### Core Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities

### Optional Dependencies

- `matplotlib` - Basic plotting
- `plotly` - Interactive visualizations
- `seaborn` - Statistical visualizations
- `networkx` - Graph operations for lineage tracking
- `tqdm` - Progress bars

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade autofex
```

### Version Conflicts

If you have version conflicts with dependencies:

```bash
pip install --upgrade autofex --force-reinstall
```

### Virtual Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install autofex
```

## Next Steps

After installation, proceed to the [Quick Start Guide](quick-start.md) to begin using AutoFE-X.

