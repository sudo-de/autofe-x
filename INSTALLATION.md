# 📦 AutoFE-X Installation Guide

## Installation Methods

### Method 1: Install from PyPI (Recommended)

Once published to PyPI:

```bash
pip install autofex
```

### Method 2: Install from GitHub

```bash
pip install git+https://github.com/autofe-x/autofe-x.git
```

### Method 3: Install from Local Source

**Prerequisites:**
- Python 3.8 or higher
- pip and setuptools

**Steps:**

1. **Clone the repository:**
```bash
git clone https://github.com/autofe-x/autofe-x.git
cd autofe-x
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Or install with development dependencies:**
```bash
pip install -e ".[dev]"
```

### Method 4: Install in Google Colab / Jupyter

**Option A: From GitHub (if repository is public):**
```python
!pip install git+https://github.com/autofe-x/autofe-x.git
```

**Option B: Upload and install from local:**
```python
# Upload the autofe-x folder to Colab
# Then run:
!pip install -e /content/autofe-x
```

**Option C: Install from PyPI (once published):**
```python
!pip install autofex
```

## Troubleshooting

### Error: "file:///content does not appear to be a Python project"

**Cause:** You're not in the correct directory or the package files are missing.

**Solution:**

1. **Verify you're in the right directory:**
```bash
# Check current directory
pwd

# List files to verify setup.py or pyproject.toml exists
ls -la | grep -E "(setup.py|pyproject.toml)"
```

2. **Navigate to the package root:**
```bash
cd /path/to/autofe-x
```

3. **Verify package structure:**
```bash
# Should show: setup.py, pyproject.toml, autofex/, README.md
ls -la
```

4. **Install from the correct location:**
```bash
pip install -e .
```

### Error: "No module named 'setuptools'"

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
```

### Error: "Package not found"

**Solution:**
```bash
# Make sure you're in the package root directory
cd /path/to/autofe-x

# Verify setup.py exists
ls setup.py

# Try installing again
pip install -e .
```

## Verification

After installation, verify it works:

```python
import autofex
print(autofex.__version__)

from autofex import AutoFEX
print("✅ AutoFE-X installed successfully!")
```

## Development Installation

For contributing to AutoFE-X:

```bash
# Clone repository
git clone https://github.com/autofe-x/autofe-x.git
cd autofe-x

# Install with all development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black autofex/
flake8 autofex/
mypy autofex/
```

## Requirements

- Python >= 3.8
- pip >= 21.0
- setuptools >= 61.0

## Dependencies

Core dependencies (installed automatically):
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- networkx >= 2.6.0

Optional dependencies:
- matplotlib >= 3.4.0 (for visualization)
- seaborn >= 0.11.0 (for visualization)
