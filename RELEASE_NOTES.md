# Release Notes

## What are Release Notes?

Release notes are documents that describe what changed in each version of your software. They help users understand:
- What's new
- What bugs were fixed
- What improvements were made
- How to upgrade
- Any breaking changes

---

## Version 0.1.4 (2024-12-09)

### 🐛 Bug Fixes

- **Fixed CI Integration Test**: Resolved AssertionError in integration tests caused by stale cache from previous runs
  - Disabled caching in CI test workflows to ensure deterministic test results
  - Tests now run with `enable_cache=False` to prevent cache-related failures

### 🔧 Technical Improvements

- **CI/CD Improvements**: 
  - Updated integration test to disable caching for reliable test execution
  - Updated smoke test to disable caching for consistency
  - Fixed workflow YAML formatting issues

### 🚀 Installation

```bash
pip install autofex==0.1.4
```

### 🔄 Migration from 0.1.3

No breaking changes. This is a patch release that fixes CI test reliability.

---

## Version 0.1.3 (2024-12-09)

### 🔧 Technical Improvements

- **Repository URLs**: Updated all repository URLs to point to correct GitHub repository (`sudo-de/autofe-x`)

### 🚀 Installation

```bash
pip install autofex==0.1.3
```

### 🔄 Migration from 0.1.2

No breaking changes. This is a patch release that updates repository URLs.

---

## Version 0.1.2 (2024-12-09)

### ✨ New Features

- **CLI Command**: Added command-line interface with `autofex` command
  - `autofex --version` / `autofex -v`: Show version
  - `autofex --info` / `autofex -i`: Show package information and quick start guide
  - `autofex --components` / `autofex -c`: List all available components
  - `autofex --help` / `autofex -h`: Show help with examples

### 🐛 Bug Fixes

- **Fixed Type Checking**: Resolved mypy type error in `ultra_stats.py` bootstrap analysis by properly handling numpy floating types and separating list/ndarray types

### 🔧 Technical Improvements

- **Type Safety**: Improved type annotations in bootstrap analysis to ensure compatibility between Python float and numpy floating types
- **CLI Enhancement**: Interactive command-line interface for exploring package capabilities

### 🚀 Installation

```bash
pip install autofex==0.1.2
```

### 📖 Usage

```bash
# Check version
autofex --version

# Get package information
autofex --info

# List available components
autofex --components

# Show help
autofex --help
```

### 🔄 Migration from 0.1.1

No breaking changes. This is a patch release that adds CLI functionality and fixes type checking issues.

---

## Version 0.1.1 (2024-12-09)

### 🐛 Bug Fixes

- **Fixed PyPI Publishing**: Updated CI workflow to build packages directly from release tags, ensuring correct version is published
- **Fixed Import Errors**: Implemented lazy loading in `__init__.py` using `importlib` to avoid pandas dependency during package build
- **Fixed Type Checking**: Added type ignore comments for FeatureEngineer import conflicts to resolve mypy errors
- **Fixed Code Formatting**: Applied black formatting to ensure consistent code style
- **Fixed Dependency Constraints**: Updated numpy version constraint to `<2.1.0` to avoid compatibility issues with numba and Google Colab

### 🔧 Technical Improvements

- **Lazy Loading**: All package exports now use lazy loading via `__getattr__`, preventing import errors when dependencies aren't available
- **Build System**: Added pandas to `build-system.requires` to ensure availability during build process
- **CI/CD**: 
  - Fixed PyPI publish action parameter (`username` → `user`)
  - Updated workflow to build from release tag directly
  - Improved version checking using `pkg_resources` instead of direct imports

### 📦 Dependencies

- **numpy**: Constrained to `<2.1.0` for better compatibility with numba and Google Colab
- **pandas**: Constrained to `>=1.3.0,<2.3.0` for stability

### 📝 Documentation

- Fixed Python code examples in README.md
- Updated all code blocks to use correct class names (removed "Advanced" prefix)
- Fixed custom transformer examples with working code

### 🚀 Installation

```bash
pip install autofex==0.1.1
```

### 🔄 Migration from 0.1.0

No breaking changes. This is a patch release that fixes build and publishing issues.

---

## Version 0.1.0 (Initial Release)

### ✨ Features

- **Automated Feature Engineering**: Classic mathematical transformations, interactions, and encodings
- **Data Profiling**: Comprehensive data quality analysis with outlier detection and statistical summaries
- **Leakage Detection**: Advanced algorithms to detect target leakage, train-test contamination, and statistical anomalies
- **Auto-Benchmarking**: Automatically compare feature sets across multiple models with ablation studies
- **Graph-based Lineage**: Track feature transformations and dependencies with full provenance
- **Progress Tracking**: Real-time progress bars with ETA for long-running operations
- **Intelligent Caching**: Operation-based caching with TTL and size management

### 📦 Core Components

- `AutoFEX`: Main orchestration class
- `FeatureEngineer`: Feature engineering pipeline
- `DataProfiler`: Data quality analysis
- `LeakageDetector`: Leakage detection algorithms
- `FeatureBenchmarker`: Model benchmarking
- `FeatureLineageTracker`: Feature lineage tracking

### 🎯 Supported Python Versions

Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

### 📚 Documentation

- Comprehensive README with examples
- API documentation
- Usage guides

---

## How to Write Release Notes

### Structure

1. **Version Number & Date**: Clear version identifier and release date
2. **Categories**: Organize changes by type:
   - 🐛 Bug Fixes
   - ✨ New Features
   - 🔧 Improvements
   - 📦 Dependencies
   - 📝 Documentation
   - ⚠️ Breaking Changes
   - 🔄 Migration Guide

### Best Practices

- **Be Specific**: Describe what changed and why
- **User-Focused**: Explain impact on users
- **Include Examples**: Show code examples for new features
- **Migration Notes**: Guide users on upgrading
- **Link to Issues**: Reference GitHub issues/PRs when relevant
- **Use Emojis**: Make sections easy to scan (optional but helpful)

### Example Format

```markdown
## Version X.Y.Z (YYYY-MM-DD)

### 🐛 Bug Fixes
- Fixed issue with [description] (#issue-number)

### ✨ New Features
- Added [feature name] ([#pr-number])
  - Description of feature
  - Usage example

### 🔧 Improvements
- Improved [component] performance by X%

### ⚠️ Breaking Changes
- [Component] now requires [change]
  - Migration: [how to update]
```

### Version Numbering

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

### For Your Project

You can create release notes:
1. **Manually**: Write in `RELEASE_NOTES.md` or `CHANGELOG.md`
2. **GitHub Releases**: Use the release notes section when creating a release
3. **Automated**: Use tools like `towncrier` or `release-notes-generator`

### Next Steps

1. Create a GitHub release with tag `v0.1.2`
2. Copy the release notes content to the GitHub release description
3. Publish the release to trigger PyPI publishing

