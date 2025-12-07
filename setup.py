"""
Setup script for AutoFE-X package.
"""

from setuptools import setup, find_packages
import os

# Read README
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = """
    AutoFE-X: Automated Feature Engineering + Data Profiling + Leakage Detection

    A next-gen toolkit that becomes the brain of any ML pipeline by combining:
    - Automatic feature engineering (classic + deep)
    - Data quality analysis
    - Leakage detection
    - Auto-benchmarking of feature sets
    - Graph-based feature lineage

    Lightweight, fast, and interpretable.
    """

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

try:
    requirements = read_requirements('requirements.txt')
except FileNotFoundError:
    requirements = [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'networkx>=2.6.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
    ]

setup(
    name="autofex",
    version="0.1.0",
    author="AutoFE-X Team",
    author_email="contact@autofe-x.com",
    description="Automated Feature Engineering + Data Profiling + Leakage Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autofe-x/autofe-x",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8,<3.16",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.800",
            "sphinx>=4.0.0",
            "jupyter>=1.0.0",
            "codecov>=2.1.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "graphviz>=0.17.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autofex=autofex.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
