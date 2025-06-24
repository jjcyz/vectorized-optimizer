"""
Setup script for Opt2Vec package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="opt2vec",
    version="0.1.0",
    author="Opt2Vec Team",
    author_email="",
    description="A lightweight meta-learning optimizer that learns to optimize by creating embedding vectors from optimization history",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/opt2vec",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "opt2vec-test=test_opt2vec:main",
            "opt2vec-example=example_usage:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="optimization, meta-learning, neural networks, pytorch, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/opt2vec/issues",
        "Source": "https://github.com/yourusername/opt2vec",
        "Documentation": "https://opt2vec.readthedocs.io/",
    },
)
