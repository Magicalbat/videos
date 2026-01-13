#!/usr/bin/env python3
"""
Setup script for GenomL - Quantitative Biology ML Library
"""

from setuptools import setup, find_packages

with open("README_genoml.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="genoml",
    version="0.1.0",
    author="GenomL Contributors",
    description="A Machine Learning Library for Quantitative Biology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlfromscratch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "viz": [
            "matplotlib>=3.0",
            "seaborn",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools here if needed
        ],
    },
)
