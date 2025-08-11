"""
ML Plotter - 简化的机器学习实验可视化库
Setup script for PyPI distribution
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ml-plotter",
    version="1.0.0",
    author="ML Plotter Team",
    author_email="your-email@example.com",
    description="简化的机器学习实验结果可视化库，保留专业学术风格",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-plotter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords="machine learning, visualization, plotting, academic, research, matplotlib",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ml-plotter/issues",
        "Source": "https://github.com/yourusername/ml-plotter",
        "Documentation": "https://github.com/yourusername/ml-plotter#readme",
    },
)