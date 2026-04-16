"""
Setup script for quantum-walk-mvc package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="quantum-walk-mvc",
    version="1.0.0",
    author="Fabricio de Souza Luiz",
    author_email="fsluiz@unicamp.br",
    description="Quantum walk algorithms for Minimum Vertex Cover problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fsluiz/quantum-walk-mvc",
    project_urls={
        "Bug Tracker": "https://github.com/fsluiz/quantum-walk-mvc/issues",
        "Documentation": "https://github.com/fsluiz/quantum-walk-mvc#readme",
        "Source Code": "https://github.com/fsluiz/quantum-walk-mvc",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black",
            "flake8",
        ],
        "quantum": [
            "qutip>=4.6.0",
        ],
        "bloqade": [
            "bloqade>=0.15.0",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qw-mvc-ba=experiments.run_barabasi_albert:main",
            "qw-mvc-er=experiments.run_erdos_renyi:main",
            "qw-mvc-regular=experiments.run_regular:main",
        ],
    },
)
