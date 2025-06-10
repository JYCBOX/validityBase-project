from setuptools import setup, find_packages

setup(
    name="validityBase_project",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas>=1.0",
        "numpy",
        "yfinance",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pre-commit",
        ],
    },
)
