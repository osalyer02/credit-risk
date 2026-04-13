from setuptools import find_packages, setup


setup(
    name="credit-risk-platform",
    version="0.1.0",
    description="Production-style credit risk modeling and decision API",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "boto3>=1.34.0",
        "fastapi>=0.110.0",
        "joblib>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pydantic>=2.6.0",
        "pyyaml>=6.0.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "uvicorn>=0.29.0",
    ],
    extras_require={
        "aws": ["mangum>=0.17.0"],
        "explainability": ["shap>=0.46.0"],
        "dev": [
            "httpx>=0.27.0",
            "mypy>=1.9.0",
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
            "ruff>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crp-train=credit_risk.models.train:main",
            "crp-api=credit_risk.api.app:main",
        ]
    },
)
