from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-fuzzy-pso-dbscan",
    version="1.0.0",
    author="Seid Mehammed Abdu",
    author_email="seidmda@gmail.com",
    description="An enhanced density-based clustering approach for smart city data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-fuzzy-pso-dbscan",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "folium>=0.14.0",
        "streamlit-folium>=0.13.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "adaptive-fuzzy-pso-dbscan=app:main",
        ],
    },
    keywords="clustering, dbscan, pso, fuzzy-logic, smart-city, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/adaptive-fuzzy-pso-dbscan/issues",
        "Source": "https://github.com/yourusername/adaptive-fuzzy-pso-dbscan",
        "Documentation": "https://github.com/yourusername/adaptive-fuzzy-pso-dbscan#readme",
    },
)