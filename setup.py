# Path: /home/chanakya/sound_classification/setup.py

from setuptools import setup, find_packages

setup(
    name="pump-net",
    version="1.0.0",
    description="Industrial Pump Anomaly Detection System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "librosa>=0.10.1",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "streamlit>=1.27.0",
        "plotly>=5.17.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)