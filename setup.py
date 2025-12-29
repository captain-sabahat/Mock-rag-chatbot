#================================================================================
# Setup.py - Traditional Python Package Setup
# ================================================================================
# This file supports traditional pip installation and is often used for
# backward compatibility. Modern projects prefer pyproject.toml.
#
# Install with: pip install -e .
#

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="user-chatbot-backend",
    version="1.0.0",
    author="Your Team",
    author_email="team@example.com",
    description="Production-grade RAG backend for user chatbot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/user-chatbot-backend",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "redis[asyncio]>=5.0.0",
        "qdrant-client[fastembed]>=2.5.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "python-magic>=0.4.27",
        "httpx[http2]>=0.25.0",
        "python-dotenv>=1.0.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": ["torch[cuda12]>=2.0.0"],
        "apple-silicon": ["torch[mps]>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "user-chatbot=src.main:main",
        ],
    },
)
