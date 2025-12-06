"""
================================================================================
TOOLS PACKAGE INITIALIZATION
src/tools/__init__.py

Purpose:
- Marks the 'tools' directory as a Python package
- Optionally sets version and package-level metadata
- Defines a clean public API for all tools submodules

Exports:
    - chunking
    - embeddings
    - ingestion
    - preprocessors
    - vectordb

Usage example:
    from tools import chunking, embeddings, ingestion, preprocessors, vectordb

================================================================================
"""

__version__ = "1.0.0"

# Optionally simplify top-level imports
from . import chunking
from . import embeddings
from . import ingestion
from . import preprocessors
from . import vectordb

# Suggest intended public exports
__all__ = ["chunking", "embeddings", "ingestion", "preprocessors", "vectordb"]
