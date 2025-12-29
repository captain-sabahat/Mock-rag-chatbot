"""
FILE: src/providers/embeddings/__init__.py

Embeddings providers package.

This file re-exports:
- IEmbeddingsProvider interface
- default provider(s)

ServiceContainer will import directly from:
  src.providers.embeddings.<provider_name>  (e.g., huggingface.py)

Optionally, other code can do:
  from src.providers.embeddings import default_provider
"""

from .base import IEmbeddingsProvider
from .huggingface import default_provider as huggingface_default

__all__ = [
    "IEmbeddingsProvider",
    "huggingface_default",
]
