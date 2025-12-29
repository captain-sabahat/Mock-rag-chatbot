# src/__init__.py

"""
User-side RAG chatbot backend package.

This package contains:
- api: FastAPI routes and dependencies
- config: settings and constants
- core: tool handlers (LLM, SLM, vector DB, Redis, etc.)
- pipeline: orchestration and RAG logic
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
