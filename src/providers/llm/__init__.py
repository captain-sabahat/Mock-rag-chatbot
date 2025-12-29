"""
FILE: src/providers/llm/__init__.py

LLM providers package.

Re-exports the interface + default providers for convenience.
ServiceContainer normally imports directly from:
  src.providers.llm.<provider_name>  (e.g., gemini, huggingface)

But other code/tests may import defaults from here.
"""

from .base import ILLMProvider
from .gemini import default_provider as gemini_default
from .huggingface import default_provider as huggingface_default

__all__ = [
    "ILLMProvider",
    "gemini_default",
    "huggingface_default",
]
