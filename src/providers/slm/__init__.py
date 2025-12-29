"""
SLM Providers package.

Re-exports default providers from individual implementation files.
"""

from src.providers.slm.base import ISLMProvider
from src.providers.slm.hf_causal import default_provider as hf_causal_default
from src.providers.slm.hf_seq2seq import default_provider as hf_seq2seq_default

__all__ = [
    "ISLMProvider",
    "hf_causal_default",
    "hf_seq2seq_default",
]
