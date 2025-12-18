"""
================================================================================
EMBEDDINGS PACKAGE - Config-driven, Factory-based
src/tools/embeddings/__init__.py

ARCHITECTURE:
- registry_embed.py: Config loading + Factory + Base class (TOOLS LAYER)
- bge_embedder.py: BGE implementation (TOOLS LAYER)
- embedding_node.py: Node orchestration + metrics (PIPELINE LAYER)

CLEAN SEPARATION:
✅ Tools layer: Config + Factory + Embedders (no pipeline logic)
✅ Pipeline layer: Node logic + State management + Metrics (no config parsing)
✅ Single source of truth: Config loaded once, used everywhere
✅ No redundancy: Each file has one clear purpose
✅ Easy to add new embedders: Just inherit + register
================================================================================
"""

import logging

from .embed_registry import (
    EmbeddingConfig,
    EmbeddingResult,
    BaseEmbedder,
    EmbedderFactory,
    load_embedding_config,
)
from .bge_embedder import BGEEmbedder

logger = logging.getLogger(__name__)

# Register embedders with factory
EmbedderFactory.register("huggingface", BGEEmbedder)

logger.info("✅ Embeddings package initialized")
logger.info(f"   Registered providers: {EmbedderFactory.list_providers()}")

__all__ = [
    'EmbeddingConfig',
    'EmbeddingResult',
    'BaseEmbedder',
    'BGEEmbedder',
    'EmbedderFactory',
    'load_embedding_config',
]