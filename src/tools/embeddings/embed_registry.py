"""
================================================================================
EMBEDDINGS REGISTRY & FACTORY (CONFIG-DRIVEN)
src/tools/embeddings/registry_embed.py

PURPOSE:
- Unified config loading from YAML
- Factory for creating embedders
- NO node logic (separation of concerns)
- Config-driven provider switching
- Single source of truth

CRITICAL: This file handles CONFIG LOADING & FACTORY ONLY.
Node logic goes in embedding_node.py in pipeline folder.
================================================================================
"""

import logging
import yaml
import os
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MODEL
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings (from YAML)."""
    active_provider: str          # "huggingface", "openai", etc.
    provider_config: Dict[str, Any]  # Config for that provider
    dimension: int
    batch_size: int
    normalize_embeddings: bool
    max_seq_length: int
    timeout_seconds: int

# ============================================================================
# CONFIG LOADER (YAML → EmbeddingConfig)
# ============================================================================

def load_embedding_config(config_path: str = "config/defaults/embeddings.yaml") -> EmbeddingConfig:
    """
    Load embedding configuration from YAML file.
    
    Reads YAML structure:
    ```yaml
    embeddings:
      active_provider: "huggingface"
      huggingface:
        model_name: "BAAI/bge-base-en-v1.5"
        dimension: 768
        batch_size: 32
        normalize_embeddings: true
        max_seq_length: 512
        # ... more provider-specific config
    ```
    
    Args:
        config_path: Path to embeddings.yaml
        
    Returns:
        EmbeddingConfig instance
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If active_provider not found in config
    """
    try:
        with open(config_path, 'r') as f:
            raw_yaml = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        raise
    
    embeddings_section = raw_yaml.get('embeddings', {})
    active_provider = embeddings_section.get('active_provider')
    
    if not active_provider:
        raise ValueError("❌ 'active_provider' not found in embeddings.yaml")
    
    # Get provider-specific config
    provider_config = embeddings_section.get(active_provider, {})
    
    if not provider_config:
        raise ValueError(
            f"❌ No config found for provider '{active_provider}' in embeddings.yaml"
        )
    
    # Build EmbeddingConfig object
    config = EmbeddingConfig(
        active_provider=active_provider,
        provider_config=provider_config,
        dimension=provider_config.get('dimension', 768),
        batch_size=provider_config.get('batch_size', 32),
        normalize_embeddings=provider_config.get('normalize_embeddings', True),
        max_seq_length=provider_config.get('max_seq_length', 512),
        timeout_seconds=provider_config.get('timeout_seconds', 60),
    )
    
    logger.info(
        f"✅ EmbeddingConfig loaded: provider={config.active_provider}, "
        f"dimension={config.dimension}, batch_size={config.batch_size}"
    )
    
    return config

# ============================================================================
# EMBEDDING RESULT
# ============================================================================

@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    text: str
    vector: List[float]
    dimension: int
    provider: str
    processing_time_ms: float
    normalized: bool
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# BASE EMBEDDER CLASS
# ============================================================================

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.
    
    Subclasses receive config dict and use it (no hardcoding).
    """
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        """
        Initialize embedder.
        
        Args:
            provider_name: "huggingface", "openai", etc.
            config: Provider-specific configuration dict
        """
        self.provider_name = provider_name
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Read from config (never hardcode)
        self.dimension = config.get("dimension", 768)
        self.model_name = config.get("model_name", "unknown")
        self.batch_size = config.get("batch_size", 32)
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.max_seq_length = config.get("max_seq_length", 512)
        
        self.logger.info(
            f"✅ {self.__class__.__name__} initialized: "
            f"model={self.model_name}, dimension={self.dimension}, "
            f"normalize={self.normalize_embeddings}"
        )
    
    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed single text. Must implement in subclass."""
        pass
    
    async def embed_batch(self, texts: List[str]) -> tuple:
        """
        Embed multiple texts.
        
        Returns:
            (results list, total_embeddings count)
        """
        results = []
        start_time = time.time()
        
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return results, len(results)
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """L2 normalize a vector."""
        if not self.normalize_embeddings:
            return vector
        
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        
        if norm == 0:
            return vector
        
        return (arr / norm).tolist()

# ============================================================================
# EMBEDDER FACTORY
# ============================================================================

class EmbedderFactory:
    """
    Factory for creating embedders from config.
    
    Config-driven: if config says huggingface, creates HuggingFaceEmbedder.
    No hardcoding, no manual if/elif switching.
    """
    
    _registry: Dict[str, Type[BaseEmbedder]] = {}
    
    @classmethod
    def register(cls, provider_name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """Register an embedder implementation."""
        cls._registry[provider_name] = embedder_class
        logger.info(f"✅ Registered embedder: {provider_name}")
    
    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbedder:
        """
        Create embedder instance from configuration.
        
        Args:
            config: EmbeddingConfig loaded from YAML
            
        Returns:
            BaseEmbedder subclass instance
            
        Raises:
            ValueError: If provider not registered
        """
        provider = config.active_provider
        
        if provider not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"❌ Unknown provider: {provider}\n"
                f"Available: {available}"
            )
        
        embedder_class = cls._registry[provider]
        
        logger.info(f"🏭 Creating {provider} embedder from config...")
        
        # Pass config dict to embedder
        return embedder_class(provider, config.provider_config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._registry.keys())

__all__ = [
    'EmbeddingConfig',
    'EmbeddingResult',
    'BaseEmbedder',
    'EmbedderFactory',
    'load_embedding_config',
]