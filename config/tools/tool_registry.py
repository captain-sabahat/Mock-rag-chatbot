# ============================================================================
# TOOL REGISTRY - Metadata from .env Configuration
# ============================================================================

"""
Tool metadata registry - Central catalog of available tools with .env models.

This registry provides METADATA about which tools are available and which is active.
It reads tool names and model identifiers from .env variables.

RESPONSIBILITY:
- Store metadata about what tools are available
- Track which tools are currently active (from .env config)
- Validate that active tools are available
- Provide tool information to pipeline
- NO tool imports (that's src/ job)
- NO implementation code

ARCHITECTURE RULE:
config/ → knows about configuration ONLY
src/ → imports from config/ AND imports tool implementations

ROLE IN PIPELINE:
- Single source of truth for active tools and available tools
- Which tool strategies/providers are available
- Which one is currently active (from .env)
- Model identifiers for each tool (from .env)
- Validation that active tool is available

BENEFITS:
- Centralized tool metadata
- Clean architecture (no circular imports)
- Easy to add/remove tools
- Easy to see what's active vs available
- All tool names and models from .env
"""

from typing import Dict, Any, Optional, List
from config.settings import get_settings
import logging
import os

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Tool metadata registry - Central catalog of available tools.
    
    Reads tool configuration from .env and provides metadata about:
    - What tools are available (not active)
    - What tools are active (from .env)
    - Model identifiers for each tool (from .env)
    - Whether a specific tool is available
    """

    def __init__(self) -> None:
        """Initialize tool registry metadata from settings."""
        self.settings = get_settings()
        self._init_metadata()
        self._validate()
        logger.info("📋 Tool registry initialized")

    def _init_metadata(self) -> None:
        """Initialize static lists of available tools with .env models."""
        
        # ===================================================================
        # CHUNKING STRATEGIES - What text chunkers are available
        # ===================================================================
        
        self.available_chunkers = {
            "recursive": {
                "description": "Recursive character-based splitting",
                "module": "src.tools.chunking.recursive_chunker",
                "class": "RecursiveChunker",
                "production_ready": True,
                "requires": ["langchain-text-splitters"],
            },
            "token": {
                "description": "Token-aware chunking",
                "module": "src.tools.chunking.token_chunker",
                "class": "TokenChunker",
                "production_ready": False,
                "requires": ["tiktoken"],
            },
            "semantic": {
                "description": "Semantic similarity-based chunking",
                "module": "src.tools.chunking.semantic_chunker",
                "class": "SemanticChunker",
                "production_ready": False,
                "requires": ["sentence-transformers"],
            },
        }
        
        # ===================================================================
        # EMBEDDERS - What embedding providers are available with .env models
        # ===================================================================
        
        self.available_embedders = {
            "huggingface": {
                "description": "HuggingFace sentence-transformers (free, open-source)",
                "module": "src.tools.embeddings.huggingface_embedder",
                "class": "HuggingFaceEmbedder",
                "production_ready": True,
                "requires": ["sentence-transformers"],
                "tier": "development",
                "model": self.settings.embeddings.huggingface.model_name,  # From .env
                "dimensions": self.settings.embeddings.huggingface.dimension,  # From .env
            },
            "openai": {
                "description": "OpenAI embeddings API (paid, high quality)",
                "module": "src.tools.embeddings.openai_embedder",
                "class": "OpenAIEmbedder",
                "production_ready": True,
                "requires": ["openai"],
                "tier": "production",
                "model": self.settings.embeddings.openai.model,  # From .env
                "dimensions": self.settings.embeddings.openai.dimension,  # From .env
                "requires_api_key": True,
            },
            "cohere": {
                "description": "Cohere embeddings API (paid, alternative)",
                "module": "src.tools.embeddings.cohere_embedder",
                "class": "CohereEmbedder",
                "production_ready": False,
                "requires": ["cohere"],
                "tier": "production",
                "model": self.settings.embeddings.cohere.model,  # From .env
                "dimensions": self.settings.embeddings.cohere.dimension,  # From .env
                "requires_api_key": True,
            },
        }
        
        # ===================================================================
        # VECTOR DATABASES - What vector DBs are available with .env config
        # ===================================================================
        
        self.available_vectordbs = {
            "faiss": {
                "description": "FAISS in-memory vector index (free, fast, local)",
                "module": "src.tools.vectordb.faiss_client",
                "class": "FAISSClient",
                "production_ready": True,
                "requires": ["faiss-cpu"],  # or faiss-gpu
                "tier": "development",
                "data_path": self.settings.vectordb.faiss.data_path,  # From .env
            },
            "qdrant": {
                "description": "Qdrant persistent vector DB (open-source, cloud)",
                "module": "src.tools.vectordb.qdrant_client",
                "class": "QdrantClient",
                "production_ready": True,
                "requires": ["qdrant-client"],
                "tier": "production",
                "url": self.settings.vectordb.qdrant.url,  # From .env
                "requires_api_key": self.settings.vectordb.qdrant.api_key is not None,
            },
            "pinecone": {
                "description": "Pinecone managed vector DB (paid, serverless)",
                "module": "src.tools.vectordb.pinecone_client",
                "class": "PineconeClient",
                "production_ready": False,
                "requires": ["pinecone-client"],
                "tier": "production",
                "environment": self.settings.vectordb.pinecone.environment,  # From .env
                "requires_api_key": True,
            },
            "weaviate": {
                "description": "Weaviate open-source vector DB (distributed)",
                "module": "src.tools.vectordb.weaviate_client",
                "class": "WeaviateClient",
                "production_ready": False,
                "requires": ["weaviate-client"],
                "tier": "production",
            },
            "milvus": {
                "description": "Milvus open-source vector DB (scalable)",
                "module": "src.tools.vectordb.milvus_client",
                "class": "MilvusClient",
                "production_ready": False,
                "requires": ["pymilvus"],
                "tier": "production",
            },
        }
        
        # ===================================================================
        # PARSERS - What file parsers are available
        # ===================================================================
        
        self.available_parsers = {
            "pdf": {
                "description": "PDF file parser",
                "module": "src.tools.parsing.pdf_parser",
                "class": "PDFParser",
                "production_ready": True,
                "requires": ["pymupdf", "pypdf"],
            },
            "txt": {
                "description": "Plain text file parser",
                "module": "src.tools.parsing.text_parser",
                "class": "TextParser",
                "production_ready": True,
                "requires": [],
            },
            "json": {
                "description": "JSON file parser",
                "module": "src.tools.parsing.json_parser",
                "class": "JSONParser",
                "production_ready": True,
                "requires": [],
            },
            "docx": {
                "description": "DOCX file parser",
                "module": "src.tools.parsing.docx_parser",
                "class": "DOCXParser",
                "production_ready": False,
                "requires": ["python-docx"],
            },
        }

    def _validate(self) -> None:
        """Validate that configured tools are available."""
        errors = []
        
        # Check chunker
        active_chunker = self.settings.chunking.strategy
        if active_chunker not in self.available_chunkers:
            errors.append(
                f"❌ Chunker '{active_chunker}' not available. "
                f"Available: {list(self.available_chunkers.keys())}"
            )
        
        # Check embedder
        active_embedder = self.settings.embeddings.active_provider
        if active_embedder not in self.available_embedders:
            errors.append(
                f"❌ Embedder '{active_embedder}' not available. "
                f"Available: {list(self.available_embedders.keys())}"
            )
        
        # Check vector DB
        active_vectordb = self.settings.vectordb.active_provider
        if active_vectordb not in self.available_vectordbs:
            errors.append(
                f"❌ Vector DB '{active_vectordb}' not available. "
                f"Available: {list(self.available_vectordbs.keys())}"
            )
        
        if errors:
            for error in errors:
                logger.error(error)
            raise ValueError("\n".join(errors))
        
        logger.info(
            f"✅ Tools validated - "
            f"Active: {active_chunker} / {active_embedder} / {active_vectordb}"
        )

    # =========================================================================
    # GETTERS - What's Active?
    # =========================================================================

    def get_active_chunker(self) -> str:
        """Get currently active chunking strategy."""
        return self.settings.chunking.strategy

    def get_active_embedder(self) -> str:
        """Get currently active embedding provider."""
        return self.settings.embeddings.active_provider

    def get_active_vectordb(self) -> str:
        """Get currently active vector DB provider."""
        return self.settings.vectordb.active_provider

    # =========================================================================
    # CHECKERS - Is Available?
    # =========================================================================

    def is_chunker_available(self, strategy: str) -> bool:
        """Check if a chunking strategy is available."""
        return strategy in self.available_chunkers

    def is_embedder_available(self, provider: str) -> bool:
        """Check if an embedding provider is available."""
        return provider in self.available_embedders

    def is_vectordb_available(self, provider: str) -> bool:
        """Check if a vector DB provider is available."""
        return provider in self.available_vectordbs

    def is_parser_available(self, parser_type: str) -> bool:
        """Check if a file parser is available."""
        return parser_type in self.available_parsers

    # =========================================================================
    # GETTERS - Tool Info with .env Models
    # =========================================================================

    def get_chunker_info(self, strategy: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a chunking strategy."""
        return self.available_chunkers.get(strategy)

    def get_embedder_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get metadata about an embedding provider (includes .env model)."""
        return self.available_embedders.get(provider)

    def get_vectordb_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a vector DB provider (includes .env config)."""
        return self.available_vectordbs.get(provider)

    # =========================================================================
    # LISTERS - What's Available?
    # =========================================================================

    def list_chunkers(self) -> List[str]:
        """List all available chunking strategies."""
        return list(self.available_chunkers.keys())

    def list_embedders(self) -> List[str]:
        """List all available embedding providers."""
        return list(self.available_embedders.keys())

    def list_vectordbs(self) -> List[str]:
        """List all available vector DB providers."""
        return list(self.available_vectordbs.keys())

    def list_parsers(self) -> List[str]:
        """List all available file parsers."""
        return list(self.available_parsers.keys())

    # =========================================================================
    # SUMMARY - Full Overview with .env Tool Names
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Get full summary of tool registry state with .env models."""
        active_embedder = self.get_active_embedder()
        active_vectordb = self.get_active_vectordb()
        
        return {
            "development_mode": True,
            "active": {
                "chunker": self.get_active_chunker(),
                "embedder": {
                    "name": active_embedder,
                    "model": self.available_embedders[active_embedder].get("model"),
                    "dimensions": self.available_embedders[active_embedder].get("dimensions"),
                },
                "vectordb": {
                    "name": active_vectordb,
                    "config": {
                        k: v for k, v in self.available_vectordbs[active_vectordb].items()
                        if k not in ["description", "module", "class", "requires", "production_ready", "tier"]
                    }
                },
            },
            "available": {
                "chunkers": [
                    {
                        "name": name,
                        "description": info["description"],
                        "production_ready": info["production_ready"],
                    }
                    for name, info in self.available_chunkers.items()
                ],
                "embedders": [
                    {
                        "name": name,
                        "description": info["description"],
                        "model": info.get("model"),
                        "dimensions": info.get("dimensions"),
                        "production_ready": info["production_ready"],
                        "tier": info.get("tier"),
                    }
                    for name, info in self.available_embedders.items()
                ],
                "vectordbs": [
                    {
                        "name": name,
                        "description": info["description"],
                        "production_ready": info["production_ready"],
                        "tier": info.get("tier"),
                    }
                    for name, info in self.available_vectordbs.items()
                ],
                "parsers": [
                    {
                        "name": name,
                        "description": info["description"],
                    }
                    for name, info in self.available_parsers.items()
                ],
            },
            "production_ready": {
                "chunker": self.available_chunkers[self.get_active_chunker()].get("production_ready"),
                "embedder": self.available_embedders[self.get_active_embedder()].get("production_ready"),
                "vectordb": self.available_vectordbs[self.get_active_vectordb()].get("production_ready"),
            },
        }

# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_registry: Optional["ToolRegistry"] = None

def get_tool_registry() -> ToolRegistry:
    """
    Get global ToolRegistry instance (singleton).
    
    Initialized once on first access, cached and reused.
    
    Returns:
        ToolRegistry: Global tool metadata registry with .env tools
    
    Raises:
        ValueError: If configured tools are not available
    """
    global _registry
    
    if _registry is None:
        _registry = ToolRegistry()
    
    return _registry