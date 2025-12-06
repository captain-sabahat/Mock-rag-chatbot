# ============================================================================
# SETTINGS - Type-safe, validated configuration structures from .env
# ============================================================================

"""
Type-safe, validated configuration structures with .env integration.

Loads configurations from:
1. .env file (environment variables - highest priority)
2. YAML config files (base configurations)
3. Pydantic validation (type checking and validation)

RESPONSIBILITY:
- Define configuration schema using Pydantic models
- Validate types and required fields
- Provide type-safe access to config throughout the app
- NO file I/O here (that's loader.py)
- NO business logic (that's src/)

BENEFITS:
- IDE autocomplete: settings.session_store.backend ✓
- Type checking: mypy catches wrong types ✓
- Validation: Pydantic validates on init ✓
- Single source of truth ✓
- All values from .env file ✓

USAGE:
from config import get_settings

settings = get_settings()
backend = settings.session_store.backend # Type-safe!
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import os
from config.loader import get_config_loader

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS - Configuration Options (What's valid?)
# ============================================================================

class SessionStoreBackend(str, Enum):
    """Supported session storage backends."""
    SQLITE = "sqlite"      # Development: File-based, no setup
    REDIS = "redis"        # Production: In-memory cache
    MONGODB = "mongodb"    # Production: Distributed, persistent
    HYBRID = "hybrid"      # Production: SQLite + Redis combo

class ChunkingStrategy(str, Enum):
    """Supported text chunking strategies."""
    RECURSIVE = "recursive"   # Development: Default, fast
    TOKEN = "token"          # Optional: Token-aware
    SEMANTIC = "semantic"    # Future: ML-based (requires model)

class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"           # Production: High quality, paid
    HUGGINGFACE = "huggingface" # Development: Free, open-source
    COHERE = "cohere"           # Production: Alternative, paid

class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    FAISS = "faiss"         # Development: In-memory, fast, free
    QDRANT = "qdrant"       # Production: Persistent, cloud-ready
    PINECONE = "pinecone"   # Production: Managed, paid
    WEAVIATE = "weaviate"   # Production: Open-source, distributed
    MILVUS = "milvus"       # Production: Open-source, scalable

# ============================================================================
# SESSION STORE SETTINGS
# ============================================================================

class SQLiteSettings(BaseModel):
    """SQLite session store configuration from .env"""
    enabled: bool = True
    db_path: str = Field(default="./data/sessions/sessions.db")
    max_connections: int = 10
    enable_wal: bool = True
    auto_vacuum: bool = True
    journal_mode: str = "WAL"
    timeout_seconds: int = 10
    check_same_thread: bool = False

class RedisSettings(BaseModel):
    """Redis session store configuration from .env"""
    enabled: bool = False
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = 0
    password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    use_tls: bool = False
    max_connections: int = 50
    timeout_seconds: int = 30
    socket_keepalive: bool = True
    session_ttl_seconds: int = 604800  # 7 days

class MongoDBSettings(BaseModel):
    """MongoDB session store configuration from .env"""
    enabled: bool = False
    uri: str = Field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    database: str = "rag_pipeline"
    collection: str = "sessions"
    username: Optional[str] = Field(default_factory=lambda: os.getenv("MONGODB_USERNAME"))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("MONGODB_PASSWORD"))
    max_pool_size: int = 50
    min_pool_size: int = 10

class SessionStoreSettings(BaseModel):
    """Master session store configuration."""
    backend: SessionStoreBackend = Field(default=SessionStoreBackend.SQLITE)
    operation_timeout_seconds: int = 600
    max_retries: int = 5
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)

    class Config:
        use_enum_values = True

# ============================================================================
# CHUNKING SETTINGS
# ============================================================================

class ChunkingSettings(BaseModel):
    """Text chunking configuration from .env"""
    strategy: ChunkingStrategy = Field(
        default_factory=lambda: ChunkingStrategy(os.getenv("CHUNKING_STRATEGY", "recursive"))
    )
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    timeout_seconds: int = 30
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
        description="Separators for recursive chunking (tried in order)"
    )

    class Config:
        use_enum_values = True

# ============================================================================
# EMBEDDINGS SETTINGS
# ============================================================================

class HuggingFaceEmbedderSettings(BaseModel):
    """HuggingFace embedder configuration from .env"""
    enabled: bool = True
    model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDINGS_HUGGINGFACE_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDINGS_HUGGINGFACE_DIMENSIONS", "384"))
    )
    device: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_HUGGINGFACE_DEVICE", "cpu")
    )
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_folder: str = "./models/embeddings"

class OpenAIEmbedderSettings(BaseModel):
    """OpenAI embedder configuration from .env"""
    enabled: bool = False
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("EMBEDDINGS_OPENAI_API_KEY"))
    model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_OPENAI_MODEL", "text-embedding-3-small")
    )
    dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDINGS_OPENAI_DIMENSIONS", "1536"))
    )
    batch_size: int = 20

class CohereEmbedderSettings(BaseModel):
    """Cohere embedder configuration from .env"""
    enabled: bool = False
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("EMBEDDINGS_COHERE_API_KEY"))
    model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_COHERE_MODEL", "embed-english-v3.0")
    )
    dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDINGS_COHERE_DIMENSIONS", "1024"))
    )
    batch_size: int = 20

class EmbeddingsSettings(BaseModel):
    """Embeddings generation configuration from .env"""
    active_provider: EmbeddingProvider = Field(
        default_factory=lambda: EmbeddingProvider(os.getenv("EMBEDDINGS_PROVIDER", "huggingface"))
    )
    huggingface: HuggingFaceEmbedderSettings = Field(default_factory=HuggingFaceEmbedderSettings)
    openai: OpenAIEmbedderSettings = Field(default_factory=OpenAIEmbedderSettings)
    cohere: CohereEmbedderSettings = Field(default_factory=CohereEmbedderSettings)
    cache_enabled: bool = False

    class Config:
        use_enum_values = True

# ============================================================================
# VECTOR DATABASE SETTINGS
# ============================================================================

class FAISSSettings(BaseModel):
    """FAISS vector database configuration from .env"""
    enabled: bool = True
    index_type: str = "IVFFlat"
    metric_type: str = "L2"
    nlist: int = 100
    nprobe: int = 10
    data_path: str = Field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/vectordb/faiss_index")
    )
    save_index: bool = True
    normalize: bool = True

class QdrantSettings(BaseModel):
    """Qdrant vector database configuration from .env"""
    enabled: bool = False
    url: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    collection_name: str = "documents"
    vector_size: int = 384
    distance: str = "Cosine"

class PineconeSettings(BaseModel):
    """Pinecone vector database configuration from .env"""
    enabled: bool = False
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    environment: str = Field(
        default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    )
    index_name: str = "rag-documents"
    dimension: int = 384

class VectorDBSettings(BaseModel):
    """Vector database configuration from .env"""
    active_provider: VectorDBProvider = Field(
        default_factory=lambda: VectorDBProvider(os.getenv("VECTORDB_PROVIDER", "faiss"))
    )
    faiss: FAISSSettings = Field(default_factory=FAISSSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    timeout_seconds: int = 30

    class Config:
        use_enum_values = True

# ============================================================================
# ROOT SETTINGS - Main Configuration Container
# ============================================================================

class Settings(BaseModel):
    """
    Application Settings - Main configuration container.
    
    Type-safe, validated configuration for the entire application.
    Loads from:
    1. .env file (highest priority)
    2. YAML config files (base)
    3. Pydantic validation (type checking)
    
    ARCHITECTURE:
    - .env → Raw environment variables
    - config/loader.py → Load YAML + merge .env
    - config/settings.py → Validate with Pydantic (THIS FILE)
    - config/__init__.py → Expose via get_settings()
    
    USAGE:
    from config import get_settings
    
    settings = get_settings()
    backend = settings.session_store.backend  # Type-safe access
    embedder = settings.embeddings.active_provider
    vectordb = settings.vectordb.active_provider
    """
    
    session_store: SessionStoreSettings = Field(
        default_factory=SessionStoreSettings,
        description="Session storage configuration"
    )
    chunking: ChunkingSettings = Field(
        default_factory=ChunkingSettings,
        description="Text chunking configuration"
    )
    embeddings: EmbeddingsSettings = Field(
        default_factory=EmbeddingsSettings,
        description="Embeddings generation configuration"
    )
    vectordb: VectorDBSettings = Field(
        default_factory=VectorDBSettings,
        description="Vector database configuration"
    )

    class Config:
        use_enum_values = True
        title = "RAG Pipeline Configuration"
        description = "Type-safe application configuration from .env"

# ============================================================================
# SINGLETON PATTERN - Global Settings Instance
# ============================================================================

_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Get or create the global Settings instance (singleton).
    
    Implements singleton pattern for efficient config access.
    Configuration is loaded once on first call and cached.
    
    Returns:
        Settings: Global settings instance (type-safe config)
    
    Example:
        settings = get_settings()
        timeout = settings.session_store.operation_timeout_seconds
        embedder = settings.embeddings.active_provider
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = Settings()
        logger.info("✅ Settings initialized from .env (singleton)")
    
    return _settings_instance

def reset_settings() -> None:
    """Reset settings instance (for testing purposes)."""
    global _settings_instance
    _settings_instance = None
    logger.debug("Settings reset for testing")