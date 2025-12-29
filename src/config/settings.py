"""
================================================================================
FILE: src/config/settings.py
================================================================================

PURPOSE:
    Application settings and configuration loaded from environment variables.
    Uses Pydantic BaseSettings for automatic validation and type hints.
    Single source of truth for all application configuration.

WORKFLOW:
    1. At startup, load from environment variables (.env file or system env)
    2. Validate all settings (type checking, range validation)
    3. Fail fast if required settings missing or invalid
    4. Cache settings in memory (zero I/O after startup)
    5. Access throughout app via: settings.device, settings.llm_model_name, etc.

IMPORTS:
    - pydantic BaseSettings: Configuration management
    - os: Environment variable access
    - typing: Type hints

INPUTS:
    - Environment variables (from .env file or system env)
    - Examples:
        DEVICE=cuda
        LLM_MODEL_NAME=gpt-3.5-turbo
        SLM_MODEL_NAME=distilbart-cnn-6-6
        EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
        REDIS_URL=redis://localhost:6379
        VECTOR_DB_URL=http://localhost:6333
        QUANTIZATION=int8
        etc.

OUTPUTS:
    - Settings object with validated configuration
    - Accessed as: settings.device, settings.llm_timeout, etc.

CONFIGURATION CATEGORIES:
    1. Device Configuration
       - device: CPU/CUDA/Metal/NPU selection
       - optimization: GPU optimizations
    
    2. Model Configuration
       - llm_model_name: Fine-tuned LLM name
       - slm_model_name: SLM for summarization
       - embeddings_model_name: Embeddings model
    
    3. API Keys & Credentials
       - llm_api_key: API key for fine-tuned LLM
       - (other API keys as needed)
    
    4. Timeouts & Limits
       - llm_timeout: Max time for LLM inference
       - slm_timeout: Max time for SLM summarization
       - embeddings_timeout: Max time for embeddings
       - vector_db_timeout: Max time for vector search
    
    5. Cache Configuration
       - redis_url: Redis connection string
       - cache_ttl_default: Default cache TTL in seconds
    
    6. Vector DB Configuration
       - vector_db_url: Qdrant/Vector DB URL
       - vector_db_top_k: Number of results to return
       - vector_db_score_threshold: Min similarity score
    
    7. Memory & Optimization
       - slm_quantization: none/int8/int4 quantization
       - embeddings_batch_size: Batch size for embeddings
       - document_chunk_size: Size of document chunks
    
    8. Server Configuration
       - server_host: Server host (0.0.0.0)
       - server_port: Server port (8001)
       - workers: Number of worker processes
    
    9. Logging Configuration
       - log_level: DEBUG/INFO/WARNING/ERROR
       - log_format: json/text logging format

KEY FACTS:
    - All settings loaded at startup (zero runtime I/O)
    - Pydantic automatically validates types and ranges
    - Missing required settings cause startup to fail (fail-fast)
    - All values immutable after loading
    - Environment variables override defaults
    - Supports .env file (python-dotenv)

VALIDATION RULES:
    - llm_timeout: > 0, <= 60 seconds
    - slm_timeout: > 0, <= 120 seconds
    - vector_db_top_k: > 0, <= 100
    - embeddings_batch_size: > 0, <= 1000
    - device: must be one of [cpu, cuda, mps, npu, auto]
    - quantization: must be one of [none, int8, int4]

DEFAULTS:
    - device: auto (auto-detect best device)
    - llm_timeout: 30 seconds
    - slm_timeout: 30 seconds
    - vector_db_top_k: 5
    - embeddings_batch_size: 32
    - cache_ttl_default: 3600 seconds (1 hour)
    - log_level: INFO

FUTURE SCOPE (Phase 2+):
    - Add settings hot-reload (update without restart)
    - Add A/B testing configuration
    - Add feature flags
    - Add rate limiting configuration
    - Add monitoring thresholds (latency, error rates)
    - Add multi-environment support (dev, staging, prod)
    - Add settings encryption (for sensitive values)
    - Add settings versioning
    - Add settings documentation generation
    - Add settings audit logging (who changed what)

TESTING ENVIRONMENT:
    - Override settings in tests: Settings(device="cpu", llm_timeout=5)
    - Use .env.test file for test configuration
    - Mock settings in unit tests

PRODUCTION DEPLOYMENT:
    - Load from environment variables (not hardcoded)
    - Use managed secrets (AWS Secrets Manager, Vault, etc.)
    - Validate all settings on startup
    - Alert if required settings missing
    - Use appropriate values for production:
        device: cuda (if available)
        llm_timeout: 30 seconds
        slm_timeout: 30 seconds
        log_level: INFO (not DEBUG)
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ for legacy code paths that call os.getenv(...)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Robust .env path resolution (Windows-safe)
_CWD_ENV = Path(os.getcwd()) / ".env"
_REPO_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
_ENV_PATH = _CWD_ENV if _CWD_ENV.exists() else _REPO_ROOT_ENV

if load_dotenv is not None:
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables + .env.

    All fields have aliases to match .env variable names.

    Pydantic automatically reads from environment variables.

    Supports both Redis URL and HOST/PORT/DB/PASSWORD configurations.
    """

    # ========================================================================
    # Pydantic v2 config
    # ========================================================================

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "ENVIRONMENT": "development",
                    "DEBUG": "True",
                    "LLM_PROVIDER": "gemini",
                    "LLM_MODEL_NAME": "gemini-2.0-flash",
                    "GEMINI_API_KEY": "your_key_here",
                },
            ],
        },
    )

    # ========================================================================
    # DEVICE
    # ========================================================================

    device: Literal["cpu", "cuda", "mps", "npu", "auto"] = Field(
        default="auto",
        alias="DEVICE",
        description="Device for model inference: cpu, cuda, mps, npu, auto",
    )

    # ========================================================================
    # TOOL SWAP CONFIGURATION (Tool-Agnostic)
    # ========================================================================

    llm_provider: Literal["gemini", "openai", "huggingface", "anthropic"] = Field(
        default="gemini",
        alias="LLM_PROVIDER",
        description="LLM provider: gemini | openai | anthropic | huggingface",
    )

    vector_db_provider: Literal["qdrant", "faiss"] = Field(
        default="faiss",
        alias="VECTORDB_PROVIDER",
        description="Vector database: qdrant | faiss",
    )

    slm_provider: Literal["hf_causal", "hf_seq2seq"] = Field(
        default="hf_causal",
        alias="SLM_PROVIDER",
        description="SLM type: hf_causal (Phi-3) | hf_seq2seq (Distilbart/BART)",
    )

    cache_provider: Literal["redis", "none"] = Field(
        default="redis",
        alias="CACHE_PROVIDER",
        description="Cache backend: redis | none",
    )

    # ========================================================================
    # MODELS
    # ========================================================================

    llm_model_name: str = Field(
        default="gemini-2.0-flash",
        alias="LLM_MODEL_NAME",
        description="LLM model name/id",
    )

    slm_model_name: str = Field(
        default="microsoft/phi-3-mini-4k-instruct",
        alias="SLM_MODEL",
        description="SLM model (HF repo id)",
    )

    embeddings_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDINGS_MODEL",
        description="Embeddings model (HF repo id)",
    )

    # ========================================================================
    # API KEYS & CREDENTIALS (PROVIDER-SPECIFIC)
    # ========================================================================

    gemini_api_key: Optional[str] = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Gemini API key",
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key",
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key",
    )

    hf_token: Optional[str] = Field(
        default=None,
        alias="HF_TOKEN",
        description="HuggingFace token for gated models",
    )

    # Gemini extra params (optional)
    gemini_temperature: float = Field(
        default=0.7,
        alias="GEMINI_TEMPERATURE",
        description="Gemini generation temperature",
    )

    gemini_max_tokens: int = Field(
        default=1024,
        alias="GEMINI_MAX_TOKENS",
        description="Gemini max output tokens",
    )

    # ========================================================================
    # TIMEOUTS
    # ========================================================================

    llm_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        alias="LLM_TIMEOUT",
        description="LLM inference timeout (seconds)",
    )

    slm_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        alias="SLM_TIMEOUT",
        description="SLM inference timeout (seconds)",
    )

    embeddings_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        alias="EMBEDDINGS_TIMEOUT",
        description="Embeddings generation timeout (seconds)",
    )

    vector_db_timeout: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        alias="VECTOR_DB_TIMEOUT",
        description="Vector DB query timeout (seconds)",
    )

    redis_timeout: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        alias="REDIS_TIMEOUT",
        description="Redis operation timeout (seconds)",
    )

    # ========================================================================
    # REDIS / CACHE - Support BOTH URL and HOST/PORT patterns
    # ========================================================================

    # Pattern 1: URL-based (preferred, takes precedence)
    redis_url: Optional[str] = Field(
        default=None,
        alias="REDIS_URL",
        description="Redis connection URL (e.g., redis://localhost:6379)",
    )

    # Pattern 2: Individual HOST/PORT/DB/PASSWORD (backward compatible)
    redis_host: str = Field(
        default="localhost",
        alias="REDIS_HOST",
        description="Redis host",
    )

    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        alias="REDIS_PORT",
        description="Redis port",
    )

    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        alias="REDIS_DB",
        description="Redis database number",
    )

    redis_password: Optional[str] = Field(
        default=None,
        alias="REDIS_PASSWORD",
        description="Redis password (optional)",
    )

    redis_pool_size: int = Field(
        default=50,
        ge=5,
        le=500,
        alias="REDIS_POOL_SIZE",
        description="Redis connection pool size",
    )

    cache_ttl_default: int = Field(
        default=3600,
        ge=60,
        le=86400,
        alias="REDIS_CACHE_TTL",
        description="Default cache TTL (seconds)",
    )

    enable_redis_cache: bool = Field(
        default=True,
        alias="ENABLE_REDIS_CACHE",
        description="Enable Redis caching",
    )

    cache_embeddings: bool = Field(
        default=True,
        alias="CACHE_EMBEDDINGS",
        description="Cache embedding results",
    )

    # ========================================================================
    # COMPUTED: Resolve redis_url from components if not provided
    # ========================================================================

    @computed_field  # type: ignore[misc]
    @property
    def resolved_redis_url(self) -> str:
        """
        Resolve Redis URL from either explicit URL or HOST/PORT/PASSWORD.

        Priority:
        1. If REDIS_URL provided → use it directly
        2. Else → construct from REDIS_HOST:REDIS_PORT

        Returns:
            Full Redis connection URL
        """
        if self.redis_url:
            return self.redis_url

        # Construct from components
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ========================================================================
    # VECTOR DB
    # ========================================================================

    vector_db_url: str = Field(
        default="http://localhost:6333",
        alias="VECTOR_DB_URL",
        description="Vector DB (Qdrant) URL",
    )

    qdrant_host: str = Field(
        default="localhost",
        alias="QDRANT_HOST",
        description="Qdrant host",
    )

    qdrant_port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        alias="QDRANT_PORT",
        description="Qdrant port",
    )

    qdrant_api_key: Optional[str] = Field(
        default=None,
        alias="QDRANT_API_KEY",
        description="Qdrant API key (optional)",
    )

    vector_db_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        alias="VECTOR_DB_TOP_K",
        description="Top-K results to return from vector search",
    )

    vector_db_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="VECTOR_DB_SCORE_THRESHOLD",
        description="Minimum similarity score threshold",
    )

    qdrant_collection_name: str = Field(
        default="documents",
        alias="QDRANT_COLLECTION_NAME",
        description="Qdrant collection name",
    )

    # ========================================================================
    # OPTIMIZATION
    # ========================================================================

    slm_quantization: Literal["none", "int8", "int4"] = Field(
        default="int8",
        alias="SLM_QUANTIZATION",
        description="SLM quantization: none, int8, int4",
    )

    embeddings_batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        alias="EMBEDDINGS_BATCH_SIZE",
        description="Embeddings batch size",
    )

    embeddings_dimension: int = Field(
        default=384,
        alias="EMBEDDINGS_DIMENSION",
        description="Embeddings output dimension",
    )

    embeddings_device: str = Field(
        default="cpu",
        alias="EMBEDDINGS_DEVICE",
        description="Device for embeddings (cpu, cuda, mps)",
    )

    document_chunk_size: int = Field(
        default=512,
        ge=100,
        le=10000,
        alias="CHUNK_SIZE",
        description="Document chunk size for ingestion",
    )

    document_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=1000,
        alias="CHUNK_OVERLAP",
        description="Document chunk overlap",
    )

    document_max_size_mb: int = Field(
        default=200,
        ge=1,
        le=500,
        alias="MAX_FILE_SIZE_MB",
        description="Max document size (MB)",
    )

    # ========================================================================
    # SERVER
    # ========================================================================

    server_host: str = Field(
        default="127.0.0.1",
        alias="BACKEND_HOST",
        description="Server host",
    )

    server_port: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        alias="BACKEND_PORT",
        description="Server port",
    )

    workers: int = Field(
        default=4,
        ge=1,
        le=32,
        alias="MAX_WORKERS",
        description="Number of worker processes",
    )

    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        alias="API_TIMEOUT",
        description="API timeout (seconds)",
    )

    # ========================================================================
    # LOGGING / ENVIRONMENT
    # ========================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )

    log_format: Literal["json", "text"] = Field(
        default="text",
        alias="LOG_FORMAT",
        description="Logging format: json or text",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Environment: development, staging, production",
    )

    debug: bool = Field(
        default=False,
        alias="DEBUG",
        description="Debug mode enabled",
    )

    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================

    enable_streaming: bool = Field(
        default=True,
        alias="ENABLE_STREAMING",
        description="Enable streaming responses",
    )

    enable_slm: bool = Field(
        default=True,
        alias="ENABLE_SLM",
        description="Enable SLM for summarization",
    )

    slm_for_summarization: bool = Field(
        default=True,
        alias="SLM_FOR_SUMMARIZATION",
        description="Use SLM for query summarization",
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator("llm_model_name", "slm_model_name", "embeddings_model_name")
    @classmethod
    def _validate_model_names(cls, v: str) -> str:
        """Validate model names are non-empty strings."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be non-empty string")
        return v

    @field_validator(
        "gemini_api_key", "openai_api_key", "anthropic_api_key", "hf_token"
    )
    @classmethod
    def _validate_api_keys(cls, v: Optional[str]) -> Optional[str]:
        """Validate API keys are non-empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("API key must be non-empty if provided")
        return v

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def resolve_llm_api_key(self) -> Optional[str]:
        """
        Resolve LLM API key based on configured provider.

        Returns:
            API key for the configured provider, or None if not available
        """
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        elif self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        elif self.llm_provider == "huggingface":
            return self.hf_token
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary with secrets redacted.

        Returns:
            Settings dictionary with API keys masked
        """
        d = self.model_dump()

        # Redact sensitive fields
        for k in (
            "gemini_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "hf_token",
            "redis_password",
            "qdrant_api_key",
        ):
            if d.get(k):
                d[k] = "***REDACTED***"

        return d

    def get_cache_config(self) -> Dict[str, Any]:
        """
        Get cache configuration for ServiceContainer.

        Returns:
            Dict with redis_url and other cache settings
        """
        return {
            "redis_url": self.resolved_redis_url,
            "redis_timeout": self.redis_timeout,
            "redis_pool_size": self.redis_pool_size,
            "cache_ttl": self.cache_ttl_default,
            "enable_cache": self.enable_redis_cache,
            "cache_embeddings": self.cache_embeddings,
        }

    def get_vector_db_config(self) -> Dict[str, Any]:
        """
        Get vector database configuration.

        Returns:
            Dict with vector DB settings
        """
        return {
            "url": self.vector_db_url,
            "top_k": self.vector_db_top_k,
            "score_threshold": self.vector_db_score_threshold,
            "collection_name": self.qdrant_collection_name,
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for the active provider.

        Returns:
            Dict with LLM settings
        """
        return {
            "provider": self.llm_provider,
            "model": self.llm_model_name,
            "api_key": self.resolve_llm_api_key(),
            "timeout": self.llm_timeout,
            "temperature": self.gemini_temperature,
            "max_tokens": self.gemini_max_tokens,
        }

    def get_embeddings_config(self) -> Dict[str, Any]:
        """
        Get embeddings model configuration.

        Returns:
            Dict with embeddings settings
        """
        return {
            "model_name": self.embeddings_model_name,
            "device": self.embeddings_device,
            "batch_size": self.embeddings_batch_size,
            "dimension": self.embeddings_dimension,
            "timeout": self.embeddings_timeout,
        }
