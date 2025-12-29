"""
================================================================================
SERVICE CONTAINER - PROVIDER DISCOVERY & INITIALIZATION
================================================================================

Main dependency injection container.

Implements TWO-LAYER SWAPPABILITY:

Layer 1: .env determines which PROVIDER FILE to import
  Example: LLM_PROVIDER=gemini  →  Import from src.providers.llm.gemini

Layer 2: Provider file contains DEFAULT with internal customization
  Example: src/providers/llm/gemini.py has DEFAULT_MODEL, DEFAULT_CONFIG
           ServiceContainer imports and uses these defaults

USAGE:

  container = ServiceContainer(settings)
  await container.initialize()
  llm = container.get_llm()
  response = await llm.generate(prompt)

FLOW:

  .env: LLM_PROVIDER=gemini
    ↓
  ServiceContainer reads settings.llm_provider = "gemini"
    ↓
  Dynamically import: from src.providers.llm.gemini import default_provider
    ↓
  Get: default_provider (GeminiProvider with DEFAULT_MODEL baked in)
    ↓
  Initialize: await default_provider.initialize()
    ↓
  Return to application
"""

import logging
import importlib
from typing import Optional, Any

from src.config.settings import Settings
from src.core.exceptions import ServiceInitializationError

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Dependency injection container for all tool providers.

    Implements two-layer swappability:
    - Layer 1 (.env): Select provider TYPE
    - Layer 2 (provider file): Configure provider DEFAULTS
    """

    def __init__(self, settings: Settings, cache_provider: Optional[Any] = None) -> None:
        """
        Initialize container with settings.

        Args:
            settings: Configuration object (from .env)
            cache_provider: Optional pre-initialized cache provider
                           (e.g., RedisCacheProvider) passed from main.py.
        """
        self.settings = settings

        self._llm: Optional[Any] = None
        self._slm: Optional[Any] = None
        self._embeddings: Optional[Any] = None
        self._vectordb: Optional[Any] = None
        self._cache: Optional[Any] = cache_provider  # accept pre-initialized cache

        logger.info("ServiceContainer instantiated")

    async def initialize(self) -> None:
        """
        Initialize all providers.

        Layer 1: Reads .env to determine provider file
        Layer 2: Imports provider file and gets default instance
        """
        try:
            logger.info("=" * 80)
            logger.info("INITIALIZING SERVICE CONTAINER")
            logger.info("=" * 80)

            # LLM
            self._llm = await self._load_provider(
                provider_type="llm",
                provider_name=self.settings.llm_provider,
                module_path="src.providers.llm",
            )

            # SLM
            self._slm = await self._load_provider(
                provider_type="slm",
                provider_name=self.settings.slm_provider,
                module_path="src.providers.slm",
            )

            # Embeddings
            self._embeddings = await self._load_provider(
                provider_type="embeddings",
                provider_name=self.settings.embeddings_provider,
                module_path="src.providers.embeddings",
            )

            # VectorDB
            self._vectordb = await self._load_provider(
                provider_type="vectordb",
                provider_name=self.settings.vector_db_provider,
                module_path="src.providers.vectordb",
            )

            # Cache is expected to be initialized outside and passed in.
            logger.info("=" * 80)
            logger.info("✓ ServiceContainer initialized successfully")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(
                f"ServiceContainer initialization failed: {str(e)}",
                exc_info=True,
            )
            raise ServiceInitializationError(
                f"Failed to initialize container: {str(e)}"
            )

    async def _load_provider(
        self,
        provider_type: str,
        provider_name: str,
        module_path: str,
    ) -> Any:
        """
        Load a provider using two-layer logic.

        Layer 1: provider_name from .env determines which file
        Layer 2: Import file and get default_provider from it

        Args:
            provider_type: Type (llm, slm, embeddings, vectordb)
            provider_name: Name from .env (gemini, openai, hf_causal, qdrant, etc.)
            module_path: Import base path (src.providers.llm, etc.)

        Returns:
            Initialized provider instance
        """
        try:
            logger.info(f"\n[Layer 1] Loading {provider_type} provider: {provider_name}")

            # Layer 1: Build import path from provider name
            full_path = f"{module_path}.{provider_name}"
            logger.info(f"[Layer 1] Import path: {full_path}")

            # Layer 2: Import the provider module and get default
            logger.info(f"[Layer 2] Importing {full_path}...")
            provider_module = importlib.import_module(full_path)

            if not hasattr(provider_module, "default_provider"):
                raise AttributeError(
                    f"Provider module {full_path} does not export 'default_provider'. "
                    f"Ensure {full_path}.py has: default_provider = {provider_name.title()}Provider(...)"
                )

            provider_instance = getattr(provider_module, "default_provider")
            logger.info(
                f"[Layer 2] Got default_provider: {provider_instance.__class__.__name__}"
            )

            # Initialize the provider
            logger.info(
                f"[Layer 2] Initializing {provider_instance.__class__.__name__}..."
            )
            if hasattr(provider_instance, "initialize"):
                await provider_instance.initialize()

            logger.info(f"✓ {provider_type.upper()} initialized: {provider_name}")
            return provider_instance

        except ImportError as e:
            raise ServiceInitializationError(
                f"Failed to import {provider_type} provider '{provider_name}' "
                f"from {module_path}.{provider_name}: {str(e)}"
            )
        except AttributeError as e:
            raise ServiceInitializationError(
                f"Provider configuration error: {str(e)}"
            )
        except Exception as e:
            raise ServiceInitializationError(
                f"Failed to initialize {provider_type} provider '{provider_name}': {str(e)}"
            )

    async def shutdown(self) -> None:
        """Shutdown all providers."""
        logger.info("Shutting down ServiceContainer...")

        providers = [
            ("LLM", self._llm),
            ("SLM", self._slm),
            ("Embeddings", self._embeddings),
            ("VectorDB", self._vectordb),
            ("Cache", self._cache),  # ensure cache also gets shutdown
        ]

        for name, provider in providers:
            if provider:
                try:
                    if hasattr(provider, "shutdown"):
                        await provider.shutdown()
                    logger.info(f"✓ {name} shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down {name}: {str(e)}")

        logger.info("✓ ServiceContainer shutdown complete")

    # ========================================================================
    # ACCESSOR METHODS
    # ========================================================================

    def get_llm(self) -> Any:
        """Get LLM provider instance."""
        if self._llm is None:
            raise RuntimeError("LLM provider not initialized")
        return self._llm

    def get_slm(self) -> Any:
        """Get SLM provider instance."""
        if self._slm is None:
            raise RuntimeError("SLM provider not initialized")
        return self._slm

    def get_embeddings(self) -> Any:
        """Get Embeddings provider instance."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings provider not initialized")
        return self._embeddings

    def get_vector_db(self) -> Any:
        """Get VectorDB provider instance."""
        if self._vectordb is None:
            raise RuntimeError("VectorDB provider not initialized")
        return self._vectordb

    def get_all(self) -> dict:
        """Get all providers as dict."""
        return {
            "llm": self._llm,
            "slm": self._slm,
            "embeddings": self._embeddings,
            "vectordb": self._vectordb,
        }

    def get_cache(self):
        """Get cache provider instance."""
        if getattr(self, "_cache", None) is None:
            raise RuntimeError(
                "Cache not initialized. Check ServiceContainer.initialize()."
            )
        return self._cache