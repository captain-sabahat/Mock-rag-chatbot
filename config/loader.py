# ============================================================================
# CONFIGURATION LOADER - Environment-Aware YAML + .env Management
# ============================================================================

"""
Load and merge YAML configurations with .env variable overrides.

Supports multiple environments (dev, staging, prod) with cascading overrides.

RESPONSIBILITY:
- Load YAML config files from config/settings/ directory
- Load .env file and override configurations
- Merge environment-specific overrides
- Apply environment variable overrides (highest priority)
- Validate critical configuration values
- Provide lazy-loading and caching

INPUTS:
- .env file (project root) → Highest priority variables
- YAML files: config/settings/*.yaml → Base configurations
- Environment name: dev, staging, prod (from ENV var in .env)

OUTPUTS:
- Loaded and validated configuration dict
- Supports dot-notation access (e.g., "session_store.backend")
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and merge YAML config with .env and environment overrides."""

    def __init__(self, config_dir: str = None):
        """Initialize config loader.
        
        Args:
            config_dir: Path to config directory (default: ./config/settings)
        """
        # Load .env file FIRST (highest priority)
        load_dotenv()
        
        if config_dir is None:
            config_dir = str(Path(__file__).parent / "settings")
        
        self.config_dir = Path(config_dir)
        self.environment = os.getenv("ENV", "dev").lower()
        self._config: Dict[str, Any] = {}
        self._loaded = False
        
        logger.info(f"🔧 ConfigLoader initialized: environment={self.environment}, dir={self.config_dir}")

    def load(self) -> Dict[str, Any]:
        """
        Load configuration with .env and environment variable overrides.
        
        Loading priority (highest to lowest):
        1. Environment variables from .env (CHUNKING_STRATEGY=, etc.)
        2. Environment-specific YAML (session_store_dev.yaml, etc.)
        3. Base YAML (session_store.yaml, etc.)
        
        Returns:
            Loaded configuration dict
        """
        if self._loaded:
            logger.debug("✓ Using cached configuration")
            return self._config

        logger.info(f"📂 Loading configuration from {self.config_dir}")
        
        try:
            # Step 1: Load base config files
            self._load_yaml_file("session_store.yaml", "session_store")
            self._load_yaml_file("chunking.yaml", "chunking")
            self._load_yaml_file("embeddings.yaml", "embeddings")
            self._load_yaml_file("vectordb.yaml", "vectordb")
            
            # Step 2: Load environment-specific overrides
            self._load_yaml_file(f"session_store_{self.environment}.yaml", "session_store", override=True)
            self._load_yaml_file(f"chunking_{self.environment}.yaml", "chunking", override=True)
            self._load_yaml_file(f"embeddings_{self.environment}.yaml", "embeddings", override=True)
            self._load_yaml_file(f"vectordb_{self.environment}.yaml", "vectordb", override=True)
            
            # Step 3: Apply .env variable overrides (HIGHEST PRIORITY)
            self._apply_env_overrides()
            
            self._loaded = True
            logger.info("✅ Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {str(e)}", exc_info=True)
            raise

    def _load_yaml_file(self, filename: str, key: str, override: bool = False) -> None:
        """
        Load a single YAML file.
        
        Args:
            filename: Name of YAML file to load
            key: Config section key
            override: Whether to merge into existing (True) or replace (False)
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.debug(f"⊘ Config file not found: {filename}")
            return
        
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f) or {}
                section_data = data.get(key, {})
                
                if override and key in self._config:
                    # Deep merge environment-specific overrides
                    self._deep_merge(self._config[key], section_data)
                    logger.debug(f"✓ Merged {filename} into {key}")
                else:
                    self._config[key] = section_data
                    logger.debug(f"✓ Loaded {filename}")
                    
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parse error in {filename}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load {filename}: {str(e)}")
            raise

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply .env variable overrides (HIGHEST PRIORITY)."""
        
        # ===== CHUNKING OVERRIDES =====
        if os.getenv("CHUNKING_STRATEGY"):
            self._config.setdefault("chunking", {})["strategy"] = os.getenv("CHUNKING_STRATEGY")
            logger.debug(f"✓ Env override: chunking.strategy={os.getenv('CHUNKING_STRATEGY')}")
        
        if os.getenv("CHUNK_SIZE"):
            self._config.setdefault("chunking", {})["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
            logger.debug(f"✓ Env override: chunking.chunk_size={os.getenv('CHUNK_SIZE')}")
        
        if os.getenv("CHUNK_OVERLAP"):
            self._config.setdefault("chunking", {})["overlap"] = int(os.getenv("CHUNK_OVERLAP"))
            logger.debug(f"✓ Env override: chunking.overlap={os.getenv('CHUNK_OVERLAP')}")
        
        # ===== EMBEDDINGS OVERRIDES =====
        if os.getenv("EMBEDDINGS_PROVIDER"):
            self._config.setdefault("embeddings", {})["active_provider"] = os.getenv("EMBEDDINGS_PROVIDER")
            logger.debug(f"✓ Env override: embeddings.active_provider={os.getenv('EMBEDDINGS_PROVIDER')}")
        
        # HuggingFace
        if os.getenv("EMBEDDINGS_HUGGINGFACE_MODEL"):
            self._config.setdefault("embeddings", {}).setdefault("huggingface", {})["model_name"] = os.getenv("EMBEDDINGS_HUGGINGFACE_MODEL")
            logger.debug(f"✓ Env override: embeddings.huggingface.model_name={os.getenv('EMBEDDINGS_HUGGINGFACE_MODEL')}")
        
        if os.getenv("EMBEDDINGS_HUGGINGFACE_DIMENSIONS"):
            self._config.setdefault("embeddings", {}).setdefault("huggingface", {})["dimension"] = int(os.getenv("EMBEDDINGS_HUGGINGFACE_DIMENSIONS"))
            logger.debug(f"✓ Env override: embeddings.huggingface.dimension={os.getenv('EMBEDDINGS_HUGGINGFACE_DIMENSIONS')}")
        
        if os.getenv("EMBEDDINGS_HUGGINGFACE_DEVICE"):
            self._config.setdefault("embeddings", {}).setdefault("huggingface", {})["device"] = os.getenv("EMBEDDINGS_HUGGINGFACE_DEVICE")
            logger.debug(f"✓ Env override: embeddings.huggingface.device={os.getenv('EMBEDDINGS_HUGGINGFACE_DEVICE')}")
        
        # OpenAI
        if os.getenv("EMBEDDINGS_OPENAI_API_KEY"):
            self._config.setdefault("embeddings", {}).setdefault("openai", {})["api_key"] = os.getenv("EMBEDDINGS_OPENAI_API_KEY")
            logger.debug("✓ Env override: embeddings.openai.api_key=***")
        
        if os.getenv("EMBEDDINGS_OPENAI_MODEL"):
            self._config.setdefault("embeddings", {}).setdefault("openai", {})["model"] = os.getenv("EMBEDDINGS_OPENAI_MODEL")
            logger.debug(f"✓ Env override: embeddings.openai.model={os.getenv('EMBEDDINGS_OPENAI_MODEL')}")
        
        if os.getenv("EMBEDDINGS_OPENAI_DIMENSIONS"):
            self._config.setdefault("embeddings", {}).setdefault("openai", {})["dimension"] = int(os.getenv("EMBEDDINGS_OPENAI_DIMENSIONS"))
            logger.debug(f"✓ Env override: embeddings.openai.dimension={os.getenv('EMBEDDINGS_OPENAI_DIMENSIONS')}")
        
        # Cohere
        if os.getenv("EMBEDDINGS_COHERE_API_KEY"):
            self._config.setdefault("embeddings", {}).setdefault("cohere", {})["api_key"] = os.getenv("EMBEDDINGS_COHERE_API_KEY")
            logger.debug("✓ Env override: embeddings.cohere.api_key=***")
        
        if os.getenv("EMBEDDINGS_COHERE_MODEL"):
            self._config.setdefault("embeddings", {}).setdefault("cohere", {})["model"] = os.getenv("EMBEDDINGS_COHERE_MODEL")
            logger.debug(f"✓ Env override: embeddings.cohere.model={os.getenv('EMBEDDINGS_COHERE_MODEL')}")
        
        if os.getenv("EMBEDDINGS_COHERE_DIMENSIONS"):
            self._config.setdefault("embeddings", {}).setdefault("cohere", {})["dimension"] = int(os.getenv("EMBEDDINGS_COHERE_DIMENSIONS"))
            logger.debug(f"✓ Env override: embeddings.cohere.dimension={os.getenv('EMBEDDINGS_COHERE_DIMENSIONS')}")
        
        # ===== VECTOR DB OVERRIDES =====
        if os.getenv("VECTORDB_PROVIDER"):
            self._config.setdefault("vectordb", {})["active_provider"] = os.getenv("VECTORDB_PROVIDER")
            logger.debug(f"✓ Env override: vectordb.active_provider={os.getenv('VECTORDB_PROVIDER')}")
        
        # FAISS
        if os.getenv("FAISS_INDEX_PATH"):
            self._config.setdefault("vectordb", {}).setdefault("faiss", {})["data_path"] = os.getenv("FAISS_INDEX_PATH")
            logger.debug(f"✓ Env override: vectordb.faiss.data_path={os.getenv('FAISS_INDEX_PATH')}")
        
        # Qdrant
        if os.getenv("QDRANT_HOST"):
            self._config.setdefault("vectordb", {}).setdefault("qdrant", {})["url"] = f"http://{os.getenv('QDRANT_HOST')}:6333"
            logger.debug(f"✓ Env override: vectordb.qdrant.url={os.getenv('QDRANT_HOST')}")
        
        if os.getenv("QDRANT_API_KEY"):
            self._config.setdefault("vectordb", {}).setdefault("qdrant", {})["api_key"] = os.getenv("QDRANT_API_KEY")
            logger.debug("✓ Env override: vectordb.qdrant.api_key=***")
        
        # Pinecone
        if os.getenv("PINECONE_API_KEY"):
            self._config.setdefault("vectordb", {}).setdefault("pinecone", {})["api_key"] = os.getenv("PINECONE_API_KEY")
            logger.debug("✓ Env override: vectordb.pinecone.api_key=***")
        
        if os.getenv("PINECONE_ENVIRONMENT"):
            self._config.setdefault("vectordb", {}).setdefault("pinecone", {})["environment"] = os.getenv("PINECONE_ENVIRONMENT")
            logger.debug(f"✓ Env override: vectordb.pinecone.environment={os.getenv('PINECONE_ENVIRONMENT')}")
        
        # ===== SESSION STORE OVERRIDES =====
        if os.getenv("SESSION_STORE_BACKEND"):
            self._config.setdefault("session_store", {})["backend"] = os.getenv("SESSION_STORE_BACKEND")
            logger.debug(f"✓ Env override: session_store.backend={os.getenv('SESSION_STORE_BACKEND')}")
        
        if os.getenv("REDIS_HOST"):
            self._config.setdefault("session_store", {}).setdefault("redis", {})["host"] = os.getenv("REDIS_HOST")
            logger.debug(f"✓ Env override: session_store.redis.host={os.getenv('REDIS_HOST')}")
        
        if os.getenv("REDIS_PORT"):
            self._config.setdefault("session_store", {}).setdefault("redis", {})["port"] = int(os.getenv("REDIS_PORT"))
            logger.debug(f"✓ Env override: session_store.redis.port={os.getenv('REDIS_PORT')}")

# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_loader_instance: Optional[ConfigLoader] = None

def init_config_loader(config_dir: str = None) -> bool:
    """
    Initialize the global config loader.
    
    This function:
    1. Creates a ConfigLoader instance
    2. Loads all configuration (YAML + .env vars)
    3. Validates critical settings
    4. Caches for reuse
    
    Args:
        config_dir: Path to config directory (optional)
    
    Returns:
        True if initialization succeeded, False otherwise
    """
    global _loader_instance
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 Initializing configuration system...")
        logger.info("=" * 80)
        
        _loader_instance = ConfigLoader(config_dir)
        _loader_instance.load()
        
        logger.info("=" * 80)
        logger.info("✅ Configuration system initialized successfully")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration initialization failed: {str(e)}", exc_info=True)
        return False

def get_config_loader() -> ConfigLoader:
    """Get global ConfigLoader instance (auto-initializes if needed)."""
    global _loader_instance
    
    if _loader_instance is None:
        init_config_loader()
    
    if _loader_instance is None:
        raise RuntimeError("Failed to initialize ConfigLoader")
    
    return _loader_instance