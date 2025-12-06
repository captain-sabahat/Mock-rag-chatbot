"""
================================================================================
CONFIG PACKAGE - Central Configuration Management
================================================================================

SUMMARY
-------
Central entry point for all configuration in the RAG pipeline.

Provides type-safe, environment-aware configuration loading.

EXPORTS
-------
    get_settings()      - Get typed Settings object
    get_tool_registry() - Get tool metadata registry
    init_config_loader() - Initialize config system (call at startup)
    Settings            - Pydantic models for type hints

ARCHITECTURE
------------
config/
├── __init__.py        ← This file (exports everything)
├── settings.py        ← Pydantic type-safe models
├── loader.py          ← YAML loading + env overrides
├── tools/
│   ├── __init__.py    ← Re-exports tool registry
│   └── tool_registry.py ← Tool metadata (no src imports)
└── settings/          ← YAML config files
    ├── session_store.yaml
    ├── chunking.yaml
    ├── embeddings.yaml
    ├── vectordb.yaml
    ├── session_store_dev.yaml
    ├── chunking_dev.yaml
    └── ... (environment overrides)

KEY PRINCIPLE
-------------
Config layer NEVER imports from src/.
Pipeline imports from config, never the reverse.
Tool implementations stay in src/tools/
Tool registry metadata stays in config/tools/

USAGE
-----
# At application startup:
from config import init_config_loader, get_settings, get_tool_registry

if not init_config_loader():
    raise RuntimeError("Failed to initialize configuration")

settings = get_settings()
registry = get_tool_registry()

# In pipeline nodes:
from config import get_settings, get_tool_registry

settings = get_settings()
backend = settings.session_store.backend
embedder = registry.get_active_embedder()

================================================================================
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Import configuration components
from config.settings import get_settings, reset_settings, Settings
from config.loader import init_config_loader, get_config_loader
from config.tools import get_tool_registry, ToolRegistry

# Public API
__all__ = [
    'get_settings',
    'reset_settings',
    'Settings',
    'get_tool_registry',
    'ToolRegistry',
    'init_config_loader',
    'get_config_loader',
]


# Optional: Auto-initialize on import (comment out if you want explicit init)
# _initialized = False
# try:
#     if not _initialized:
#         init_config_loader()
#         _initialized = True
# except Exception as e:
#     logger.warning(f"⚠️  Config auto-initialization failed: {str(e)}")
