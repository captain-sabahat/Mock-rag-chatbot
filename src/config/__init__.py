# 5 files: Settings layer
"""
================================================================================
FILE: src/config/__init__.py
================================================================================

PURPOSE:
    Package initialization for configuration layer. Exports main Settings class
    and constants for easy imports throughout codebase.

WORKFLOW:
    1. Import Settings from settings.py
    2. Import constants from constants.py
    3. Export as public API
    4. Enable clean imports: from src.config import Settings, CONSTANTS

IMPORTS:
    - Settings from settings.py
    - Constants from constants.py

OUTPUTS:
    - Settings: Configuration class instance
    - CONSTANTS: Global constants dict

KEY FACTS:
    - Minimal file (just re-exports)
    - Enables clean package structure
    - Settings loaded once at startup

FUTURE SCOPE (Phase 2+):
    - Add config validation helper
    - Add config diff utilities (for debugging)

TESTING ENVIRONMENT:
    - Import: from src.config import Settings

PRODUCTION DEPLOYMENT:
    - Settings auto-loads from environment variables
"""

from src.config.settings import Settings
from src.config.constants import CONSTANTS

__all__ = ["Settings", "CONSTANTS"]
