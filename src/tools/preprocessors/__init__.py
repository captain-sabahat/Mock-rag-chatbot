"""
================================================================================
PREPROCESSORS PACKAGE INITIALIZATION
src/tools/preprocessors/__init__.py

MODULE PURPOSE:
---------------
Package initialization that exports preprocessor components and provides
unified interface for text preprocessing of official documents.

WORKING & METHODOLOGY:
----------------------
This module orchestrates package initialization by:
1. Importing base preprocessor class
2. Importing all preprocessor implementations
3. Importing utility components (language detector, text cleaner)
4. Exposing public API for easy imports
5. Managing version tracking

HOW IT CONTRIBUTES TO ADMIN PIPELINE:
-------------------------------------
- Single import point for preprocessing functionality
- Provides clean API: from preprocessors import TextCleaner
- Manages internal organization
- Enables tool factory discovery
- Centralizes version tracking

PUBLIC API EXPORTS:
-------------------
Base Classes:
  • BasePreprocessor: Abstract base for all preprocessors
  • ExtractionDepth: Enum for depth levels (MINIMAL, STANDARD, DETAILED)
  • EntityType: Enum for entity types

Utilities:
  • LanguageDetector: Language and encoding detection
  • TextCleaner: Text cleaning and normalization
  • CleaningLevel: Enum for cleaning intensity

Configuration Enums:
  • ProcessingStatus: Processing status values

EXTERNAL IMPORTS & DEPENDENCIES:
---------------------------------
Internal Submodules:
  • .base_preprocessor: BasePreprocessor, ExtractionDepth, EntityType
  • .language_detector: LanguageDetector, LanguageCode, EncodingType
  • .text_cleaner: TextCleaner, CleaningLevel

FUTURE WORK & CONTRIBUTIONS:
-----------------------------
TODO: Advanced preprocessors
  1. Add format-specific preprocessors (PDFPreprocessor, etc.)
  2. Implement machine learning enhancement
  3. Add fuzzy matching for entity validation
  4. Support custom preprocessing rules

TODO: Performance optimization
  1. GPU acceleration for text processing
  2. Distributed preprocessing
  3. Advanced caching strategies
  4. Adaptive preprocessing

TODO: Integration improvements
  1. Add preprocessing pipeline builder
  2. Implement streaming architecture
  3. Add result caching layer
  4. Create preprocessing metrics dashboard

CIRCUIT BREAKER CONSIDERATIONS:
-------------------------------
RISK POINT 1: IMPORT FAILURE
  Risk Level: #CIRCUIT_BREAK:CRITICAL
  Scenario: Any submodule import fails
  Impact: Entire package unavailable
  Detection: ImportError on first import
  Recovery: Check dependencies, validate syntax
  Prevention: Import validation on startup

RISK POINT 2: VERSION MISMATCH
  Risk Level: #CIRCUIT_BREAK:MEDIUM
  Scenario: Component versions incompatible
  Impact: Preprocessing fails or gives wrong results
  Detection: Version check function
  Recovery: Update components
  Prevention: Strict version pinning

MONITORING & HEALTH CHECKS:
----------------------------
Metrics to Track:
  # #MLFLOW:PREPROCESSOR_IMPORT: Track import success
  # mlflow.log_metric("preprocessor_import_success", 1)

Health Checks:
  - All submodules importable (verify on startup)
  - Version compatibility check (component versions)
  - Detector and cleaner functional (basic test)

================================================================================
"""

# Package version tracking
__version__ = "1.0.0"

# #CRITICAL: Import base classes and exceptions
# Must import in correct order (base before specific)
try:
    from .base_preprocessor import (
        BasePreprocessor,
        ExtractionDepth,
        EntityType,
    )
except ImportError as e:
    raise ImportError("Failed to import base preprocessor: " + str(e))

# #CRITICAL: Import utility classes
try:
    from .language_detector import (
        LanguageDetector,
        LanguageCode,
        EncodingType,
    )
except ImportError as e:
    raise ImportError("Failed to import language detector: " + str(e))

# #CRITICAL: Import text cleaning
try:
    from .text_cleaner import (
        TextCleaner,
        CleaningLevel,
    )
except ImportError as e:
    raise ImportError("Failed to import text cleaner: " + str(e))

# Public API exports - #CRITICAL: All must be listed for tool factory discovery
__all__ = [
    # Base classes
    "BasePreprocessor",
    "ExtractionDepth",
    "EntityType",
    # Language detection
    "LanguageDetector",
    "LanguageCode",
    "EncodingType",
    # Text cleaning
    "TextCleaner",
    "CleaningLevel",
    # Version
    "__version__",
]

# #MLFLOW:PACKAGE_INIT: Log successful package initialization
import logging
logger = logging.getLogger(__name__)
logger.info("Preprocessors package initialized successfully (v%s)" % __version__)