"""
===============================================================================
Ingestion Package Initialization
===============================================================================

SUMMARY:
--------
This package initializes the ingestion module for document processing.
It exports core classes, functions, and models to facilitate use by
higher-level orchestrator code.

WORKING & METHODOLOGY:
----------------------
- Imports key classes, functions, and models from submodules:
    - document_processor
    - models
    - parsers package
- Ensures a single point for external import
- Designed with extensibility in mind for future parsers or processing steps

INPUTS:
-------
- No external inputs at import time.

OUTPUTS:
--------
- Exposes:
    - DocumentProcessor class
    - Data models
    - Parsers subpackage

GLOBAL & CONFIG VARIABLES:
--------------------------
- None at this level unless explicitly initialized.

FUTURE WORK:
------------
- Add plugin registration system for new parsers or process steps.
- Integrate monitoring hooks (MLFlow, Prometheus) with minimal changes.

CIRCUIT_BREAK:
---------------
- Not directly involved. Future work can include health checks here.

MONITORING & HEALTH:
--------------------
- Placeholder comments for monitoring hooks (e.g., MLFlow) for activity metrics.

"""

from .document_processor import DocumentProcessor
from .models import (
    DocumentInput,
    ProcessingResult,
    ExtractedDocument,
)
from .parsers import (
    BaseParser,
    PDFParser,
    TextParser,
    JSONParser,
    MarkdownParser,
)

__all__ = [
    "DocumentProcessor",
    "DocumentInput",
    "ProcessingResult",
    "ExtractedDocument",
    "BaseParser",
    "PDFParser",
    "TextParser",
    "JSONParser",
    "MarkdownParser",
]
