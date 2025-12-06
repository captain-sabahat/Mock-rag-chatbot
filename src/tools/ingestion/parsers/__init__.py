"""
===============================================================================
parsers/__init__.py
===============================================================================

This package initializes the parsers module. It re-exports core classes 
and factory functions for easy access and dynamic instantiation of parser 
classes in the ingestion pipeline.

WORK:
- Imports abstract class BaseParser
- Imports parser classes (PDFParser, TextParser, JSONParser, MarkdownParser)
- Provides parser_factory for dynamic object creation based on config
- Designed for plugin extensibility and module swappability

REMARKS:
- Future work: Implement plugin registry for parser discovery
- No static dependencies; all parser classes are loaded dynamically as needed

MLFLOW:
- Metrics placeholders (e.g., parse_time, input_size, success_flag) commented for future logging
- Artifacts: raw parsing time, success rate, input size

CIRCUIT BREAK:
- Errors during parser instantiation or parsing can trigger circuit break via exception handling upstream
"""

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .text_parser import TextParser
from .json_parser import JSONParser
from .markdown_parser import MarkdownParser

# Factory method to instantiate parser based on config
def parser_factory(parser_type: str, config: dict):
    """Create parser instance based on parser_type string."""
    parser_classes = {
        "pdf": PDFParser,
        "text": TextParser,
        "json": JSONParser,
        "markdown": MarkdownParser,
    }
    cls = parser_classes.get(parser_type.lower())
    if cls:
        return cls(config=config)
    # #MLFLOW:METRIC_NAME -- parser_type
    # raise NotImplementedError(f"Parser type '{parser_type}' not implemented.")
    return None

__all__ = [
    "BaseParser",
    "PDFParser",
    "TextParser",
    "JSONParser",
    "MarkdownParser",
    "parser_factory",
]
