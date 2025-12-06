"""
===============================================================================
Abstract Base Class for Document Parsers
===============================================================================

SUMMARY:
--------
Defines the ABC pattern for document parsing modules, enforcing a consistent
interface and error handling strategy.

WORKING & METHODOLOGY:
----------------------
- Inherits from BaseTool for uniformity and extensibility.
- Declares an abstract async method `parse()` which actual parsers must implement.
- Provides common metrics framework to track parsing activity.
- Supports configuration pass-through for flexible parameterization.
- Capable of raising exceptions that can be caught by circuit breaker or orchestrator.
- Designed with future ML monitoring hooks (placeholders provided).

INPUTS:
-------
- `content`: bytes - raw uploaded file content
- `config`: dict - parser-specific configuration, from global settings

OUTPUTS:
--------
- dict: contains parsed text and optional metadata
- Causes exceptions on failure, flagged for circuit breaker

GLOBAL & CONFIG VARIABLES:
--------------------------
- parse_timeout_seconds: default 300s, configurable
- MLflow placeholders for metrics

FUTURE WORK:
------------
- Extend with format detection pre-parser
- Add plugin system for new parsers
- Integrate MLflow metrics tracking (placeholder)

CIRCUIT_BREAK:
---------------
- Exceptions thrown here can trigger circuit break in orchestrator

MONITORING & HEALTH:
--------------------
- Placeholder comments for MLflow metrics
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

class BaseParser(BaseModel, ABC):
    """
    Abstract base class for all parsers.
    """
    parser_name: str = "base_parser"
    parse_timeout_seconds: int = 300  # Default timeout
    # #MLFLOW:METRIC_name placeholders for parse durations

    @abstractmethod
    async def parse(self, content: bytes, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Parse method to be implemented by all subclasses.
        Inputs:
            - content: bytes - raw file bytes
            - config: dict - parser configurations
        Outputs:
            - dict with parsed text and metadata
        Raises:
            - ParseError on failure to parse
        """
        pass

    # Future: add error handling, retries, and MLflow tracking here.
