# 7 files: Orchestration logic
"""
================================================================================
FILE: src/pipelines/__init__.py
================================================================================

PURPOSE:
    Package initialization for pipelines layer. Exports main orchestrator
    and pipeline classes for easy imports.

WORKFLOW:
    1. Import orchestrator and pipeline classes
    2. Export as public API
    3. Enable clean imports: from src.pipelines import Orchestrator

IMPORTS:
    - Orchestrator from orchestrator.py
    - Pipeline classes from individual pipeline files

OUTPUTS:
    - Orchestrator: Main coordinator class
    - Pipeline classes: Individual line orchestrators

KEY FACTS:
    - Minimal file (just re-exports)
    - Enables clean package structure

FUTURE SCOPE (Phase 2+):
    - Add pipeline factory methods
    - Add pipeline registry
    - Add pipeline composition utilities

TESTING ENVIRONMENT:
    - Import: from src.pipelines import Orchestrator

PRODUCTION DEPLOYMENT:
    - Orchestrator loaded at startup
"""

from src.pipeline.orchestrator import Orchestrator
from src.pipeline.base import BasePipeline
from src.pipeline.logic_router import LogicRouter

__all__ = [
    "Orchestrator",
    "BasePipeline",
    "LogicRouter"
]
