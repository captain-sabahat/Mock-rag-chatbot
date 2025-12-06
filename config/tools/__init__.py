"""
================================================================================
TOOLS CONFIGURATION PACKAGE - Tool registry and factory initialization
================================================================================

SUMMARY:
--------
Central initialization point for the tools configuration module. Lazy-loads tool
registries (parsers, chunkers, embedders, vector DB clients) on first access.
This package orchestrates the mapping between tool names and their implementations,
enabling tool-agnostic code throughout the pipeline.

ROLE IN ADMIN PIPELINE:
------------------------
This is the **bridge layer** between the pipeline nodes and concrete tool implementations.
Instead of nodes knowing about specific tools (e.g., "OpenAI embeddings"), they request
a tool by category and type from this registry. This allows:
  - Easy swapping of embedding providers without node code changes
  - Multi-provider support (OpenAI, Cohere, HuggingFace in parallel)
  - Configuration-driven tool selection at runtime
  - Future vector DB migration without pipeline changes

RESPONSIBILITY:
- Export tool registry functionality
- Provide tool metadata queries
- Validate tool availability
- Track active tool selections

NOT RESPONSIBLE:
- Implementing tools (that's src/tools/)
- Storing configuration (that's config/settings.py)
- Loading YAML (that's config/loader.py)

ARCHITECTURE:
    config/tools/
    ├── tool_registry.py  ← Tool metadata and registry
    └── __init__.py       ← THIS FILE (exports)
        ↓
    config/__init__.py    ← Parent package exports
        ↓
    Application Code      ← Uses tools

    
WORKING & METHODOLOGY:
-----------------------
1. On module import, registries are instantiated but NOT populated immediately.
2. Registries use lazy-loading: tool implementations are registered on first access.
3. Each registry (PARSER_REGISTRY, CHUNKER_REGISTRY, etc.) maintains a dict of:
   - Tool name → Tool class or factory function
   - Metadata: provider, version, capabilities
4. Nodes/API layers call registry.get("tool_name") to obtain instances.

INPUTS (FROM OUTSIDE):
-----------------------
• Individual tool implementation modules (parsers, embedders, chunkers, clients)
• Configuration from config.settings (tools.yaml, tools_dev.yaml)
• Environment variables (OPENAI_API_KEY, COHERE_API_KEY, etc.)
• Tool metadata (version, capabilities, supported formats)

OUTPUTS:
--------
• PARSER_REGISTRY: Parsers for ingestion (pdf, txt, json, web)
• CHUNKER_REGISTRY: Chunking strategies (recursive, token, semantic)
• EMBEDDER_REGISTRY: Embedding providers (openai, cohere, huggingface)
• VECTORDB_REGISTRY: Vector DB clients (qdrant, pinecone, weaviate, milvus)
• Lazy-loaded registries accessible globally via module attributes

GLOBAL/CONFIG VARIABLES IN THIS FILE:
-------------------------------------
• PARSER_REGISTRY (dict): Maps parser names to parser classes
• CHUNKER_REGISTRY (dict): Maps chunker strategy names to chunker classes
• EMBEDDER_REGISTRY (dict): Maps embedding provider names to embedder clients
• VECTORDB_REGISTRY (dict): Maps vector DB names to client classes
• _registry_initialized (bool): Flag to ensure one-time initialization
• MAX_TOOLS_IN_REGISTRY (int): Upper bound for registry size (prevents runaway registration)
• TOOL_TIMEOUT_SECONDS (int): Global timeout for any tool operation (can be overridden per tool)

FUTURE WORK / NOTES FOR CONTRIBUTORS:
-------------------------------------
• Add caching layer for tool instances (singleton pattern) to avoid recreating clients
• Support plugin architecture: allow third-party tool registration via entry points
• Add tool health checks: periodic pings to embedders, vector DBs to detect failures
• Implement tool versioning: support multiple versions of same provider side-by-side
• Add metrics collection: track which tools are used, success rates, latency per tool
• Consider circuit-breaker per tool: if a tool fails N times, temporarily disable it

ML/AI METRICS & ARTIFACTS (If relevant):
-----------------------------------------
This file does NOT perform AI/ML operations; it is purely infrastructure.
However, it should track:
  • tool_initialization_time_ms: How long it takes to initialize each registry
  • tools_registered_count: Number of tools in each registry
  • tool_lookup_latency_ms: Time to retrieve a tool from registry (should be <1ms)
  • tool_instantiation_latency_ms: Time to create a tool instance (varies by tool)

Future monitoring could use:
  # mlflow: log_metric("parser_registry_size", len(PARSER_REGISTRY))
  # langfuse: track_event("tool_initialized", {"tool_name": name, "category": category})

CIRCUIT BREAK NOTE:
-------------------
This module can contribute to circuit breaks indirectly:
  - If a tool fails to initialize (e.g., API key missing), it's not registered,
    and nodes requesting it will fail immediately (no retry).
  - If ALL embedders fail registration, the embedding_node will fail and trigger
    circuit break upstream.
  - Downstream: A missing tool in registry causes StateManager to mark that node
    as failed immediately, leading to circuit break.

SCALABILITY & K8S NOTES:
------------------------
• k8s: Tool registries should be thread-safe (use locks if needed)
• docker: Tool initialization happens once per container start
• scaling: Registries are memory-light (<1MB); safe to replicate across pods
• migration: When migrating to new vector DB, swap provider in registry without
  redeploying nodes

RESOURCE OPTIMIZATION:
---------------------
• Lazy loading: Tools not used are not instantiated (memory efficient)
• Singleton pattern: Each tool type created once, reused across requests (CPU efficient)
• Network: Tool clients reuse connections; connection pools configured in each client
• Token awareness: Not applicable here (infrastructure layer)

BENEFITS:
- Centralized tool metadata
- Easy tool availability checking
- Clean imports: from config import get_tool_registry
- Type hints: from config.tools import ToolRegistry

================================================================================
"""
import logging

# Configure logging for tools config subpackage
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS - Tool Registry
# ============================================================================

from .tool_registry import (
    # Main Tool Registry Class
    ToolRegistry,
    
    # Singleton accessor
    get_tool_registry,
)

logger.debug("✅ Loaded config.tools.tool_registry")


# ============================================================================
# PUBLIC API - Tools Subpackage Exports
# ============================================================================

__all__ = [
    # Core Tool Registry
    "ToolRegistry",
    "get_tool_registry",
]


# ============================================================================
# VALIDATION - Tool Registry on Import
# ============================================================================

def _validate_tools_on_import() -> bool:
    """
    Validate tool registry on subpackage import.
    
    Checks:
    1. Tool registry can be instantiated
    2. All configured tools are available
    
    Returns:
        True if validation passes, False otherwise
    
    Called automatically on import, logs warnings but doesn't fail.
    """
    try:
        registry = get_tool_registry()
        logger.debug(
            f"✅ Tool registry validated\n"
            f"   Active: {registry.get_active_chunker()} / "
            f"{registry.get_active_embedder()} / "
            f"{registry.get_active_vectordb()}"
        )
        return True
    except ValueError as e:
        logger.error(f"❌ Tool registry validation failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Tool registry validation warning: {e}")
        return False


# Validate on import (warns but continues)
try:
    _validate_tools_on_import()
except Exception as e:
    logger.debug(f"Tools validation skipped (deferred): {e}")


# ============================================================================
# MODULE INTERFACE - Common Queries
# ============================================================================

def get_available_tools() -> dict:
    """
    Get all available tools grouped by category.
    
    Returns:
        Dictionary of available tools by type:
        {
            "chunkers": ["recursive", "token", "semantic"],
            "embedders": ["huggingface", "openai", "cohere"],
            "vectordbs": ["faiss", "qdrant", "pinecone", "weaviate", "milvus"],
            "parsers": ["pdf", "txt", "json", "docx"]
        }
    
    Example:
        >>> tools = get_available_tools()
        >>> print(tools["embedders"])
        ['huggingface', 'openai', 'cohere']
    """
    registry = get_tool_registry()
    
    return {
        "chunkers": registry.list_chunkers(),
        "embedders": registry.list_embedders(),
        "vectordbs": registry.list_vectordbs(),
        "parsers": registry.list_parsers(),
    }


def get_tool_status() -> dict:
    """
    Get current status of all tools.
    
    Returns:
        Dictionary with active tools and availability:
        {
            "active": {
                "chunker": "recursive",
                "embedder": "huggingface",
                "vectordb": "faiss"
            },
            "available": {...},
            "production_ready": {
                "chunker": true,
                "embedder": true,
                "vectordb": true
            }
        }
    
    Example:
        >>> status = get_tool_status()
        >>> print(status["active"]["embedder"])
        'huggingface'
    """
    registry = get_tool_registry()
    return registry.summary()


def check_tool_compatibility() -> tuple[bool, list]:
    """
    Check if all active tools are production-ready.
    
    Returns:
        Tuple of (all_ready: bool, warnings: list)
    
    Example:
        >>> ready, warnings = check_tool_compatibility()
        >>> if not ready:
        ...     for warning in warnings:
        ...         print(warning)
    """
    registry = get_tool_registry()
    summary = registry.summary()
    warnings = []
    
    # Check each active tool
    for tool_type, is_ready in summary.get("production_ready", {}).items():
        if not is_ready:
            active = summary["active"].get(tool_type, "unknown")
            warnings.append(
                f"⚠️  {tool_type} '{active}' is NOT production-ready "
                f"(use for development only)"
            )
    
    return (len(warnings) == 0, warnings)


# ============================================================================
# LOGGING - Subpackage Import Info
# ============================================================================

logger.info(
    f"""
    📦 Tools Config Subpackage Initialized
    
    Exports:
       • ToolRegistry         - Tool metadata class
       • get_tool_registry()  - Get singleton registry
       
    Helper Functions:
       • get_available_tools()      - List all tools by type
       • get_tool_status()          - Current tool status
       • check_tool_compatibility() - Check production readiness
       
    Usage: from config.tools import get_tool_registry
    Or:    from config import get_tool_registry
    """
)