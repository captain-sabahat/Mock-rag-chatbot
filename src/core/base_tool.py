"""
================================================================================
BASE TOOL - Abstract Base Class for All Tools
================================================================================

PURPOSE:
--------
Define the interface that all tool implementations must follow.

This enforces consistency across:
  - Chunkers (RecursiveChunker, SlidingWindowChunker, etc.)
  - Embedders (OpenAIEmbedder, BGEEmbedder, etc.)
  - VectorDB clients (QdrantClient, FAISSClient, etc.)
  - Preprocessors (TextCleaner, LanguageDetector, etc.)

BENEFITS:
---------
✅ Interchangeable implementations
✅ Type safety with Pydantic
✅ Easy to mock for testing
✅ Clear contract for developers
✅ Config-driven tool selection

USAGE:
------
    class MyChunker(BaseTool):
        async def execute(self, **kwargs):
            # Your implementation
            pass

================================================================================
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ToolConfig(BaseModel):
    """Base configuration for all tools."""
    
    tool_name: str = Field(..., description="Name of the tool")
    tool_type: str = Field(..., description="Type (chunker, embedder, vectordb, etc.)")
    enabled: bool = Field(True, description="Is tool enabled?")
    timeout_seconds: float = Field(300.0, description="Execution timeout")
    retry_count: int = Field(3, ge=0, description="Number of retries on failure")
    
    class Config:
        extra = "allow"  # Allow additional fields


class BaseTool(ABC):
    """
    Abstract base class for all RAG pipeline tools.
    
    All tools (chunkers, embedders, vectordb clients, etc.)
    must inherit from this and implement execute().
    """

    def __init__(self, config: ToolConfig):
        """
        Initialize tool with configuration.
        
        Args:
            config: ToolConfig instance
        """
        self.config = config
        self.tool_name = config.tool_name
        self.tool_type = config.tool_type
        self.enabled = config.enabled
        self.timeout = config.timeout_seconds
        self.retry_count = config.retry_count
        
        logger.info(f"🔧 Initialized {self.tool_type} tool: {self.tool_name}")

    # ========================================================================
    # ABSTRACT METHOD - MUST BE IMPLEMENTED
    # ========================================================================

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        This method MUST be implemented by all subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict with execution results
            
        Raises:
            ProcessingError: If execution fails
            ValidationError: If inputs invalid
            
        Example:
            # Chunker
            result = await chunker.execute(
                text="Large document text...",
                chunk_size=512,
                overlap=50
            )
            
            # Embedder
            result = await embedder.execute(
                texts=["text1", "text2"],
                model="openai"
            )
            
            # VectorDB
            result = await vectordb.execute(
                operation="store",
                vectors=[[0.1, 0.2, ...], ...],
                metadata=[...]
            )
        """
        pass

    # ========================================================================
    # OPTIONAL: LIFECYCLE HOOKS
    # ========================================================================

    async def setup(self) -> None:
        """
        Setup hook called before first use.
        
        Override if your tool needs:
          - Load models
          - Initialize connections
          - Create resources
          
        Example:
            async def setup(self):
                self.model = await load_model(self.config.model_name)
                self.client = redis.Redis(...)
        """
        pass

    async def cleanup(self) -> None:
        """
        Cleanup hook called on shutdown.
        
        Override if your tool needs:
          - Close connections
          - Release resources
          - Save state
          
        Example:
            async def cleanup(self):
                await self.client.close()
                self.model.unload()
        """
        pass

    # ========================================================================
    # OPTIONAL: VALIDATION
    # ========================================================================

    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.
        
        Override in subclasses to enforce constraints.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If invalid
            
        Example:
            def validate_input(self, **kwargs):
                if 'text' not in kwargs:
                    raise ValidationError("Missing 'text' parameter")
                if len(kwargs['text']) == 0:
                    raise ValidationError("Text cannot be empty")
                return True
        """
        return True

    # ========================================================================
    # HELPER METHODS (Common functionality)
    # ========================================================================

    async def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """
        Execute with automatic retry on failure.
        
        Uses retry_count from config.
        
        Args:
            **kwargs: Parameters for execute()
            
        Returns:
            Execution result
            
        Raises:
            Last exception if all retries exhausted
        """
        from .exceptions import ProcessingError
        
        last_error = None
        
        for attempt in range(self.retry_count + 1):
            try:
                logger.debug(f"Execution attempt {attempt + 1}/{self.retry_count + 1}")
                return await self.execute(**kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retries remaining: {self.retry_count - attempt}"
                )
                
                if attempt < self.retry_count:
                    # Wait before retry (exponential backoff)
                    import asyncio
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, ...
                    await asyncio.sleep(wait_time)
        
        raise ProcessingError(
            f"Tool execution failed after {self.retry_count + 1} attempts. "
            f"Last error: {str(last_error)}"
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this tool.
        
        Returns:
            Dict with tool metadata
        """
        return {
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self.enabled

    async def health_check(self) -> bool:
        """
        Check if tool is healthy and ready to use.
        
        Override in subclasses to implement specific health checks.
        
        Returns:
            bool: True if healthy
        """
        return self.enabled
