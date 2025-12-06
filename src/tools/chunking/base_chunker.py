"""
================================================================================
BASE CHUNKER MODULE
src/tools/chunking/base_chunker.py

MODULE PURPOSE:
───────────────
Abstract base class defining interface for text chunking.
Supports multiple chunking strategies.

WORKING & METHODOLOGY:
──────────────────────
1. CHUNKING STRATEGIES:
   - FIXED_SIZE: Fixed-size non-overlapping chunks
   - SLIDING_WINDOW: Fixed-size overlapping chunks
   - SEMANTIC: Intelligent segmentation (sentence/paragraph)
   - RECURSIVE: Hierarchical chunking

2. CORE OPERATIONS:
   - chunk(): Split text into chunks
   - batch_chunk(): Process multiple documents
   - validate_chunk(): Check chunk properties
   - get_stats(): Get chunking statistics

3. OPTIMIZATION:
   - Batch processing
   - Memory efficiency
   - Overlap support
   - Metadata preservation

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Convert documents to indexed chunks
- Preserve document structure
- Enable semantic search
- Support embedding generation

CHUNK PROPERTIES:
─────────────────
- ID: Unique identifier per chunk
- Content: Text content
- Indices: Start/end positions in original text
- Metadata: Document attribution and context

================================================================================
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time


class ChunkingStrategy(Enum):
    """Chunking strategy enumeration."""
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class Chunk:
    """Chunk data class with metadata."""
    chunk_id: str
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]


class BaseChunker(ABC):
    """
    Abstract base class for chunking implementations.
    
    Defines interface for:
    - Text splitting strategies
    - Batch processing
    - Statistics tracking
    - Metadata management
    
    Example:
        >>> class MyChunker(BaseChunker):
        >>>     async def chunk(self, text, metadata):
        >>>         # Implementation
        >>>         pass
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters (config-driven, not hardcoded)
        self.chunk_size = config.get("chunk_size", 512) if config else 512
        self.overlap = config.get("overlap", 50) if config else 50
        self.min_chunk_size = config.get("min_chunk_size", 50) if config else 50
        self.max_chunk_size = config.get("max_chunk_size", 2048) if config else 2048
        
        # Statistics
        self.total_chunks = 0
        self.total_characters = 0
        self.chunking_operations = 0
        
        self.logger.info(
            "BaseChunker initialized (chunk_size=%d, overlap=%d)" %
            (self.chunk_size, self.overlap)
        )
    
    @abstractmethod
    async def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            metadata: Optional document metadata
            
        Returns:
            List of Chunk objects
        """
        pass
    
    async def batch_chunk(
        self,
        documents: List[str],
        base_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[Chunk]]:
        """
        Process multiple documents.
        
        Args:
            documents: List of documents
            base_metadata: Optional metadata per document
            
        Returns:
            List of chunk lists
        """
        results = []
        for i, doc in enumerate(documents):
            meta = base_metadata[i] if base_metadata and i < len(base_metadata) else None
            chunks = await self.chunk(doc, meta)
            results.append(chunks)
        
        return results
    
    def _normalize_newlines(self, text: str) -> str:
        """
        Normalize line endings.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        text = text.replace("\\r\\n", "\\n")
        text = text.replace("\\r", "\\n")
        return text
    
    def _validate_chunk(self, chunk: Chunk) -> bool:
        """
        Validate chunk properties.
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if valid, False otherwise
        """
        if len(chunk.content) < self.min_chunk_size:
            self.logger.warning(
                "Chunk too small: %d bytes (min %d)" %
                (len(chunk.content), self.min_chunk_size)
            )
            return False
        
        if len(chunk.content) > self.max_chunk_size:
            self.logger.warning(
                "Chunk too large: %d bytes (max %d)" %
                (len(chunk.content), self.max_chunk_size)
            )
            return False
        
        return True
    
    async def _record_chunk(self, chunks: List[Chunk]):
        """Record chunking operation."""
        self.total_chunks += len(chunks)
        self.total_characters += sum(len(c.content) for c in chunks)
        self.chunking_operations += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get chunking statistics.
        
        Returns:
            Statistics dictionary
        """
        avg_chunk_size = (self.total_characters / self.total_chunks
                         if self.total_chunks > 0 else 0)
        
        return {
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "operations": self.chunking_operations,
            "avg_chunk_size": avg_chunk_size,
            "config_chunk_size": self.chunk_size,
            "config_overlap": self.overlap,
        }
    
    async def health_check(self) -> bool:
        """
        Check chunker health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to chunk test text
            test_text = "This is a test chunk. It contains multiple sentences. " * 10
            result = await self.chunk(test_text)
            
            if result and len(result) > 0:
                return True
            return False
        except Exception as e:
            self.logger.error("Health check failed: %s" % str(e))
            return False