"""
================================================================================
CHUNKING NODE - Split Text into Chunks
================================================================================

PURPOSE:
--------
Third node in pipeline. Splits cleaned text into manageable chunks.

Chunking strategies:
  1. Recursive - Split by separators, preserving context
  2. Token-based - Split by token count
  3. Sliding window - Overlapping chunks
  4. Sentence-based - Split by sentences

Responsibilities:
  - Split text by configured strategy
  - Add overlap for context
  - Validate chunk sizes
  - Add chunk metadata
  - Track statistics

INPUT:
------
  PipelineState.cleaned_text = str (cleaned text)
  PipelineState.chunk_size = int (target chunk size)
  PipelineState.chunk_overlap = int (overlap size)
  PipelineState.chunking_strategy = str (strategy name)

OUTPUT:
-------
  PipelineState.chunks = List[str] (text chunks)
  PipelineState.chunk_metadata = List[Dict] (metadata)
  PipelineState.checkpoints["chunking"] = updated

================================================================================
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from src.pipeline.schemas import PipelineState, NodeStatus
from src.core import ValidationError

logger = logging.getLogger(__name__)


async def chunking_node(state: PipelineState) -> PipelineState:
    """
    Split cleaned text into chunks.
    
    Args:
        state: Pipeline state with cleaned_text
        
    Returns:
        Updated state with chunks
    """
    start_time = datetime.utcnow()
    
    logger.info(
        f"✂️  Chunking: Using {state.chunking_strategy} strategy "
        f"(size={state.chunk_size}, overlap={state.chunk_overlap})"
    )
    
    try:
        # Validate input
        if not state.cleaned_text:
            raise ValidationError("No cleaned text to chunk")
        
        # Select chunking strategy
        if state.chunking_strategy == "recursive":
            chunks = await _recursive_chunking(
                state.cleaned_text,
                state.chunk_size,
                state.chunk_overlap
            )
        elif state.chunking_strategy == "token":
            chunks = await _token_chunking(
                state.cleaned_text,
                state.chunk_size,
                state.chunk_overlap
            )
        elif state.chunking_strategy == "sliding_window":
            chunks = await _sliding_window_chunking(
                state.cleaned_text,
                state.chunk_size,
                state.chunk_overlap
            )
        elif state.chunking_strategy == "sentence":
            chunks = await _sentence_chunking(
                state.cleaned_text,
                state.chunk_size
            )
        else:
            raise ValidationError(f"Unknown strategy: {state.chunking_strategy}")
        
        # Validate chunks
        if not chunks:
            raise ValidationError("No chunks created")
        
        # Add metadata to chunks
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
                "start_char": sum(len(c) for c in chunks[:i]) + i,  # Approximate
            })
        
        # Update state
        state.chunks = chunks
        state.chunk_metadata = metadata
        
        state.add_message(
            f"✅ Chunking: Created {len(chunks)} chunks "
            f"(avg size: {sum(len(c) for c in chunks) // len(chunks)} chars)"
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        state.update_checkpoint(
            "chunking",
            status=NodeStatus.COMPLETED,
            output_ready=True,
            output_data={
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks),
                "strategy": state.chunking_strategy
            },
            duration_ms=duration_ms
        )
        
        logger.info(f"✅ Chunking complete: {len(chunks)} chunks created")
        return state
    
    except ValidationError as e:
        logger.error(f"❌ Chunking validation error: {e}")
        state.status = "error"
        state.add_error(f"Chunking validation: {e.message}")
        state.update_checkpoint(
            "chunking",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except Exception as e:
        logger.error(f"❌ Chunking failed: {str(e)}")
        state.status = "error"
        state.add_error(f"Chunking error: {str(e)}")
        state.update_checkpoint(
            "chunking",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=str(e)
        )
        return state


async def _recursive_chunking(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """Recursive chunking strategy."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    
    def split_text(text: str, separators: List[str]) -> List[str]:
        good_splits = []
        for i, separator in enumerate(separators):
            if separator == "":
                splits = list(text)
            else:
                splits = text.split(separator)
            
            good_splits = [s for s in splits if len(s) > chunk_size]
            
            if good_splits:
                break
        
        if not good_splits:
            return [text]
        
        result = []
        for s in good_splits:
            if len(s) > chunk_size:
                result.extend(split_text(s, separators[separators.index(separator) + 1:]))
            else:
                result.append(s)
        
        return result
    
    splits = split_text(text, separators)
    
    # Combine with overlap
    current_chunk = ""
    for split in splits:
        if len(current_chunk) + len(split) + 1 <= chunk_size:
            current_chunk += split + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = split
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


async def _token_chunking(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """Token-based chunking (approximate)."""
    words = text.split()
    chunks = []
    
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Add overlap
            current_chunk = current_chunk[-max(1, overlap // 10):]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


async def _sliding_window_chunking(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """Sliding window chunking."""
    chunks = []
    step_size = chunk_size - overlap
    
    for i in range(0, len(text), step_size):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    return chunks


async def _sentence_chunking(
    text: str,
    chunk_size: int
) -> List[str]:
    """Sentence-based chunking."""
    import re
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
