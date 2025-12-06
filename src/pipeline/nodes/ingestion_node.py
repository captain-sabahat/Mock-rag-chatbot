"""
================================================================================
INGESTION NODE - Parse File Content to Text
================================================================================

PURPOSE:
--------
First node in pipeline. Converts raw file bytes to parsed text.

Supported formats:
  - PDF (.pdf)
  - Plain text (.txt)
  - JSON (.json)
  - Markdown (.md)
  - DOCX (.docx) - optional

Responsibilities:
  - Detect file format
  - Parse file content
  - Extract text
  - Validate output
  - Track statistics (page count, word count, etc.)

INPUT:
------
  PipelineState.raw_content = bytes (file content)
  PipelineState.file_name = str (file name)

OUTPUT:
-------
  PipelineState.parsed_text = str (extracted text)
  PipelineState.checkpoints["ingestion"] = updated

ERROR HANDLING:
---------------
  - Invalid format → ValidationError
  - Parsing failure → ProcessingError
  - Empty content → ValidationError

================================================================================
"""

import logging
from datetime import datetime
from typing import Optional
import fitz  # PyMuPDF for PDF parsing

from src.pipeline.schemas import PipelineState, NodeStatus
from src.core import ValidationError, ProcessingError

logger = logging.getLogger(__name__)


async def ingestion_node(state: PipelineState) -> PipelineState:
    """
    Parse raw file content to text.
    
    Args:
        state: Pipeline state with raw_content
        
    Returns:
        Updated state with parsed_text
    """
    start_time = datetime.utcnow()
    
    logger.info(f"📄 Ingestion: Processing {state.file_name}")
    
    try:
        # Validate input
        if not state.raw_content:
            raise ValidationError("No file content provided")
        
        if not state.file_name:
            raise ValidationError("No file name provided")
        
        # Detect file format
        file_ext = state.file_name.lower().split('.')[-1]
        
        logger.debug(f"   File type: .{file_ext}")
        
        # Parse based on format
        if file_ext == "pdf":
            parsed_text = await _parse_pdf(state.raw_content)
        elif file_ext == "txt":
            parsed_text = await _parse_txt(state.raw_content)
        elif file_ext == "json":
            parsed_text = await _parse_json(state.raw_content)
        elif file_ext == "md":
            parsed_text = await _parse_markdown(state.raw_content)
        else:
            raise ValidationError(f"Unsupported file format: .{file_ext}")
        
        # Validate parsed content
        if not parsed_text or len(parsed_text.strip()) == 0:
            raise ValidationError(f"No text extracted from {state.file_name}")
        
        # Update state
        state.parsed_text = parsed_text
        
        # Calculate statistics
        word_count = len(parsed_text.split())
        char_count = len(parsed_text)
        
        state.add_message(
            f"✅ Ingestion: Extracted {word_count} words, {char_count} chars"
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        state.update_checkpoint(
            "ingestion",
            status=NodeStatus.COMPLETED,
            output_ready=True,
            output_data={
                "word_count": word_count,
                "char_count": char_count,
                "file_format": file_ext
            },
            duration_ms=duration_ms
        )
        
        logger.info(f"✅ Ingestion complete: {word_count} words extracted")
        return state
    
    except ValidationError as e:
        logger.error(f"❌ Ingestion validation error: {e}")
        state.status = "error"
        state.add_error(f"Ingestion validation: {e.message}")
        state.update_checkpoint(
            "ingestion",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}")
        state.status = "error"
        state.add_error(f"Ingestion error: {str(e)}")
        state.update_checkpoint(
            "ingestion",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=str(e)
        )
        return state


async def _parse_pdf(content: bytes) -> str:
    """Parse PDF file to text."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        return text
    
    except Exception as e:
        raise ProcessingError(f"PDF parsing failed: {str(e)}")


async def _parse_txt(content: bytes) -> str:
    """Parse plain text file."""
    try:
        # Try UTF-8 first
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            return content.decode('latin-1')
    
    except Exception as e:
        raise ProcessingError(f"Text parsing failed: {str(e)}")


async def _parse_json(content: bytes) -> str:
    """Parse JSON file to text."""
    import json
    
    try:
        text_bytes = content.decode('utf-8')
        data = json.loads(text_bytes)
        
        # Convert to readable text
        if isinstance(data, dict):
            text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        elif isinstance(data, list):
            text = "\n".join([str(item) for item in data])
        else:
            text = str(data)
        
        return text
    
    except Exception as e:
        raise ProcessingError(f"JSON parsing failed: {str(e)}")


async def _parse_markdown(content: bytes) -> str:
    """Parse Markdown file to text."""
    try:
        return content.decode('utf-8')
    
    except Exception as e:
        raise ProcessingError(f"Markdown parsing failed: {str(e)}")
