"""
================================================================================
INGESTION NODE - Parse File Content to Text (v2.5 FIXED)
================================================================================

v2.5 FIXES:
✅ Removed: state.ingestion_status_flag = "COMPLETED" (causes Pydantic error)
✅ Added: Safe enrichment fields to state (ingestion_char_count, etc.)
✅ Preserved: All parsing logic, error handling, A-B-C pattern
✅ Preserved: NodeStatus creation and monitoring file writing

KEY PATTERN CHANGE:
- OLD (BROKEN): state.ingestion_status_flag = "COMPLETED"  ❌
- NEW (FIXED): state.ingestion_char_count = char_count  ✅

Nodes enrich state with METRICS (counts, samples), not FLAGS.
Orchestrator reads enriched state to build monitoring context.

================================================================================
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.pipeline.schemas import PipelineState, NodeStatus, NodeStatusEnum
from src.core.exceptions import ValidationError, ProcessingError

logger = logging.getLogger(__name__)

async def ingestion_node(state: PipelineState) -> PipelineState:
    """
    Parse raw file content to text.
    
    Implements A-B-C pattern:
    A: Validate input (file_content, file_name)
    B: Execute parsing based on file format
    C: Validate output (parsed_text non-empty)
    
    Args:
        state: Pipeline state with raw_content and file_name
    
    Returns:
        Updated state with parsed_text and node status
    """
    node_name = "ingestion"
    start_time = datetime.utcnow()
    logger.info(f"📄 Ingestion Node: Processing {state.file_name}")

    # Initialize NodeStatus object
    status = NodeStatus(
        node_name=node_name,
        status=NodeStatusEnum.PROCESSING,
        request_id=state.request_id,
        timestamp=start_time,
        input_received=False,
        input_valid=False,
        output_generated=False,
        output_valid=False,
        start_time=start_time,
    )

    try:
        # ====== A METHOD: INPUT VALIDATION ======
        logger.info("🔍 A Method: Validating input...")

        # Check 1: File content received
        if not state.raw_content:
            logger.error("❌ No file content received")
            status.input_received = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No file content provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No file content provided")

        status.input_received = True
        logger.debug(f"✅ Input received: {len(state.raw_content)} bytes")

        # Check 2: File name exists
        if not state.file_name:
            logger.error("❌ No file name provided")
            status.input_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No file name provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No file name provided")

        status.input_valid = True
        logger.debug(f"✅ File name valid: {state.file_name}")

        # ====== B METHOD: EXECUTE PARSING ======
        logger.info("🚀 B Method: Executing parsing...")

        # Detect file format
        file_ext = state.file_name.lower().split('.')[-1]
        logger.debug(f"Detected format: .{file_ext}")

        # Parse based on format (delegate to tools)
        if file_ext == "pdf":
            parsed_text = await _parse_pdf(state.raw_content)
        elif file_ext == "txt":
            parsed_text = await _parse_txt(state.raw_content)
        elif file_ext == "json":
            parsed_text = await _parse_json(state.raw_content)
        elif file_ext == "md":
            parsed_text = await _parse_markdown(state.raw_content)
        else:
            logger.error(f"❌ Unsupported file format: .{file_ext}")
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "UnsupportedFormatError"
            status.exception_message = f"Unsupported file format: .{file_ext}"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError(f"Unsupported file format: .{file_ext}")

        logger.debug(f"Parsing complete: {len(parsed_text)} chars extracted")

        # ====== C METHOD: OUTPUT VALIDATION ======
        logger.info("✅ C Method: Validating output...")

        # Check 1: Output generated
        if not parsed_text:
            logger.error("❌ No text extracted")
            status.output_generated = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "EmptyOutputError"
            status.exception_message = f"No text extracted from {state.file_name}"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError(f"No text extracted from {state.file_name}")

        status.output_generated = True
        logger.debug(f"✅ Output generated: {len(parsed_text)} chars")

        # Check 2: Output is valid (non-empty, valid type)
        if not isinstance(parsed_text, str):
            logger.error(f"❌ Invalid output type: {type(parsed_text)}")
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "OutputValidationError"
            status.exception_message = f"Expected str, got {type(parsed_text)}"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError(f"Expected str output, got {type(parsed_text)}")

        if len(parsed_text.strip()) == 0:
            logger.error("❌ Output is empty after stripping")
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "EmptyOutputError"
            status.exception_message = "Parsed text is empty"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Parsed text is empty")

        status.output_valid = True
        logger.debug("✅ Output valid")

        # ====== SUCCESS: Update state and status ======
        state.parsed_text = parsed_text
        word_count = len(parsed_text.split())
        char_count = len(parsed_text)

        state.add_message(
            f"✅ Ingestion: Extracted {word_count} words, {char_count} chars"
        )

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # These are safe optional fields that orchestrator reads for monitoring
        state.ingestion_size_bytes = len(state.raw_content or b"")
        state.ingestion_char_count = char_count  # ✅ Not a flag, safe to set
        state.ingestion_word_count = word_count  # ✅ Not a flag, safe to set
        state.ingestion_format_detected = file_ext  # ✅ Not a flag, safe to set
        state.ingestion_encoding = "utf-8"  # ✅ Not a flag, safe to set

        status.status = NodeStatusEnum.COMPLETED
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (
            status.end_time - start_time
        ).total_seconds() * 1000

        logger.info(
            f"✅ Ingestion COMPLETED | status=COMPLETED | "
            f"size={state.ingestion_size_bytes}B | chars={char_count}"
        )

        # ====== WRITE MONITORING FILE ======
        await _write_node_monitoring(state.request_id, status)

        # Update state
        state.node_statuses[node_name] = status
        return state

    except Exception as e:
        # ====== ERROR: Catch and report ======
        logger.error(f"❌ Ingestion failed: {str(e)}", exc_info=True)

        # ====== v2.5 FIX: Use ENRICHMENT fields, not status flags ======
        state.ingestion_char_count = 0  # ✅ Safe to set
        state.ingestion_word_count = 0  # ✅ Safe to set

        status.status = NodeStatusEnum.FAILED
        status.exception_type = type(e).__name__
        status.exception_message = str(e)

        # Determine severity
        if isinstance(e, ValidationError):
            status.exception_severity = "CRITICAL"
        else:
            status.exception_severity = "CRITICAL"

        status.end_time = datetime.utcnow()
        status.execution_time_ms = (
            status.end_time - start_time
        ).total_seconds() * 1000

        logger.error(f"❌ Ingestion failed: {str(e)} | status=FAILED")

        # Write monitoring file
        await _write_node_monitoring(state.request_id, status)

        # Update state
        state.node_statuses[node_name] = status
        state.add_error(f"Ingestion error: {str(e)}")

        # Return state (don't raise) so orchestrator can handle CB
        return state

async def _parse_pdf(content: bytes) -> str:
    """
    Parse PDF file to text.
    Preferred: src.tools.ingestion.parsers.PDFParser
    Fallback: fitz (PyMuPDF) direct extraction
    """
    # 1) Try tool-based PDFParser
    try:
        from src.tools.ingestion.parsers import PDFParser
        parser = PDFParser()
        result = await parser.parse(content, config={})
        # Expecting {"parsed_text": str, "metadata": {...}}
        text = result.get("parsed_text", "")
        if not isinstance(text, str) or not text.strip():
            raise ProcessingError("PDFParser returned empty or invalid parsed_text")
        return text
    except ImportError:
        # Only log this if the import actually fails
        logger.warning("⚠️ PDFParser not available, using fallback")
    except Exception as e:
        # PDFParser present but failed → log and fall back
        logger.warning(f"⚠️ PDFParser failed, using fallback: {str(e)}")

    # 2) Fallback implementation using fitz
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text() or ""
        doc.close()
        return text
    except Exception as e:
        raise ProcessingError(f"PDF parsing failed: {str(e)}")

async def _parse_txt(content: bytes) -> str:
    """Parse plain text file."""
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('latin-1')
    except Exception as e:
        raise ProcessingError(f"Text parsing failed: {str(e)}")

async def _parse_json(content: bytes) -> str:
    """Parse JSON file to text."""
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

async def _write_node_monitoring(request_id: str, status: NodeStatus) -> None:
    """
    Write node status to monitoring file immediately.
    Creates: data/monitoring/nodes/{request_id}/ingestion_node.json
    """
    try:
        monitoring_dir = Path(f"./data/monitoring/nodes/{request_id}")
        monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Write node-specific status file
        node_file = monitoring_dir / f"{status.node_name}_node.json"
        with open(node_file, "w") as f:
            json.dump(status.to_dict(), f, indent=2, default=str)

        logger.debug(f"📝 Wrote monitoring: {node_file}")
    except Exception as e:
        logger.error(f"⚠️ Failed to write monitoring file: {str(e)}")
        # Don't raise, monitoring failure shouldn't crash pipeline
