"""
================================================================================
CHUNKING NODE - Split Text into Chunks (v2.5 FIXED)
================================================================================

v2.5 FIXES:
✅ Removed: state.chunking_status_flag = "COMPLETED" (causes Pydantic error)
✅ Removed: state.chunking_error, state.chunking_error_type (Pydantic errors)
✅ Added: Safe enrichment fields (num_chunks, chunk_size_min, chunk_size_max)
✅ Preserved: All chunking logic, strategies, validation, monitoring

KEY PATTERN CHANGE:
- OLD (BROKEN): state.chunking_status_flag = "COMPLETED"  ❌
- NEW (FIXED): state.num_chunks = len(chunks)  ✅

Nodes enrich state with METRICS (counts, ranges), not FLAGS.

================================================================================
"""

import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List

from src.pipeline.schemas import PipelineState, NodeStatus, NodeStatusEnum
from src.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

async def chunking_node(state: PipelineState) -> PipelineState:
    """Split cleaned text into chunks."""
    node_name = "chunking"
    start_time = datetime.utcnow()
    logger.info(
        f"✂️ Chunking: Using {state.chunking_strategy} strategy "
        f"(size={state.chunk_size}, overlap={state.chunk_overlap})"
    )

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

        if not state.cleaned_text:
            logger.error("❌ No cleaned text to chunk")
            status.input_received = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No cleaned text provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No cleaned text to chunk")

        status.input_received = True
        status.input_valid = True

        # ====== B METHOD: EXECUTE CHUNKING ======
        logger.info("🚀 B Method: Executing chunking...")

        # Select chunking strategy
        if state.chunking_strategy == "recursive":
            chunks = await _recursive_chunking(state.cleaned_text, state.chunk_size, state.chunk_overlap)
        elif state.chunking_strategy == "token":
            chunks = await _token_chunking(state.cleaned_text, state.chunk_size, state.chunk_overlap)
        elif state.chunking_strategy == "sliding_window":
            chunks = await _sliding_window_chunking(state.cleaned_text, state.chunk_size, state.chunk_overlap)
        elif state.chunking_strategy == "sentence":
            chunks = await _sentence_chunking(state.cleaned_text, state.chunk_size)
        else:
            raise ValidationError(f"Unknown strategy: {state.chunking_strategy}")

        # ====== C METHOD: OUTPUT VALIDATION ======
        logger.info("✅ C Method: Validating output...")

        if not chunks:
            logger.error("❌ No chunks created")
            status.output_generated = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "EmptyOutputError"
            status.exception_message = "No chunks created"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No chunks created")

        status.output_generated = True
        status.output_valid = True

        # Add metadata
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
            })

        # ====== SUCCESS ======
        state.chunks = chunks
        state.chunk_metadata = metadata
        state.add_message(
            f"✅ Chunking: Created {len(chunks)} chunks "
            f"(avg size: {sum(len(c) for c in chunks) // len(chunks)} chars)"
        )

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Use safe enrichment fields
        state.num_chunks = len(chunks)
        if chunks:
            state.chunk_size_min = min(len(c) for c in chunks)
            state.chunk_size_max = max(len(c) for c in chunks)
        else:
            state.chunk_size_min = 0
            state.chunk_size_max = 0

        status.status = NodeStatusEnum.COMPLETED
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000

        logger.info(
            f"✅ Chunking COMPLETED | status=COMPLETED | chunks={len(chunks)} | "
            f"min_size={state.chunk_size_min} | max_size={state.chunk_size_max}"
        )

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        return state

    except Exception as e:
        logger.error(f"❌ Chunking failed: {str(e)}", exc_info=True)

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Safe enrichment fields on error too
        state.num_chunks = 0
        state.chunk_size_min = 0
        state.chunk_size_max = 0

        status.status = NodeStatusEnum.FAILED
        status.exception_type = type(e).__name__
        status.exception_message = str(e)
        status.exception_severity = "CRITICAL"
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000

        logger.error(f"❌ Chunking failed: {str(e)} | status=FAILED")

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        state.add_error(f"Chunking error: {str(e)}")

        return state


async def _recursive_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
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


async def _token_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Token-based chunking."""
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-max(1, overlap // 10):]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


async def _sliding_window_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window chunking."""
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(text), step_size):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


async def _sentence_chunking(text: str, chunk_size: int) -> List[str]:
    """Sentence-based chunking."""
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


async def _write_node_monitoring(request_id: str, status: NodeStatus) -> None:
    """Write node status to monitoring file."""
    try:
        monitoring_dir = Path(f"./data/monitoring/nodes/{request_id}")
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        node_file = monitoring_dir / f"{status.node_name}_node.json"
        with open(node_file, "w") as f:
            json.dump(status.to_dict(), f, indent=2, default=str)
        logger.debug(f"📝 Monitoring written: {node_file}")
    except Exception as e:
        logger.error(f"⚠️ Monitoring write failed: {str(e)}")
