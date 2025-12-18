"""
================================================================================
PREPROCESSING NODE - Clean and Normalize Text (v2.5 HOTFIX)
================================================================================

v2.5 HOTFIX:
✅ Removed: state.preprocess_language = "en" (field doesn't exist in schema)
✅ Kept: All other enrichment fields (preprocess_item_count, head, tail)
✅ Preserved: All text cleaning logic, validation, monitoring

SCHEMA COMPLIANCE FIX:
- OLD: state.preprocess_language = "en"  ❌ Field not defined
- NEW: Removed - not in schema  ✅

Use ONLY fields defined in schemas.py

================================================================================
"""

import logging
import re
import json
from datetime import datetime
from pathlib import Path

from src.pipeline.schemas import PipelineState, NodeStatus, NodeStatusEnum
from src.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

async def preprocessing_node(state: PipelineState) -> PipelineState:
    """Clean and normalize parsed text."""
    node_name = "preprocessing"
    start_time = datetime.utcnow()
    logger.info("🧹 Preprocessing Node: Cleaning text")

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

        if not state.parsed_text:
            logger.error("❌ No parsed text to preprocess")
            status.input_received = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No parsed text provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No parsed text to preprocess")

        status.input_received = True
        status.input_valid = True
        text = state.parsed_text
        original_length = len(text)

        # ====== B METHOD: EXECUTE CLEANING ======
        logger.info("🚀 B Method: Executing cleaning...")

        # Step 1: Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Step 2: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Step 3: Remove control characters
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')

        # Step 4: Remove extra punctuation
        text = re.sub(r'([!?.])\\1{2,}', r'\1', text)

        # Step 5: Remove empty lines and duplicates
        seen = set()
        unique_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        text = '\n'.join(unique_lines)

        # ====== C METHOD: OUTPUT VALIDATION ======
        logger.info("✅ C Method: Validating output...")

        cleaned_length = len(text)
        quality_ratio = cleaned_length / original_length if original_length > 0 else 0

        if not text:
            logger.error("❌ Cleaned text is empty")
            status.output_generated = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "EmptyOutputError"
            status.exception_message = "Text cleaning produced empty result"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Text cleaning produced empty result")

        status.output_generated = True
        status.output_valid = True

        # ====== SUCCESS ======
        state.cleaned_text = text
        state.add_message(
            f"✅ Preprocessing: {original_length} → {cleaned_length} chars, "
            f"{quality_ratio:.1%} retained"
        )

        # ====== v2.5 HOTFIX: Use ONLY schema-defined enrichment fields ======
        # Count unique items from cleaned text
        items = unique_lines  # Already extracted above

        # ✅ HOTFIX: Use safe enrichment fields (removed preprocess_language)
        state.preprocess_item_count = len(items)
        state.preprocess_head = items[:3] if len(items) >= 3 else items
        state.preprocess_tail = items[-3:] if len(items) >= 3 else items
        # ❌ REMOVED: state.preprocess_language = "en" (not in schema)

        status.status = NodeStatusEnum.COMPLETED
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000

        logger.info(
            f"✅ Preprocessing COMPLETED | status=COMPLETED | items={len(items)}"
        )

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        return state

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}", exc_info=True)

        # ====== v2.5 HOTFIX: Use ONLY schema-defined enrichment fields ======
        # ✅ HOTFIX: Safe enrichment fields on error too
        state.preprocess_item_count = 0
        state.preprocess_head = []
        state.preprocess_tail = []

        status.status = NodeStatusEnum.FAILED
        status.exception_type = type(e).__name__
        status.exception_message = str(e)
        status.exception_severity = "CRITICAL"
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000

        logger.error(f"❌ Preprocessing failed: {str(e)} | status=FAILED")

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        state.add_error(f"Preprocessing error: {str(e)}")

        return state


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
