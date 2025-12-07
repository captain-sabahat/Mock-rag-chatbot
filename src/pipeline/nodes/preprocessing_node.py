"""

================================================================================

PREPROCESSING NODE - Clean and Normalize Text

================================================================================

PURPOSE:

--------

Second node in pipeline. Cleans, normalizes, and validates text.

Responsibilities:

- Remove special characters/control characters
- Normalize whitespace
- Remove duplicates/boilerplate
- Validate text quality
- Language detection (optional)
- Remove PII (optional)

INPUT:

------

PipelineState.parsed_text = str (raw parsed text)

OUTPUT:

-------

PipelineState.cleaned_text = str (cleaned text)
PipelineState.checkpoints["preprocessing"] = updated

TECHNIQUES:

-----------

- HTML/XML tag removal
- Whitespace normalization
- Special character handling
- Duplicate line removal
- Quality validation

================================================================================

"""

import logging
import re
from datetime import datetime
from src.pipeline.schemas import PipelineState, NodeStatus
from src.core import ValidationError

logger = logging.getLogger(__name__)


async def preprocessing_node(state: PipelineState) -> PipelineState:
    """
    Clean and normalize parsed text.

    Args:
        state: Pipeline state with parsed_text

    Returns:
        Updated state with cleaned_text
    """
    start_time = datetime.utcnow()
    logger.info("🧹 Preprocessing: Cleaning text")

    try:
        # Validate input
        if not state.parsed_text:
            raise ValidationError("No parsed text to preprocess")

        text = state.parsed_text
        original_length = len(text)

        # Step 1: Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Step 2: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Step 3: Remove control characters
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')

        # Step 4: Remove extra punctuation
        text = re.sub(r'([!?.])\\1{2,}', r'\1', text)

        # Step 5: Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Step 6: Remove duplicates (preserve order)
        seen = set()
        unique_lines = []
        for line in text.split('\n'):
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        text = '\n'.join(unique_lines)

        # Validate quality
        cleaned_length = len(text)
        quality_ratio = cleaned_length / original_length if original_length > 0 else 0

        if quality_ratio < 0.1:  # Less than 10% of original
            logger.warning(f"⚠️ Low text quality after cleaning: {quality_ratio:.1%}")

        # ✅ LOGGING BLOCK - INSIDE FUNCTION AFTER TEXT CLEANING
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 PREPROCESSING NODE - DATA OUTPUT")
        logger.info(f"{'='*80}")
        logger.info(f"📥 INPUT: {original_length} chars")
        logger.info(f"   First 200 chars: {state.parsed_text[:200]!r}")
        logger.info(f"📤 OUTPUT: {cleaned_length} chars")
        logger.info(f"   Cleaned: {text[:200]!r}")
        logger.info(f"   Retention: {(cleaned_length/original_length*100):.1f}%")
        logger.info(f"{'='*80}\n")

        # Update state
        state.cleaned_text = text
        state.add_message(
            f"✅ Preprocessing: Cleaned text "
            f"({original_length} → {cleaned_length} chars, {quality_ratio:.1%} retained)"
        )

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        state.update_checkpoint(
            "preprocessing",
            status=NodeStatus.COMPLETED,
            output_ready=True,
            output_data={
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "quality_ratio": quality_ratio
            },
            duration_ms=duration_ms
        )

        logger.info(f"✅ Preprocessing complete: {cleaned_length} chars retained")
        return state

    except ValidationError as e:
        logger.error(f"❌ Preprocessing validation error: {e}")
        state.status = "error"
        state.add_error(f"Preprocessing validation: {e.message}")
        state.update_checkpoint(
            "preprocessing",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        state.status = "error"
        state.add_error(f"Preprocessing error: {str(e)}")
        state.update_checkpoint(
            "preprocessing",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=str(e)
        )
        return state