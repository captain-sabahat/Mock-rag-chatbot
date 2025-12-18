"""
================================================================================
EMBEDDING NODE - Pipeline Node with Metrics (v2.5 FIXED)
================================================================================

v2.5 FIXES:
✅ Removed: state.embedding_status_flag = "COMPLETED" (causes Pydantic error)
✅ Removed: state.embedding_error, state.embedding_error_type (Pydantic errors)
✅ Removed: state.embedding_dimension = len(state.embeddings) (wrong value, should use config)
✅ Added: Safe enrichment fields (num_embeddings, embedding_samples)
✅ Preserved: All embedding logic, validation, monitoring

KEY PATTERN CHANGE:
- OLD (BROKEN): state.embedding_status_flag = "COMPLETED"  ❌
- NEW (FIXED): state.num_embeddings = len(state.embeddings)  ✅

Nodes enrich state with METRICS (counts, samples), not FLAGS.

================================================================================
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.pipeline.schemas import PipelineState, NodeStatus, NodeStatusEnum
from src.core.exceptions import ValidationError
from src.tools.embeddings.embed_registry import (
    load_embedding_config,
    EmbedderFactory,
)

logger = logging.getLogger(__name__)

async def embedding_node(state: PipelineState) -> PipelineState:
    """
    Embed chunks using config-driven provider selection.

    Flow:
    1. Load embedding config from YAML (registry_embed.py)
    2. A Method: Validate input (chunks exist)
    3. B Method: Create embedder from factory + embed
    4. C Method: Validate embeddings (count, dimension)
    5. Track metrics: total count, speed, time
    6. Write node status with metrics
    """
    node_name = "embedding"
    start_time = datetime.utcnow()

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
        # ====== LOAD CONFIG (from tools layer) ======
        logger.info("📋 Loading embedding configuration...")
        try:
            config = load_embedding_config("config/defaults/embeddings.yaml")
        except Exception as config_error:
            logger.error(f"❌ Config load failed: {str(config_error)}")
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ConfigError"
            status.exception_message = str(config_error)
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise

        logger.info(
            f"✅ Config loaded: provider={config.active_provider}, "
            f"dimension={config.dimension}, batch_size={config.batch_size}"
        )

        # ====== A METHOD: INPUT VALIDATION ======
        logger.info("🔍 A Method: Validating input...")

        if not state.chunks:
            logger.error("❌ No chunks to embed")
            status.input_received = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No chunks provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No chunks to embed")

        status.input_received = True
        logger.info(f"✅ Received: {len(state.chunks)} chunks")

        if any(not chunk or len(chunk.strip()) == 0 for chunk in state.chunks):
            logger.error("❌ Some chunks are empty")
            status.input_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ValidationError"
            status.exception_message = "Some chunks are empty"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Some chunks are empty")

        status.input_valid = True
        logger.info(f"✅ Input validated: {len(state.chunks)} non-empty chunks")

        # ====== B METHOD: CREATE EMBEDDER & EMBED ======
        logger.info(f"🚀 B Method: Creating {config.active_provider} embedder...")

        # Factory creates correct embedder (config-driven)
        embedder = EmbedderFactory.create(config)

        logger.info(
            f"📦 Starting embedding: {len(state.chunks)} chunks with "
            f"{config.active_provider} embedder (batch_size={config.batch_size})..."
        )

        # Embed all chunks
        embedding_results, total_embeddings = await embedder.embed_batch(state.chunks)

        logger.info(
            f"✅ Embedding complete: {total_embeddings} embeddings generated"
        )

        # Extract vectors from results
        state.embeddings = [result.vector for result in embedding_results]

        # ====== C METHOD: OUTPUT VALIDATION ======
        logger.info("✅ C Method: Validating output...")

        if len(state.embeddings) == 0:
            logger.error("❌ No embeddings generated")
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ValidationError"
            status.exception_message = "No embeddings generated"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No embeddings generated")

        if len(state.embeddings) != len(state.chunks):
            logger.error(
                f"❌ Embedding count mismatch: {len(state.embeddings)} vs "
                f"{len(state.chunks)} chunks"
            )
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ValidationError"
            status.exception_message = "Embedding count mismatch"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Embedding count mismatch")

        if any(len(emb) != config.dimension for emb in state.embeddings):
            logger.error(
                f"❌ Embedding dimension mismatch: expected {config.dimension}"
            )
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "DimensionError"
            status.exception_message = f"Dimension mismatch: expected {config.dimension}"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Embedding dimension mismatch")

        status.output_generated = True
        status.output_valid = True

        # ====== METRICS & SUCCESS ======
        elapsed_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        embeddings_per_sec = total_embeddings / (elapsed_time_ms / 1000) if elapsed_time_ms > 0 else 0

        success_msg = (
            f"✅ Embedding complete: {total_embeddings} embeddings "
            f"from {len(state.chunks)} chunks in {elapsed_time_ms:.1f}ms "
            f"({embeddings_per_sec:.1f} embeddings/sec)"
        )

        state.add_message(success_msg)
        logger.info(f"🎉 {success_msg}")

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Use safe enrichment fields
        state.num_embeddings = len(state.embeddings)
        
        if state.embeddings:
            # Store sample embeddings (first 10 dims of first, middle, last)
            state.embedding_samples = [
                state.embeddings[:10] if len(state.embeddings) >= 10 else state.embeddings,
                state.embeddings[len(state.embeddings)//2][:10] if len(state.embeddings[len(state.embeddings)//2]) >= 10 else state.embeddings[len(state.embeddings)//2],
                state.embeddings[-1][:10] if len(state.embeddings[-1]) >= 10 else state.embeddings[-1],
            ]
        else:
            state.embedding_samples = []

        status.status = NodeStatusEnum.COMPLETED
        status.end_time = datetime.utcnow()
        status.execution_time_ms = elapsed_time_ms

        # ====== METRICS (for monitoring) ======
        logger.info(
            f"✅ Embedding COMPLETED | status=COMPLETED | embeddings={len(state.embeddings)} | dim={config.dimension}"
        )

        logger.info(
            f"📊 EMBEDDING METRICS: "
            f"total_embeddings={total_embeddings}, "
            f"provider={config.active_provider}, "
            f"dimension={config.dimension}, "
            f"batch_size={config.batch_size}, "
            f"execution_time_ms={elapsed_time_ms:.1f}, "
            f"embeddings_per_second={embeddings_per_sec:.1f}"
        )

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        return state

    except Exception as e:
        logger.error(f"❌ Embedding node failed: {str(e)}", exc_info=True)

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Safe enrichment fields on error too
        state.num_embeddings = 0
        state.embedding_samples = []

        status.status = NodeStatusEnum.FAILED
        status.exception_type = type(e).__name__
        status.exception_message = str(e)
        status.exception_severity = "CRITICAL"
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (status.end_time - start_time).total_seconds() * 1000

        logger.error(f"❌ Embedding failed: {str(e)} | status=FAILED")

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        state.add_error(f"Embedding error: {str(e)}")

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
