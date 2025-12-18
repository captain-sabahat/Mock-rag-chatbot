"""
================================================================================
VECTORDB NODE - Store Vectors (v2.5 FIXED + DIMENSION BUG FIX)
================================================================================

v2.5 FIXES:
✅ Removed: state.vectordb_status_flag = "COMPLETED" (Pydantic error)
✅ Removed: state.vectordb_error, state.vectordb_error_type (Pydantic errors)
✅ Added: Safe enrichment fields (vectordb_batches_total, vectordb_batches_done)
✅ Preserved: All vectordb logic, config, batch upload, validation, monitoring

DIMENSION BUG FIX:
✅ OLD BROKEN: if config.dimension != len(state.embeddings) # Wrong!
✅ NEW FIXED: compare config.dimension with embedding vector dimension (len(first_vector))

KEY PATTERN:
- OLD (BROKEN): state.vectordb_status_flag = "COMPLETED" ❌
- NEW (FIXED): state.vectordb_batches_total = upload_result.batch_count ✅

================================================================================
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline.schemas import PipelineState, NodeStatus, NodeStatusEnum
from src.core.exceptions import ValidationError
from src.tools.vectordb.vectordb_registry import (
    load_vectordb_config,
    VectorDBFactory,
    VectorStoreResult,
)

logger = logging.getLogger(__name__)


def _print_first_vector_top5(embeddings: List[List[float]]) -> None:
    """
    Print top 5 values of the FIRST embedding vector.
    Uses print() as requested (in addition to logging).
    Safe no-op if embeddings are missing/invalid.
    """
    try:
        if not embeddings or not embeddings[0]:
            print("[vectordb_node] first_vector_top5: <no embeddings>")
            return

        # "top 5 values" interpreted as first 5 values (common debug view)
        first_vec = embeddings[0]
        top5 = first_vec[:5]
        print(f"[vectordb_node] first_vector_top5 (first 5 vals): {top5}")
    except Exception as e:
        # Do not fail the pipeline for debug print
        print(f"[vectordb_node] first_vector_top5: <print failed: {e}>")


async def vectordb_node(state: PipelineState) -> PipelineState:
    """
    Store embeddings in vector database (config-driven backend selection).

    Flow:
    1. Load VectorDB config from YAML
    2. A Method: Validate input (embeddings, chunks, metadata)
    3. B Method: Create backend instance + batch upload with progress
    4. C Method: Validate all vectors stored
    5. Write monitoring file with upload stats
    """

    node_name = "vectordb"
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
        # ====== LOAD CONFIG ======
        logger.info("📋 Loading vector DB configuration...")

        try:
            config = load_vectordb_config("config/defaults/vectordb.yaml")
        except Exception as config_error:
            logger.error(f"❌ Config load failed: {str(config_error)}")
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ConfigError"
            status.exception_message = str(config_error)
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (
                status.end_time - start_time
            ).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise

        logger.info(
            f"✅ Config loaded: backend={config.backend}, "
            f"dimension={config.dimension}, batch_size={config.batch_size}"
        )

        # ====== A METHOD: INPUT VALIDATION ======
        logger.info("🔍 A Method: Validating input...")

        if not state.embeddings:
            logger.error("❌ No embeddings to store")
            status.input_received = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "MissingInputError"
            status.exception_message = "No embeddings provided"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (
                status.end_time - start_time
            ).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No embeddings to store")

        status.input_received = True
        logger.info(f"✅ Received: {len(state.embeddings)} embeddings")

        # Debug print: first vector top 5
        _print_first_vector_top5(state.embeddings)

        # ====== DIMENSION VALIDATION (FIX: Check embedding vector dimension, not count) ======
        if state.embeddings:
            # FIX: dimension is based on the vector length, not number of vectors
            actual_dim = len(state.embeddings[0]) if state.embeddings[0] else 0
            logger.info(
                f"🔍 Checking embedding dimensions: config={config.dimension}, actual={actual_dim}"
            )

            if actual_dim != config.dimension:
                logger.error(
                    f"❌ Dimension mismatch: config={config.dimension}, "
                    f"actual_embedding_dim={actual_dim}"
                )
                status.input_valid = False
                status.status = NodeStatusEnum.FAILED
                status.exception_type = "DimensionError"
                status.exception_message = (
                    f"Embedding dimension mismatch: config={config.dimension}, "
                    f"actual={actual_dim}"
                )
                status.exception_severity = "CRITICAL"
                status.end_time = datetime.utcnow()
                status.execution_time_ms = (
                    status.end_time - start_time
                ).total_seconds() * 1000
                await _write_node_monitoring(state.request_id, status)
                raise ValidationError("Embedding dimension mismatch")

        if len(state.embeddings) != len(state.chunks):
            logger.error("❌ Embedding/chunk count mismatch")
            status.input_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "ValidationError"
            status.exception_message = (
                f"Embedding/chunk mismatch: {len(state.embeddings)} vs {len(state.chunks)}"
            )
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (
                status.end_time - start_time
            ).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("Embedding/chunk count mismatch")

        status.input_valid = True
        logger.info(
            f"✅ Input validated: {len(state.chunks)} chunks, dimension={config.dimension}"
        )

        # ====== B METHOD: EXECUTE BATCH UPLOAD ======
        logger.info(f"🚀 B Method: Creating {config.backend} vector DB...")

        # Factory creates correct backend (config-driven switching)
        vectordb = VectorDBFactory.create(config)

        logger.info(f"📦 Starting batch upload to {config.backend}...")

        # Generate IDs for vectors (chunk_index-based)
        vector_ids = [
            f"{state.request_id}_chunk_{i}" for i in range(len(state.chunks))
        ]

        # Metadata: chunk text, index, request ID, etc.
        metadata = []
        for i, chunk in enumerate(state.chunks):
            metadata.append(
                {
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "request_id": state.request_id,
                    "file_name": getattr(state, "file_name", "unknown"),
                    "embedding_model": getattr(state, "embedding_model", "huggingface"),
                    "stored_at": datetime.utcnow().isoformat(),
                }
            )

        # Define progress callback
        upload_stats = {
            "batches_processed": 0,
            "vectors_stored": 0,
            "batches_total": 0,
        }

        async def progress_callback(
            batch_num: int, total_batches: int, cumulative: int, batch_stored: int
        ):
            """Called after each batch completes."""
            upload_stats["batches_processed"] = batch_num
            upload_stats["batches_total"] = total_batches
            upload_stats["vectors_stored"] = cumulative
            logger.info(
                f"📊 Upload progress: Batch {batch_num}/{total_batches}, "
                f"Total: {cumulative}/{len(state.embeddings)} vectors ✅"
            )

        # Execute batch upload
        upload_result = await vectordb.batch_upload(
            vectors=state.embeddings,
            ids=vector_ids,
            metadata=metadata,
            progress_callback=progress_callback,
        )

        logger.info(
            f"✅ Batch upload complete: {upload_result.total_stored}/"
            f"{upload_result.total_attempted} vectors in "
            f"{upload_result.storage_time_ms:.1f}ms "
            f"({upload_result.vectors_per_second:.1f} vec/sec)"
        )

        # ====== C METHOD: OUTPUT VALIDATION ======
        logger.info("✅ C Method: Validating output...")

        if upload_result.total_stored == 0:
            logger.error("❌ No vectors were stored")
            status.output_valid = False
            status.status = NodeStatusEnum.FAILED
            status.exception_type = "StorageError"
            status.exception_message = "No vectors stored"
            status.exception_severity = "CRITICAL"
            status.end_time = datetime.utcnow()
            status.execution_time_ms = (
                status.end_time - start_time
            ).total_seconds() * 1000
            await _write_node_monitoring(state.request_id, status)
            raise ValidationError("No vectors stored")

        if upload_result.total_failed > 0:
            logger.warning(
                f"⚠️ Some vectors failed to store: " f"{upload_result.total_failed} failures"
            )
            for error in upload_result.errors[:5]:  # Log first 5 errors
                logger.warning(f" - {error}")

            if upload_result.total_stored == 0:
                status.output_valid = False
                status.status = NodeStatusEnum.FAILED
                status.exception_type = "PartialStorageError"
                status.exception_message = f"{upload_result.total_failed} vectors failed"
                status.exception_severity = "CRITICAL"
                raise ValidationError(f"All {upload_result.total_failed} vectors failed")

        status.output_generated = True
        status.output_valid = True

        # ====== SUCCESS ======
        success_msg = (
            f"✅ VectorDB complete: {upload_result.total_stored}/{upload_result.total_attempted} "
            f"vectors stored in {config.backend} in {upload_result.storage_time_ms:.1f}ms "
            f"({upload_result.vectors_per_second:.1f} vectors/sec)"
        )

        state.add_message(success_msg)
        logger.info(f"🎉 {success_msg}")

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Use safe enrichment fields
        state.vectordb_batches_total = upload_result.batch_count
        state.vectordb_batches_done = upload_result.batch_count
        state.vectordb_upsert_count = upload_result.total_stored

        # Note: vectordb_collection_name and vectordb_vector_dimension
        # are already set in schema defaults if needed

        status.status = NodeStatusEnum.COMPLETED
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (
            status.end_time - start_time
        ).total_seconds() * 1000

        # ====== METRICS (for monitoring) ======
        logger.info(
            f"✅ Vectordb COMPLETED | status=COMPLETED | batches={upload_result.batch_count} | total={state.vectordb_upsert_count}"
        )

        logger.info(
            f"📊 VectorDB stats: backend={upload_result.backend}, "
            f"total_vectors={upload_result.total_stored}, "
            f"batch_count={upload_result.batch_count}, "
            f"storage_time_ms={upload_result.storage_time_ms:.1f}, "
            f"vectors_per_second={upload_result.vectors_per_second:.1f}, "
            f"failed_vectors={upload_result.total_failed}"
        )

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status

        return state

    except Exception as e:
        logger.error(f"❌ VectorDB node failed: {str(e)}", exc_info=True)

        # ====== v2.5 FIX: Use ENRICHMENT fields instead of status flags ======
        # ✅ NEW (v2.5): Safe enrichment fields on error too
        state.vectordb_batches_total = 0
        state.vectordb_batches_done = 0
        state.vectordb_upsert_count = 0

        status.status = NodeStatusEnum.FAILED
        status.exception_type = type(e).__name__
        status.exception_message = str(e)
        status.exception_severity = "CRITICAL"
        status.end_time = datetime.utcnow()
        status.execution_time_ms = (
            status.end_time - start_time
        ).total_seconds() * 1000

        logger.error(f"❌ Vectordb failed: {str(e)} | status=FAILED")

        await _write_node_monitoring(state.request_id, status)
        state.node_statuses[node_name] = status
        state.add_error(f"VectorDB error: {str(e)}")

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
