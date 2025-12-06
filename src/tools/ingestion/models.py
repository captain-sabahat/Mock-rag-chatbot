"""
===============================================================================
Models for Ingestion Pipeline
===============================================================================

SUMMARY:
--------
Defines all Pydantic models used for data exchange within the ingestion pipeline.
Includes request/response schemas, parser configs, metrics, and execution data.

WORKING & METHODOLOGY:
----------------------
- Provides data validation and type safety.
- Structures data flow: input files, parsed content, metrics, errors.
- Facilitates easy extension for additional models, e.g., ML metrics.
- Designed to be integrated with MLFlow or other monitoring tools via
  including metrics and artifacts fields (placeholders).

INPUTS:
-------
- External file inputs (bytes, filename).
- Configurations (dict or class).

OUTPUTS:
--------
- Structured data for pipeline steps.
- Return types for core orchestrator logic.

GLOBAL VARIABLES:
-----------------
- None; all schemas are self-contained.

FUTURE WORK:
------------
- Extend for MLFlow or other ML metrics with active hooks.
- Add storage for logs/artifacts.

CIRCUIT_BREAK:
---------------
- Not directly involved, but models could include status flags.

MONITORING & HEALTH:
--------------------
- Placeholder for metrics collection fields such as `processing_time`, `error_code`.
- To be integrated with monitoring hooks later.

"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class DocumentInput(BaseModel):
    """
    Represents the current processing state for a given document/request.
    Tracks progress, checkpoints, messages, errors, and content data.
    """
    request_id: str = Field(..., description="Unique request identifier.")
    file_name: str = Field(..., description="Original filename.")
    raw_content: bytes = Field(..., description="Raw file bytes.")
    parsed_text: Optional[str] = Field(None, description="Extracted text from file.")
    cleaned_text: Optional[str] = Field(None, description="Preprocessed text.")
    chunks: Optional[List[str]] = Field(None, description="Text chunks for embedding.")
    embeddings: Optional[List[List[float]]] = Field(None, description="Generated vectors.")
    status: str = Field("processing", description="Processing status.")
    progress_percent: int = Field(0, description="Progress % (0-100).")
    checkpoints: Dict[str, Dict[str, object]] = Field(default_factory=dict, description="Stage checkpoints.")
    messages: List[str] = Field(default_factory=list, description="Processing logs/messages.")
    errors: List[str] = Field(default_factory=list, description="Error messages.")
    circuit_breaker_triggered: bool = Field(False, description="Flag for circuit breaker activation.")
    # Additional fields for MLOps metrics (placeholders)
    # #MLFLOW:METRIC:TotalProcessingTimeSeconds

class ProcessingResult(BaseModel):
    """
    Represents the final or interim processing output for a document.
    Used for API response or session storage.
    """
    request_id: str
    status: str
    progress_percent: int
    parsed_text: Optional[str]
    chunks: Optional[List[str]]
    embeddings: Optional[List[List[float]]]
    message: str = "Processing completed successfully."
    errors: List[str] = []

class ExtractedDocument(BaseModel):
    """
    Schema for extracted structured content details.
    """
    content_type: str
    content: Dict[str, object]
    extraction_confidence: Optional[float] = None

# Future schemas for configuration settings
class ParserConfig(BaseModel):
    """
    Configuration schema for individual parser parameters.
    """
    max_pages: Optional[int] = None
    enable_ocr: Optional[bool] = False
    language: Optional[str] = "en"
    # Add more parser-specific settings

class ValidationConfig(BaseModel):
    """
    Validation parameters for the ingestion process.
    """
    max_file_size_bytes: int = Field(50_000_000)
    allowed_types: List[str] = ["pdf", "txt", "json", "md"]
    # Add more validation rules

class PipelineConfig(BaseModel):
    """
    Overall pipeline parameters.
    """
    chunk_size: int = 1000
    chunk_strategy: str = "recursive"  # or token, sentence
    embedding_model: str = "bge"
    vectordb_backend: str = "faiss"
    # Add more pipeline parameters
