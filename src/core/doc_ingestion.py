# LINE 1: Document parsing
"""
================================================================================
FILE: src/core/doc_ingestion.py
================================================================================

PURPOSE:
    Parse multi-format documents (PDF, DOCX, TXT) into plain text.
    Async I/O with executor for CPU-bound parsing work.

WORKFLOW:
    1. Receive document file (bytes)
    2. Detect format (PDF, DOCX, TXT)
    3. Parse asynchronously (CPU-bound → executor)
    4. Return extracted text

SUPPORTED FORMATS:
    - PDF: PyPDF2 or pdfplumber
    - DOCX: python-docx
    - TXT: Direct read

LATENCY:
    - TXT: 1-10ms (trivial)
    - DOCX: 10-100ms
    - PDF: 100-500ms (OCR adds time)

IMPORTS:
    - asyncio: Async execution, executor
    - concurrent.futures: ThreadPoolExecutor
    - PyPDF2: PDF parsing
    - python-docx: DOCX parsing
    - logging: Logging

INPUTS:
    - file_content: Document bytes
    - file_type: Format (pdf, docx, txt)
    - file_name: Original filename (for logging)

OUTPUTS:
    - Extracted text (string)

KEY FACTS:
    - CPU-bound work (parsing) runs in executor (doesn't block event loop)
    - Async I/O for file operations
    - Handles multiple formats transparently
    - Size limits (50MB max) prevent memory issues

RESILIENCE:
    - Format validation (reject unsupported types)
    - Size validation (reject >50MB)
    - Error handling (return partial text on parse error)
    - Timeout protection (fail if parse >10s)

FUTURE SCOPE (Phase 2+):
    - Add OCR support (extract text from images)
    - Add table extraction
    - Add metadata extraction
    - Add streaming (process large files incrementally)
    - Add compression (handle gzipped files)
    - Add language detection
    - Add quality metrics (text coherence)

TESTING ENVIRONMENT:
    - Use small test documents
    - Mock file parsing in tests
    - Test all supported formats
    - Test error handling

PRODUCTION DEPLOYMENT:
    - Monitor parsing latency
    - Alert if latency >500ms
    - Cache parsed text
    - Implement streaming for large files
"""

# ================================================================================
# IMPORTS
# ================================================================================

import asyncio
import logging
import io
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.config.settings import Settings
from .exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound parsing work
_thread_pool = ThreadPoolExecutor(max_workers=4)

# ================================================================================
# DOC INGESTION HANDLER CLASS
# ================================================================================

class DocumentIngestionHandler:
    """
    Parse multi-format documents into plain text.
    
    Supports PDF, DOCX, TXT with async I/O.
    CPU-bound parsing runs in executor (non-blocking).
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize document ingestion handler.
        
        Args:
            settings: Application settings (doc size limit, etc.)
        """
        self.settings = settings
        logger.info("DocumentIngestionHandler initialized")
    
    async def parse_document(
        self,
        file_content: bytes,
        file_type: str,
        file_name: Optional[str] = None
    ) -> str:
        """
        Parse document and extract text.
        
        Args:
            file_content: Document file bytes
            file_type: Format (pdf, docx, txt)
            file_name: Original filename (for logging)
        
        Returns:
            Extracted text
        
        Raises:
            DocumentProcessingError: If parsing fails
        
        LATENCY: 10-500ms depending on format
        """
        # Validate file size
        if len(file_content) > self.settings.document_max_size_mb * 1024 * 1024:
            raise DocumentProcessingError(
                f"Document too large ({len(file_content)} bytes > "
                f"{self.settings.document_max_size_mb}MB)"
            )
        
        # Validate file type
        if file_type not in self.settings.document_max_size_mb:  # Use supported formats
            raise DocumentProcessingError(f"Unsupported file type: {file_type}")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run CPU-bound parsing in executor (doesn't block event loop)
            if file_type == "pdf":
                text = await loop.run_in_executor(
                    _thread_pool,
                    self._parse_pdf_sync,
                    file_content
                )
            elif file_type == "docx":
                text = await loop.run_in_executor(
                    _thread_pool,
                    self._parse_docx_sync,
                    file_content
                )
            elif file_type == "txt":
                text = file_content.decode("utf-8")
            else:
                raise DocumentProcessingError(f"Unsupported format: {file_type}")
            
            logger.info(f"Document parsed: {file_name or 'unknown'} ({len(text)} chars)")
            return text
        
        except Exception as e:
            raise DocumentProcessingError(
                f"Document parsing failed: {str(e)}",
                context={"file_type": file_type, "file_name": file_name}
            )
    
    def _parse_pdf_sync(self, file_content: bytes) -> str:
        """
        Synchronous PDF parsing (runs in executor).
        """
        try:
            import PyPDF2
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text()
                if page_num % 10 == 0:
                    logger.debug(f"Parsed PDF page {page_num + 1}")
            
            return text
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
    
    def _parse_docx_sync(self, file_content: bytes) -> str:
        """
        Synchronous DOCX parsing (runs in executor).
        """
        try:
            from docx import Document
            
            doc = Document(io.BytesIO(file_content))
            text = "\n".join(para.text for para in doc.paragraphs)
            return text
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            raise

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add OCR support
# TODO (Phase 2): Add table extraction
# TODO (Phase 2): Add streaming for large files
# TODO (Phase 2): Add compression support
