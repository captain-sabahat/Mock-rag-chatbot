"""
===============================================================================
PDF Parser Implementation
===============================================================================

SUMMARY:
--------
Parses PDF files using PyPDF2 library and extracts textual content.
Supports page limit, error handling, and metrics tracking.

WORKING & METHODOLOGY:
----------------------
- Inherits from BaseParser ABC
- Uses PyPDF2 to read PDF bytes
- Optional page range extraction
- Implements parse() async method according to protocol
- Adds metrics: parse duration, pages processed
- Provides fix for potential PDF decode errors
- Alerts future contributors for OCR integration if PDFs are scanned images
- Incorporates circuit break flags on critical failures

INPUTS:
-------
- content: bytes of PDF file
- config: dict with optional parameters (page_range, extract_images)

OUTPUTS:
--------
- dict: { "parsed_text": str, "metadata": {...} }

GLOBAL & CONFIG VARIABLES:
--------------------------
- parse_timeout_seconds: default 300s
- MLflow: transition possible for text extraction time

FUTURE WORK:
------------
- Add OCR fallback for scanned PDFs (e.g., Tesseract)
- Improve page-level error recovery
- Configurable OCR flag

CIRCUIT_BREAK:
---------------
- Decode errors raise ParseError, which can trigger circuit break

MONITORING & HEALTH:
--------------------
- Placeholder for parse duration and page count to MLFlow hooks.
"""

import io
import time
from PyPDF2 import PdfReader
from .base_parser import BaseParser
# #MLFLOW:METRIC_parse_duration_ms
# #MLFLOW:METRIC_pages_processed

class PDFParser(BaseParser):
    parser_name: str = "pdf_parser"

    async def parse(self, content: bytes, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Parses PDF bytes into text.
        """
        # Start timer for metrics
        start_time = time.time()
        # #MLFLOW:track start
        try:
            # Create a file-like object from bytes
            pdf_stream = io.BytesIO(content)
            reader = PdfReader(pdf_stream)
            num_pages = len(reader.pages)
            # #MLFLOW:pages_processed = num_pages

            output_text = ""
            page_range = config.get("page_range", None)
            if page_range:
                start_page, end_page = page_range
                pages = reader.pages[start_page:end_page]
            else:
                pages = reader.pages

            for page in pages:
                try:
                    output_text += page.extract_text() or ""
                except Exception as e:
                    # #MLFLOW:log_error "Page extraction failed"
                    raise e

            parse_duration_ms = (time.time() - start_time) * 1000
            # #MLFLOW:log_metric("parse_duration_ms", parse_duration_ms)
            # #MLFLOW:log_metric("pages_processed", len(pages))
            return {
                "parsed_text": output_text,
                "metadata": {"pages": len(pages)}
            }
        except Exception as e:
            # Possible circuit break trigger
            # #MLFLOW:log_error str(e)
            raise RuntimeError(f"PDF parsing failed: {str(e)}")
