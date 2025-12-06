"""
===============================================================================
Plain Text Parser Implementation
===============================================================================

SUMMARY:
--------
Handles plain text files, normalizes encoding, whitespace, and filtering.
Simple but effective for .txt inputs.

WORKING & METHODOLOGY:
----------------------
- Inherits from BaseParser ABC
- Reads bytes, decodes with fallback encodings
- Normalizes whitespace, removes extra newlines
- Supports optional encoding detection
- Implements parse() async method
- Collects metrics: parse time, size
- Placeholder for future language detection or correction modules.
- Designed keeping circuit break points in mind for decoding errors.

INPUTS:
-------
- bytes: raw file bytes
- config: optional dict for encoding and normalization options

OUTPUTS:
--------
- dict containing `"parsed_text"` (str) and `"metadata"` (dict with sizes)

GLOBAL & CONFIG VARIABLES:
--------------------------
- parse_timeout_seconds: default 300 seconds
- MLflow: placeholders for duration and size

FUTURE WORK:
------------
- Language detection
- Spell correction
- Enable multi-encoding support
- Integrate MLFlow metrics later

CIRCUIT_BREAK:
---------------
- Decoding errors can raise exceptions sparking circuit break

MONITORING & HEALTH:
--------------------
- Duration and size logging for MLFlow later.
"""

from typing import Dict, Any
import chardet
import time
from .base_parser import BaseParser

# #MLFLOW:METRIC_parse_duration_ms
# #MLFLOW:METRIC_input_size_bytes

class TextParser(BaseParser):
    parser_name: str = "text_parser"

    async def parse(self, content: bytes, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Parses plain text bytes, normalizes content.
        """
        start_time = time.time()
        # #MLFLOW:track start
        try:
            # Detect encoding if not specified
            # #MLFLOW:log_placeholder for encoding detection if needed
            detected_encoding = chardet.detect(content)['encoding']
            decoded_text = content.decode(detected_encoding or 'utf-8', errors='replace')
            # Normalize whitespace
            normalized_text = ' '.join(decoded_text.split())
            parse_time_ms = (time.time() - start_time) * 1000
            # #MLFLOW:log_metric("parse_duration_ms", parse_time_ms)
            # #MLFLOW:log_metric("input_size_bytes", len(content))
            return {
                "parsed_text": normalized_text,
                "metadata": {
                    "decoded_encoding": detected_encoding,
                    "original_size_bytes": len(content),
                }
            }
        except Exception as e:
            # #MLFLOW:log_error str(e)
            raise RuntimeError(f"Text parsing failed: {str(e)}")
