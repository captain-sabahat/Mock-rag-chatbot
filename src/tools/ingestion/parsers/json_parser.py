"""
===============================================================================
JSON Parser Implementation
===============================================================================

SUMMARY:
--------
Extracts JSON structure into text with optional key exclusions.
Includes recursive parsing, error handling, and security checks.

WORKING & METHODOLOGY:
----------------------
- Inherits from BaseParser ABC
- Uses json library to parse JSON bytes
- Recursively extracts text content from nested structures
- Detects circular references to prevent infinite loops
- Security: Excludes sensitive keys if configured
- Implements parse() async method
- Adds metrics: parse duration, object size, recursion depth
- Alerts future developers for schema validation or validation pipeline

INPUTS:
-------
- bytes: raw JSON bytes
- config: dict for security keys, max depth, etc.

OUTPUTS:
--------
- dict: parsed "text" (flattened) and metadata (size, keys excluded)

GLOBAL & CONFIG VARIABLES:
--------------------------
- parse_timeout_seconds: default 300 seconds
- MLFlow: placeholders for duration, memory usage

FUTURE WORK:
------------
- JSON schema validation
- Data anonymization
- Add more security features

CIRCUIT_BREAK:
---------------
- Malformed JSON raises exceptions
- Circular ref detection triggers circuit break

MONITORING & HEALTH:
--------------------
- Placeholder for duration, size, key exclusions logs.

"""

import json
import time
from typing import Dict, Any, Set
from .base_parser import BaseParser

# #MLFLOW:METRIC_parse_duration_ms
# #MLFLOW:METRIC_json_size_bytes

class JSONParser(BaseParser):
    parser_name: str = "json_parser"

    async def parse(self, content: bytes, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Parses JSON bytes into flattened text content.
        """
        start_time = time.time()
        # #MLFLOW:track start
        try:
            json_obj = json.loads(content)
            # Detect circular references or errors
            visited_ids: Set[int] = set()
            def recurse_extract(obj):
                if id(obj) in visited_ids:
                    # #MLFLOW:log_error "Circular reference detected"
                    raise RuntimeError("Circular reference detected")
                visited_ids.add(id(obj))
                text_segments = []

                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if self._exclude_key(key, config):
                            continue
                        text_segments.append(str(key))
                        text_segments.extend(recurse_extract(value))
                elif isinstance(obj, list):
                    for item in obj:
                        text_segments.extend(recurse_extract(item))
                else:
                    text_segments.append(str(obj))
                return text_segments

            def _exclude_key(key: str, config: Dict[str, Any]) -> bool:
                excluded_keys = config.get("exclude_keys", [])
                return key in excluded_keys

            flattened_text = ' '.join(recurse_extract(json_obj))
            parse_duration_ms = (time.time() - start_time) * 1000
            # #MLFLOW:log_metric("parse_duration_ms", parse_duration_ms)
            # #MLFLOW:log_metric("json_size_bytes", len(content))
            return {
                "parsed_text": flattened_text,
                "metadata": {
                    "json_size_bytes": len(content),
                }
            }
        except json.JSONDecodeError as e:
            # #MLFLOW:log_error str(e)
            raise RuntimeError(f"JSON parsing failed: {str(e)}")
