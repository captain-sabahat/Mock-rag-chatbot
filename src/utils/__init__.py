"""
================================================================================
UTILS PACKAGE - Utility Functions and Helpers
================================================================================

PURPOSE:
--------
Provide utility functions for:
  - File operations
  - Validation
  - Formatting
  - Helpers

Modules:
  - file_validator: File validation (size, type, content)
  - helpers: Common helper functions (logging, timing, formatting)

USAGE:
------
    from src.utils import validate_file, measure_time, format_size
    
    # Validate file
    is_valid = validate_file("document.pdf", max_size=10_000_000)
    
    # Measure execution time
    with measure_time("operation"):
        # Do something
        pass
    
    # Format size
    size_str = format_size(1_000_000)  # "976.56 KB"

================================================================================
"""

from .file_validator import (
    validate_file,
    validate_file_type,
    validate_file_size,
    get_file_type,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
)

from .helpers import (
    measure_time,
    format_size,
    format_duration,
    safe_json_dumps,
    retry_on_exception,
    sanitize_filename,
)

__all__ = [
    # file_validator
    "validate_file",
    "validate_file_type",
    "validate_file_size",
    "get_file_type",
    "ALLOWED_EXTENSIONS",
    "MAX_FILE_SIZE",
    # helpers
    "measure_time",
    "format_size",
    "format_duration",
    "safe_json_dumps",
    "retry_on_exception",
    "sanitize_filename",
]
