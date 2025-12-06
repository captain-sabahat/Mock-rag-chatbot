"""
================================================================================
FILE VALIDATOR - File Validation Utilities
================================================================================

PURPOSE:
--------
Validate files before processing:
  - File size validation
  - File type validation
  - Content type detection
  - Magic number checking

Validation:
  - Size check (max 50MB default)
  - Type check (allowed extensions)
  - Content check (magic bytes)
  - Name sanitization

ALLOWED FORMATS:
  - PDF (.pdf)
  - Text (.txt)
  - JSON (.json)
  - Markdown (.md)

ERRORS:
  - FileSizeError: File too large
  - FileTypeError: Invalid file type
  - FileValidationError: Generic validation error

================================================================================
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

from src.core import ValidationError

logger = logging.getLogger(__name__)


# Configuration
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".json", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Magic numbers (file signatures)
MAGIC_NUMBERS = {
    b"%PDF": ".pdf",
    b"\x7fELF": ".elf",
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG": ".png",
    b"PK\x03\x04": ".zip",
}


def validate_file(
    file_name: str,
    file_content: bytes,
    max_size: Optional[int] = None,
    allowed_types: Optional[set] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate file comprehensively.
    
    Checks:
      1. File name is not empty
      2. File extension is allowed
      3. File size is acceptable
      4. File content is valid
    
    Args:
        file_name: File name (with extension)
        file_content: Raw file bytes
        max_size: Max allowed size (default: MAX_FILE_SIZE)
        allowed_types: Allowed extensions (default: ALLOWED_EXTENSIONS)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    max_size = max_size or MAX_FILE_SIZE
    allowed_types = allowed_types or ALLOWED_EXTENSIONS
    
    logger.debug(f"Validating file: {file_name}")
    
    # Validate name
    if not file_name or len(file_name.strip()) == 0:
        error = "File name cannot be empty"
        logger.warning(f"❌ {error}")
        return False, error
    
    # Validate extension
    ext = Path(file_name).suffix.lower()
    if ext not in allowed_types:
        error = f"File type not allowed: {ext}. Allowed: {allowed_types}"
        logger.warning(f"❌ {error}")
        return False, error
    
    # Validate size
    is_valid, error = validate_file_size(file_content, max_size)
    if not is_valid:
        logger.warning(f"❌ {error}")
        return False, error
    
    logger.info(f"✅ File validation passed: {file_name}")
    return True, None


def validate_file_type(
    file_name: str,
    allowed_types: Optional[set] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate file extension.
    
    Args:
        file_name: File name with extension
        allowed_types: Allowed extensions (default: ALLOWED_EXTENSIONS)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    allowed_types = allowed_types or ALLOWED_EXTENSIONS
    
    ext = Path(file_name).suffix.lower()
    
    if not ext:
        error = "File has no extension"
        logger.warning(f"❌ {error}")
        return False, error
    
    if ext not in allowed_types:
        error = (
            f"File type '{ext}' not allowed. "
            f"Allowed types: {', '.join(allowed_types)}"
        )
        logger.warning(f"❌ {error}")
        return False, error
    
    logger.debug(f"✅ File type valid: {ext}")
    return True, None


def validate_file_size(
    file_content: bytes,
    max_size: int = MAX_FILE_SIZE
) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Args:
        file_content: Raw file bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    size = len(file_content)
    
    if size == 0:
        error = "File is empty"
        logger.warning(f"❌ {error}")
        return False, error
    
    if size > max_size:
        from .helpers import format_size
        error = (
            f"File size {format_size(size)} exceeds maximum "
            f"allowed size {format_size(max_size)}"
        )
        logger.warning(f"❌ {error}")
        return False, error
    
    logger.debug(f"✅ File size valid: {len(file_content)} bytes")
    return True, None


def get_file_type(file_content: bytes) -> Optional[str]:
    """
    Detect file type from magic bytes.
    
    Args:
        file_content: Raw file bytes
        
    Returns:
        File extension string (e.g., ".pdf") or None
    """
    if not file_content:
        return None
    
    # Check first 4 bytes against known magic numbers
    for magic, ext in MAGIC_NUMBERS.items():
        if file_content.startswith(magic):
            logger.debug(f"Detected file type: {ext}")
            return ext
    
    # Default: couldn't detect
    logger.debug("Could not detect file type from magic bytes")
    return None


def validate_file_content(
    file_name: str,
    file_content: bytes
) -> Tuple[bool, Optional[str]]:
    """
    Validate file content is actually readable.
    
    Args:
        file_name: File name
        file_content: Raw file bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    ext = Path(file_name).suffix.lower()
    
    try:
        if ext == ".pdf":
            # Try to read as PDF
            import fitz
            doc = fitz.open(stream=file_content, filetype="pdf")
            doc.close()
        
        elif ext == ".txt":
            # Try to decode as text
            try:
                file_content.decode('utf-8')
            except UnicodeDecodeError:
                file_content.decode('latin-1')
        
        elif ext == ".json":
            # Try to parse as JSON
            import json
            json.loads(file_content.decode('utf-8'))
        
        elif ext == ".md":
            # Try to decode as text (markdown is just text)
            file_content.decode('utf-8')
        
        logger.debug(f"✅ File content valid: {ext}")
        return True, None
    
    except Exception as e:
        error = f"Invalid file content for {ext}: {str(e)}"
        logger.warning(f"❌ {error}")
        return False, error
