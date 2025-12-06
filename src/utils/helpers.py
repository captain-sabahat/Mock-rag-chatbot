"""
================================================================================
HELPERS - Common Utility Functions
================================================================================

PURPOSE:
--------
Provide common helper utilities:
  - Time measurement
  - Size formatting
  - JSON serialization
  - Retry logic
  - File name sanitization

FUNCTIONS:
  - measure_time: Context manager for measuring execution time
  - format_size: Format bytes to human-readable size
  - format_duration: Format milliseconds to human-readable duration
  - safe_json_dumps: JSON serialization with error handling
  - retry_on_exception: Decorator for retry logic
  - sanitize_filename: Clean up file names

================================================================================
"""

import logging
import json
import re
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional
import asyncio

logger = logging.getLogger(__name__)


@contextmanager
def measure_time(operation_name: str):
    """
    Context manager to measure execution time.
    
    Usage:
        with measure_time("file parsing"):
            # do something
            pass
        # Logs: "✅ file parsing completed in 125.5ms"
    
    Args:
        operation_name: Name of operation being measured
    """
    start_time = time.time()
    logger.debug(f"⏱️  Starting: {operation_name}")
    
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ {operation_name} completed in {duration_ms:.1f}ms")


def format_size(bytes_value: int) -> str:
    """
    Format bytes to human-readable size string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    
    Examples:
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1_048_576)
        '1.00 MB'
        >>> format_size(1_073_741_824)
        '1.00 GB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.2f} PB"


def format_duration(milliseconds: float) -> str:
    """
    Format milliseconds to human-readable duration.
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        Formatted duration string
    
    Examples:
        >>> format_duration(125)
        '125.0ms'
        >>> format_duration(1250)
        '1.25s'
        >>> format_duration(65000)
        '1m 5s'
    """
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    
    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def safe_json_dumps(
    obj: Any,
    default_value: str = '{"error": "serialization failed"}',
    **kwargs
) -> str:
    """
    Safely serialize object to JSON with fallback.
    
    Args:
        obj: Object to serialize
        default_value: Value if serialization fails
        **kwargs: Additional json.dumps arguments
        
    Returns:
        JSON string or default_value on error
    
    Examples:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
        >>> safe_json_dumps({"date": datetime.now()})
        '{"error": "serialization failed"}'
    """
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {str(e)}")
        return default_value


def retry_on_exception(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on exception.
    
    Implements exponential backoff:
      - Attempt 1: immediate
      - Attempt 2: 1s delay
      - Attempt 3: 2s delay
      - Attempt 4: 4s delay
    
    Args:
        max_attempts: Max number of attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
    
    Examples:
        @retry_on_exception(max_attempts=3, delay_seconds=1.0)
        def flaky_function():
            # This might fail, but will retry
            pass
        
        @retry_on_exception(
            max_attempts=5,
            delay_seconds=0.5,
            exceptions=(ConnectionError, TimeoutError)
        )
        def api_call():
            # Retry only on connection/timeout errors
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = delay_seconds * (backoff_multiplier ** (attempt - 1))
                    logger.warning(
                        f"⚠️  {func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {delay:.1f}s... Error: {str(e)}"
                    )
                    time.sleep(delay)
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    
    return decorator


def sanitize_filename(filename: str) -> str:
    """
    Sanitize file name for safe storage.
    
    Removes/replaces:
      - Special characters (keep only alphanumeric, dot, hyphen, underscore)
      - Leading/trailing spaces
      - Multiple spaces
      - Path traversal attempts (../)
    
    Args:
        filename: Original file name
        
    Returns:
        Sanitized file name
    
    Examples:
        >>> sanitize_filename("my document.pdf")
        'my_document.pdf'
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("file@#$%.txt")
        'file.txt'
    """
    # Remove path traversal attempts
    filename = filename.replace("../", "").replace("..\\", "")
    
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Keep only safe characters (alphanumeric, dot, hyphen, underscore)
    filename = re.sub(r"[^\w\.\-]", "", filename)
    
    # Remove leading/trailing dots
    filename = filename.strip(".")
    
    # Remove consecutive underscores
    filename = re.sub(r"_+", "_", filename)
    
    # Minimum length
    if not filename or len(filename) == 0:
        filename = "file"
    
    # Maximum length (keep extension)
    if len(filename) > 200:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        name = name[:180]
        filename = f"{name}.{ext}" if ext else name
    
    return filename


def get_memory_usage() -> dict:
    """
    Get current memory usage info.
    
    Returns:
        Dict with memory stats (requires psutil)
    
    Examples:
        >>> stats = get_memory_usage()
        >>> print(f"Using {stats['percent']}% of memory")
    """
    try:
        import psutil
        process = psutil.Process()
        
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage")
        return {}


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to max length with suffix.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    
    Examples:
        >>> truncate_string("Hello world", 5)
        'He...'
        >>> truncate_string("Hello", 10)
        'Hello'
    """
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(suffix)] + suffix


def chunks(iterable, size: int):
    """
    Split iterable into chunks of given size.
    
    Args:
        iterable: Sequence to split
        size: Chunk size
        
    Yields:
        Chunks of specified size
    
    Examples:
        >>> list(chunks([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], ]
    """
    items = []
    for item in iterable:
        items.append(item)
        if len(items) == size:
            yield items
            items = []
    
    if items:
        yield items
