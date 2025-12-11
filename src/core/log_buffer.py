# ============================================================================
# Log Buffer - In-Memory + JSON Persistence for Real-Time Log Display
# ============================================================================

"""
Custom logging handler that:
1. Captures logs to in-memory buffer (does NOT affect console output)
2. Writes logs to JSON file for frontend access
3. Maintains separate buffers per log level
4. Circular buffer (keeps last N logs to prevent memory bloat)
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Dict, Any
from threading import Lock

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_BUFFER_SIZE = 1000  # Keep last 1000 logs in memory
LOG_FILE_PATH = Path("./data/monitoring/pipeline_logs.json")

# ============================================================================
# LOG BUFFER CLASS
# ============================================================================

class LogBuffer:
    """In-memory circular buffer for logs with JSON persistence."""
    
    def __init__(self, max_size: int = MAX_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = Lock()
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Create monitoring directory if needed."""
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    def add_log(self, log_record: logging.LogRecord):
        """Add log to buffer and persist to JSON."""
        with self.lock:
            # Create structured log entry
            log_entry = {
                "timestamp": datetime.fromtimestamp(log_record.created).isoformat() + "Z",
                "level": log_record.levelname,
                "logger": log_record.name,
                "message": log_record.getMessage(),
                "module": log_record.module,
                "function": log_record.funcName,
                "line": log_record.lineno,
            }
            
            # Add to in-memory buffer
            self.buffer.append(log_entry)
            
            # Write to JSON file (append mode)
            self._write_to_json(log_entry)
    
    def _write_to_json(self, log_entry: Dict[str, Any]):
        """Append single log entry to JSON file."""
        try:
            # Read existing logs
            logs = []
            if LOG_FILE_PATH.exists():
                with open(LOG_FILE_PATH, "r") as f:
                    data = json.load(f)
                    logs = data.get("logs", [])
            
            # Append new log
            logs.append(log_entry)
            
            # Keep only last MAX_BUFFER_SIZE logs
            if len(logs) > self.max_size:
                logs = logs[-self.max_size:]
            
            # Write back
            with open(LOG_FILE_PATH, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat() + "Z",
                        "total_logs": len(logs),
                        "logs": logs,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            # Silently fail to not break logging
            pass
    
    def get_logs(
        self,
        level: str = None,
        search: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get filtered logs from buffer."""
        with self.lock:
            logs = list(self.buffer)
        
        # Filter by level
        if level:
            logs = [l for l in logs if l["level"] == level.upper()]
        
        # Filter by search term
        if search:
            search_lower = search.lower()
            logs = [
                l
                for l in logs
                if search_lower in l["message"].lower()
                or search_lower in l["logger"].lower()
            ]
        
        # Return last N logs
        return logs[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logs in buffer."""
        with self.lock:
            logs = list(self.buffer)
        
        summary = {
            "total": len(logs),
            "by_level": {},
            "by_logger": {},
        }
        
        for log in logs:
            # Count by level
            level = log["level"]
            summary["by_level"][level] = summary["by_level"].get(level, 0) + 1
            
            # Count by logger
            logger = log["logger"]
            summary["by_logger"][logger] = summary["by_logger"].get(logger, 0) + 1
        
        return summary
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

# ============================================================================
# CUSTOM LOGGING HANDLER
# ============================================================================

class BufferHandler(logging.Handler):
    """Custom logging handler that writes to LogBuffer."""
    
    def __init__(self, log_buffer: LogBuffer):
        super().__init__()
        self.log_buffer = log_buffer
    
    def emit(self, record: logging.LogRecord):
        """Handle log record."""
        try:
            self.log_buffer.add_log(record)
        except Exception:
            # Never raise exceptions in logging handlers
            pass

# ============================================================================
# GLOBAL LOG BUFFER INSTANCE
# ============================================================================

_log_buffer = LogBuffer(max_size=MAX_BUFFER_SIZE)

def get_log_buffer() -> LogBuffer:
    """Get global log buffer instance."""
    return _log_buffer

def setup_log_buffer(logger: logging.Logger):
    """Attach buffer handler to logger (call once per logger)."""
    handler = BufferHandler(_log_buffer)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
