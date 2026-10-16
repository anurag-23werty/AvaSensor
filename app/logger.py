import os
import sys
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import List, Dict
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "aeroshield.log")
LOG_BUFFER_MAX = 500
in_memory_log_buffer: deque = deque(maxlen=LOG_BUFFER_MAX)

class MemoryLogHandler(logging.Handler):
    """Stores structured log records in memory for real-time frontend/API streaming."""
    def emit(self, record):
        try:
            if self.formatter:
                formatted_time = self.formatter.formatTime(record, "%Y-%m-%d %H:%M:%S")
                exc_str = self.formatter.formatException(record.exc_info) if record.exc_info else None
            else:
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
                exc_str = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else None

            log_entry = {
                "timestamp": formatted_time,
                "level": record.levelname,
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
                "message": record.getMessage(),
                "exc_info": exc_str
            }
            in_memory_log_buffer.append(log_entry)
        except Exception:
            self.handleError(record)

def setup_root_logger():
    """Configures the root application logger."""
    root_logger = logging.getLogger("aeroshield")
    
    if root_logger.hasHandlers():
        return root_logger

    root_logger.setLevel(logging.INFO)

    log_format = "[%(asctime)s] [%(levelname)-8s] [%(filename)s:%(lineno)d in %(funcName)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # 1. Console Output Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler (Max 10MB per file, 5 backup archives)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. In-Memory Ring Buffer Handler (for Web UI & REST API)
    mem_handler = MemoryLogHandler()
    mem_handler.setLevel(logging.INFO)
    mem_handler.setFormatter(formatter)
    root_logger.addHandler(mem_handler)

    root_logger.info(f"Production Logger initialized. Log file path: {LOG_FILE}")
    return root_logger

# Initialize on import
_logger = setup_root_logger()

def get_logger(module_name: str) -> logging.Logger:
    """Returns child logger inheriting root configuration."""
    return logging.getLogger(f"aeroshield.{module_name}")

def get_recent_logs(limit: int = 100) -> List[Dict]:
    """Returns recent log entries from the in-memory ring buffer."""
    items = list(in_memory_log_buffer)
    return items[-limit:]
