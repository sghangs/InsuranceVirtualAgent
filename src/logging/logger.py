import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def get_log_level():
    return os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y_%m_%d')}.log")

    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(get_log_level())
    logger.handlers.clear()

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s"
    ))
    console_handler.setLevel(get_log_level())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Example: send critical errors to alerting system (placeholder)
    class AlertHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.CRITICAL:
                # Integrate with alerting/monitoring system here
                pass

    logger.addHandler(AlertHandler())

    return logger

logging = setup_logger()

# Usage example:
if __name__ == "__main__":
    try:
        logging.info("Logger initialized for RAG chatbot.")
        # Simulate error
        1 / 0
    except Exception:
        logging.exception("Unhandled exception occurred")