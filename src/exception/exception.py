import sys
import traceback
import datetime
from typing import Optional, Any, Dict
from src.logging import logger

class InsuranceAgentException(Exception):
    """
    Custom exception class for the Insurance Agent RAG Chatbot.
    Captures detailed error context for production-grade debugging.
    """
    def __init__(
        self,
        error_message: str,
        error_details: sys,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_message)
        self.error_message = error_message
        self.context = context or {}
        self.timestamp = datetime.datetime.utcnow().isoformat()
        exc_type, exc_obj, exc_tb = error_details.exc_info()
        self.exc_type = exc_type.__name__ if exc_type else "Unknown"
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None
        self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_obj, exc_tb)) if exc_tb else ""
        self.log_exception()

    def log_exception(self):
        logger.logging.error(
            f"[{self.timestamp}] Exception occurred: {self.exc_type} in {self.file_name} at line {self.lineno}\n"
            f"Message: {self.error_message}\n"
            f"Context: {self.context}\n"
            f"Traceback:\n{self.traceback_str}"
        )

    def __str__(self):
        return (
            f"[{self.timestamp}] Error in [{self.file_name}] at line [{self.lineno}]: "
            f"{self.error_message} | Type: {self.exc_type} | Context: {self.context}"
        )

if __name__ == "__main__":
    try:
        logger.logging.info("Enter the try block")
        a = 1 / 0
        print("This will not be printed", a)
    except Exception as e:
        raise InsuranceAgentException(
            "Division by zero in main block",
            sys,
            context={"operation": "1/0", "user": "test_user"}
        )