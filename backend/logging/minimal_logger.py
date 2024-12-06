import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

class MinimalLogger:
    """
    Minimal logging infrastructure for MVP
    
    Key Features:
    - Simple file-based logging
    - Basic log rotation
    - Minimal configuration
    """
    
    def __init__(
        self, 
        log_dir: Optional[str] = None, 
        log_level: int = logging.INFO
    ):
        """
        Initialize logger
        
        Args:
            log_dir (Optional[str]): Directory to store log files
            log_level (int): Logging level
        """
        # Create log directory if not exists
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure main logger
        self.logger = logging.getLogger('DAICN_MVP')
        self.logger.setLevel(log_level)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'daicn_mvp_{timestamp}.log')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log informational message
        
        Args:
            message (str): Log message
            context (Optional[Dict]): Additional context information
        """
        log_message = message
        if context:
            log_message += f" | Context: {context}"
        self.logger.info(log_message)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log warning message
        
        Args:
            message (str): Log message
            context (Optional[Dict]): Additional context information
        """
        log_message = message
        if context:
            log_message += f" | Context: {context}"
        self.logger.warning(log_message)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log error message
        
        Args:
            message (str): Log message
            context (Optional[Dict]): Additional context information
        """
        log_message = message
        if context:
            log_message += f" | Context: {context}"
        self.logger.error(log_message)
    
    def log_system_event(
        self, 
        event_type: str, 
        details: Dict[str, Any]
    ):
        """
        Log a system-wide event
        
        Args:
            event_type (str): Type of system event
            details (Dict): Event details
        """
        log_message = f"SYSTEM EVENT: {event_type}"
        self.logger.info(f"{log_message} | Details: {details}")

def main():
    """
    Demonstration of Minimal Logger
    """
    logger = MinimalLogger()
    
    # Log different types of messages
    logger.info("System startup", {"version": "0.1.0", "mode": "MVP"})
    logger.warning("Low resource warning", {"cpu_usage": 85, "memory_usage": 75})
    logger.error("Authentication failure", {"username": "test_user", "ip": "192.168.1.100"})
    
    # Log a system event
    logger.log_system_event("USER_REGISTRATION", {
        "username": "new_user",
        "registration_method": "basic"
    })

if __name__ == '__main__':
    main()
