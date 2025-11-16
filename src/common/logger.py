import logging
import os
import sys
from datetime import datetime

class FileLogger:
    """Handles logging to files with timestamped names."""
    
    @staticmethod
    def setup(module_name: str, log_dir: str = "logs") -> logging.Logger:
        """Create logger that writes to timestamped log file. Args: module_name: Name of the module (e.g., 'train_st53'), log_dir: Directory to store log files. Returns: Configured logger instance."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"{log_dir}/{module_name}_{timestamp}.log"
        
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # File handler (UTF-8 encoding to support emoji characters like ✅, ⚠️, ❌)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Redirect stdout and stderr to log file as well
        sys.stdout = FileLogger.TeeOutput(sys.stdout, log_file)
        sys.stderr = FileLogger.TeeOutput(sys.stderr, log_file)
        
        return logger
    
    class TeeOutput:
        """Duplicates output to both console and log file."""
        def __init__(self, original_stream, log_file):
            self.original_stream = original_stream
            self.log_file = log_file
        
        def write(self, message):
            self.original_stream.write(message)
            if message.strip():  # Only write non-empty messages
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message)
        
        def flush(self):
            self.original_stream.flush()
