"""Logging configuration for the project."""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console logging
        file: Enable file logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger("llm_finetuning")
