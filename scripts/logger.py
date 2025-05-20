#!/usr/bin/env python
"""
Logging setup for the T2 Factor Clusters project.
"""
import os
import sys
import yaml
from pathlib import Path
from loguru import logger

def setup_logger(config_path="config.yml"):
    """
    Set up the logger based on configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        log_level = config.get('log_level', 'INFO')
        log_rotation = config.get('log_rotation', '1 day')
    except Exception as e:
        print(f"Error loading config: {e}")
        log_level = 'INFO'
        log_rotation = '1 day'
    
    # Configure logger
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler with rotation
    logger.add(
        "logs/T2_processing.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=log_rotation,
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Logger initialized")
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger setup test successful")