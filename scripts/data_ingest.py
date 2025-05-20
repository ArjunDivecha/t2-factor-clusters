#!/usr/bin/env python
"""
Data ingestion script to convert Excel to Parquet format.
"""
import os
import yaml
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset, init_catalog

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def excel_to_parquet(config):
    """
    Convert the Excel file to Parquet format.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the created Parquet file
    """
    logger = setup_logger()
    logger.info("Starting Excel to Parquet conversion")
    
    input_file = config['input_file']
    input_sheet = config['input_sheet']
    output_path = config['output_paths']['raw']
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Reading Excel file: {input_file}, sheet: {input_sheet}")
        df = pd.read_excel(input_file, sheet_name=input_sheet)
        
        # Basic data validation
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Convert to parquet
        logger.info(f"Writing to Parquet: {output_path}")
        df.to_parquet(output_path, index=False)
        
        # Register in catalog
        init_catalog()
        register_asset(output_path, metadata={"source_file": input_file, "sheet": input_sheet})
        
        logger.info("Excel to Parquet conversion complete")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting Excel to Parquet: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    excel_to_parquet(config)