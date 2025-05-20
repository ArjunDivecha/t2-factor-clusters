#!/usr/bin/env python
"""
Data cleaning script for T2 Factor Clusters project.
"""
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_factor_data(config):
    """
    Clean the factor returns data.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the cleaned Parquet file
    """
    logger = setup_logger()
    logger.info("Starting data cleaning process")
    
    input_path = config['output_paths']['raw']
    output_path = config['output_paths']['cleaned']
    winsorize_threshold = config.get('winsorize_threshold', 0.5)
    interpolation_method = config.get('interpolation_method', 'linear')
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Reading Parquet file: {input_path}")
        df = pd.read_parquet(input_path)
        
        # Assuming the first column is Date
        logger.info("Converting dates to datetime format")
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set date as index
        df = df.set_index(date_col)
        
        # Generate data availability summary
        logger.info("Generating data availability summary")
        availability = pd.DataFrame({
            'first_date': df.apply(lambda x: x.first_valid_index()),
            'last_date': df.apply(lambda x: x.last_valid_index()),
            'percent_missing': df.isnull().mean() * 100,
            'count_missing': df.isnull().sum()
        })
        
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save availability summary
        availability.to_csv("reports/data_availability.csv")
        
        # Handle missing values
        logger.info("Handling missing values")
        # For isolated gaps (if rare), use the specified interpolation method
        df_cleaned = df.copy()
        
        # Only interpolate series with less than 5% missing values
        columns_to_interpolate = availability[availability['percent_missing'] < 5].index.tolist()
        logger.info(f"Interpolating {len(columns_to_interpolate)} columns with < 5% missing values")
        
        for col in columns_to_interpolate:
            df_cleaned[col] = df_cleaned[col].interpolate(method=interpolation_method)
        
        # Winsorize extreme returns
        logger.info(f"Winsorizing extreme returns > |{winsorize_threshold}|")
        for col in df_cleaned.columns:
            extreme_values = (df_cleaned[col].abs() > winsorize_threshold)
            if extreme_values.any():
                extreme_count = extreme_values.sum()
                logger.warning(f"Found {extreme_count} extreme values in {col}")
                
                # Replace with sign(x) * threshold
                df_cleaned.loc[extreme_values, col] = df_cleaned.loc[extreme_values, col].apply(
                    lambda x: np.sign(x) * winsorize_threshold
                )
        
        # Generate data quality report
        logger.info("Generating data quality plots")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(availability)), availability['percent_missing'])
        plt.xticks([])
        plt.xlabel('Factors')
        plt.ylabel('Percent Missing (%)')
        plt.title('Missing Data by Factor')
        plt.tight_layout()
        plt.savefig("reports/missing_data_by_factor.pdf")
        
        # Save cleaned data
        logger.info(f"Writing cleaned data to {output_path}")
        df_cleaned.to_parquet(output_path)
        
        # Register in catalog
        register_asset(output_path, metadata={"source_file": input_path, "cleaning_applied": True})
        
        logger.info("Data cleaning complete")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    clean_factor_data(config)