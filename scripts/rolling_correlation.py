#!/usr/bin/env python
"""
Compute rolling correlation matrices for factor returns.
"""
import os
import yaml
import pandas as pd
import numpy as np
import zarr
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

def compute_rolling_correlations(config):
    """
    Compute rolling correlation matrices.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the zarr store with correlation matrices
    """
    logger = setup_logger()
    logger.info("Starting rolling correlation computation")
    
    input_path = config['output_paths']['cleaned']
    output_path = config['output_paths']['rolling_corr']
    window_length = config['window_length']
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Reading cleaned data from {input_path}")
        df = pd.read_parquet(input_path)
        
        # Sort by index (date) to ensure proper time order
        df = df.sort_index()
        
        # Create zarr store
        logger.info(f"Creating zarr store at {output_path}")
        store = zarr.open(output_path, mode='w')
        
        # List to store window information
        window_info = []
        
        # List to store average correlation per window
        avg_correlations = []
        
        # Compute rolling windows
        logger.info(f"Computing rolling correlation matrices with window={window_length} months")
        
        # Get unique years and months
        dates = df.index
        unique_year_months = sorted(set([(d.year, d.month) for d in dates]))
        
        # For each month, compute trailing window correlation
        for i, (year, month) in enumerate(unique_year_months[window_length-1:]):
            end_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
            start_date = end_date - pd.DateOffset(months=window_length-1)
            
            # Filter data for this window
            window_data = df.loc[start_date:end_date]
            
            # Skip windows with too few factors (require at least 5)
            available_columns = window_data.dropna(axis=1, how='any').columns
            if len(available_columns) < 5:
                logger.warning(f"Window {start_date} to {end_date} has fewer than 5 factors with complete data, skipping")
                continue
            
            # Compute correlation matrix for complete factors
            window_data_complete = window_data[available_columns]
            corr_matrix = window_data_complete.corr()
            
            # Store correlation matrix
            window_name = f"{year}_{month:02d}"
            data = corr_matrix.values
            store.create_dataset(
                window_name, 
                shape=data.shape,
                data=data, 
                dtype='float32'
            )
            
            # Store average correlation (excluding self-correlations)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = corr_matrix.values[mask].mean()
            
            # Store window information
            window_info.append({
                'window_name': window_name,
                'start_date': start_date,
                'end_date': end_date,
                'num_factors': len(available_columns),
                'avg_correlation': avg_corr
            })
            
            avg_correlations.append({
                'date': end_date,
                'avg_correlation': avg_corr,
                'num_factors': len(available_columns)
            })
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(unique_year_months[window_length-1:])} windows")
        
        # Store window information
        window_info_df = pd.DataFrame(window_info)
        
        # Convert datetime objects to strings for JSON serialization
        window_info_df['start_date'] = window_info_df['start_date'].astype(str)
        window_info_df['end_date'] = window_info_df['end_date'].astype(str)
        
        # Save as CSV (we'll use this instead of zarr attributes since there's a serialization issue)
        window_info_df.to_csv("data/derived/window_info.csv", index=False)
        
        # Note: Not storing in zarr attributes due to serialization issues with Timestamp objects
        
        # Save average correlations for AI upload
        avg_corr_df = pd.DataFrame(avg_correlations)
        upload_bundle_dir = Path("outputs/upload_bundle")
        upload_bundle_dir.mkdir(exist_ok=True, parents=True)
        avg_corr_df.to_csv("outputs/upload_bundle/avg_corr_series.csv", index=False)
        
        # Plot average correlation over time
        logger.info("Generating average correlation timeline plot")
        plt.figure(figsize=(12, 6))
        plt.plot(avg_corr_df['date'], avg_corr_df['avg_correlation'])
        plt.title('Average Pairwise Correlation Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Average_Correlation_Timeline.pdf")
        plt.savefig("outputs/upload_bundle/average_corr_timeline.pdf")
        
        # Register in catalog
        register_asset(output_path, metadata={"source_file": input_path, "window_length": window_length})
        
        logger.info("Rolling correlation computation complete")
        return output_path
        
    except Exception as e:
        logger.error(f"Error computing rolling correlations: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    compute_rolling_correlations(config)