#!/usr/bin/env python
"""
Main script to run the T2 Factor Clusters analysis pipeline.
"""
import os
import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

# Import modules
from scripts.logger import setup_logger
from scripts.init_catalog import init_catalog
from scripts.data_ingest import excel_to_parquet, load_config
from scripts.data_cleaning import clean_factor_data
from scripts.rolling_correlation import compute_rolling_correlations
from scripts.hierarchical_clustering import hierarchical_clustering
from scripts.super_clusters import identify_superclusters
from scripts.pca_analysis import perform_pca_analysis
from scripts.network_analysis import perform_network_analysis
from scripts.structural_change import detect_structural_changes
from scripts.visualization_suite import generate_visualizations
from scripts.factor_selection import select_representative_factors

def run_pipeline(config_path="config.yml", start_phase=0, end_phase=None):
    """
    Run the T2 Factor Clusters analysis pipeline.
    
    Args:
        config_path: Path to the configuration file
        start_phase: Phase to start from (0-indexed)
        end_phase: Phase to end at (0-indexed, inclusive)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    logger = setup_logger()
    logger.info(f"Starting pipeline from phase {start_phase}")
    
    # Define phases
    phases = [
        {"name": "Initialize Catalog", "func": init_catalog},
        {"name": "Data Ingestion", "func": lambda: excel_to_parquet(config)},
        {"name": "Data Cleaning", "func": lambda: clean_factor_data(config)},
        {"name": "Rolling Correlation", "func": lambda: compute_rolling_correlations(config)},
        {"name": "Hierarchical Clustering", "func": lambda: hierarchical_clustering(config)},
        {"name": "Super-Cluster Identification", "func": lambda: identify_superclusters(config)},
        {"name": "PCA Analysis", "func": lambda: perform_pca_analysis(config)},
        {"name": "Network Analysis", "func": lambda: perform_network_analysis(config)},
        {"name": "Structural Change Detection", "func": lambda: detect_structural_changes(config)},
        {"name": "Visualization Suite", "func": lambda: generate_visualizations(config)},
        {"name": "Representative Factor Selection", "func": lambda: select_representative_factors(config)},
    ]
    
    if end_phase is None:
        end_phase = len(phases) - 1
    
    # Run selected phases
    for i, phase in enumerate(phases):
        if start_phase <= i <= end_phase:
            logger.info(f"Running phase {i}: {phase['name']}")
            try:
                result = phase["func"]()
                logger.info(f"Phase {i}: {phase['name']} completed successfully")
            except Exception as e:
                logger.error(f"Error in phase {i}: {phase['name']} - {e}")
                raise
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T2 Factor Clusters analysis pipeline")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to configuration file")
    parser.add_argument("--start", type=int, default=0, help="Phase to start from (0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Phase to end at (0-indexed, inclusive)")
    
    args = parser.parse_args()
    run_pipeline(config_path=args.config, start_phase=args.start, end_phase=args.end)