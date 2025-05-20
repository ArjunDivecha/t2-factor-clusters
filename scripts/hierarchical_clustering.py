#!/usr/bin/env python
"""
Perform hierarchical clustering on rolling correlation matrices.
"""
import os
import yaml
import pandas as pd
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.colors as mcolors

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def hierarchical_clustering(config):
    """
    Perform hierarchical clustering on rolling correlation matrices.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the cluster labels parquet file
    """
    logger = setup_logger()
    logger.info("Starting hierarchical clustering")
    
    corr_path = config['output_paths']['rolling_corr']
    output_path = config['output_paths']['cluster_labels']
    corr_threshold = config['corr_threshold']
    dist_threshold = 1 - corr_threshold  # convert correlation to distance
    key_periods = config.get('key_periods', [])
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open zarr store
        logger.info(f"Reading correlation matrices from {corr_path}")
        store = zarr.open(corr_path, mode='r')
        
        # Read window info from CSV
        logger.info("Reading window info from CSV")
        window_info = pd.read_csv("data/derived/window_info.csv")
        
        # Convert date strings back to datetime
        window_info['start_date'] = pd.to_datetime(window_info['start_date'])
        window_info['end_date'] = pd.to_datetime(window_info['end_date'])
        
        # Load cleaned data to get factor names
        cleaned_data_path = config['output_paths']['cleaned']
        df_cleaned = pd.read_parquet(cleaned_data_path)
        
        # Dictionary to store cluster labels for each window
        all_cluster_labels = {}
        
        # For each window, perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering with distance threshold {dist_threshold:.2f}")
        
        for i, row in window_info.iterrows():
            window_name = row['window_name']
            
            # Get correlation matrix from zarr
            corr_matrix = store[window_name][:]
            
            # Get available factors for this window
            start_date = pd.Timestamp(row['start_date'])
            end_date = pd.Timestamp(row['end_date'])
            window_data = df_cleaned.loc[start_date:end_date]
            available_columns = window_data.dropna(axis=1, how='any').columns.tolist()
            
            # Convert correlation to distance: d = 1 - ρ
            distance_matrix = 1 - corr_matrix
            
            # Convert to condensed distance matrix (required by linkage)
            # Only take upper triangle (excluding diagonal)
            condensed_distance = squareform(distance_matrix)
            
            # Perform hierarchical clustering (average linkage)
            Z = linkage(condensed_distance, method='average')
            
            # Cut the dendrogram at the specified threshold
            cluster_labels = fcluster(Z, t=dist_threshold, criterion='distance')
            
            # Store cluster labels
            all_cluster_labels[window_name] = {
                'start_date': start_date,
                'end_date': end_date,
                'factors': available_columns,
                'cluster_labels': cluster_labels.tolist(),
                'num_clusters': len(np.unique(cluster_labels))
            }
            
            # Generate dendrograms for key periods
            if window_name.replace('_', '-') in key_periods:
                logger.info(f"Generating dendrogram for key period {window_name}")
                plt.figure(figsize=(14, 8))
                plt.title(f"Hierarchical Clustering Dendrogram - {window_name}")
                
                # Create a custom color map for clusters
                num_clusters = len(np.unique(cluster_labels))
                colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
                cluster_colors = {cluster: color for cluster, color in zip(np.unique(cluster_labels), colors)}
                
                # Plot dendrogram
                dendrogram(
                    Z,
                    labels=available_columns,
                    color_threshold=dist_threshold,
                    leaf_rotation=90,
                    leaf_font_size=8,
                )
                plt.axhline(y=dist_threshold, color='r', linestyle='--', 
                           label=f'Threshold (τ={dist_threshold:.2f}, ρ={corr_threshold:.2f})')
                plt.ylabel('Distance (1 - ρ)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"reports/Dendrogram_{window_name}.pdf")
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(window_info)} windows")
        
        # Convert to DataFrame for easier manipulation
        # We'll create a long-format DataFrame first
        cluster_records = []
        
        for window_name, data in all_cluster_labels.items():
            for factor_idx, factor_name in enumerate(data['factors']):
                cluster_records.append({
                    'window_name': window_name,
                    'date': data['end_date'],
                    'factor': factor_name,
                    'cluster': data['cluster_labels'][factor_idx]
                })
        
        # Create long-format DataFrame
        cluster_df_long = pd.DataFrame(cluster_records)
        
        # Also create wide-format DataFrame (factor × window)
        cluster_df_wide = cluster_df_long.pivot(index='factor', columns='window_name', values='cluster')
        
        # Save cluster labels
        logger.info(f"Saving cluster labels to {output_path}")
        cluster_df_long.to_parquet(output_path)
        
        # Save CSV version for AI upload
        upload_bundle_dir = Path("outputs/upload_bundle")
        upload_bundle_dir.mkdir(exist_ok=True, parents=True)
        cluster_df_long.to_csv("outputs/upload_bundle/cluster_labels.csv", index=False)
        
        # Also save metadata about number of clusters per window
        num_clusters_df = pd.DataFrame([
            {'window_name': window_name, 'date': data['end_date'], 'num_clusters': data['num_clusters']}
            for window_name, data in all_cluster_labels.items()
        ])
        num_clusters_df.to_csv("data/derived/num_clusters_per_window.csv", index=False)
        
        # Register in catalog
        register_asset(output_path, metadata={"source_file": corr_path, "threshold": corr_threshold})
        
        logger.info("Hierarchical clustering complete")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in hierarchical clustering: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    hierarchical_clustering(config)