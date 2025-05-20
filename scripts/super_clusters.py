#!/usr/bin/env python
"""
Identify persistent super-clusters of factors based on co-occurrence of cluster membership.
"""
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def identify_superclusters(config):
    """
    Identify persistent super-clusters of factors.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the superclusters parquet file
    """
    logger = setup_logger()
    logger.info("Starting super-cluster identification")
    
    cluster_labels_path = config['output_paths']['cluster_labels']
    output_path = config['output_paths']['superclusters']
    corr_threshold = config['corr_threshold']
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load cluster labels
        logger.info(f"Reading cluster labels from {cluster_labels_path}")
        cluster_df_long = pd.read_parquet(cluster_labels_path)
        
        # Create a pivot table: factor × window with cluster values
        logger.info("Creating factor-window pivot table")
        pivot_df = cluster_df_long.pivot(index='factor', columns='window_name', values='cluster')
        
        # Calculate co-occurrence matrix
        logger.info("Calculating co-occurrence matrix")
        n_factors = len(pivot_df)
        factors = pivot_df.index.tolist()
        
        # Initialize co-occurrence matrix and counts matrix
        co_occurrence = np.zeros((n_factors, n_factors))
        valid_window_counts = np.zeros((n_factors, n_factors))
        
        # For each window, count pairs of factors in the same cluster
        for window in pivot_df.columns:
            window_data = pivot_df[window].dropna()
            window_factors = window_data.index
            
            for i, factor_i in enumerate(factors):
                if factor_i not in window_factors:
                    continue
                    
                for j, factor_j in enumerate(factors):
                    if j <= i or factor_j not in window_factors:
                        continue
                    
                    # Increment valid window count
                    valid_window_counts[i, j] += 1
                    valid_window_counts[j, i] += 1
                    
                    # Check if both factors are in the same cluster
                    if window_data[factor_i] == window_data[factor_j]:
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1
        
        # Calculate co-occurrence percentage (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            co_occurrence_pct = np.where(valid_window_counts > 0, 
                                        co_occurrence / valid_window_counts, 
                                        0)
        
        # Create co-occurrence DataFrame
        co_occurrence_df = pd.DataFrame(co_occurrence_pct, index=factors, columns=factors)
        
        # Create distance matrix: distance = 1 - co_occurrence_pct
        distance_matrix = 1 - co_occurrence_pct
        np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0
        
        # Convert to condensed distance matrix (upper triangle, no diagonal)
        condensed_distance = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        logger.info("Performing hierarchical clustering on co-occurrence matrix")
        Z = linkage(condensed_distance, method='average')
        
        # Cut the dendrogram at the threshold
        supercluster_threshold = 0.5  # Factors together in 50% of windows
        supercluster_labels = fcluster(Z, t=supercluster_threshold, criterion='distance')
        
        # Create super-cluster DataFrame
        supercluster_df = pd.DataFrame({
            'factor': factors,
            'supercluster': supercluster_labels
        })
        
        # Save super-clusters
        logger.info(f"Saving super-clusters to {output_path}")
        supercluster_df.to_parquet(output_path)
        
        # Export for AI upload
        logger.info("Exporting super-cluster stability for AI upload")
        upload_bundle_dir = Path("outputs/upload_bundle")
        upload_bundle_dir.mkdir(exist_ok=True, parents=True)
        
        # Create stability metric for each factor pair
        stability_records = []
        for i, factor_i in enumerate(factors):
            for j, factor_j in enumerate(factors):
                if j <= i:
                    continue
                
                stability_records.append({
                    'factor_i': factor_i,
                    'factor_j': factor_j,
                    'co_occurrence_pct': co_occurrence_pct[i, j],
                    'valid_windows': int(valid_window_counts[i, j]),
                    'same_supercluster': supercluster_labels[i] == supercluster_labels[j]
                })
        
        stability_df = pd.DataFrame(stability_records)
        stability_df.to_csv("outputs/upload_bundle/supercluster_stability.csv", index=False)
        
        # Generate co-occurrence heatmap
        logger.info("Generating co-occurrence heatmap")
        plt.figure(figsize=(14, 12))
        
        # Reorder matrix by super-cluster
        ordered_factors = supercluster_df.sort_values('supercluster').factor.tolist()
        co_occurrence_reordered = co_occurrence_df.loc[ordered_factors, ordered_factors]
        
        # Plot heatmap
        sns.heatmap(co_occurrence_reordered, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Factor Co-occurrence Matrix (ordered by super-clusters)')
        plt.tight_layout()
        plt.savefig("reports/Cooccurrence_Heatmap.pdf")
        
        # Save PNG version for AI upload
        plt.savefig("outputs/upload_bundle/cooccurrence_heatmap.png", dpi=150)
        
        # Generate super-cluster membership visualization
        logger.info("Generating super-cluster membership visualization")
        plt.figure(figsize=(10, 8))
        
        # Group by supercluster and count
        supercluster_counts = supercluster_df.groupby('supercluster').count()
        supercluster_counts = supercluster_counts.sort_values('factor', ascending=False)
        
        plt.bar(supercluster_counts.index, supercluster_counts.factor)
        plt.title('Number of Factors per Super-cluster')
        plt.xlabel('Super-cluster ID')
        plt.ylabel('Number of Factors')
        plt.xticks(supercluster_counts.index)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Supercluster_Membership.pdf")
        
        # Register in catalog
        register_asset(output_path, metadata={"source_file": cluster_labels_path})
        
        logger.info("Super-cluster identification complete")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in super-cluster identification: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    identify_superclusters(config)