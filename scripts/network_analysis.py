#!/usr/bin/env python
"""
Perform network analysis on correlation matrices.
"""
import os
import yaml
import pandas as pd
import numpy as np
import zarr
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
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

def perform_network_analysis(config):
    """
    Perform network analysis on correlation matrices.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the modularity metrics file
    """
    logger = setup_logger()
    logger.info("Starting network analysis")
    
    corr_path = config['output_paths']['rolling_corr']
    output_dir = Path(config['output_paths']['metrics']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, "modularity.parquet")
    
    edge_threshold = config.get('edge_threshold', 0.5)
    random_seed = config.get('random_seed', 42)
    
    try:
        # Open zarr store
        logger.info(f"Reading correlation matrices from {corr_path}")
        store = zarr.open(corr_path, mode='r')
        
        # Read window info
        logger.info("Reading window info from CSV")
        window_info = pd.read_csv("data/derived/window_info.csv")
        
        # Convert date strings back to datetime
        window_info['start_date'] = pd.to_datetime(window_info['start_date'])
        window_info['end_date'] = pd.to_datetime(window_info['end_date'])
        
        # Load cleaned data to get factor names
        cleaned_data_path = config['output_paths']['cleaned']
        logger.info(f"Reading cleaned data from {cleaned_data_path}")
        df_cleaned = pd.read_parquet(cleaned_data_path)
        
        # List to store network metrics for each window
        network_metrics = []
        
        # Set random seed for community detection
        np.random.seed(random_seed)
        
        # For each window, create network and detect communities
        logger.info(f"Creating correlation networks with edge threshold {edge_threshold}")
        
        for i, row in window_info.iterrows():
            window_name = row['window_name']
            
            # Get correlation matrix from zarr
            corr_matrix = store[window_name][:]
            
            # Get available factors for this window
            start_date = pd.Timestamp(row['start_date'])
            end_date = pd.Timestamp(row['end_date'])
            window_data = df_cleaned.loc[start_date:end_date]
            available_columns = window_data.dropna(axis=1, how='any').columns.tolist()
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (factors)
            for j, factor in enumerate(available_columns):
                G.add_node(factor)
            
            # Add edges between factors with correlation above threshold
            edge_count = 0
            for j in range(len(available_columns)):
                for k in range(j+1, len(available_columns)):
                    correlation = abs(corr_matrix[j, k])  # Use absolute correlation
                    if correlation > edge_threshold:
                        G.add_edge(available_columns[j], available_columns[k], weight=correlation)
                        edge_count += 1
            
            # Detect communities
            communities = community_louvain.best_partition(G, random_state=random_seed)
            
            # Calculate modularity
            modularity = community_louvain.modularity(communities, G)
            
            # Store network metrics
            network_metrics.append({
                'window_name': window_name,
                'date': end_date,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'edge_density': edge_count / (len(available_columns) * (len(available_columns) - 1) / 2),
                'num_communities': len(set(communities.values())),
                'modularity': modularity,
                'avg_clustering': nx.average_clustering(G)
            })
            
            # Generate network graphs for key periods
            if window_name.replace('_', '-') in config.get('key_periods', []):
                logger.info(f"Generating network visualization for {window_name}")
                
                # Create plot
                plt.figure(figsize=(14, 14))
                
                # Position nodes using force-directed layout
                pos = nx.spring_layout(G, seed=random_seed)
                
                # Draw nodes, color by community
                community_colors = {}
                for node, community_id in communities.items():
                    if community_id not in community_colors:
                        community_colors[community_id] = np.random.rand(3,)
                
                for community_id in set(communities.values()):
                    nodelist = [node for node, com in communities.items() if com == community_id]
                    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, 
                                          node_color=[community_colors[community_id]],
                                          node_size=200, alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8)
                
                plt.title(f'Factor Correlation Network - {window_name} (Modularity: {modularity:.3f})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"reports/Network_{window_name}.pdf")
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(window_info)} windows")
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(network_metrics)
        
        # Save metrics
        logger.info(f"Saving network metrics to {metrics_path}")
        metrics_df.to_parquet(metrics_path)
        
        # Export for AI upload
        logger.info("Exporting modularity metrics for AI upload")
        modularity_df = metrics_df[['window_name', 'date', 'modularity', 'num_communities', 'edge_density']]
        modularity_df.to_csv("outputs/upload_bundle/modularity.csv", index=False)
        
        # Plot modularity timeline
        logger.info("Generating modularity timeline")
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['date'], metrics_df['modularity'])
        plt.title('Network Modularity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Modularity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Modularity_Timeline.pdf")
        plt.savefig("outputs/upload_bundle/modularity_timeline.pdf")
        
        # Plot number of communities timeline
        logger.info("Generating communities timeline")
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['date'], metrics_df['num_communities'])
        plt.title('Number of Communities Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Communities')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Communities_Timeline.pdf")
        
        # Create combined plot: modularity and edge density
        logger.info("Generating combined network metrics plot")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot modularity on left y-axis
        ax1.plot(metrics_df['date'], metrics_df['modularity'], 'b-', label='Modularity')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Modularity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot edge density on right y-axis
        ax2 = ax1.twinx()
        ax2.plot(metrics_df['date'], metrics_df['edge_density'], 'r-', label='Edge Density')
        ax2.set_ylabel('Edge Density', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Network Modularity and Edge Density Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Network_Metrics_Combined.pdf")
        
        # Register in catalog
        register_asset(metrics_path, metadata={"source_file": corr_path})
        
        logger.info("Network analysis complete")
        return metrics_path
        
    except Exception as e:
        logger.error(f"Error in network analysis: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    perform_network_analysis(config)