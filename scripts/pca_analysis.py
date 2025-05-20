#!/usr/bin/env python
"""
Perform Principal Component Analysis on correlation matrices.
"""
import os
import yaml
import pandas as pd
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def perform_pca_analysis(config):
    """
    Perform PCA analysis on correlation matrices.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of paths to the eigenvalues and eigenvectors files
    """
    logger = setup_logger()
    logger.info("Starting PCA analysis")
    
    corr_path = config['output_paths']['rolling_corr']
    eigenvalues_path = config['output_paths']['pca']['eigenvalues']
    eigenvectors_path = config['output_paths']['pca']['eigenvectors']
    
    # Ensure output directories exist
    eigenvalues_dir = Path(eigenvalues_path).parent
    eigenvalues_dir.mkdir(parents=True, exist_ok=True)
    
    eigenvectors_dir = Path(eigenvectors_path).parent
    eigenvectors_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # List to store eigenvalues for each window
        eigenvalues_records = []
        
        # List to store eigenvectors for each window
        eigenvectors_zarr = zarr.open(eigenvectors_path, mode='w')
        
        # Dictionary to store similarity between consecutive eigenvectors
        eigenvector_similarity = {}
        
        # Previous eigenvectors (for similarity calculation)
        prev_eigenvectors = None
        prev_window = None
        
        # For each window, perform PCA
        logger.info("Performing PCA for each window")
        
        for i, row in window_info.iterrows():
            window_name = row['window_name']
            
            # Get correlation matrix from zarr
            corr_matrix = store[window_name][:]
            
            # Get available factors for this window
            start_date = pd.Timestamp(row['start_date'])
            end_date = pd.Timestamp(row['end_date'])
            window_data = df_cleaned.loc[start_date:end_date]
            available_columns = window_data.dropna(axis=1, how='any').columns.tolist()
            
            # Perform PCA on correlation matrix
            pca = PCA(n_components=min(len(available_columns), 10))
            pca.fit(corr_matrix)
            
            # Store eigenvalues
            eigenvalues_record = {
                'window_name': window_name,
                'date': end_date
            }
            
            # Store first 10 eigenvalues (or fewer if there are fewer components)
            for j in range(min(len(available_columns), 10)):
                eigenvalues_record[f'eigenvalue_{j+1}'] = pca.explained_variance_ratio_[j]
                eigenvalues_record[f'cumulative_variance_{j+1}'] = pca.explained_variance_ratio_[:j+1].sum()
            
            eigenvalues_records.append(eigenvalues_record)
            
            # Store eigenvectors
            eigenvectors_zarr.create_dataset(
                window_name, 
                shape=pca.components_.shape,
                data=pca.components_, 
                dtype='float32'
            )
            
            # Calculate eigenvector similarity with previous window
            if prev_eigenvectors is not None:
                # Only compare first eigenvector
                similarity = np.abs(np.dot(prev_eigenvectors[0], pca.components_[0]))
                eigenvector_similarity[window_name] = {
                    'prev_window': prev_window,
                    'similarity': similarity
                }
            
            prev_eigenvectors = pca.components_
            prev_window = window_name
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(window_info)} windows")
        
        # Create eigenvalues DataFrame
        eigenvalues_df = pd.DataFrame(eigenvalues_records)
        eigenvalues_df.to_parquet(eigenvalues_path)
        
        # Also save eigenvalues for AI upload
        logger.info("Exporting eigenvalues for AI upload")
        eigenvalues_for_upload = eigenvalues_df[['window_name', 'date', 'eigenvalue_1', 'eigenvalue_2', 
                                              'cumulative_variance_1', 'cumulative_variance_2']]
        eigenvalues_for_upload.to_csv("outputs/upload_bundle/eigenvalues.csv", index=False)
        
        # Create eigenvector similarity DataFrame
        similarity_df = pd.DataFrame([
            {'window_name': window, 'prev_window': data['prev_window'], 'similarity': data['similarity']}
            for window, data in eigenvector_similarity.items()
        ])
        similarity_df['date'] = similarity_df['window_name'].map(
            eigenvalues_df.set_index('window_name')['date'])
        
        # Plot eigenvalue trends
        logger.info("Generating eigenvalue trend plots")
        plt.figure(figsize=(12, 6))
        plt.plot(eigenvalues_df['date'], eigenvalues_df['eigenvalue_1'], label='PC1')
        plt.plot(eigenvalues_df['date'], eigenvalues_df['eigenvalue_2'], label='PC2')
        plt.plot(eigenvalues_df['date'], eigenvalues_df['eigenvalue_3'], label='PC3')
        plt.title('Eigenvalue Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Explained Variance Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Eigenvalue_Trend.pdf")
        
        # Plot cumulative variance
        plt.figure(figsize=(12, 6))
        plt.plot(eigenvalues_df['date'], eigenvalues_df['cumulative_variance_1'], label='PC1')
        plt.plot(eigenvalues_df['date'], eigenvalues_df['cumulative_variance_2'], label='PC1+PC2')
        plt.plot(eigenvalues_df['date'], eigenvalues_df['cumulative_variance_3'], label='PC1+PC2+PC3')
        plt.title('Cumulative Explained Variance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/Cumulative_Variance_Trend.pdf")
        
        # Plot PC1 similarity timeline
        plt.figure(figsize=(12, 6))
        plt.plot(similarity_df['date'], similarity_df['similarity'])
        plt.title('PC1 Eigenvector Similarity Between Consecutive Windows')
        plt.xlabel('Date')
        plt.ylabel('Similarity (absolute dot product)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/PC1_Similarity_Timeline.pdf")
        
        # Plot scatter of factors in PC1-PC2 space for a few selected periods
        logger.info("Generating PC scatter plots for selected periods")
        key_periods = config.get('key_periods', [])
        
        for period in key_periods:
            period_name = period.replace('-', '_')
            if period_name in store:
                logger.info(f"Generating PCA scatter for period {period}")
                
                # Get correlation matrix
                corr_matrix = store[period_name][:]
                
                # Get available factors
                period_end = pd.Timestamp(period)
                period_start = period_end - pd.DateOffset(months=config['window_length']-1)
                period_data = df_cleaned.loc[period_start:period_end]
                available_columns = period_data.dropna(axis=1, how='any').columns.tolist()
                
                # Perform PCA
                pca = PCA(n_components=2)
                pca.fit(corr_matrix)
                
                # Get factor positions in PC1-PC2 space
                pca_coords = pca.transform(corr_matrix)
                
                # Create DataFrame for plotting
                pca_df = pd.DataFrame({
                    'factor': available_columns,
                    'PC1': pca_coords[:, 0],
                    'PC2': pca_coords[:, 1]
                })
                
                # Plot scatter
                plt.figure(figsize=(14, 10))
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                
                # Add factor labels
                for _, row in pca_df.iterrows():
                    plt.annotate(row['factor'], 
                                (row['PC1'], row['PC2']),
                                fontsize=8,
                                alpha=0.8)
                
                plt.title(f'Factors in PC1-PC2 Space - {period}')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"reports/PCA_Scatter_{period_name}.pdf")
                
                # Save one for AI upload
                if period_name == key_periods[0].replace('-', '_'):
                    plt.savefig("outputs/upload_bundle/pca_scatter.pdf")
        
        # Register in catalog
        register_asset(eigenvalues_path, metadata={"source_file": corr_path})
        register_asset(eigenvectors_path, metadata={"source_file": corr_path})
        
        logger.info("PCA analysis complete")
        return eigenvalues_path, eigenvectors_path
        
    except Exception as e:
        logger.error(f"Error in PCA analysis: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    perform_pca_analysis(config)