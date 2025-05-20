#!/usr/bin/env python
"""
Select a parsimonious, low-collinearity subset of approximately 15 factors.

Steps:
1. Read cleaned factor returns and superclusters
2. Choose one anchor factor from every persistent super-cluster (highest information content)
3. Build a pool of factors that add unique variance beyond anchors
4. Keep only factors with high representative power
5. Iteratively prune by Variance-Inflation-Factor until VIF < threshold
6. Write the final list to outputs
"""
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.logger import setup_logger
from scripts.init_catalog import register_asset

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def single_factor_tstat(x: pd.Series, y: pd.Series) -> float:
    """OLS t-stat of slope in simple regression y ~ x (no intercept, monthly data)."""
    beta = np.dot(x, y) / np.dot(x, x)
    resid = y - beta * x
    se = resid.std(ddof=1) / (np.sqrt(len(y)) * x.std(ddof=1))
    return beta / se

def variance_inflation(df: pd.DataFrame) -> pd.Series:
    """Return VIF for each column of *standardized* df."""
    std_df = (df - df.mean()) / df.std(ddof=0)
    return pd.Series(
        [variance_inflation_factor(std_df.values, i) for i in range(std_df.shape[1])],
        index=std_df.columns,
    )

def select_representative_factors(config):
    """
    Select a subset of representative factors.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to the selected factors output file
    """
    logger = setup_logger()
    logger.info("Starting factor selection process")
    
    # Parameters for selection
    THRESH_UNIQUENESS = 0.6  # Uniqueness threshold
    VIF_THRESHOLD = 5.0      # Variance inflation factor threshold
    TARGET_COUNT = 15        # Target number of factors to select
    SEED = config.get('random_seed', 42)
    np.random.seed(SEED)
    
    # Paths
    returns_path = config['output_paths']['cleaned']
    superclusters_path = config['output_paths']['superclusters']
    output_path = "outputs/upload_bundle/selected_factors.yml"
    
    try:
        # Load data
        logger.info(f"Loading cleaned returns from {returns_path}")
        returns = pd.read_parquet(returns_path).sort_index()
        
        logger.info(f"Loading superclusters from {superclusters_path}")
        superclusters = pd.read_parquet(superclusters_path)
        
        # Check if superclusters are in correct format
        if 'supercluster' not in superclusters.columns:
            logger.error("Superclusters file does not contain 'supercluster' column")
            return None
        
        # Step 1: Select anchor factors from each supercluster
        # Pick the factor with highest information content (variance) from each cluster
        logger.info("Selecting anchor factors from superclusters")
        anchors = []
        num_superclusters = superclusters['supercluster'].nunique()
        
        # If we have too many superclusters, limit to the largest ones
        if num_superclusters > TARGET_COUNT:
            # Count factors per supercluster
            sc_counts = superclusters.groupby('supercluster').count().sort_values('factor', ascending=False)
            # Take the N largest superclusters
            top_superclusters = sc_counts.head(TARGET_COUNT).index.tolist()
            logger.info(f"Limiting to top {TARGET_COUNT} superclusters out of {num_superclusters}")
        else:
            top_superclusters = superclusters['supercluster'].unique()
        
        # For each supercluster, find the factor with highest variance (information content)
        for sc in top_superclusters:
            factors_in_cluster = superclusters[superclusters['supercluster'] == sc]['factor'].tolist()
            # Filter to only factors that exist in returns
            available_factors = [f for f in factors_in_cluster if f in returns.columns]
            if not available_factors:
                continue
            
            # Select factor with highest variance
            variances = returns[available_factors].var()
            best_factor = variances.idxmax()
            anchors.append(best_factor)
        
        logger.info(f"Selected {len(anchors)} anchor factors, one from each major supercluster")
        
        # Step 2: Uniqueness screen
        logger.info("Conducting uniqueness screening")
        unique_pool = []
        anchor_corr = returns[anchors].corr().abs()
        
        for factor in returns.columns:
            if factor in anchors:
                continue
            
            # Calculate uniqueness ratio (1 - avg squared correlation with anchors)
            corr_vec = returns[anchors + [factor]].corr().abs()[factor].drop(factor)
            uniqueness_ratio = 1 - (corr_vec.pow(2).mean())
            
            if uniqueness_ratio > THRESH_UNIQUENESS:
                unique_pool.append(factor)
        
        logger.info(f"Found {len(unique_pool)} factors with high uniqueness")
        
        # Step 3: Combine anchors and unique factors
        combined_pool = anchors + unique_pool
        
        # Step 4: Apply VIF pruning
        logger.info("Applying VIF pruning")
        X = returns[combined_pool].copy()
        
        # Drop any columns with NaN values
        X = X.dropna(axis=1)
        
        # If we have too many factors, prune based on VIF
        while len(X.columns) > TARGET_COUNT:
            vifs = variance_inflation(X)
            max_vif = vifs.max()
            
            if max_vif <= VIF_THRESHOLD:
                break
            
            drop_candidate = vifs.idxmax()
            logger.info(f"Dropping {drop_candidate} with VIF = {max_vif:.2f}")
            X = X.drop(columns=drop_candidate)
        
        # Final selected factors
        final_factors = X.columns.tolist()
        
        # If we still have more than target count, drop lowest variance factors
        if len(final_factors) > TARGET_COUNT:
            logger.info(f"Still have {len(final_factors)} factors, pruning to {TARGET_COUNT}")
            factor_var = returns[final_factors].var()
            sorted_factors = factor_var.sort_values(ascending=False)
            final_factors = sorted_factors.head(TARGET_COUNT).index.tolist()
        
        # Save output
        logger.info(f"Final selection: {len(final_factors)} factors")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to YAML
        with open(output_path, 'w') as f:
            yaml.dump(final_factors, f)
        
        # Also save as CSV for easier reading
        pd.DataFrame({'factor': final_factors}).to_csv(
            "outputs/upload_bundle/selected_factors.csv", index=False
        )
        
        # Print the selected factors
        logger.info("Selected factors:")
        for factor in final_factors:
            logger.info(f"- {factor}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in factor selection: {e}")
        raise

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)
    select_representative_factors(config)