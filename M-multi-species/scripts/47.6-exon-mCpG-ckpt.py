#!/usr/bin/env python3
"""
47.6-exon-mCpG.py

Build predictive models to identify mCpGs that can accurately predict changes in 
exon expression for each of three species (Apul, Peve, Ptua).

This script uses machine learning approaches (Random Forest, Elastic Net) to:
1. Identify CpG sites whose methylation levels predict exon expression
2. Evaluate model performance using cross-validation
3. Rank CpGs by their predictive importance

PARALLELIZED VERSION - Uses all available CPUs for model training.

Data Sources:
-------------
Exon Expression Matrices:
- Apul: https://gannet.fish.washington.edu/.../apul-exon_gene_count_matrix.csv
- Peve: https://gannet.fish.washington.edu/.../peve-exon_gene_count_matrix.csv
- Ptua: https://gannet.fish.washington.edu/.../ptua-exon_gene_count_matrix.csv

mCpG Methylation Data:
- Apul: https://gannet.fish.washington.edu/.../merged-WGBS-CpG-counts_filtered_n20.csv
- Peve: https://gannet.fish.washington.edu/.../merged-WGBS-CpG-counts_filtered_n20.csv
- Ptua: https://gannet.fish.washington.edu/.../merged-WGBS-CpG-counts_filtered_n20.csv
"""

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Data URLs
EXON_URLS = {
    'apul': 'https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries-molecular-calcification/M-multi-species/output/40-exon-count-matrix/apul-exon_gene_count_matrix.csv',
    'peve': 'https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries-molecular-calcification/M-multi-species/output/40-exon-count-matrix/peve-exon_gene_count_matrix.csv',
    'ptua': 'https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries-molecular-calcification/M-multi-species/output/40-exon-count-matrix/ptua-exon_gene_count_matrix.csv',
}

CPG_URLS = {
    'apul': 'https://gannet.fish.washington.edu/metacarcinus/E5/20250903_meth_Apul/merged-WGBS-CpG-counts_filtered_n20.csv',
    'peve': 'https://gannet.fish.washington.edu/metacarcinus/E5/Pevermanni/20250821_meth_Peve/merged-WGBS-CpG-counts_filtered_n20.csv',
    'ptua': 'https://gannet.fish.washington.edu/metacarcinus/E5/Ptuahiniensis/20250821_meth_Ptua/merged-WGBS-CpG-counts_filtered_n20.csv',
}

# Output directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / '47.6-exon-mCpG'

# Model parameters
MIN_SAMPLES = 5
MIN_VARIANCE_THRESHOLD = 0.01  # Minimum variance for features
TOP_CPGS_PER_EXON = 100  # Top CpGs to select per exon based on univariate correlation
MAX_EXONS_TO_MODEL = 5000  # Maximum number of exons to model (for computational tractability)
CV_FOLDS = 5  # Number of cross-validation folds
N_TOP_PREDICTIVE_CPGS = 50  # Number of top predictive CpGs to report per exon

# Number of CPUs to use (None = all available)
N_CPUS = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_n_cpus():
    """Get number of CPUs to use for parallel processing."""
    if N_CPUS is not None:
        return N_CPUS
    return mp.cpu_count()


def parse_cpg_ids_vectorized(cpg_ids: pd.Series) -> pd.DataFrame:
    """
    Parse multiple CpG IDs to extract chromosome and position (vectorized).
    
    Parameters
    ----------
    cpg_ids : pd.Series
        Series of CpG identifier strings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cpg_id, cpg_chr, cpg_pos
    """
    # Remove "CpG_" prefix
    stripped = cpg_ids.str.replace('^CpG_', '', regex=True)
    
    # Extract chromosome (everything before last underscore) and position (last numeric segment)
    pattern = r'^(.+)_(\d+)$'
    extracted = stripped.str.extract(pattern)
    
    return pd.DataFrame({
        'cpg_id': cpg_ids,
        'cpg_chr': extracted[0],
        'cpg_pos': pd.to_numeric(extracted[1], errors='coerce')
    })


def filter_low_variance_features(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Filter out features with variance below threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features)
    threshold : float
        Minimum variance threshold
        
    Returns
    -------
    np.ndarray
        Boolean mask of features to keep
    """
    variances = np.var(X, axis=0)
    return variances > threshold


def compute_univariate_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman correlation between each feature and target.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features)
    y : np.ndarray
        Target vector
        
    Returns
    -------
    np.ndarray
        Absolute correlation values for each feature
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    
    for i in range(n_features):
        if np.std(X[:, i]) > 0:
            corr, _ = stats.spearmanr(X[:, i], y)
            correlations[i] = abs(corr) if not np.isnan(corr) else 0
        else:
            correlations[i] = 0
    
    return correlations


def build_predictive_model_for_exon(
    exon_id: str,
    exon_expr: np.ndarray,
    cpg_matrix: np.ndarray,
    cpg_names: np.ndarray,
    n_top_cpgs: int = 100,
    cv_folds: int = 5
) -> Optional[Dict]:
    """
    Build a predictive model for a single exon using CpG methylation as features.
    
    Parameters
    ----------
    exon_id : str
        Identifier for the exon
    exon_expr : np.ndarray
        Expression values for this exon across samples
    cpg_matrix : np.ndarray
        CpG methylation matrix (samples x CpGs)
    cpg_names : np.ndarray
        Names of CpG sites
    n_top_cpgs : int
        Number of top CpGs to use based on univariate correlation
    cv_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    dict or None
        Model results dictionary or None if modeling fails
    """
    n_samples = len(exon_expr)
    
    # Need minimum samples for cross-validation
    if n_samples < MIN_SAMPLES:
        return None
    
    # Check variance in expression
    if np.std(exon_expr) < MIN_VARIANCE_THRESHOLD:
        return None
    
    # Filter low variance CpGs
    var_mask = filter_low_variance_features(cpg_matrix, MIN_VARIANCE_THRESHOLD)
    if var_mask.sum() < 5:
        return None
    
    X_filtered = cpg_matrix[:, var_mask]
    cpg_names_filtered = cpg_names[var_mask]
    
    # Pre-filter by univariate correlation to reduce dimensionality
    correlations = compute_univariate_correlations(X_filtered, exon_expr)
    
    # Select top correlated CpGs
    n_select = min(n_top_cpgs, len(correlations))
    top_indices = np.argsort(correlations)[-n_select:]
    
    X_selected = X_filtered[:, top_indices]
    cpg_names_selected = cpg_names_filtered[top_indices]
    correlations_selected = correlations[top_indices]
    
    if X_selected.shape[1] < 2:
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Determine CV strategy based on sample size
    if n_samples < cv_folds * 2:
        cv = LeaveOneOut()
        cv_method = 'LOO'
    else:
        cv = KFold(n_splits=min(cv_folds, n_samples // 2), shuffle=True, random_state=42)
        cv_method = f'{cv_folds}-fold'
    
    results = {
        'exon_id': exon_id,
        'n_samples': n_samples,
        'n_cpgs_tested': len(cpg_names_filtered),
        'n_cpgs_selected': len(cpg_names_selected),
        'cv_method': cv_method,
    }
    
    # Try Random Forest first
    try:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=max(1, n_samples // 10),
            random_state=42,
            n_jobs=1  # Single job since we parallelize at exon level
        )
        
        rf_scores = cross_val_score(rf, X_scaled, exon_expr, cv=cv, scoring='r2')
        rf_r2 = np.mean(rf_scores)
        rf_r2_std = np.std(rf_scores)
        
        # Fit final model to get feature importances
        rf.fit(X_scaled, exon_expr)
        rf_importances = rf.feature_importances_
        
        results['rf_r2_cv'] = rf_r2
        results['rf_r2_std'] = rf_r2_std
        results['rf_importances'] = rf_importances
        
    except Exception as e:
        results['rf_r2_cv'] = np.nan
        results['rf_r2_std'] = np.nan
        results['rf_importances'] = None
    
    # Try Elastic Net for sparse feature selection
    try:
        if n_samples > 10:
            enet = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                cv=min(5, n_samples // 2),
                random_state=42,
                max_iter=5000
            )
            enet.fit(X_scaled, exon_expr)
            
            # Get predictions for R2
            enet_scores = cross_val_score(enet, X_scaled, exon_expr, cv=cv, scoring='r2')
            enet_r2 = np.mean(enet_scores)
            enet_r2_std = np.std(enet_scores)
            
            results['enet_r2_cv'] = enet_r2
            results['enet_r2_std'] = enet_r2_std
            results['enet_coefs'] = enet.coef_
            results['enet_alpha'] = enet.alpha_
            results['enet_l1_ratio'] = enet.l1_ratio_
            results['n_nonzero_coefs'] = np.sum(np.abs(enet.coef_) > 1e-6)
        else:
            results['enet_r2_cv'] = np.nan
            results['enet_r2_std'] = np.nan
            results['enet_coefs'] = None
            
    except Exception as e:
        results['enet_r2_cv'] = np.nan
        results['enet_r2_std'] = np.nan
        results['enet_coefs'] = None
    
    # Best model R2
    rf_r2 = results.get('rf_r2_cv', -np.inf) if not np.isnan(results.get('rf_r2_cv', np.nan)) else -np.inf
    enet_r2 = results.get('enet_r2_cv', -np.inf) if not np.isnan(results.get('enet_r2_cv', np.nan)) else -np.inf
    
    results['best_r2_cv'] = max(rf_r2, enet_r2)
    results['best_model'] = 'RF' if rf_r2 >= enet_r2 else 'ElasticNet'
    
    # Identify most predictive CpGs
    predictive_cpgs = []
    
    # From Random Forest importances
    if results.get('rf_importances') is not None:
        rf_imp = results['rf_importances']
        top_rf_idx = np.argsort(rf_imp)[-N_TOP_PREDICTIVE_CPGS:]
        for idx in reversed(top_rf_idx):
            predictive_cpgs.append({
                'cpg_id': cpg_names_selected[idx],
                'importance_rf': rf_imp[idx],
                'correlation': correlations_selected[idx],
                'source': 'RF'
            })
    
    # From Elastic Net coefficients
    if results.get('enet_coefs') is not None:
        enet_coef = np.abs(results['enet_coefs'])
        nonzero_mask = enet_coef > 1e-6
        if nonzero_mask.sum() > 0:
            for idx in np.where(nonzero_mask)[0]:
                # Check if already added from RF
                cpg_name = cpg_names_selected[idx]
                existing = [p for p in predictive_cpgs if p['cpg_id'] == cpg_name]
                if existing:
                    existing[0]['importance_enet'] = enet_coef[idx]
                else:
                    predictive_cpgs.append({
                        'cpg_id': cpg_name,
                        'importance_enet': enet_coef[idx],
                        'correlation': correlations_selected[idx],
                        'source': 'ElasticNet'
                    })
    
    results['predictive_cpgs'] = predictive_cpgs[:N_TOP_PREDICTIVE_CPGS]
    results['cpg_names_selected'] = cpg_names_selected.tolist()
    
    return results


def process_exon_wrapper(args: Tuple) -> Optional[Dict]:
    """Wrapper function for parallel processing."""
    exon_id, exon_expr, cpg_matrix, cpg_names = args
    return build_predictive_model_for_exon(
        exon_id=exon_id,
        exon_expr=exon_expr,
        cpg_matrix=cpg_matrix,
        cpg_names=cpg_names,
        n_top_cpgs=TOP_CPGS_PER_EXON,
        cv_folds=CV_FOLDS
    )


def plot_model_performance(results_df: pd.DataFrame, species: str, output_dir: Path) -> None:
    """Create visualization of model performance."""
    if results_df.empty:
        return
    
    colors = {'apul': 'steelblue', 'peve': 'coral', 'ptua': 'seagreen'}
    color = colors.get(species.lower(), 'gray')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution of R2 scores
    ax1 = axes[0, 0]
    valid_r2 = results_df['best_r2_cv'].dropna()
    ax1.hist(valid_r2, bins=50, color=color, edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, linestyle='--', color='red', linewidth=1.5, label='R² = 0')
    ax1.axvline(x=valid_r2.median(), linestyle='--', color='green', linewidth=1.5, 
                label=f'Median = {valid_r2.median():.3f}')
    ax1.set_xlabel('Cross-validated R²')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{species}: Distribution of Model R² Scores')
    ax1.legend()
    
    # 2. R2 vs number of CpGs tested
    ax2 = axes[0, 1]
    ax2.scatter(results_df['n_cpgs_tested'], results_df['best_r2_cv'], 
                alpha=0.5, s=10, c=color)
    ax2.set_xlabel('Number of CpGs Tested')
    ax2.set_ylabel('Cross-validated R²')
    ax2.set_title(f'{species}: Model Performance vs CpG Count')
    ax2.axhline(y=0, linestyle='--', color='red', linewidth=1)
    
    # 3. Comparison of RF vs ElasticNet
    ax3 = axes[1, 0]
    valid_both = results_df.dropna(subset=['rf_r2_cv', 'enet_r2_cv'])
    if not valid_both.empty:
        ax3.scatter(valid_both['rf_r2_cv'], valid_both['enet_r2_cv'], 
                    alpha=0.5, s=10, c=color)
        lim_min = min(valid_both['rf_r2_cv'].min(), valid_both['enet_r2_cv'].min())
        lim_max = max(valid_both['rf_r2_cv'].max(), valid_both['enet_r2_cv'].max())
        ax3.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
        ax3.set_xlabel('Random Forest R²')
        ax3.set_ylabel('Elastic Net R²')
        ax3.set_title(f'{species}: RF vs Elastic Net Performance')
    
    # 4. Top predictive exons
    ax4 = axes[1, 1]
    top_exons = results_df.nlargest(20, 'best_r2_cv')
    if not top_exons.empty:
        y_pos = range(len(top_exons))
        ax4.barh(y_pos, top_exons['best_r2_cv'], color=color, alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([eid[:30] + '...' if len(eid) > 30 else eid 
                            for eid in top_exons['exon_id']], fontsize=8)
        ax4.set_xlabel('Cross-validated R²')
        ax4.set_title(f'{species}: Top 20 Predictable Exons')
        ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species.lower()}_model_performance.png', dpi=150)
    plt.close()


def plot_top_predictive_cpgs(cpg_summary: pd.DataFrame, species: str, output_dir: Path) -> None:
    """Plot summary of most predictive CpGs across all exons."""
    if cpg_summary.empty:
        return
    
    colors = {'apul': 'steelblue', 'peve': 'coral', 'ptua': 'seagreen'}
    color = colors.get(species.lower(), 'gray')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. CpGs appearing in most models
    ax1 = axes[0]
    cpg_counts = cpg_summary['cpg_id'].value_counts().head(30)
    if not cpg_counts.empty:
        y_pos = range(len(cpg_counts))
        ax1.barh(y_pos, cpg_counts.values, color=color, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([cid[:25] + '...' if len(cid) > 25 else cid 
                            for cid in cpg_counts.index], fontsize=8)
        ax1.set_xlabel('Number of Exons')
        ax1.set_title(f'{species}: CpGs Predictive of Most Exons')
        ax1.invert_yaxis()
    
    # 2. Average importance of top CpGs
    ax2 = axes[1]
    cpg_avg_imp = cpg_summary.groupby('cpg_id')['importance_rf'].mean().nlargest(30)
    if not cpg_avg_imp.empty:
        y_pos = range(len(cpg_avg_imp))
        ax2.barh(y_pos, cpg_avg_imp.values, color=color, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([cid[:25] + '...' if len(cid) > 25 else cid 
                            for cid in cpg_avg_imp.index], fontsize=8)
        ax2.set_xlabel('Average RF Importance')
        ax2.set_title(f'{species}: CpGs with Highest Average Importance')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species.lower()}_top_predictive_cpgs.png', dpi=150)
    plt.close()


def plot_cross_species_comparison(all_results: pd.DataFrame, output_dir: Path) -> None:
    """Create cross-species comparison plots."""
    if all_results.empty:
        return
    
    colors = {'Apul': 'steelblue', 'Peve': 'coral', 'Ptua': 'seagreen'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. R2 distribution comparison
    ax1 = axes[0]
    for species in all_results['species'].unique():
        species_data = all_results[all_results['species'] == species]
        valid_r2 = species_data['best_r2_cv'].dropna()
        if len(valid_r2) > 0:
            sns.kdeplot(data=valid_r2, fill=True, alpha=0.4, 
                       color=colors.get(species, 'gray'), label=species, ax=ax1)
    ax1.axvline(x=0, linestyle='--', color='black', linewidth=1)
    ax1.set_xlabel('Cross-validated R²')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Model R² Scores Across Species')
    ax1.legend()
    
    # 2. Box plot comparison
    ax2 = axes[1]
    species_order = ['Apul', 'Peve', 'Ptua']
    valid_species = [s for s in species_order if s in all_results['species'].unique()]
    plot_data = all_results[all_results['species'].isin(valid_species)]
    
    box_colors = [colors.get(s, 'gray') for s in valid_species]
    bp = ax2.boxplot([plot_data[plot_data['species'] == s]['best_r2_cv'].dropna() 
                      for s in valid_species],
                     labels=valid_species, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(y=0, linestyle='--', color='red', linewidth=1)
    ax2.set_ylabel('Cross-validated R²')
    ax2.set_title('Model Performance Comparison')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_species_model_comparison.png', dpi=150)
    plt.close()


def process_species(species: str, exon_url: str, cpg_url: str, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Process a single species and build predictive models.
    """
    species_upper = species.capitalize()
    print(f"\n{'='*60}")
    print(f"Processing {species_upper}")
    print('='*60)
    
    # Load exon expression data
    print(f"  Loading exon expression data...")
    try:
        exon_df = pd.read_csv(exon_url)
        print(f"  Exon matrix: {len(exon_df)} exons")
    except Exception as e:
        print(f"  ERROR loading exon data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Load CpG methylation data
    print(f"  Loading CpG methylation data...")
    try:
        cpg_df = pd.read_csv(cpg_url)
        print(f"  CpG matrix: {len(cpg_df)} CpGs")
    except Exception as e:
        print(f"  ERROR loading CpG data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Identify common samples
    exon_meta_cols = ['gene_id', 'e_id', 'chr', 'strand', 'start', 'end']
    exon_sample_cols = [c for c in exon_df.columns if c not in exon_meta_cols]
    cpg_sample_cols = [c for c in cpg_df.columns if c != 'CpG']
    common_samples = sorted(list(set(exon_sample_cols) & set(cpg_sample_cols)))
    
    print(f"  Exon samples: {len(exon_sample_cols)}")
    print(f"  CpG samples: {len(cpg_sample_cols)}")
    print(f"  Common samples: {len(common_samples)}")
    
    if len(common_samples) < MIN_SAMPLES:
        print(f"  WARNING: Not enough common samples (need >= {MIN_SAMPLES})")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Create unique exon identifiers
    exon_df['exon_id'] = exon_df['gene_id'].astype(str) + '_' + exon_df['e_id'].astype(str)
    
    # Prepare matrices
    print(f"  Preparing data matrices...")
    
    # CpG matrix: samples x CpGs
    cpg_df_indexed = cpg_df.set_index('CpG')
    cpg_matrix = cpg_df_indexed[common_samples].T.values.astype(float)
    cpg_names = cpg_df_indexed.index.values
    
    # Handle missing values in CpG matrix (impute with column mean)
    col_means = np.nanmean(cpg_matrix, axis=0)
    nan_mask = np.isnan(cpg_matrix)
    cpg_matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    print(f"  CpG matrix shape: {cpg_matrix.shape}")
    
    # Exon expression matrix
    exon_expr_matrix = exon_df.set_index('exon_id')[common_samples].T.values.astype(float)
    exon_ids = exon_df['exon_id'].values
    
    print(f"  Exon matrix shape: {exon_expr_matrix.shape}")
    
    # Limit number of exons for computational tractability
    n_exons = len(exon_ids)
    if n_exons > MAX_EXONS_TO_MODEL:
        print(f"  Limiting to {MAX_EXONS_TO_MODEL} exons (from {n_exons})")
        # Select exons with highest variance
        exon_variances = np.var(exon_expr_matrix, axis=0)
        top_var_indices = np.argsort(exon_variances)[-MAX_EXONS_TO_MODEL:]
        exon_ids = exon_ids[top_var_indices]
        exon_expr_matrix = exon_expr_matrix[:, top_var_indices]
        n_exons = MAX_EXONS_TO_MODEL
    
    # Prepare arguments for parallel processing
    print(f"  Building predictive models for {n_exons} exons...")
    args_list = []
    for i, exon_id in enumerate(exon_ids):
        exon_expr = exon_expr_matrix[:, i]
        args_list.append((exon_id, exon_expr, cpg_matrix, cpg_names))
    
    # Run models in parallel
    n_workers = get_n_cpus()
    print(f"  Using {n_workers} CPUs for parallel processing...")
    
    results = []
    
    # Use process pool for parallel execution
    if n_exons > 100:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_exon_wrapper, args): i 
                      for i, args in enumerate(args_list)}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1
                
                if completed % max(1, n_exons // 10) == 0:
                    pct = 100 * completed / n_exons
                    print(f"    Progress: {completed}/{n_exons} ({pct:.0f}%)")
    else:
        # For small datasets, run sequentially
        for i, args in enumerate(args_list):
            result = process_exon_wrapper(args)
            if result is not None:
                results.append(result)
            if (i + 1) % max(1, n_exons // 10) == 0:
                print(f"    Progress: {i+1}/{n_exons}")
    
    print(f"  Completed: {len(results)} models built successfully")
    
    if not results:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Create results DataFrame
    results_df = pd.DataFrame([{
        'exon_id': r['exon_id'],
        'n_samples': r['n_samples'],
        'n_cpgs_tested': r['n_cpgs_tested'],
        'n_cpgs_selected': r['n_cpgs_selected'],
        'cv_method': r['cv_method'],
        'rf_r2_cv': r.get('rf_r2_cv'),
        'rf_r2_std': r.get('rf_r2_std'),
        'enet_r2_cv': r.get('enet_r2_cv'),
        'enet_r2_std': r.get('enet_r2_std'),
        'enet_alpha': r.get('enet_alpha'),
        'enet_l1_ratio': r.get('enet_l1_ratio'),
        'n_nonzero_coefs': r.get('n_nonzero_coefs'),
        'best_r2_cv': r['best_r2_cv'],
        'best_model': r['best_model'],
    } for r in results])
    
    results_df['species'] = species_upper
    
    # Extract predictive CpGs
    cpg_summary_list = []
    for r in results:
        exon_id = r['exon_id']
        for cpg_info in r.get('predictive_cpgs', []):
            cpg_summary_list.append({
                'exon_id': exon_id,
                'cpg_id': cpg_info['cpg_id'],
                'importance_rf': cpg_info.get('importance_rf', np.nan),
                'importance_enet': cpg_info.get('importance_enet', np.nan),
                'correlation': cpg_info.get('correlation', np.nan),
                'source': cpg_info.get('source', ''),
            })
    
    cpg_summary_df = pd.DataFrame(cpg_summary_list) if cpg_summary_list else pd.DataFrame()
    if not cpg_summary_df.empty:
        cpg_summary_df['species'] = species_upper
    
    # Summary statistics
    valid_r2 = results_df['best_r2_cv'].dropna()
    stats_dict = {
        'species': species_upper,
        'total_exons': len(exon_df),
        'total_cpgs': len(cpg_df),
        'models_built': len(results_df),
        'mean_r2': float(valid_r2.mean()) if len(valid_r2) > 0 else np.nan,
        'median_r2': float(valid_r2.median()) if len(valid_r2) > 0 else np.nan,
        'positive_r2_count': int((valid_r2 > 0).sum()),
        'r2_gt_0.1': int((valid_r2 > 0.1).sum()),
        'r2_gt_0.2': int((valid_r2 > 0.2).sum()),
        'r2_gt_0.3': int((valid_r2 > 0.3).sum()),
        'best_model_rf': int((results_df['best_model'] == 'RF').sum()),
        'best_model_enet': int((results_df['best_model'] == 'ElasticNet').sum()),
    }
    
    print(f"\n  === {species_upper} Summary ===")
    print(f"  Models built: {stats_dict['models_built']}")
    print(f"  Mean R² (CV): {stats_dict['mean_r2']:.4f}")
    print(f"  Median R² (CV): {stats_dict['median_r2']:.4f}")
    print(f"  Exons with R² > 0: {stats_dict['positive_r2_count']}")
    print(f"  Exons with R² > 0.1: {stats_dict['r2_gt_0.1']}")
    print(f"  Exons with R² > 0.2: {stats_dict['r2_gt_0.2']}")
    print(f"  Exons with R² > 0.3: {stats_dict['r2_gt_0.3']}")
    print(f"  Best model: RF={stats_dict['best_model_rf']}, ElasticNet={stats_dict['best_model_enet']}")
    
    # Save results
    results_df.to_csv(output_dir / f'{species.lower()}_exon_model_results.csv', index=False)
    if not cpg_summary_df.empty:
        cpg_summary_df.to_csv(output_dir / f'{species.lower()}_predictive_cpgs.csv', index=False)
    
    # Create plots
    print(f"  Creating plots...")
    plot_model_performance(results_df, species, output_dir)
    if not cpg_summary_df.empty:
        plot_top_predictive_cpgs(cpg_summary_df, species, output_dir)
    
    # Print top predictable exons
    print(f"\n  Top 10 Most Predictable Exons:")
    top_exons = results_df.nlargest(10, 'best_r2_cv')
    print(top_exons[['exon_id', 'best_r2_cv', 'best_model', 'n_cpgs_selected']].to_string(index=False))
    
    return results_df, cpg_summary_df, stats_dict


def main():
    """Main function to run the analysis."""
    print("="*70)
    print("Predictive Modeling: CpG Methylation → Exon Expression")
    print("="*70)
    print(f"Using {get_n_cpus()} CPUs for parallel processing")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each species
    all_results = []
    all_cpg_summaries = []
    all_stats = []
    
    for species in ['apul', 'peve', 'ptua']:
        results_df, cpg_df, stats_dict = process_species(
            species=species,
            exon_url=EXON_URLS[species],
            cpg_url=CPG_URLS[species],
            output_dir=OUTPUT_DIR
        )
        
        if not results_df.empty:
            all_results.append(results_df)
        if not cpg_df.empty:
            all_cpg_summaries.append(cpg_df)
        if stats_dict:
            all_stats.append(stats_dict)
    
    # Cross-species comparison
    print("\n" + "="*60)
    print("Cross-Species Comparison")
    print("="*60)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(OUTPUT_DIR / 'all_species_model_results.csv', index=False)
        plot_cross_species_comparison(combined_results, OUTPUT_DIR)
    
    if all_cpg_summaries:
        combined_cpgs = pd.concat(all_cpg_summaries, ignore_index=True)
        combined_cpgs.to_csv(OUTPUT_DIR / 'all_species_predictive_cpgs.csv', index=False)
    
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df.to_csv(OUTPUT_DIR / 'cross_species_summary.csv', index=False)
        print("\nCross-species summary:")
        print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
