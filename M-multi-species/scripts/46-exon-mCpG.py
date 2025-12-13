#!/usr/bin/env python3
"""
46-exon-mCpG.py

Analyze the relationship between local CpG methylation and exon-level expression.
For each exon, identify CpG sites that fall within the exon coordinates and compute
correlations between methylation levels and exon expression across samples.

PARALLELIZED VERSION - Uses all available CPUs for correlation computation.

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
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / '46-exon-mCpG'

# Minimum samples required for correlation
MIN_SAMPLES = 5

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


def find_cpgs_in_exons(exon_coords: pd.DataFrame, cpg_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Find CpG sites that fall within exon coordinates.
    Uses chunked merge for memory efficiency.
    
    Parameters
    ----------
    exon_coords : pd.DataFrame
        DataFrame with columns: gene_id, e_id, chr, start, end
    cpg_coords : pd.DataFrame
        DataFrame with columns: cpg_id, cpg_chr, cpg_pos
        
    Returns
    -------
    pd.DataFrame
        Mapping of exons to CpGs within them
    """
    # Get unique chromosomes present in both datasets
    common_chrs = set(exon_coords['chr'].unique()) & set(cpg_coords['cpg_chr'].unique())
    
    results = []
    
    for chrom in common_chrs:
        # Filter to this chromosome
        exon_chr = exon_coords[exon_coords['chr'] == chrom]
        cpg_chr = cpg_coords[cpg_coords['cpg_chr'] == chrom]
        
        if exon_chr.empty or cpg_chr.empty:
            continue
        
        # For each exon, find CpGs within bounds using numpy broadcasting
        exon_starts = exon_chr['start'].values
        exon_ends = exon_chr['end'].values
        cpg_positions = cpg_chr['cpg_pos'].values
        
        # Create mask for CpGs within each exon
        # This uses memory but is much faster than row-by-row iteration
        for i, (gene_id, e_id, start, end) in enumerate(
            zip(exon_chr['gene_id'], exon_chr['e_id'], exon_starts, exon_ends)
        ):
            mask = (cpg_positions >= start) & (cpg_positions <= end)
            matching_cpgs = cpg_chr[mask]
            
            if len(matching_cpgs) > 0:
                for _, cpg_row in matching_cpgs.iterrows():
                    results.append({
                        'gene_id': gene_id,
                        'e_id': e_id,
                        'chr': chrom,
                        'start': start,
                        'end': end,
                        'cpg_id': cpg_row['cpg_id'],
                        'cpg_pos': cpg_row['cpg_pos']
                    })
    
    if not results:
        return pd.DataFrame(columns=['gene_id', 'e_id', 'chr', 'start', 'end', 'cpg_id', 'cpg_pos'])
    
    return pd.DataFrame(results)


def compute_single_correlation(args: Tuple) -> Optional[Dict]:
    """
    Compute correlation for a single exon-CpG pair.
    Designed for parallel execution.
    
    Parameters
    ----------
    args : tuple
        (gene_id, e_id, chr, start, end, cpg_id, cpg_pos, exon_vals, cpg_vals)
        
    Returns
    -------
    dict or None
        Correlation result dictionary or None if computation fails
    """
    gene_id, e_id, chrom, start, end, cpg_id, cpg_pos, exon_vals, cpg_vals = args
    
    # Remove NA pairs
    valid_mask = ~(np.isnan(exon_vals) | np.isnan(cpg_vals))
    exon_clean = exon_vals[valid_mask]
    cpg_clean = cpg_vals[valid_mask]
    
    n_samples = len(exon_clean)
    
    if n_samples < MIN_SAMPLES:
        return None
    
    # Check for zero variance
    if np.std(exon_clean) == 0 or np.std(cpg_clean) == 0:
        return None
    
    try:
        corr, pval = stats.spearmanr(exon_clean, cpg_clean)
        if np.isnan(corr):
            return None
            
        return {
            'gene_id': gene_id,
            'e_id': e_id,
            'chr': chrom,
            'exon_start': start,
            'exon_end': end,
            'cpg_id': cpg_id,
            'cpg_pos': cpg_pos,
            'n_samples': n_samples,
            'correlation': corr,
            'p_value': pval
        }
    except Exception:
        return None


def prepare_correlation_args(
    exon_expr: pd.DataFrame,
    cpg_meth: pd.DataFrame,
    exon_cpg_map: pd.DataFrame,
    sample_cols: List[str]
) -> List[Tuple]:
    """
    Prepare arguments for parallel correlation computation.
    
    Parameters
    ----------
    exon_expr : pd.DataFrame
        Exon expression matrix
    cpg_meth : pd.DataFrame
        CpG methylation matrix  
    exon_cpg_map : pd.DataFrame
        Mapping of exons to CpGs
    sample_cols : list
        Common sample column names
        
    Returns
    -------
    list
        List of argument tuples for compute_single_correlation
    """
    # Create indexed lookups
    exon_expr = exon_expr.copy()
    exon_expr['exon_key'] = exon_expr['gene_id'].astype(str) + '_' + exon_expr['e_id'].astype(str)
    exon_expr_indexed = exon_expr.set_index('exon_key')
    cpg_meth_indexed = cpg_meth.set_index('CpG')
    
    # Get unique pairs
    pairs = exon_cpg_map.drop_duplicates(subset=['gene_id', 'e_id', 'cpg_id'])
    
    args_list = []
    
    for _, row in pairs.iterrows():
        exon_key = f"{row['gene_id']}_{row['e_id']}"
        cpg_id = row['cpg_id']
        
        # Check if data exists
        if exon_key not in exon_expr_indexed.index:
            continue
        if cpg_id not in cpg_meth_indexed.index:
            continue
        
        try:
            exon_vals = exon_expr_indexed.loc[exon_key, sample_cols].values.astype(float)
            cpg_vals = cpg_meth_indexed.loc[cpg_id, sample_cols].values.astype(float)
        except (KeyError, ValueError):
            continue
        
        args_list.append((
            row['gene_id'],
            row['e_id'],
            row['chr'],
            row['start'],
            row['end'],
            cpg_id,
            row['cpg_pos'],
            exon_vals,
            cpg_vals
        ))
    
    return args_list


def compute_correlations_parallel(args_list: List[Tuple], n_workers: int = None) -> pd.DataFrame:
    """
    Compute correlations in parallel using all available CPUs.
    
    Parameters
    ----------
    args_list : list
        List of argument tuples for compute_single_correlation
    n_workers : int, optional
        Number of worker processes (default: all CPUs)
        
    Returns
    -------
    pd.DataFrame
        Correlation results
    """
    if n_workers is None:
        n_workers = get_n_cpus()
    
    n_pairs = len(args_list)
    print(f"  Computing {n_pairs} correlations using {n_workers} CPUs...")
    
    if n_pairs == 0:
        return pd.DataFrame()
    
    # For small datasets, use single process to avoid overhead
    if n_pairs < 1000:
        results = [compute_single_correlation(args) for args in args_list]
        results = [r for r in results if r is not None]
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    # Use process pool for larger datasets
    results = []
    chunk_size = max(100, n_pairs // (n_workers * 10))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(compute_single_correlation, args): i 
                   for i, args in enumerate(args_list)}
        
        # Collect results with progress
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            completed += 1
            
            # Progress update every 10%
            if completed % max(1, n_pairs // 10) == 0:
                pct = 100 * completed / n_pairs
                print(f"    Progress: {completed}/{n_pairs} ({pct:.0f}%)")
    
    print(f"  Completed: {len(results)} valid correlations")
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def adjust_pvalues(pvalues: pd.Series) -> pd.Series:
    """
    Adjust p-values for multiple testing using Benjamini-Hochberg method.
    
    Parameters
    ----------
    pvalues : pd.Series
        Raw p-values
        
    Returns
    -------
    pd.Series
        Adjusted p-values
    """
    valid_mask = ~pvalues.isna()
    adjusted = pd.Series(np.nan, index=pvalues.index)
    
    if valid_mask.sum() == 0:
        return adjusted
    
    pvals = pvalues[valid_mask].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    
    # BH adjustment
    ranks = np.arange(1, n + 1)
    adj = sorted_pvals * n / ranks
    
    # Ensure monotonicity (cumulative minimum from end)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    
    # Put back in original order
    result = np.empty(n)
    result[sorted_idx] = adj
    adjusted.loc[valid_mask] = result
    
    return adjusted


def summarize_gene_correlations(corr_df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    Aggregate exon-level correlations to gene level.
    """
    if corr_df.empty:
        return pd.DataFrame()
    
    summary = corr_df.groupby('gene_id').agg(
        n_exons=('e_id', 'nunique'),
        n_cpgs=('cpg_id', 'nunique'),
        n_pairs=('correlation', 'count'),
        mean_correlation=('correlation', 'mean'),
        median_correlation=('correlation', 'median'),
        sd_correlation=('correlation', 'std'),
        n_sig_p05=('p_value', lambda x: (x < 0.05).sum()),
        n_sig_fdr10=('p_adj', lambda x: (x < 0.1).sum()),
        pct_positive=('correlation', lambda x: (x > 0).mean() * 100),
        min_p_adj=('p_adj', 'min')
    ).reset_index()
    
    summary = summary.sort_values('min_p_adj')
    summary['species'] = species
    
    return summary


def plot_correlation_distribution(corr_df: pd.DataFrame, species: str, output_dir: Path) -> None:
    """Create histogram and volcano plot of correlations."""
    if corr_df.empty:
        print(f"  No correlations to plot for {species}")
        return
    
    colors = {'apul': 'steelblue', 'peve': 'coral', 'ptua': 'seagreen'}
    color = colors.get(species.lower(), 'gray')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(corr_df['correlation'], bins=50, color=color, edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, linestyle='--', color='red', linewidth=1.5)
    ax1.set_xlabel('Spearman Correlation')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{species}: Distribution of Exon-CpG Correlations')
    
    # Volcano plot
    ax2 = axes[1]
    sig_mask = corr_df['p_adj'] < 0.1
    ax2.scatter(
        corr_df.loc[~sig_mask, 'correlation'],
        -np.log10(corr_df.loc[~sig_mask, 'p_value'].clip(lower=1e-300)),
        alpha=0.5, s=10, c='gray', label='Not significant'
    )
    if sig_mask.sum() > 0:
        ax2.scatter(
            corr_df.loc[sig_mask, 'correlation'],
            -np.log10(corr_df.loc[sig_mask, 'p_value'].clip(lower=1e-300)),
            alpha=0.7, s=15, c='red', label='FDR < 0.1'
        )
    ax2.axhline(y=-np.log10(0.05), linestyle='--', color='blue', linewidth=1)
    ax2.set_xlabel('Spearman Correlation')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title(f'{species}: Volcano Plot')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species.lower()}_exon_cpg_correlation_plots.png', dpi=150)
    plt.close()


def plot_cross_species_comparison(all_correlations: pd.DataFrame, output_dir: Path) -> None:
    """Create cross-species comparison plot."""
    if all_correlations.empty:
        return
    
    colors = {'Apul': 'steelblue', 'Peve': 'coral', 'Ptua': 'seagreen'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for species in all_correlations['species'].unique():
        species_data = all_correlations[all_correlations['species'] == species]
        sns.kdeplot(
            data=species_data,
            x='correlation',
            fill=True,
            alpha=0.4,
            color=colors.get(species, 'gray'),
            label=species,
            ax=ax
        )
    
    ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
    ax.set_xlabel('Spearman Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Exon-CpG Correlations Across Species')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_species_correlation_density.png', dpi=150)
    plt.close()


def process_species(species: str, exon_url: str, cpg_url: str, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Process a single species.
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
    
    # Parse CpG coordinates
    print(f"  Parsing CpG coordinates...")
    cpg_coords = parse_cpg_ids_vectorized(cpg_df['CpG'])
    
    # Show chromosome distribution
    chr_counts = cpg_coords['cpg_chr'].value_counts().head(5)
    print(f"  Top 5 chromosomes: {dict(chr_counts)}")
    
    # Extract exon coordinates
    exon_coords = exon_df[['gene_id', 'e_id', 'chr', 'start', 'end']].copy()
    
    # Find CpGs within exons
    print(f"  Finding CpGs within exons...")
    exon_cpg_map = find_cpgs_in_exons(exon_coords, cpg_coords)
    
    n_pairs = len(exon_cpg_map)
    print(f"  Found {n_pairs} exon-CpG pairs")
    
    if exon_cpg_map.empty:
        print(f"  WARNING: No CpGs found within exons")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    print(f"  Unique exons with CpGs: {exon_cpg_map[['gene_id', 'e_id']].drop_duplicates().shape[0]}")
    print(f"  Unique CpGs in exons: {exon_cpg_map['cpg_id'].nunique()}")
    
    # Identify common samples
    exon_sample_cols = [c for c in exon_df.columns if c not in ['gene_id', 'e_id', 'chr', 'strand', 'start', 'end']]
    cpg_sample_cols = [c for c in cpg_df.columns if c != 'CpG']
    common_samples = list(set(exon_sample_cols) & set(cpg_sample_cols))
    
    print(f"  Exon samples: {len(exon_sample_cols)}")
    print(f"  CpG samples: {len(cpg_sample_cols)}")
    print(f"  Common samples: {len(common_samples)}")
    
    if len(common_samples) < MIN_SAMPLES:
        print(f"  WARNING: Not enough common samples")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Prepare arguments for parallel computation
    print(f"  Preparing data for parallel processing...")
    args_list = prepare_correlation_args(exon_df, cpg_df, exon_cpg_map, common_samples)
    
    # Compute correlations in parallel
    correlations = compute_correlations_parallel(args_list, n_workers=get_n_cpus())
    
    print(f"  Total valid correlations: {len(correlations)}")
    
    if correlations.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Adjust p-values
    correlations['p_adj'] = adjust_pvalues(correlations['p_value'])
    
    # Summary statistics
    stats = {
        'species': species_upper,
        'total_exons': len(exon_df),
        'total_cpgs': len(cpg_df),
        'exon_cpg_pairs': n_pairs,
        'correlations_tested': len(correlations),
        'sig_p05': int((correlations['p_value'] < 0.05).sum()),
        'sig_fdr10': int((correlations['p_adj'] < 0.1).sum()),
        'mean_correlation': float(correlations['correlation'].mean()),
        'positive_correlations': int((correlations['correlation'] > 0).sum()),
        'negative_correlations': int((correlations['correlation'] < 0).sum()),
    }
    
    print(f"\n  === {species_upper} Summary ===")
    print(f"  Correlations tested: {stats['correlations_tested']}")
    print(f"  Significant (p < 0.05): {stats['sig_p05']}")
    print(f"  Significant (FDR < 0.1): {stats['sig_fdr10']}")
    print(f"  Positive: {stats['positive_correlations']}, Negative: {stats['negative_correlations']}")
    print(f"  Mean correlation: {stats['mean_correlation']:.4f}")
    
    # Save results
    correlations.to_csv(output_dir / f'{species.lower()}_exon_cpg_correlations.csv', index=False)
    
    # Create plots
    print(f"  Creating plots...")
    plot_correlation_distribution(correlations, species_upper, output_dir)
    
    # Top correlations
    print(f"\n  Top 10 Positive Correlations:")
    top_pos = correlations[correlations['correlation'] > 0].nsmallest(10, 'p_adj')
    if not top_pos.empty:
        print(top_pos[['gene_id', 'e_id', 'cpg_id', 'correlation', 'p_adj']].to_string(index=False))
    
    print(f"\n  Top 10 Negative Correlations:")
    top_neg = correlations[correlations['correlation'] < 0].nsmallest(10, 'p_adj')
    if not top_neg.empty:
        print(top_neg[['gene_id', 'e_id', 'cpg_id', 'correlation', 'p_adj']].to_string(index=False))
    
    # Gene-level summary
    gene_summary = summarize_gene_correlations(correlations, species_upper)
    if not gene_summary.empty:
        gene_summary.to_csv(output_dir / f'{species.lower()}_gene_level_summary.csv', index=False)
    
    correlations['species'] = species_upper
    
    return correlations, gene_summary, stats


def main():
    """Main function to run the analysis."""
    print("="*70)
    print("Exon-Level Expression and CpG Methylation Correlation Analysis")
    print("="*70)
    print(f"Using {get_n_cpus()} CPUs for parallel processing")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each species
    all_correlations = []
    all_gene_summaries = []
    all_stats = []
    
    for species in ['apul', 'peve', 'ptua']:
        corr_df, gene_df, stats = process_species(
            species=species,
            exon_url=EXON_URLS[species],
            cpg_url=CPG_URLS[species],
            output_dir=OUTPUT_DIR
        )
        
        if not corr_df.empty:
            all_correlations.append(corr_df)
        if not gene_df.empty:
            all_gene_summaries.append(gene_df)
        if stats:
            all_stats.append(stats)
    
    # Cross-species comparison
    print("\n" + "="*60)
    print("Cross-Species Comparison")
    print("="*60)
    
    if all_correlations:
        combined_corr = pd.concat(all_correlations, ignore_index=True)
        combined_corr.to_csv(OUTPUT_DIR / 'all_species_exon_cpg_correlations.csv', index=False)
        plot_cross_species_comparison(combined_corr, OUTPUT_DIR)
    
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df.to_csv(OUTPUT_DIR / 'cross_species_summary.csv', index=False)
        print("\nCross-species summary:")
        print(summary_df.to_string(index=False))
    
    if all_gene_summaries:
        combined_genes = pd.concat(all_gene_summaries, ignore_index=True)
        combined_genes.to_csv(OUTPUT_DIR / 'all_species_gene_level_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
