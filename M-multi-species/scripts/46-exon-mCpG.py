#!/usr/bin/env python3
"""
46-exon-mCpG.py

Analyze the relationship between local CpG methylation and exon-level expression.
For each exon, identify CpG sites that fall within the exon coordinates and compute
correlations between methylation levels and exon expression across samples.

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

CpG ID Format:
- Apul: CpG_ntLink_0_90500 → chr=ntLink_0, pos=90500
- Peve: CpG_Porites_evermani_scaffold_1000_100536 → chr=Porites_evermani_scaffold_1000, pos=100536
- Ptua: CpG_Pocillopora_meandrina_HIv1___Sc0000000_1000005 → chr=..., pos=1000005
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# =============================================================================
# Helper Functions
# =============================================================================

def parse_cpg_id(cpg_id: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a CpG ID to extract chromosome and position.
    
    CpG ID format: CpG_{chromosome}_{position}
    where position is the last underscore-separated numeric segment.
    
    Examples:
    - CpG_ntLink_0_90500 → ('ntLink_0', 90500)
    - CpG_Porites_evermani_scaffold_1000_100536 → ('Porites_evermani_scaffold_1000', 100536)
    - CpG_Pocillopora_meandrina_HIv1___Sc0000000_1000005 → ('Pocillopora_meandrina_HIv1___Sc0000000', 1000005)
    
    Parameters
    ----------
    cpg_id : str
        The CpG identifier string
        
    Returns
    -------
    tuple
        (chromosome, position) or (None, None) if parsing fails
    """
    # Remove "CpG_" prefix
    stripped = cpg_id.replace('CpG_', '', 1) if cpg_id.startswith('CpG_') else cpg_id
    
    # Match pattern: everything before last underscore = chromosome, after = position
    match = re.match(r'^(.+)_(\d+)$', stripped)
    
    if match:
        return match.group(1), int(match.group(2))
    return None, None


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
    # Merge on chromosome
    merged = exon_coords.merge(
        cpg_coords,
        left_on='chr',
        right_on='cpg_chr',
        how='inner'
    )
    
    # Filter to CpGs within exon boundaries
    within_exon = merged[
        (merged['cpg_pos'] >= merged['start']) & 
        (merged['cpg_pos'] <= merged['end'])
    ]
    
    return within_exon[['gene_id', 'e_id', 'chr', 'start', 'end', 'cpg_id', 'cpg_pos']].copy()


def compute_correlation(exon_vals: np.ndarray, cpg_vals: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman correlation between exon expression and CpG methylation.
    
    Parameters
    ----------
    exon_vals : np.ndarray
        Exon expression values
    cpg_vals : np.ndarray
        CpG methylation values
        
    Returns
    -------
    tuple
        (correlation coefficient, p-value)
    """
    # Remove NA pairs
    valid_mask = ~(np.isnan(exon_vals) | np.isnan(cpg_vals))
    exon_clean = exon_vals[valid_mask]
    cpg_clean = cpg_vals[valid_mask]
    
    if len(exon_clean) < MIN_SAMPLES:
        return np.nan, np.nan
    
    # Check for zero variance
    if np.std(exon_clean) == 0 or np.std(cpg_clean) == 0:
        return np.nan, np.nan
    
    try:
        corr, pval = stats.spearmanr(exon_clean, cpg_clean)
        return corr, pval
    except Exception:
        return np.nan, np.nan


def compute_exon_cpg_correlations(
    exon_expr: pd.DataFrame,
    cpg_meth: pd.DataFrame,
    exon_cpg_map: pd.DataFrame,
    sample_cols: List[str]
) -> pd.DataFrame:
    """
    Compute correlations between exon expression and CpG methylation.
    
    Parameters
    ----------
    exon_expr : pd.DataFrame
        Exon expression matrix with sample columns
    cpg_meth : pd.DataFrame
        CpG methylation matrix with sample columns
    exon_cpg_map : pd.DataFrame
        Mapping of exons to CpGs within them
    sample_cols : list
        Sample column names present in both datasets
        
    Returns
    -------
    pd.DataFrame
        Correlation results for each exon-CpG pair
    """
    results = []
    
    # Create unique exon key for faster lookup
    exon_expr = exon_expr.copy()
    exon_expr['exon_key'] = exon_expr['gene_id'].astype(str) + '_' + exon_expr['e_id'].astype(str)
    exon_expr_indexed = exon_expr.set_index('exon_key')
    
    # Index CpG data by CpG ID
    cpg_meth_indexed = cpg_meth.set_index('CpG')
    
    # Get unique exon-CpG pairs
    pairs = exon_cpg_map.drop_duplicates(subset=['gene_id', 'e_id', 'cpg_id'])
    
    print(f"  Computing correlations for {len(pairs)} exon-CpG pairs...")
    
    for idx, row in pairs.iterrows():
        exon_key = f"{row['gene_id']}_{row['e_id']}"
        cpg_id = row['cpg_id']
        
        # Get exon expression values
        if exon_key not in exon_expr_indexed.index:
            continue
        exon_row = exon_expr_indexed.loc[exon_key]
        
        # Get CpG methylation values
        if cpg_id not in cpg_meth_indexed.index:
            continue
        cpg_row = cpg_meth_indexed.loc[cpg_id]
        
        # Extract values for common samples
        try:
            exon_vals = exon_row[sample_cols].values.astype(float)
            cpg_vals = cpg_row[sample_cols].values.astype(float)
        except KeyError:
            continue
        
        # Compute correlation
        corr, pval = compute_correlation(exon_vals, cpg_vals)
        
        if not np.isnan(corr):
            # Count valid samples
            valid_mask = ~(np.isnan(exon_vals) | np.isnan(cpg_vals))
            n_samples = np.sum(valid_mask)
            
            results.append({
                'gene_id': row['gene_id'],
                'e_id': row['e_id'],
                'chr': row['chr'],
                'exon_start': row['start'],
                'exon_end': row['end'],
                'cpg_id': cpg_id,
                'cpg_pos': row['cpg_pos'],
                'n_samples': n_samples,
                'correlation': corr,
                'p_value': pval
            })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def adjust_pvalues(pvalues: pd.Series, method: str = 'fdr_bh') -> pd.Series:
    """
    Adjust p-values for multiple testing using Benjamini-Hochberg method.
    
    Parameters
    ----------
    pvalues : pd.Series
        Raw p-values
    method : str
        Correction method (default: 'fdr_bh')
        
    Returns
    -------
    pd.Series
        Adjusted p-values
    """
    from scipy.stats import false_discovery_control
    
    valid_mask = ~pvalues.isna()
    adjusted = pd.Series(np.nan, index=pvalues.index)
    
    if valid_mask.sum() > 0:
        # Use scipy's FDR control
        try:
            adj_vals = false_discovery_control(pvalues[valid_mask].values, method='bh')
            adjusted.loc[valid_mask] = adj_vals
        except Exception:
            # Fallback: simple BH correction
            pvals = pvalues[valid_mask].values
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_pvals = pvals[sorted_idx]
            
            # BH adjustment
            adj = sorted_pvals * n / (np.arange(n) + 1)
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
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        Exon-CpG correlation results
    species : str
        Species name
        
    Returns
    -------
    pd.DataFrame
        Gene-level summary
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


def plot_correlation_distribution(
    corr_df: pd.DataFrame,
    species: str,
    output_dir: Path
) -> None:
    """
    Create histogram and volcano plot of correlations.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results
    species : str
        Species name
    output_dir : Path
        Output directory for plots
    """
    if corr_df.empty:
        print(f"  No correlations to plot for {species}")
        return
    
    colors = {'apul': 'steelblue', 'peve': 'coral', 'ptua': 'seagreen'}
    color = colors.get(species.lower(), 'gray')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of correlations
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
        -np.log10(corr_df.loc[~sig_mask, 'p_value']),
        alpha=0.5, s=10, c='gray', label='Not significant'
    )
    ax2.scatter(
        corr_df.loc[sig_mask, 'correlation'],
        -np.log10(corr_df.loc[sig_mask, 'p_value']),
        alpha=0.7, s=15, c='red', label='FDR < 0.1'
    )
    ax2.axhline(y=-np.log10(0.05), linestyle='--', color='blue', linewidth=1)
    ax2.set_xlabel('Spearman Correlation')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title(f'{species}: Volcano Plot of Exon-CpG Correlations')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species.lower()}_exon_cpg_correlation_plots.png', dpi=150)
    plt.close()


def plot_cross_species_comparison(
    all_correlations: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create cross-species comparison plot.
    
    Parameters
    ----------
    all_correlations : pd.DataFrame
        Combined correlations from all species
    output_dir : Path
        Output directory for plots
    """
    if all_correlations.empty:
        print("No correlations to plot for cross-species comparison")
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


def process_species(
    species: str,
    exon_url: str,
    cpg_url: str,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Process a single species: load data, find CpGs in exons, compute correlations.
    
    Parameters
    ----------
    species : str
        Species name (e.g., 'apul', 'peve', 'ptua')
    exon_url : str
        URL to exon expression matrix
    cpg_url : str
        URL to CpG methylation matrix
    output_dir : Path
        Output directory
        
    Returns
    -------
    tuple
        (correlation_df, gene_summary_df, stats_dict)
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
    chr_counts = cpg_coords['cpg_chr'].value_counts().head(10)
    print(f"  Top chromosomes with CpGs: {dict(chr_counts)}")
    
    # Extract exon coordinates
    exon_coords = exon_df[['gene_id', 'e_id', 'chr', 'start', 'end']].copy()
    
    # Find CpGs within exons
    print(f"  Finding CpGs within exons...")
    exon_cpg_map = find_cpgs_in_exons(exon_coords, cpg_coords)
    
    print(f"  Found {len(exon_cpg_map)} exon-CpG pairs")
    print(f"  Unique exons with CpGs: {exon_cpg_map[['gene_id', 'e_id']].drop_duplicates().shape[0]}")
    print(f"  Unique CpGs in exons: {exon_cpg_map['cpg_id'].nunique()}")
    
    if exon_cpg_map.empty:
        print(f"  WARNING: No CpGs found within exons for {species_upper}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # CpGs per exon summary
    cpgs_per_exon = exon_cpg_map.groupby(['gene_id', 'e_id']).size()
    print(f"  CpGs per exon - mean: {cpgs_per_exon.mean():.2f}, median: {cpgs_per_exon.median():.1f}, max: {cpgs_per_exon.max()}")
    
    # Identify common samples
    exon_sample_cols = [c for c in exon_df.columns if c not in ['gene_id', 'e_id', 'chr', 'strand', 'start', 'end']]
    cpg_sample_cols = [c for c in cpg_df.columns if c != 'CpG']
    common_samples = list(set(exon_sample_cols) & set(cpg_sample_cols))
    
    print(f"  Exon samples: {len(exon_sample_cols)}")
    print(f"  CpG samples: {len(cpg_sample_cols)}")
    print(f"  Common samples: {len(common_samples)}")
    
    if len(common_samples) < MIN_SAMPLES:
        print(f"  WARNING: Not enough common samples for {species_upper}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Compute correlations
    correlations = compute_exon_cpg_correlations(
        exon_expr=exon_df,
        cpg_meth=cpg_df,
        exon_cpg_map=exon_cpg_map,
        sample_cols=common_samples
    )
    
    print(f"  Computed {len(correlations)} correlations")
    
    if correlations.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Adjust p-values
    correlations['p_adj'] = adjust_pvalues(correlations['p_value'])
    
    # Summary statistics
    stats = {
        'species': species_upper,
        'total_exons': len(exon_df),
        'total_cpgs': len(cpg_df),
        'exon_cpg_pairs': len(exon_cpg_map),
        'correlations_tested': len(correlations),
        'sig_p05': (correlations['p_value'] < 0.05).sum(),
        'sig_fdr10': (correlations['p_adj'] < 0.1).sum(),
        'mean_correlation': correlations['correlation'].mean(),
        'positive_correlations': (correlations['correlation'] > 0).sum(),
        'negative_correlations': (correlations['correlation'] < 0).sum(),
    }
    
    print(f"\n  === {species_upper} Correlation Summary ===")
    print(f"  Total exon-CpG pairs tested: {stats['correlations_tested']}")
    print(f"  Significant (p < 0.05): {stats['sig_p05']}")
    print(f"  Significant (FDR < 0.1): {stats['sig_fdr10']}")
    print(f"  Positive correlations: {stats['positive_correlations']}")
    print(f"  Negative correlations: {stats['negative_correlations']}")
    print(f"  Mean correlation: {stats['mean_correlation']:.4f}")
    
    # Correlation summary
    print(f"\n  Correlation distribution:")
    print(f"    {correlations['correlation'].describe()}")
    
    # Save results
    correlations.to_csv(output_dir / f'{species.lower()}_exon_cpg_correlations.csv', index=False)
    
    # Create plots
    print(f"  Creating plots...")
    plot_correlation_distribution(correlations, species_upper, output_dir)
    
    # Top correlations
    print(f"\n  Top 10 Positive Correlations:")
    top_pos = correlations[correlations['correlation'] > 0].nsmallest(10, 'p_adj')
    print(top_pos[['gene_id', 'e_id', 'cpg_id', 'correlation', 'p_value', 'p_adj']].to_string(index=False))
    
    print(f"\n  Top 10 Negative Correlations:")
    top_neg = correlations[correlations['correlation'] < 0].nsmallest(10, 'p_adj')
    print(top_neg[['gene_id', 'e_id', 'cpg_id', 'correlation', 'p_value', 'p_adj']].to_string(index=False))
    
    # Gene-level summary
    gene_summary = summarize_gene_correlations(correlations, species_upper)
    if not gene_summary.empty:
        gene_summary.to_csv(output_dir / f'{species.lower()}_gene_level_correlation_summary.csv', index=False)
        print(f"\n  Top 10 genes by significance:")
        print(gene_summary.head(10).to_string(index=False))
    
    correlations['species'] = species_upper
    
    return correlations, gene_summary, stats


def main():
    """Main function to run the exon-mCpG correlation analysis."""
    print("="*70)
    print("Exon-Level Expression and CpG Methylation Correlation Analysis")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
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
        
        # Cross-species plot
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
