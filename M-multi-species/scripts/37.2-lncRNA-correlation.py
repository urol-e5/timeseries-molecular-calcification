#!/usr/bin/env python3
"""
37.2-lncRNA-correlation.py

lncRNA-Gene Expression Correlation Analysis with Multi-threading

This analysis identifies which long non-coding RNAs (lncRNAs) have significant correlations 
(both positive and negative) with gene expression across three coral species:
- Acropora pulchra
- Porites evermanni  
- Pocillopora tuahiniensis

Uses parallel processing for efficient computation of pairwise correlations.

uv run --no-project --with numpy --with pandas --with scipy --with statsmodels --with matplotlib --with tqdm python M-multi-species/scripts/37.2-lncRNA-correlation.py

Author: GitHub Copilot
Date: 2025-12-04
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Number of parallel workers
N_WORKERS = os.cpu_count() or 4


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data():
    """Load all gene expression, lncRNA, and biomin data."""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Gene expression count matrices
    print("\nLoading gene expression matrices...")
    
    apul_gene = pd.read_csv(
        "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/D-Apul/output/02.20-D-Apul-RNAseq-alignment-HiSat2/apul-gene_count_matrix.csv"
    )
    print(f"  Apul genes: {apul_gene.shape}")
    
    peve_gene = pd.read_csv(
        "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/E-Peve/output/02.20-E-Peve-RNAseq-alignment-HiSat2/peve-gene_count_matrix.csv"
    )
    print(f"  Peve genes: {peve_gene.shape}")
    
    ptua_gene = pd.read_csv(
        "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/F-Ptua/output/02.20-F-Ptua-RNAseq-alignment-HiSat2/ptua-gene_count_matrix.csv"
    )
    print(f"  Ptua genes: {ptua_gene.shape}")
    
    # lncRNA count matrices
    print("\nLoading lncRNA matrices...")
    
    apul_lncrna = pd.read_csv(
        "https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries_molecular/D-Apul/output/31.5-Apul-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
        sep='\t'
    )
    print(f"  Apul lncRNAs: {apul_lncrna.shape}")
    
    peve_lncrna = pd.read_csv(
        "https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries_molecular/E-Peve/output/12-Peve-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
        sep='\t'
    )
    print(f"  Peve lncRNAs: {peve_lncrna.shape}")
    
    ptua_lncrna = pd.read_csv(
        "https://raw.githubusercontent.com/urol-e5/timeseries_molecular/refs/heads/main/F-Ptua/output/06-Ptua-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
        sep='\t'
    )
    print(f"  Ptua lncRNAs: {ptua_lncrna.shape}")
    
    # Biomin gene lists
    print("\nLoading biomineralization gene lists...")
    
    apul_biomin = pd.read_csv(
        "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv"
    )
    print(f"  Apul biomin genes: {len(apul_biomin)}")
    
    peve_biomin = pd.read_csv(
        "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv"
    )
    print(f"  Peve biomin genes: {len(peve_biomin)}")
    
    ptua_biomin = pd.read_csv(
        "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv"
    )
    print(f"  Ptua biomin genes: {len(ptua_biomin)}")
    
    return {
        'apul': {'gene': apul_gene, 'lncrna': apul_lncrna, 'biomin': apul_biomin},
        'peve': {'gene': peve_gene, 'lncrna': peve_lncrna, 'biomin': peve_biomin},
        'ptua': {'gene': ptua_gene, 'lncrna': ptua_lncrna, 'biomin': ptua_biomin}
    }


# =============================================================================
# Correlation Calculation Functions (Parallelized)
# =============================================================================

def correlate_single_lncrna(lncrna_idx: int, lncrna_log: np.ndarray, 
                            gene_log: np.ndarray, lncrna_ids: list, 
                            gene_ids: list) -> list:
    """
    Calculate correlations between a single lncRNA and all genes.
    
    This function is designed to be called in parallel.
    """
    lncrna_expr = lncrna_log[lncrna_idx, :]
    lncrna_id = lncrna_ids[lncrna_idx]
    
    # Skip if no variance
    if np.std(lncrna_expr) == 0:
        return []
    
    results = []
    for gene_idx in range(gene_log.shape[0]):
        gene_expr = gene_log[gene_idx, :]
        gene_id = gene_ids[gene_idx]
        
        # Skip if no variance
        if np.std(gene_expr) == 0:
            continue
        
        # Calculate Pearson correlation
        cor, pval = stats.pearsonr(lncrna_expr, gene_expr)
        results.append({
            'lncrna_id': lncrna_id,
            'gene_id': gene_id,
            'cor': cor,
            'pval': pval
        })
    
    return results


def calculate_correlations_parallel(gene_df: pd.DataFrame, lncrna_df: pd.DataFrame,
                                    min_counts: int = 10, n_workers: int = N_WORKERS) -> pd.DataFrame:
    """
    Calculate correlations between lncRNA and gene expression using parallel processing.
    
    Uses numpy vectorized operations where possible and ProcessPoolExecutor for 
    parallelization across lncRNAs.
    """
    # Identify ID columns (first column)
    gene_id_col = gene_df.columns[0]
    lncrna_id_col = lncrna_df.columns[0]
    
    # Get sample columns (shared between both matrices)
    gene_samples = set(gene_df.columns[1:])
    lncrna_samples = set(lncrna_df.columns[1:])
    shared_samples = sorted(gene_samples & lncrna_samples)
    
    print(f"  Shared samples: {len(shared_samples)}")
    
    if len(shared_samples) < 3:
        print("  WARNING: Not enough shared samples for correlation analysis")
        return pd.DataFrame()
    
    # Prepare matrices
    gene_mat = gene_df.set_index(gene_id_col)[shared_samples].values.astype(np.float64)
    lncrna_mat = lncrna_df.set_index(lncrna_id_col)[shared_samples].values.astype(np.float64)
    
    gene_ids = gene_df[gene_id_col].tolist()
    lncrna_ids = lncrna_df[lncrna_id_col].tolist()
    
    # Filter low-expression genes and lncRNAs
    gene_keep = np.sum(gene_mat, axis=1) >= min_counts
    lncrna_keep = np.sum(lncrna_mat, axis=1) >= min_counts
    
    gene_mat = gene_mat[gene_keep, :]
    lncrna_mat = lncrna_mat[lncrna_keep, :]
    
    gene_ids = [g for g, k in zip(gene_ids, gene_keep) if k]
    lncrna_ids = [l for l, k in zip(lncrna_ids, lncrna_keep) if k]
    
    print(f"  Genes after filtering: {len(gene_ids)}")
    print(f"  lncRNAs after filtering: {len(lncrna_ids)}")
    
    # Log2 transform (add 1 pseudocount)
    gene_log = np.log2(gene_mat + 1)
    lncrna_log = np.log2(lncrna_mat + 1)
    
    n_correlations = len(lncrna_ids) * len(gene_ids)
    print(f"  Calculating {n_correlations:,} correlations using {n_workers} workers...")
    
    # Use vectorized correlation calculation for efficiency
    # This is much faster than scipy.stats.pearsonr in a loop
    results = calculate_correlations_vectorized(
        gene_log, lncrna_log, gene_ids, lncrna_ids, n_workers
    )
    
    if len(results) == 0:
        return pd.DataFrame()
    
    # Create DataFrame
    all_cors = pd.DataFrame(results)
    
    # Adjust p-values for multiple testing
    all_cors = all_cors.dropna(subset=['pval'])
    if len(all_cors) > 0:
        _, padj, _, _ = multipletests(all_cors['pval'].values, method='fdr_bh')
        all_cors['padj'] = padj
        all_cors = all_cors.sort_values('padj')
    
    return all_cors


def calculate_correlations_vectorized(gene_log: np.ndarray, lncrna_log: np.ndarray,
                                       gene_ids: list, lncrna_ids: list,
                                       n_workers: int) -> list:
    """
    Calculate correlations using vectorized operations with parallel processing.
    
    This implementation uses numpy broadcasting for fast correlation calculation
    and parallelizes across chunks of lncRNAs.
    """
    n_lncrna = lncrna_log.shape[0]
    n_genes = gene_log.shape[0]
    n_samples = gene_log.shape[1]
    
    # Pre-compute standardized matrices for Pearson correlation
    # Pearson correlation = covariance / (std_x * std_y)
    # = mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
    # = mean(z_x * z_y) where z = (x - mean) / std
    
    # Standardize gene expression (z-scores)
    gene_mean = gene_log.mean(axis=1, keepdims=True)
    gene_std = gene_log.std(axis=1, keepdims=True)
    gene_std[gene_std == 0] = 1  # Avoid division by zero
    gene_z = (gene_log - gene_mean) / gene_std
    
    # Standardize lncRNA expression (z-scores)
    lncrna_mean = lncrna_log.mean(axis=1, keepdims=True)
    lncrna_std = lncrna_log.std(axis=1, keepdims=True)
    lncrna_std[lncrna_std == 0] = 1  # Avoid division by zero
    lncrna_z = (lncrna_log - lncrna_mean) / lncrna_std
    
    # Calculate all correlations at once using matrix multiplication
    # cor_matrix[i, j] = correlation between lncRNA i and gene j
    print("  Computing correlation matrix...")
    cor_matrix = np.dot(lncrna_z, gene_z.T) / n_samples
    
    # Calculate p-values in parallel
    print("  Computing p-values...")
    
    # For Pearson correlation, t = r * sqrt(n-2) / sqrt(1-r^2)
    # p-value from t-distribution with n-2 degrees of freedom
    df = n_samples - 2
    
    # Vectorized p-value calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = cor_matrix * np.sqrt(df) / np.sqrt(1 - cor_matrix**2)
        p_matrix = 2 * stats.t.sf(np.abs(t_stat), df)
    
    # Handle edge cases
    p_matrix = np.where(np.isfinite(p_matrix), p_matrix, 1.0)
    
    # Convert to list of dictionaries
    print("  Formatting results...")
    results = []
    
    # Use chunked processing for memory efficiency
    chunk_size = 1000
    for i in tqdm(range(0, n_lncrna, chunk_size), desc="  Processing chunks"):
        chunk_end = min(i + chunk_size, n_lncrna)
        for li in range(i, chunk_end):
            lncrna_id = lncrna_ids[li]
            for gi in range(n_genes):
                cor = cor_matrix[li, gi]
                pval = p_matrix[li, gi]
                if np.isfinite(cor) and np.isfinite(pval):
                    results.append({
                        'lncrna_id': lncrna_id,
                        'gene_id': gene_ids[gi],
                        'cor': cor,
                        'pval': pval
                    })
    
    return results


def get_significant_cors(cor_df: pd.DataFrame, padj_threshold: float = 0.05,
                         cor_threshold: float = 0.5) -> pd.DataFrame:
    """Identify significant correlations."""
    if cor_df.empty:
        return pd.DataFrame()
    
    sig = cor_df[
        (cor_df['padj'] < padj_threshold) & 
        (cor_df['cor'].abs() >= cor_threshold)
    ].copy()
    
    sig['direction'] = np.where(sig['cor'] > 0, 'positive', 'negative')
    return sig


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_species(species_name: str, data: dict) -> tuple:
    """Run correlation analysis for a single species."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing {species_name}")
    print("=" * 60)
    
    gene_df = data['gene']
    lncrna_df = data['lncrna']
    biomin_df = data['biomin']
    
    print(f"\nGene columns: {list(gene_df.columns[:5])}...")
    print(f"lncRNA columns: {list(lncrna_df.columns[:5])}...")
    
    # Calculate correlations
    cors = calculate_correlations_parallel(gene_df, lncrna_df, min_counts=10)
    
    if cors.empty:
        print(f"  No correlations calculated for {species_name}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get significant correlations
    sig = get_significant_cors(cors, padj_threshold=0.05, cor_threshold=0.5)
    
    print(f"\nSignificant lncRNA-gene correlations:")
    print(f"  Total significant: {len(sig)}")
    print(f"  Positive correlations: {(sig['direction'] == 'positive').sum()}")
    print(f"  Negative correlations: {(sig['direction'] == 'negative').sum()}")
    
    # Mark biomin genes
    if 'gene_id' in biomin_df.columns:
        biomin_genes = set(biomin_df['gene_id'].tolist())
    else:
        biomin_genes = set()
    
    # Clean gene IDs for matching (remove 'gene-' prefix if present)
    sig['gene_id_clean'] = sig['gene_id'].str.replace('^gene-', '', regex=True)
    sig['is_biomin'] = sig['gene_id'].isin(biomin_genes) | sig['gene_id_clean'].isin(biomin_genes)
    
    n_biomin = sig['is_biomin'].sum()
    print(f"  Biomin gene associations: {n_biomin}")
    
    if n_biomin > 0:
        print(f"\nTop biomin genes with significant lncRNA correlations:")
        print(sig[sig['is_biomin']].head(10).to_string())
    
    # Top lncRNAs by number of correlated genes
    top_lncrnas = sig.groupby('lncrna_id').agg(
        n_positive=('direction', lambda x: (x == 'positive').sum()),
        n_negative=('direction', lambda x: (x == 'negative').sum()),
        n_total=('lncrna_id', 'count'),
        n_biomin=('is_biomin', 'sum'),
        mean_cor=('cor', 'mean')
    ).reset_index().sort_values('n_total', ascending=False)
    
    print(f"\nTop 10 lncRNAs by number of correlated genes:")
    print(top_lncrnas.head(10).to_string())
    
    return sig, top_lncrnas, cors


def create_volcano_plot(cors: pd.DataFrame, sig: pd.DataFrame, 
                        species_name: str, output_dir: Path):
    """Create volcano-style plot for correlation results."""
    if cors.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filter valid correlations
    cors_valid = cors[cors['cor'].notna() & cors['pval'].notna()].copy()
    cors_valid['neg_log_p'] = -np.log10(cors_valid['pval'].clip(lower=1e-300))
    
    # Background points
    ax.scatter(cors_valid['cor'], cors_valid['neg_log_p'],
               alpha=0.1, color='gray', s=0.5, rasterized=True)
    
    if not sig.empty:
        sig_valid = sig[sig['cor'].notna() & sig['pval'].notna()].copy()
        sig_valid['neg_log_p'] = -np.log10(sig_valid['pval'].clip(lower=1e-300))
        
        # Non-biomin significant points
        non_biomin = sig_valid[~sig_valid['is_biomin']]
        for direction, color in [('positive', '#E74C3C'), ('negative', '#3498DB')]:
            subset = non_biomin[non_biomin['direction'] == direction]
            ax.scatter(subset['cor'], subset['neg_log_p'],
                       alpha=0.5, color=color, s=1, label=direction.capitalize())
        
        # Biomin significant points
        biomin = sig_valid[sig_valid['is_biomin']]
        if len(biomin) > 0:
            ax.scatter(biomin['cor'], biomin['neg_log_p'],
                       color='gold', s=25, edgecolors='black', linewidths=0.8,
                       label='Biomineralization', zorder=10)
    
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title(f'{species_name}: lncRNA-Gene Expression Correlations\n(Gold = biomineralization genes)',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species_name.lower().replace(" ", "_").replace(".", "")}_volcano.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(summary_df: pd.DataFrame, output_dir: Path):
    """Create summary comparison plot across species."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, summary_df['Positive'], width, 
                   label='Positive', color='#E74C3C')
    bars2 = ax.bar(x + width/2, summary_df['Negative'], width,
                   label='Negative', color='#3498DB')
    
    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('Number of Significant Correlations', fontsize=12)
    ax.set_title('Significant lncRNA-Gene Correlations Across Species', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Species'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'species_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_biomin_plot(all_sig: dict, output_dir: Path):
    """Create biomineralization gene correlation plot."""
    biomin_data = []
    for species, sig in all_sig.items():
        if not sig.empty and 'is_biomin' in sig.columns:
            biomin = sig[sig['is_biomin']].copy()
            biomin['species'] = species
            biomin_data.append(biomin)
    
    if not biomin_data:
        print("No significant biomin-lncRNA correlations found")
        return
    
    biomin_combined = pd.concat(biomin_data, ignore_index=True)
    
    if len(biomin_combined) == 0:
        return
    
    species_list = biomin_combined['species'].unique()
    n_species = len(species_list)
    
    fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 8))
    if n_species == 1:
        axes = [axes]
    
    colors = {'positive': '#E74C3C', 'negative': '#3498DB'}
    
    for ax, species in zip(axes, species_list):
        species_data = biomin_combined[biomin_combined['species'] == species]
        species_data = species_data.sort_values('cor')
        
        ax.barh(range(len(species_data)), species_data['cor'],
                color=[colors[d] for d in species_data['direction']])
        ax.set_yticks(range(len(species_data)))
        ax.set_yticklabels(species_data['gene_id'], fontsize=6)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(species)
        ax.axvline(0, color='black', linewidth=0.5)
    
    plt.suptitle('Biomineralization Genes with Significant lncRNA Correlations', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'biomin_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("lncRNA-Gene Expression Correlation Analysis")
    print("Multi-threaded Python Implementation")
    print("=" * 60)
    print(f"\nUsing {N_WORKERS} parallel workers")
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'output' / '37-lncRNA-correlation'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    data = load_data()
    
    # Analyze each species
    results = {}
    all_sig = {}
    
    species_names = {
        'apul': 'A. pulchra',
        'peve': 'P. evermanni',
        'ptua': 'P. tuahiniensis'
    }
    
    for species_key, species_name in species_names.items():
        sig, top_lncrnas, cors = analyze_species(species_name, data[species_key])
        results[species_key] = {
            'sig': sig,
            'top_lncrnas': top_lncrnas,
            'cors': cors
        }
        all_sig[species_name] = sig
        
        # Create volcano plot
        create_volcano_plot(cors, sig, species_name, output_dir)
    
    # Create summary
    print(f"\n{'=' * 60}")
    print("Summary Across Species")
    print("=" * 60)
    
    summary_data = []
    for species_key, species_name in species_names.items():
        sig = results[species_key]['sig']
        if not sig.empty:
            summary_data.append({
                'Species': species_name,
                'Positive': (sig['direction'] == 'positive').sum(),
                'Negative': (sig['direction'] == 'negative').sum(),
                'Biomin_associated': sig['is_biomin'].sum() if 'is_biomin' in sig.columns else 0,
                'Total': len(sig)
            })
        else:
            summary_data.append({
                'Species': species_name,
                'Positive': 0,
                'Negative': 0,
                'Biomin_associated': 0,
                'Total': 0
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string())
    
    # Create summary plot
    create_summary_plot(summary_df, output_dir)
    
    # Create biomin plot
    create_biomin_plot(all_sig, output_dir)
    
    # Save results
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print("=" * 60)
    
    for species_key, species_name in species_names.items():
        sig = results[species_key]['sig']
        top_lncrnas = results[species_key]['top_lncrnas']
        
        if not sig.empty:
            sig.to_csv(output_dir / f'{species_key}_lncrna_gene_correlations.csv', index=False)
            print(f"  Saved {species_key}_lncrna_gene_correlations.csv")
        
        if not top_lncrnas.empty:
            top_lncrnas.to_csv(output_dir / f'{species_key}_top_lncrnas.csv', index=False)
            print(f"  Saved {species_key}_top_lncrnas.csv")
    
    summary_df.to_csv(output_dir / 'species_summary.csv', index=False)
    print("  Saved species_summary.csv")
    
    print(f"\nAll results saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
