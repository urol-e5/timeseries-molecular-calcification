#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "statsmodels",
#     "joblib",
#     "matplotlib",
#     "seaborn",
#     "tqdm",
# ]
# ///
"""
38.2-lncRNA-gbm-reg.py - lncRNA and Gene Body Methylation Regulation Analysis

This analysis investigates how long non-coding RNAs (lncRNAs) and gene body methylation (GBM) 
potentially regulate gene expression across three coral species:
- Acropora pulchra
- Porites evermanni  
- Pocillopora tuahiniensis

Optimized for parallel processing using all available CPUs.

Author: GitHub Copilot
Date: 2025-12-04
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up parallel processing
N_JOBS = cpu_count()
print(f"Using {N_JOBS} CPU cores for parallel processing")

# Output directory
OUTPUT_DIR = Path("../output/38-lncRNA-gbm-reg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_gene_expression() -> Dict[str, pd.DataFrame]:
    """Load gene expression count matrices for all species."""
    urls = {
        'apul': "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/D-Apul/output/02.20-D-Apul-RNAseq-alignment-HiSat2/apul-gene_count_matrix.csv",
        'peve': "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/E-Peve/output/02.20-E-Peve-RNAseq-alignment-HiSat2/peve-gene_count_matrix.csv",
        'ptua': "https://gannet.fish.washington.edu/gitrepos/urol-e5/timeseries_molecular/F-Ptua/output/02.20-F-Ptua-RNAseq-alignment-HiSat2/ptua-gene_count_matrix.csv",
    }
    
    data = {}
    for species, url in urls.items():
        print(f"Loading {species} gene expression...")
        data[species] = pd.read_csv(url)
        print(f"  Shape: {data[species].shape}")
    
    return data


def load_lncrna() -> Dict[str, pd.DataFrame]:
    """Load lncRNA count matrices for all species."""
    urls = {
        'apul': "https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries_molecular/D-Apul/output/31.5-Apul-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
        'peve': "https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/timeseries_molecular/E-Peve/output/12-Peve-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
        'ptua': "https://raw.githubusercontent.com/urol-e5/timeseries_molecular/refs/heads/main/F-Ptua/output/06-Ptua-lncRNA-discovery/lncRNA_counts.clean.filtered.txt",
    }
    
    data = {}
    for species, url in urls.items():
        print(f"Loading {species} lncRNA...")
        data[species] = pd.read_csv(url, sep='\t')
        print(f"  Shape: {data[species].shape}")
    
    return data


def load_gbm() -> Dict[str, pd.DataFrame]:
    """Load gene body methylation data for all species."""
    urls = {
        'apul': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/D-Apul/output/40-Apul-Gene-Methylation/Apul-gene-methylation_75pct.tsv",
        'peve': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/E-Peve/output/15-Peve-Gene-Methylation/Peve-gene-methylation_75pct.tsv",
        'ptua': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/F-Ptua/output/09-Ptua-Gene-Methylation/Ptua-gene-methylation_75pct.tsv",
    }
    
    data = {}
    for species, url in urls.items():
        print(f"Loading {species} GBM...")
        data[species] = pd.read_csv(url, sep='\t')
        print(f"  Shape: {data[species].shape}")
    
    return data


def load_biomin() -> Dict[str, pd.DataFrame]:
    """Load biomineralization gene lists for all species."""
    urls = {
        'apul': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv",
        'peve': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv",
        'ptua': "https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv",
    }
    
    data = {}
    for species, url in urls.items():
        print(f"Loading {species} biomin genes...")
        data[species] = pd.read_csv(url)
        print(f"  Genes: {len(data[species])}")
    
    return data


# =============================================================================
# Correlation Calculation Functions (Parallelized)
# =============================================================================

def _compute_single_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation and p-value for two arrays."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) < 3:
        return np.nan, np.nan
    
    r, p = stats.pearsonr(x[valid_mask], y[valid_mask])
    return r, p


def _compute_lncrna_gene_correlation_chunk(
    lncrna_indices: List[int],
    lncrna_mat: np.ndarray,
    gene_mat: np.ndarray,
    lncrna_ids: List[str],
    gene_ids: List[str]
) -> List[Dict]:
    """Compute correlations for a chunk of lncRNAs against all genes."""
    results = []
    
    for i in lncrna_indices:
        lncrna_expr = lncrna_mat[i, :]
        lncrna_id = lncrna_ids[i]
        
        for j in range(gene_mat.shape[0]):
            gene_expr = gene_mat[j, :]
            cor, pval = _compute_single_correlation(lncrna_expr, gene_expr)
            
            if not np.isnan(cor):
                results.append({
                    'lncrna_id': lncrna_id,
                    'gene_id': gene_ids[j],
                    'cor': cor,
                    'pval': pval
                })
    
    return results


def calculate_lncrna_gene_correlations(
    gene_df: pd.DataFrame,
    lncrna_df: pd.DataFrame,
    gene_id_col: Optional[str] = None,
    lncrna_id_col: Optional[str] = None,
    min_counts: int = 10
) -> pd.DataFrame:
    """
    Calculate correlations between lncRNA and gene expression.
    Uses parallel processing for efficiency.
    """
    if gene_id_col is None:
        gene_id_col = gene_df.columns[0]
    if lncrna_id_col is None:
        lncrna_id_col = lncrna_df.columns[0]
    
    # Get sample columns
    gene_samples = [c for c in gene_df.columns if c != gene_id_col]
    lncrna_samples = [c for c in lncrna_df.columns if c != lncrna_id_col]
    shared_samples = list(set(gene_samples) & set(lncrna_samples))
    
    print(f"  Shared samples: {len(shared_samples)}")
    
    if len(shared_samples) < 3:
        print("  Warning: Not enough shared samples")
        return pd.DataFrame()
    
    # Prepare matrices
    gene_ids = gene_df[gene_id_col].tolist()
    gene_mat = gene_df[shared_samples].values.astype(float)
    
    lncrna_ids = lncrna_df[lncrna_id_col].tolist()
    lncrna_mat = lncrna_df[shared_samples].values.astype(float)
    
    # Filter low-count features
    gene_keep = np.sum(gene_mat, axis=1) >= min_counts
    lncrna_keep = np.sum(lncrna_mat, axis=1) >= min_counts
    
    gene_mat = gene_mat[gene_keep, :]
    gene_ids = [gene_ids[i] for i in range(len(gene_ids)) if gene_keep[i]]
    
    lncrna_mat = lncrna_mat[lncrna_keep, :]
    lncrna_ids = [lncrna_ids[i] for i in range(len(lncrna_ids)) if lncrna_keep[i]]
    
    print(f"  Genes after filtering: {len(gene_ids)}")
    print(f"  lncRNAs after filtering: {len(lncrna_ids)}")
    
    # Log2 transform
    gene_log = np.log2(gene_mat + 1)
    lncrna_log = np.log2(lncrna_mat + 1)
    
    # Parallel correlation calculation
    n_lncrna = len(lncrna_ids)
    chunk_size = max(1, n_lncrna // N_JOBS)
    chunks = [list(range(i, min(i + chunk_size, n_lncrna))) 
              for i in range(0, n_lncrna, chunk_size)]
    
    print(f"  Computing correlations in parallel ({len(chunks)} chunks)...")
    
    results_nested = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_compute_lncrna_gene_correlation_chunk)(
            chunk, lncrna_log, gene_log, lncrna_ids, gene_ids
        )
        for chunk in chunks
    )
    
    # Flatten results
    all_results = [r for chunk_results in results_nested for r in chunk_results]
    
    if not all_results:
        return pd.DataFrame()
    
    # Create DataFrame and adjust p-values
    cors_df = pd.DataFrame(all_results)
    _, cors_df['padj'], _, _ = multipletests(cors_df['pval'], method='fdr_bh')
    cors_df = cors_df.sort_values('padj')
    
    return cors_df


def _compute_gbm_gene_correlation_chunk(
    gene_indices: List[int],
    gene_mat: np.ndarray,
    gbm_mat: np.ndarray,
    gene_ids: List[str]
) -> List[Dict]:
    """Compute GBM-gene correlations for a chunk of genes."""
    results = []
    
    for i in gene_indices:
        gene_expr = gene_mat[i, :]
        gbm_vals = gbm_mat[i, :]
        
        cor, pval = _compute_single_correlation(gene_expr, gbm_vals)
        
        if not np.isnan(cor):
            results.append({
                'gene_id': gene_ids[i],
                'cor': cor,
                'pval': pval
            })
    
    return results


def calculate_gbm_gene_correlations(
    gene_df: pd.DataFrame,
    gbm_df: pd.DataFrame,
    gene_id_col: Optional[str] = None,
    gbm_gene_col: Optional[str] = None,
    min_counts: int = 10
) -> Optional[pd.DataFrame]:
    """
    Calculate correlations between GBM and gene expression.
    Uses parallel processing for efficiency.
    """
    if gene_id_col is None:
        gene_id_col = gene_df.columns[0]
    if gbm_gene_col is None:
        gbm_gene_col = gbm_df.columns[0]
    
    # Get sample columns
    gene_samples = [c for c in gene_df.columns if c != gene_id_col]
    gbm_samples = [c for c in gbm_df.columns if c != gbm_gene_col]
    
    shared_samples = list(set(gene_samples) & set(gbm_samples))
    
    print(f"  Gene samples: {len(gene_samples)}")
    print(f"  GBM samples: {len(gbm_samples)}")
    print(f"  Shared samples: {len(shared_samples)}")
    
    # Try fuzzy matching if no direct matches
    if len(shared_samples) < 3:
        print("  Attempting fuzzy sample matching...")
        
        # Create mapping based on numeric IDs
        gene_sample_ids = {s: re.search(r'(\d+)', s) for s in gene_samples}
        gbm_sample_ids = {s: re.search(r'(\d+)', s) for s in gbm_samples}
        
        sample_mapping = {}
        for gs, gm in gene_sample_ids.items():
            if gm:
                for bs, bm in gbm_sample_ids.items():
                    if bm and gm.group(1) == bm.group(1):
                        sample_mapping[bs] = gs
                        break
        
        if sample_mapping:
            gbm_df = gbm_df.rename(columns=sample_mapping)
            gbm_samples = [c for c in gbm_df.columns if c != gbm_gene_col]
            shared_samples = list(set(gene_samples) & set(gbm_samples))
            print(f"  After fuzzy matching, shared samples: {len(shared_samples)}")
    
    if len(shared_samples) < 3:
        print("  Warning: Not enough shared samples for correlation")
        return None
    
    # Clean gene IDs for matching
    gene_df = gene_df.copy()
    gbm_df = gbm_df.copy()
    
    gene_df['gene_clean'] = gene_df[gene_id_col].str.replace('^gene-', '', regex=True)
    gbm_df['gene_clean'] = gbm_df[gbm_gene_col].str.replace('^gene-', '', regex=True)
    
    shared_genes = list(set(gene_df['gene_clean']) & set(gbm_df['gene_clean']))
    print(f"  Shared genes: {len(shared_genes)}")
    
    if len(shared_genes) < 10:
        print("  Warning: Too few shared genes")
        return None
    
    # Filter and align data
    gene_df_filt = gene_df[gene_df['gene_clean'].isin(shared_genes)].copy()
    gbm_df_filt = gbm_df[gbm_df['gene_clean'].isin(shared_genes)].copy()
    
    gene_df_filt = gene_df_filt.sort_values('gene_clean')
    gbm_df_filt = gbm_df_filt.sort_values('gene_clean')
    
    # Extract matrices
    gene_ids = gene_df_filt['gene_clean'].tolist()
    gene_mat = gene_df_filt[shared_samples].values.astype(float)
    gbm_mat = gbm_df_filt[shared_samples].values.astype(float)
    
    # Filter low-expression genes
    gene_keep = np.sum(gene_mat, axis=1) >= min_counts
    gene_mat = gene_mat[gene_keep, :]
    gbm_mat = gbm_mat[gene_keep, :]
    gene_ids = [gene_ids[i] for i in range(len(gene_ids)) if gene_keep[i]]
    
    print(f"  Genes for correlation: {len(gene_ids)}")
    
    # Log2 transform gene expression
    gene_log = np.log2(gene_mat + 1)
    
    # Parallel correlation calculation
    n_genes = len(gene_ids)
    chunk_size = max(1, n_genes // N_JOBS)
    chunks = [list(range(i, min(i + chunk_size, n_genes))) 
              for i in range(0, n_genes, chunk_size)]
    
    print(f"  Computing correlations in parallel ({len(chunks)} chunks)...")
    
    results_nested = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_compute_gbm_gene_correlation_chunk)(
            chunk, gene_log, gbm_mat, gene_ids
        )
        for chunk in chunks
    )
    
    # Flatten results
    all_results = [r for chunk_results in results_nested for r in chunk_results]
    
    if not all_results:
        return None
    
    # Create DataFrame and adjust p-values
    cors_df = pd.DataFrame(all_results)
    _, cors_df['padj'], _, _ = multipletests(cors_df['pval'], method='fdr_bh')
    cors_df = cors_df.sort_values('padj')
    
    return cors_df


# =============================================================================
# Analysis Functions
# =============================================================================

def get_significant_correlations(
    cor_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    cor_threshold: float = 0.5
) -> pd.DataFrame:
    """Filter for significant correlations."""
    if cor_df.empty:
        return pd.DataFrame()
    
    sig = cor_df[
        (cor_df['padj'] < padj_threshold) & 
        (np.abs(cor_df['cor']) >= cor_threshold)
    ].copy()
    
    sig['direction'] = np.where(sig['cor'] > 0, 'positive', 'negative')
    
    return sig


def classify_regulation(
    lncrna_sig: pd.DataFrame,
    gbm_sig: pd.DataFrame,
    biomin_genes: List[str]
) -> Dict:
    """Classify genes by regulation type (lncRNA, GBM, or both)."""
    
    # Get unique genes from each source
    lncrna_genes = set(lncrna_sig['gene_id'].str.replace('^gene-', '', regex=True)) if not lncrna_sig.empty else set()
    gbm_genes = set(gbm_sig['gene_id'].str.replace('^gene-', '', regex=True)) if not gbm_sig.empty else set()
    
    # Classify genes
    both_regulated = lncrna_genes & gbm_genes
    lncrna_only = lncrna_genes - gbm_genes
    gbm_only = gbm_genes - lncrna_genes
    
    print("\n=== Regulation Classification ===")
    print(f"Regulated by both lncRNA and GBM: {len(both_regulated)}")
    print(f"Regulated by lncRNA only: {len(lncrna_only)}")
    print(f"Regulated by GBM only: {len(gbm_only)}")
    
    # Check biomin status
    biomin_set = set(biomin_genes)
    both_biomin = both_regulated & biomin_set
    lncrna_biomin = lncrna_only & biomin_set
    gbm_biomin = gbm_only & biomin_set
    
    print("\n=== Biomineralization Genes ===")
    print(f"Biomin genes regulated by both: {len(both_biomin)}")
    print(f"Biomin genes regulated by lncRNA only: {len(lncrna_biomin)}")
    print(f"Biomin genes regulated by GBM only: {len(gbm_biomin)}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'regulation_type': ['Both', 'lncRNA_only', 'GBM_only'],
        'n_genes': [len(both_regulated), len(lncrna_only), len(gbm_only)],
        'n_biomin': [len(both_biomin), len(lncrna_biomin), len(gbm_biomin)]
    })
    
    # Detailed gene list
    gene_details = pd.concat([
        pd.DataFrame({'gene_id': list(both_regulated), 'regulation': 'Both'}),
        pd.DataFrame({'gene_id': list(lncrna_only), 'regulation': 'lncRNA_only'}),
        pd.DataFrame({'gene_id': list(gbm_only), 'regulation': 'GBM_only'})
    ], ignore_index=True)
    
    gene_details['is_biomin'] = gene_details['gene_id'].isin(biomin_set)
    
    return {
        'summary': summary_df,
        'genes': gene_details,
        'both': list(both_regulated),
        'lncrna_only': list(lncrna_only),
        'gbm_only': list(gbm_only)
    }


def analyze_concordance(
    lncrna_sig: pd.DataFrame,
    gbm_sig: pd.DataFrame
) -> Dict:
    """Analyze concordance between lncRNA and GBM effects."""
    
    if lncrna_sig.empty or gbm_sig.empty:
        return {'summary': pd.DataFrame(), 'details': pd.DataFrame()}
    
    # Summarize lncRNA effects per gene
    lncrna_summary = lncrna_sig.copy()
    lncrna_summary['gene_clean'] = lncrna_summary['gene_id'].str.replace('^gene-', '', regex=True)
    lncrna_summary = lncrna_summary.groupby('gene_clean').agg({
        'cor': 'mean'
    }).reset_index()
    lncrna_summary.columns = ['gene_clean', 'lncrna_mean_cor']
    lncrna_summary['lncrna_direction'] = np.where(
        lncrna_summary['lncrna_mean_cor'] > 0, 'positive', 'negative'
    )
    
    # GBM summary
    gbm_summary = gbm_sig.copy()
    gbm_summary['gene_clean'] = gbm_summary['gene_id'].str.replace('^gene-', '', regex=True)
    gbm_summary = gbm_summary[['gene_clean', 'cor', 'direction']].copy()
    gbm_summary.columns = ['gene_clean', 'gbm_cor', 'gbm_direction']
    
    # Merge
    both = pd.merge(lncrna_summary, gbm_summary, on='gene_clean', how='inner')
    
    if both.empty:
        return {'summary': pd.DataFrame(), 'details': pd.DataFrame()}
    
    both['concordance'] = np.where(
        both['lncrna_direction'] == both['gbm_direction'],
        'Concordant', 'Discordant'
    )
    
    concordance_summary = both.groupby('concordance').size().reset_index(name='n')
    
    return {
        'summary': concordance_summary,
        'details': both
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_regulation_summary(regulation: Dict, species_name: str, output_dir: Path):
    """Plot regulation type distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'Both': '#9B59B6', 'lncRNA_only': '#E74C3C', 'GBM_only': '#3498DB'}
    summary = regulation['summary']
    
    # All genes
    ax = axes[0]
    bars = ax.bar(summary['regulation_type'], summary['n_genes'], 
                  color=[colors[r] for r in summary['regulation_type']])
    ax.bar_label(bars, padding=3)
    ax.set_xlabel('Regulation Type')
    ax.set_ylabel('Number of Genes')
    ax.set_title(f'{species_name}: Gene Regulation Types')
    
    # Biomin genes
    ax = axes[1]
    bars = ax.bar(summary['regulation_type'], summary['n_biomin'], 
                  color=[colors[r] for r in summary['regulation_type']])
    ax.bar_label(bars, padding=3)
    ax.set_xlabel('Regulation Type')
    ax.set_ylabel('Number of Biomin Genes')
    ax.set_title(f'{species_name}: Biomineralization Genes')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species_name.replace(" ", "_").replace(".", "")}_regulation_summary.png', dpi=150)
    plt.close()


def plot_volcano(cor_df: pd.DataFrame, sig_df: pd.DataFrame, 
                 title: str, output_path: Path, cor_threshold: float = 0.5):
    """Create volcano plot for correlations."""
    if cor_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # All points
    ax.scatter(cor_df['cor'], -np.log10(cor_df['pval']), 
               alpha=0.1, c='gray', s=5, label='Non-significant')
    
    # Significant points
    if not sig_df.empty:
        pos = sig_df[sig_df['direction'] == 'positive']
        neg = sig_df[sig_df['direction'] == 'negative']
        
        if not pos.empty:
            ax.scatter(pos['cor'], -np.log10(pos['pval']), 
                       alpha=0.5, c='#E74C3C', s=10, label='Positive')
        if not neg.empty:
            ax.scatter(neg['cor'], -np.log10(neg['pval']), 
                       alpha=0.5, c='#3498DB', s=10, label='Negative')
    
    ax.axvline(cor_threshold, linestyle='--', alpha=0.5, c='black')
    ax.axvline(-cor_threshold, linestyle='--', alpha=0.5, c='black')
    
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cross_species_comparison(all_regulation: pd.DataFrame, output_dir: Path):
    """Create cross-species comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {'Both': '#9B59B6', 'lncRNA_only': '#E74C3C', 'GBM_only': '#3498DB'}
    species_list = all_regulation['species'].unique()
    
    for i, species in enumerate(species_list):
        sp_data = all_regulation[all_regulation['species'] == species]
        
        # All genes
        ax = axes[0, i]
        bars = ax.bar(sp_data['regulation_type'], sp_data['n_genes'],
                      color=[colors[r] for r in sp_data['regulation_type']])
        ax.bar_label(bars, padding=3)
        ax.set_xlabel('Regulation Type')
        ax.set_ylabel('Number of Genes')
        ax.set_title(f'{species}\nAll Genes')
        ax.tick_params(axis='x', rotation=45)
        
        # Biomin genes
        ax = axes[1, i]
        bars = ax.bar(sp_data['regulation_type'], sp_data['n_biomin'],
                      color=[colors[r] for r in sp_data['regulation_type']])
        ax.bar_label(bars, padding=3)
        ax.set_xlabel('Regulation Type')
        ax.set_ylabel('Number of Biomin Genes')
        ax.set_title(f'{species}\nBiomin Genes')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_species_regulation_comparison.png', dpi=150)
    plt.close()


def plot_concordance(concordance_data: Dict, species_name: str, output_dir: Path):
    """Plot concordance of lncRNA and GBM effects."""
    details = concordance_data['details']
    
    if details.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'Concordant': '#27AE60', 'Discordant': '#E74C3C'}
    
    for conc_type in ['Concordant', 'Discordant']:
        subset = details[details['concordance'] == conc_type]
        if not subset.empty:
            ax.scatter(subset['lncrna_mean_cor'], subset['gbm_cor'],
                       alpha=0.6, c=colors[conc_type], s=30, label=conc_type)
    
    ax.axhline(0, linestyle='--', alpha=0.5, c='black')
    ax.axvline(0, linestyle='--', alpha=0.5, c='black')
    
    ax.set_xlabel('Mean lncRNA Correlation')
    ax.set_ylabel('GBM Correlation')
    ax.set_title(f'{species_name}: Concordance of lncRNA and GBM Effects')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{species_name.replace(" ", "_").replace(".", "")}_concordance.png', dpi=150)
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_species(
    species_key: str,
    species_name: str,
    gene_df: pd.DataFrame,
    lncrna_df: pd.DataFrame,
    gbm_df: pd.DataFrame,
    biomin_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """Run complete analysis for one species."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {species_name}")
    print('='*60)
    
    gene_id_col = gene_df.columns[0]
    biomin_ids = biomin_df['gene_id'].tolist()
    
    # lncRNA-Gene correlations
    print("\n## lncRNA-Gene Correlations")
    lncrna_cors = calculate_lncrna_gene_correlations(
        gene_df, lncrna_df, gene_id_col=gene_id_col, min_counts=10
    )
    lncrna_sig = get_significant_correlations(lncrna_cors, padj_threshold=0.05, cor_threshold=0.5)
    
    print(f"\nSignificant lncRNA-gene correlations: {len(lncrna_sig)}")
    if not lncrna_sig.empty:
        print(f"  Positive: {(lncrna_sig['direction'] == 'positive').sum()}")
        print(f"  Negative: {(lncrna_sig['direction'] == 'negative').sum()}")
    
    # GBM-Gene correlations
    print("\n## GBM-Gene Correlations")
    gbm_cors = calculate_gbm_gene_correlations(
        gene_df, gbm_df, gene_id_col=gene_id_col, min_counts=10
    )
    
    if gbm_cors is not None:
        gbm_sig = get_significant_correlations(gbm_cors, padj_threshold=0.05, cor_threshold=0.3)
        print(f"\nSignificant GBM-gene correlations: {len(gbm_sig)}")
        if not gbm_sig.empty:
            print(f"  Positive: {(gbm_sig['direction'] == 'positive').sum()}")
            print(f"  Negative: {(gbm_sig['direction'] == 'negative').sum()}")
    else:
        gbm_sig = pd.DataFrame()
        gbm_cors = pd.DataFrame()
    
    # Classify regulation
    regulation = classify_regulation(lncrna_sig, gbm_sig, biomin_ids)
    
    # Concordance analysis
    concordance = analyze_concordance(lncrna_sig, gbm_sig)
    
    print("\n## Concordance Analysis")
    if not concordance['summary'].empty:
        print(concordance['summary'].to_string(index=False))
    else:
        print("No genes with both lncRNA and GBM regulation found")
    
    # Create visualizations
    print("\n## Creating visualizations...")
    plot_regulation_summary(regulation, species_name, output_dir)
    plot_volcano(lncrna_cors, lncrna_sig, 
                 f'{species_name}: lncRNA-Gene Correlations',
                 output_dir / f'{species_key}_lncrna_volcano.png', cor_threshold=0.5)
    if not gbm_cors.empty:
        plot_volcano(gbm_cors, gbm_sig,
                     f'{species_name}: GBM-Gene Correlations',
                     output_dir / f'{species_key}_gbm_volcano.png', cor_threshold=0.3)
    plot_concordance(concordance, species_name, output_dir)
    
    # Save results
    print("\n## Saving results...")
    lncrna_sig.to_csv(output_dir / f'{species_key}_lncrna_gene_correlations_sig.csv', index=False)
    lncrna_cors.to_csv(output_dir / f'{species_key}_lncrna_gene_correlations_full.csv', index=False)
    
    if not gbm_sig.empty:
        gbm_sig.to_csv(output_dir / f'{species_key}_gbm_gene_correlations_sig.csv', index=False)
    if not gbm_cors.empty:
        gbm_cors.to_csv(output_dir / f'{species_key}_gbm_gene_correlations_full.csv', index=False)
    
    regulation['genes'].to_csv(output_dir / f'{species_key}_regulation_classification.csv', index=False)
    
    return {
        'lncrna_cors': lncrna_cors,
        'lncrna_sig': lncrna_sig,
        'gbm_cors': gbm_cors,
        'gbm_sig': gbm_sig,
        'regulation': regulation,
        'concordance': concordance
    }


def main():
    """Main analysis pipeline."""
    
    print("="*60)
    print("lncRNA and Gene Body Methylation Regulation Analysis")
    print("Optimized Python Version with Parallel Processing")
    print("="*60)
    
    # Load all data
    print("\n## Loading Data")
    print("-"*40)
    gene_data = load_gene_expression()
    lncrna_data = load_lncrna()
    gbm_data = load_gbm()
    biomin_data = load_biomin()
    
    # Species mapping
    species_info = {
        'apul': 'A. pulchra',
        'peve': 'P. evermanni',
        'ptua': 'P. tuahiniensis'
    }
    
    # Run analysis for each species
    results = {}
    for species_key, species_name in species_info.items():
        results[species_key] = analyze_species(
            species_key,
            species_name,
            gene_data[species_key],
            lncrna_data[species_key],
            gbm_data[species_key],
            biomin_data[species_key],
            OUTPUT_DIR
        )
    
    # Cross-species comparison
    print("\n" + "="*60)
    print("Cross-Species Comparison")
    print("="*60)
    
    all_regulation = pd.concat([
        results[sp]['regulation']['summary'].assign(species=species_info[sp])
        for sp in species_info.keys()
    ], ignore_index=True)
    
    print("\n## Regulation Summary Across Species")
    print(all_regulation.to_string(index=False))
    
    # Save cross-species summary
    all_regulation.to_csv(OUTPUT_DIR / 'cross_species_regulation_summary.csv', index=False)
    
    # Create cross-species visualization
    plot_cross_species_comparison(all_regulation, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n## Data Overview")
    print("Three coral species analyzed")
    print("- Gene expression, lncRNA expression, and gene body methylation data integrated")
    
    print("\n## lncRNA Regulation")
    for sp, name in species_info.items():
        print(f"- {name}: {len(results[sp]['lncrna_sig'])} significant lncRNA-gene correlations")
    
    print("\n## GBM Regulation")
    for sp, name in species_info.items():
        print(f"- {name}: {len(results[sp]['gbm_sig'])} significant GBM-gene correlations")
    
    print("\n## Regulation Classification")
    print(all_regulation.to_string(index=False))
    
    print(f"\n## Results saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
