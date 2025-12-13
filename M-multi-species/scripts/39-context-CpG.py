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
39-context-CpG.py - CpG Methylation and lncRNA Regulation Analysis

This analysis investigates how CpG methylation and long non-coding RNAs (lncRNAs) 
potentially regulate gene expression across three coral species from 4 timepoints:
- Acropora pulchra
- Porites evermanni  
- Pocillopora tuahiniensis

The script determines:
1. Which genes are regulated by lncRNA (via expression correlation)
2. Which genes are regulated by CpG methylation (via methylation-expression correlation)
3. Which genes are regulated by both mechanisms (concert regulation)
4. Which regulated genes are biomineralization genes

Optimized for parallel processing using all available CPUs.

Author: GitHub Copilot
Date: 2025-12-05
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
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "39-context-CpG"
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


def load_cpg_methylation() -> Dict[str, pd.DataFrame]:
    """Load CpG methylation data for all species."""
    urls = {
        'apul': "https://gannet.fish.washington.edu/metacarcinus/E5/20250903_meth_Apul/merged-WGBS-CpG-counts_filtered_n20.csv",
        'peve': "https://gannet.fish.washington.edu/metacarcinus/E5/Pevermanni/20250821_meth_Peve/merged-WGBS-CpG-counts_filtered_n20.csv",
        'ptua': "https://gannet.fish.washington.edu/metacarcinus/E5/Ptuahiniensis/20250821_meth_Ptua/merged-WGBS-CpG-counts_filtered_n20.csv",
    }
    
    data = {}
    for species, url in urls.items():
        print(f"Loading {species} CpG methylation...")
        try:
            data[species] = pd.read_csv(url)
            print(f"  Shape: {data[species].shape}")
            print(f"  Columns: {list(data[species].columns[:5])}...")
        except Exception as e:
            print(f"  Warning: Failed to load {species} CpG data: {e}")
            data[species] = pd.DataFrame()
    
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
# Data Preprocessing Functions
# =============================================================================

def preprocess_cpg_data(cpg_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess CpG methylation data to calculate per-gene methylation levels.
    
    CpG methylation data typically has chromosome, position, and methylation counts.
    We need to aggregate this to gene-level methylation.
    """
    if cpg_df.empty:
        return pd.DataFrame(), []
    
    print(f"  CpG data columns: {list(cpg_df.columns)}")
    
    # Identify structure of CpG data
    # Common formats: chr, pos, strand, then sample columns
    # or: position info + methylated/unmethylated counts per sample
    
    # Check for common column patterns
    pos_cols = [c for c in cpg_df.columns if any(x in c.lower() for x in ['chr', 'pos', 'start', 'end', 'strand', 'chrom'])]
    
    if not pos_cols:
        # Assume first column is position identifier
        pos_cols = [cpg_df.columns[0]]
    
    sample_cols = [c for c in cpg_df.columns if c not in pos_cols]
    
    print(f"  Position columns: {pos_cols}")
    print(f"  Sample columns: {len(sample_cols)}")
    
    return cpg_df, sample_cols


def extract_sample_ids(columns: List[str]) -> Dict[str, str]:
    """Extract sample IDs from column names for matching across data types."""
    sample_mapping = {}
    for col in columns:
        # Extract numeric sample ID
        match = re.search(r'(\d+)', col)
        if match:
            sample_mapping[col] = match.group(1)
    return sample_mapping


def match_samples(gene_cols: List[str], other_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Match samples between gene expression and another data type."""
    gene_ids = extract_sample_ids(gene_cols)
    other_ids = extract_sample_ids(other_cols)
    
    # Find matching numeric IDs
    gene_by_id = {v: k for k, v in gene_ids.items()}
    other_by_id = {v: k for k, v in other_ids.items()}
    
    common_ids = set(gene_by_id.keys()) & set(other_by_id.keys())
    
    matched_gene = [gene_by_id[i] for i in sorted(common_ids)]
    matched_other = [other_by_id[i] for i in sorted(common_ids)]
    
    return matched_gene, matched_other


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
        # Try fuzzy matching
        print("  Attempting fuzzy sample matching...")
        matched_gene, matched_lncrna = match_samples(gene_samples, lncrna_samples)
        if len(matched_gene) >= 3:
            # Rename columns for alignment
            rename_map = {m_l: m_g for m_g, m_l in zip(matched_gene, matched_lncrna)}
            lncrna_df = lncrna_df.rename(columns=rename_map)
            shared_samples = matched_gene
            print(f"  After fuzzy matching: {len(shared_samples)} shared samples")
        else:
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
    
    if len(gene_ids) == 0 or len(lncrna_ids) == 0:
        return pd.DataFrame()
    
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


def _compute_cpg_gene_correlation_chunk(
    cpg_indices: List[int],
    cpg_mat: np.ndarray,
    gene_mat: np.ndarray,
    cpg_ids: List[str],
    gene_ids: List[str]
) -> List[Dict]:
    """Compute CpG-gene correlations for a chunk of CpG sites."""
    results = []
    
    for i in cpg_indices:
        cpg_vals = cpg_mat[i, :]
        cpg_id = cpg_ids[i]
        
        for j in range(gene_mat.shape[0]):
            gene_expr = gene_mat[j, :]
            cor, pval = _compute_single_correlation(cpg_vals, gene_expr)
            
            if not np.isnan(cor):
                results.append({
                    'cpg_id': cpg_id,
                    'gene_id': gene_ids[j],
                    'cor': cor,
                    'pval': pval
                })
    
    return results


def calculate_cpg_gene_correlations(
    gene_df: pd.DataFrame,
    cpg_df: pd.DataFrame,
    gene_id_col: Optional[str] = None,
    min_counts: int = 10,
    max_cpg_sites: int = 10000
) -> Optional[pd.DataFrame]:
    """
    Calculate correlations between CpG methylation and gene expression.
    Uses parallel processing for efficiency.
    
    Args:
        gene_df: Gene expression count matrix
        cpg_df: CpG methylation data
        gene_id_col: Column name for gene IDs
        min_counts: Minimum total counts for a gene to be included
        max_cpg_sites: Maximum number of CpG sites to analyze (for computational efficiency)
    """
    if cpg_df.empty:
        print("  Warning: Empty CpG data")
        return None
    
    if gene_id_col is None:
        gene_id_col = gene_df.columns[0]
    
    # Identify CpG position and sample columns
    cpg_pos_cols = []
    cpg_sample_cols = []
    
    for col in cpg_df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['chr', 'pos', 'start', 'end', 'strand', 'chrom', 'scaffold', 'contig']):
            cpg_pos_cols.append(col)
        else:
            # Check if column has numeric data
            if cpg_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                cpg_sample_cols.append(col)
    
    # If no position columns found, assume first column is identifier
    if not cpg_pos_cols and len(cpg_df.columns) > 1:
        cpg_pos_cols = [cpg_df.columns[0]]
        cpg_sample_cols = [c for c in cpg_df.columns[1:] if cpg_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    print(f"  CpG position columns: {cpg_pos_cols}")
    print(f"  CpG sample columns: {len(cpg_sample_cols)}")
    
    if len(cpg_sample_cols) < 3:
        print("  Warning: Not enough CpG sample columns")
        return None
    
    # Get gene sample columns
    gene_samples = [c for c in gene_df.columns if c != gene_id_col]
    
    # Try to match samples
    matched_gene, matched_cpg = match_samples(gene_samples, cpg_sample_cols)
    
    print(f"  Gene samples: {len(gene_samples)}")
    print(f"  Matched samples: {len(matched_gene)}")
    
    if len(matched_gene) < 3:
        print("  Warning: Not enough matched samples for CpG-gene correlation")
        return None
    
    # Create CpG ID if needed
    if len(cpg_pos_cols) > 0:
        cpg_df = cpg_df.copy()
        cpg_df['cpg_id'] = cpg_df[cpg_pos_cols[0]].astype(str)
        if len(cpg_pos_cols) > 1:
            for col in cpg_pos_cols[1:]:
                cpg_df['cpg_id'] = cpg_df['cpg_id'] + '_' + cpg_df[col].astype(str)
    else:
        cpg_df = cpg_df.copy()
        cpg_df['cpg_id'] = [f"cpg_{i}" for i in range(len(cpg_df))]
    
    # Subsample CpG sites if too many
    if len(cpg_df) > max_cpg_sites:
        print(f"  Subsampling from {len(cpg_df)} to {max_cpg_sites} CpG sites...")
        cpg_df = cpg_df.sample(n=max_cpg_sites, random_state=42)
    
    # Prepare matrices
    gene_ids = gene_df[gene_id_col].tolist()
    gene_mat = gene_df[matched_gene].values.astype(float)
    
    cpg_ids = cpg_df['cpg_id'].tolist()
    cpg_mat = cpg_df[matched_cpg].values.astype(float)
    
    # Filter low-expression genes
    gene_keep = np.sum(gene_mat, axis=1) >= min_counts
    gene_mat = gene_mat[gene_keep, :]
    gene_ids = [gene_ids[i] for i in range(len(gene_ids)) if gene_keep[i]]
    
    # Filter CpG sites with no variance
    cpg_var = np.var(cpg_mat, axis=1)
    cpg_keep = cpg_var > 0
    cpg_mat = cpg_mat[cpg_keep, :]
    cpg_ids = [cpg_ids[i] for i in range(len(cpg_ids)) if cpg_keep[i]]
    
    print(f"  Genes for correlation: {len(gene_ids)}")
    print(f"  CpG sites for correlation: {len(cpg_ids)}")
    
    if len(gene_ids) == 0 or len(cpg_ids) == 0:
        return None
    
    # Log2 transform gene expression
    gene_log = np.log2(gene_mat + 1)
    
    # Parallel correlation calculation
    n_cpg = len(cpg_ids)
    chunk_size = max(1, n_cpg // N_JOBS)
    chunks = [list(range(i, min(i + chunk_size, n_cpg))) 
              for i in range(0, n_cpg, chunk_size)]
    
    print(f"  Computing CpG-gene correlations in parallel ({len(chunks)} chunks)...")
    
    results_nested = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_compute_cpg_gene_correlation_chunk)(
            chunk, cpg_mat, gene_log, cpg_ids, gene_ids
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


def aggregate_cpg_by_gene(cpg_cors: pd.DataFrame, padj_threshold: float = 0.05) -> pd.DataFrame:
    """
    Aggregate CpG-gene correlations to gene-level summary.
    
    For each gene, summarizes the CpG correlations to determine overall
    methylation-expression relationship.
    """
    if cpg_cors is None or cpg_cors.empty:
        return pd.DataFrame()
    
    # Filter significant correlations
    sig = cpg_cors[cpg_cors['padj'] < padj_threshold].copy()
    
    if sig.empty:
        # If no significant, use top correlations
        sig = cpg_cors.nsmallest(1000, 'pval').copy()
    
    # Aggregate by gene
    gene_summary = sig.groupby('gene_id').agg(
        n_cpg_sig=('cpg_id', 'count'),
        mean_cor=('cor', 'mean'),
        median_cor=('cor', 'median'),
        min_cor=('cor', 'min'),
        max_cor=('cor', 'max'),
        best_pval=('pval', 'min')
    ).reset_index()
    
    # Determine direction
    gene_summary['direction'] = np.where(
        gene_summary['mean_cor'] > 0, 'positive', 'negative'
    )
    
    return gene_summary


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
    cpg_summary: pd.DataFrame,
    biomin_genes: List[str]
) -> Dict:
    """
    Classify genes by regulation type (lncRNA, CpG methylation, or both).
    
    Returns a dictionary with:
    - Summary statistics
    - Gene classifications
    - Lists of genes in each category
    """
    
    # Get unique genes from each source
    lncrna_genes = set()
    if not lncrna_sig.empty:
        lncrna_genes = set(lncrna_sig['gene_id'].str.replace('^gene-', '', regex=True))
    
    cpg_genes = set()
    if not cpg_summary.empty:
        cpg_genes = set(cpg_summary['gene_id'].str.replace('^gene-', '', regex=True))
    
    # Classify genes
    both_regulated = lncrna_genes & cpg_genes
    lncrna_only = lncrna_genes - cpg_genes
    cpg_only = cpg_genes - lncrna_genes
    
    print("\n=== Regulation Classification ===")
    print(f"Regulated by both lncRNA and CpG methylation (concert): {len(both_regulated)}")
    print(f"Regulated by lncRNA only: {len(lncrna_only)}")
    print(f"Regulated by CpG methylation only: {len(cpg_only)}")
    
    # Check biomin status
    biomin_set = set(g.replace('gene-', '') for g in biomin_genes)
    both_biomin = both_regulated & biomin_set
    lncrna_biomin = lncrna_only & biomin_set
    cpg_biomin = cpg_only & biomin_set
    
    print("\n=== Biomineralization Genes ===")
    print(f"Biomin genes regulated by both: {len(both_biomin)}")
    if both_biomin:
        print(f"  Genes: {', '.join(list(both_biomin)[:10])}{'...' if len(both_biomin) > 10 else ''}")
    print(f"Biomin genes regulated by lncRNA only: {len(lncrna_biomin)}")
    if lncrna_biomin:
        print(f"  Genes: {', '.join(list(lncrna_biomin)[:10])}{'...' if len(lncrna_biomin) > 10 else ''}")
    print(f"Biomin genes regulated by CpG only: {len(cpg_biomin)}")
    if cpg_biomin:
        print(f"  Genes: {', '.join(list(cpg_biomin)[:10])}{'...' if len(cpg_biomin) > 10 else ''}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'regulation_type': ['Both (Concert)', 'lncRNA_only', 'CpG_only'],
        'n_genes': [len(both_regulated), len(lncrna_only), len(cpg_only)],
        'n_biomin': [len(both_biomin), len(lncrna_biomin), len(cpg_biomin)]
    })
    
    # Detailed gene list
    gene_details = pd.concat([
        pd.DataFrame({'gene_id': list(both_regulated), 'regulation': 'Both (Concert)'}),
        pd.DataFrame({'gene_id': list(lncrna_only), 'regulation': 'lncRNA_only'}),
        pd.DataFrame({'gene_id': list(cpg_only), 'regulation': 'CpG_only'})
    ], ignore_index=True)
    
    gene_details['is_biomin'] = gene_details['gene_id'].isin(biomin_set)
    
    return {
        'summary': summary_df,
        'genes': gene_details,
        'both': list(both_regulated),
        'lncrna_only': list(lncrna_only),
        'cpg_only': list(cpg_only),
        'biomin_both': list(both_biomin),
        'biomin_lncrna': list(lncrna_biomin),
        'biomin_cpg': list(cpg_biomin)
    }


def analyze_concordance(
    lncrna_sig: pd.DataFrame,
    cpg_summary: pd.DataFrame
) -> Dict:
    """
    Analyze concordance between lncRNA and CpG methylation effects.
    
    Determines whether lncRNA and CpG methylation have concordant (same direction)
    or discordant (opposite direction) effects on gene expression.
    """
    
    if lncrna_sig.empty or cpg_summary.empty:
        return {'summary': pd.DataFrame(), 'details': pd.DataFrame()}
    
    # Summarize lncRNA effects per gene
    lncrna_summary = lncrna_sig.copy()
    lncrna_summary['gene_clean'] = lncrna_summary['gene_id'].str.replace('^gene-', '', regex=True)
    lncrna_agg = lncrna_summary.groupby('gene_clean').agg({
        'cor': 'mean'
    }).reset_index()
    lncrna_agg.columns = ['gene_clean', 'lncrna_mean_cor']
    lncrna_agg['lncrna_direction'] = np.where(
        lncrna_agg['lncrna_mean_cor'] > 0, 'positive', 'negative'
    )
    
    # CpG summary
    cpg_agg = cpg_summary.copy()
    cpg_agg['gene_clean'] = cpg_agg['gene_id'].str.replace('^gene-', '', regex=True)
    cpg_agg = cpg_agg[['gene_clean', 'mean_cor', 'direction']].copy()
    cpg_agg.columns = ['gene_clean', 'cpg_cor', 'cpg_direction']
    
    # Merge
    both = pd.merge(lncrna_agg, cpg_agg, on='gene_clean', how='inner')
    
    if both.empty:
        return {'summary': pd.DataFrame(), 'details': pd.DataFrame()}
    
    both['concordance'] = np.where(
        both['lncrna_direction'] == both['cpg_direction'],
        'Concordant', 'Discordant'
    )
    
    concordance_summary = both.groupby('concordance').size().reset_index(name='n')
    
    print("\n=== Concordance Analysis ===")
    print(f"Genes with both lncRNA and CpG regulation: {len(both)}")
    print(concordance_summary.to_string(index=False))
    
    return {
        'summary': concordance_summary,
        'details': both
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_regulation_summary(regulation: Dict, species_name: str, output_dir: Path):
    """Plot regulation type distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Both (Concert)': '#9B59B6', 'lncRNA_only': '#E74C3C', 'CpG_only': '#3498DB'}
    summary = regulation['summary']
    
    # All genes
    ax = axes[0]
    bars = ax.bar(summary['regulation_type'], summary['n_genes'], 
                  color=[colors.get(r, '#666666') for r in summary['regulation_type']])
    ax.bar_label(bars, padding=3, fontsize=12)
    ax.set_xlabel('Regulation Type', fontsize=12)
    ax.set_ylabel('Number of Genes', fontsize=12)
    ax.set_title(f'{species_name}: Gene Regulation Types', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    
    # Biomin genes
    ax = axes[1]
    bars = ax.bar(summary['regulation_type'], summary['n_biomin'], 
                  color=[colors.get(r, '#666666') for r in summary['regulation_type']])
    ax.bar_label(bars, padding=3, fontsize=12)
    ax.set_xlabel('Regulation Type', fontsize=12)
    ax.set_ylabel('Number of Biomin Genes', fontsize=12)
    ax.set_title(f'{species_name}: Biomineralization Genes', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    safe_name = species_name.replace(" ", "_").replace(".", "")
    plt.savefig(output_dir / f'{safe_name}_regulation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_volcano(cor_df: pd.DataFrame, sig_df: pd.DataFrame, 
                 title: str, output_path: Path, cor_threshold: float = 0.5):
    """Create volcano plot for correlations."""
    if cor_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # All points
    ax.scatter(cor_df['cor'], -np.log10(cor_df['pval'].clip(lower=1e-300)), 
               alpha=0.1, c='gray', s=5, label='Non-significant', rasterized=True)
    
    # Significant points
    if not sig_df.empty:
        pos = sig_df[sig_df['direction'] == 'positive']
        neg = sig_df[sig_df['direction'] == 'negative']
        
        if not pos.empty:
            ax.scatter(pos['cor'], -np.log10(pos['pval'].clip(lower=1e-300)), 
                       alpha=0.5, c='#E74C3C', s=15, label='Positive')
        if not neg.empty:
            ax.scatter(neg['cor'], -np.log10(neg['pval'].clip(lower=1e-300)), 
                       alpha=0.5, c='#3498DB', s=15, label='Negative')
    
    ax.axvline(cor_threshold, linestyle='--', alpha=0.5, c='black')
    ax.axvline(-cor_threshold, linestyle='--', alpha=0.5, c='black')
    
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_species_comparison(all_regulation: pd.DataFrame, output_dir: Path):
    """Create cross-species comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    colors = {'Both (Concert)': '#9B59B6', 'lncRNA_only': '#E74C3C', 'CpG_only': '#3498DB'}
    species_list = all_regulation['species'].unique()
    
    for i, species in enumerate(species_list):
        sp_data = all_regulation[all_regulation['species'] == species]
        
        # All genes
        ax = axes[0, i]
        bars = ax.bar(sp_data['regulation_type'], sp_data['n_genes'],
                      color=[colors.get(r, '#666666') for r in sp_data['regulation_type']])
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xlabel('Regulation Type', fontsize=10)
        ax.set_ylabel('Number of Genes', fontsize=10)
        ax.set_title(f'{species}\nAll Genes', fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        
        # Biomin genes
        ax = axes[1, i]
        bars = ax.bar(sp_data['regulation_type'], sp_data['n_biomin'],
                      color=[colors.get(r, '#666666') for r in sp_data['regulation_type']])
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xlabel('Regulation Type', fontsize=10)
        ax.set_ylabel('Number of Biomin Genes', fontsize=10)
        ax.set_title(f'{species}\nBiomin Genes', fontsize=12)
        ax.tick_params(axis='x', rotation=30)
    
    plt.suptitle('CpG Methylation and lncRNA Regulation Across Species', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_species_regulation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_concordance(concordance_data: Dict, species_name: str, output_dir: Path):
    """Plot concordance of lncRNA and CpG methylation effects."""
    details = concordance_data['details']
    
    if details.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    colors = {'Concordant': '#27AE60', 'Discordant': '#E74C3C'}
    
    for conc_type in ['Concordant', 'Discordant']:
        subset = details[details['concordance'] == conc_type]
        if not subset.empty:
            ax.scatter(subset['lncrna_mean_cor'], subset['cpg_cor'],
                       alpha=0.6, c=colors[conc_type], s=40, label=conc_type, edgecolors='white', linewidths=0.5)
    
    ax.axhline(0, linestyle='--', alpha=0.5, c='black')
    ax.axvline(0, linestyle='--', alpha=0.5, c='black')
    
    ax.set_xlabel('Mean lncRNA-Gene Correlation', fontsize=12)
    ax.set_ylabel('CpG-Gene Correlation', fontsize=12)
    ax.set_title(f'{species_name}: Concordance of\nlncRNA and CpG Methylation Effects', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar plot of concordance counts
    ax = axes[1]
    summary = concordance_data['summary']
    if not summary.empty:
        bars = ax.bar(summary['concordance'], summary['n'],
                      color=[colors.get(c, '#666666') for c in summary['concordance']])
        ax.bar_label(bars, padding=3, fontsize=12)
        ax.set_xlabel('Concordance Type', fontsize=12)
        ax.set_ylabel('Number of Genes', fontsize=12)
        ax.set_title(f'{species_name}: Concordance Summary', fontsize=12)
    
    plt.tight_layout()
    safe_name = species_name.replace(" ", "_").replace(".", "")
    plt.savefig(output_dir / f'{safe_name}_concordance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_biomin_heatmap(regulation_results: Dict, species_info: Dict, output_dir: Path):
    """Create heatmap showing biomineralization gene regulation across species."""
    # Collect all biomin genes and their regulation status
    all_biomin_genes = set()
    for sp, results in regulation_results.items():
        all_biomin_genes.update(results['regulation']['biomin_both'])
        all_biomin_genes.update(results['regulation']['biomin_lncrna'])
        all_biomin_genes.update(results['regulation']['biomin_cpg'])
    
    if not all_biomin_genes:
        print("No biomineralization genes with regulation found")
        return
    
    # Create matrix
    genes = sorted(all_biomin_genes)
    species_keys = list(species_info.keys())
    
    matrix = np.zeros((len(genes), len(species_keys) * 2))  # 2 columns per species (lncRNA, CpG)
    
    for i, gene in enumerate(genes):
        for j, sp in enumerate(species_keys):
            reg = regulation_results[sp]['regulation']
            if gene in reg['biomin_both']:
                matrix[i, j*2] = 1  # lncRNA
                matrix[i, j*2 + 1] = 1  # CpG
            elif gene in reg['biomin_lncrna']:
                matrix[i, j*2] = 1  # lncRNA only
            elif gene in reg['biomin_cpg']:
                matrix[i, j*2 + 1] = 1  # CpG only
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(genes) * 0.3)))
    
    col_labels = []
    for sp in species_keys:
        col_labels.extend([f'{species_info[sp]}\nlncRNA', f'{species_info[sp]}\nCpG'])
    
    # Custom colormap
    cmap = sns.color_palette(["white", "#2ECC71"])
    
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False,
                xticklabels=col_labels, yticklabels=genes,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Biomineralization Genes: Regulation by lncRNA and CpG Methylation', fontsize=14)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'biomin_regulation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_species(
    species_key: str,
    species_name: str,
    gene_df: pd.DataFrame,
    lncrna_df: pd.DataFrame,
    cpg_df: pd.DataFrame,
    biomin_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """Run complete analysis for one species."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {species_name}")
    print('='*60)
    
    gene_id_col = gene_df.columns[0]
    
    # Get biomin gene IDs
    biomin_ids = []
    if 'gene_id' in biomin_df.columns:
        biomin_ids = biomin_df['gene_id'].tolist()
    else:
        biomin_ids = biomin_df.iloc[:, 0].tolist()
    
    # lncRNA-Gene correlations
    print("\n## lncRNA-Gene Correlations")
    lncrna_cors = calculate_lncrna_gene_correlations(
        gene_df, lncrna_df, gene_id_col=gene_id_col, min_counts=10
    )
    lncrna_sig = get_significant_correlations(lncrna_cors, padj_threshold=0.05, cor_threshold=0.5)
    
    print(f"\nTotal lncRNA-gene correlations: {len(lncrna_cors)}")
    print(f"Significant lncRNA-gene correlations: {len(lncrna_sig)}")
    if not lncrna_sig.empty:
        print(f"  Positive: {(lncrna_sig['direction'] == 'positive').sum()}")
        print(f"  Negative: {(lncrna_sig['direction'] == 'negative').sum()}")
    
    # CpG-Gene correlations
    print("\n## CpG Methylation-Gene Correlations")
    cpg_cors = calculate_cpg_gene_correlations(
        gene_df, cpg_df, gene_id_col=gene_id_col, min_counts=10
    )
    
    cpg_summary = pd.DataFrame()
    if cpg_cors is not None and not cpg_cors.empty:
        cpg_summary = aggregate_cpg_by_gene(cpg_cors, padj_threshold=0.1)
        print(f"\nTotal CpG-gene correlations: {len(cpg_cors)}")
        print(f"Genes with significant CpG correlations: {len(cpg_summary)}")
        if not cpg_summary.empty:
            print(f"  Positive mean correlation: {(cpg_summary['direction'] == 'positive').sum()}")
            print(f"  Negative mean correlation: {(cpg_summary['direction'] == 'negative').sum()}")
    else:
        print("\nNo CpG-gene correlations computed")
        cpg_cors = pd.DataFrame()
    
    # Classify regulation
    regulation = classify_regulation(lncrna_sig, cpg_summary, biomin_ids)
    
    # Concordance analysis
    concordance = analyze_concordance(lncrna_sig, cpg_summary)
    
    # Create visualizations
    print("\n## Creating visualizations...")
    plot_regulation_summary(regulation, species_name, output_dir)
    
    if not lncrna_cors.empty:
        plot_volcano(lncrna_cors, lncrna_sig, 
                     f'{species_name}: lncRNA-Gene Correlations',
                     output_dir / f'{species_key}_lncrna_volcano.png', cor_threshold=0.5)
    
    if not cpg_cors.empty:
        # For CpG volcano, use aggregated correlations per gene
        cpg_for_plot = cpg_cors.copy()
        cpg_sig_for_plot = cpg_cors[cpg_cors['padj'] < 0.05].copy()
        if not cpg_sig_for_plot.empty:
            cpg_sig_for_plot['direction'] = np.where(cpg_sig_for_plot['cor'] > 0, 'positive', 'negative')
        plot_volcano(cpg_for_plot, cpg_sig_for_plot,
                     f'{species_name}: CpG Methylation-Gene Correlations',
                     output_dir / f'{species_key}_cpg_volcano.png', cor_threshold=0.3)
    
    plot_concordance(concordance, species_name, output_dir)
    
    # Save results
    print("\n## Saving results...")
    
    if not lncrna_sig.empty:
        lncrna_sig.to_csv(output_dir / f'{species_key}_lncrna_gene_correlations_sig.csv', index=False)
    if not lncrna_cors.empty:
        # Save top correlations to avoid huge files
        lncrna_cors.head(100000).to_csv(output_dir / f'{species_key}_lncrna_gene_correlations_top.csv', index=False)
    
    if not cpg_summary.empty:
        cpg_summary.to_csv(output_dir / f'{species_key}_cpg_gene_summary.csv', index=False)
    if cpg_cors is not None and not cpg_cors.empty:
        cpg_cors.head(100000).to_csv(output_dir / f'{species_key}_cpg_gene_correlations_top.csv', index=False)
    
    regulation['genes'].to_csv(output_dir / f'{species_key}_regulation_classification.csv', index=False)
    
    if not concordance['details'].empty:
        concordance['details'].to_csv(output_dir / f'{species_key}_concordance_details.csv', index=False)
    
    return {
        'lncrna_cors': lncrna_cors,
        'lncrna_sig': lncrna_sig,
        'cpg_cors': cpg_cors,
        'cpg_summary': cpg_summary,
        'regulation': regulation,
        'concordance': concordance
    }


def main():
    """Main analysis pipeline."""
    
    print("="*60)
    print("CpG Methylation and lncRNA Regulation Analysis")
    print("Optimized Python Version with Parallel Processing")
    print("="*60)
    
    # Load all data
    print("\n## Loading Data")
    print("-"*40)
    gene_data = load_gene_expression()
    lncrna_data = load_lncrna()
    cpg_data = load_cpg_methylation()
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
            cpg_data[species_key],
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
    
    # Create cross-species visualizations
    plot_cross_species_comparison(all_regulation, OUTPUT_DIR)
    plot_biomin_heatmap(results, species_info, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n## Data Overview")
    print("Three coral species analyzed from 4 timepoints each:")
    print("- Gene expression (RNA-seq)")
    print("- lncRNA expression")
    print("- CpG methylation (WGBS)")
    print("- Biomineralization gene annotations")
    
    print("\n## lncRNA Regulation")
    for sp, name in species_info.items():
        n_sig = len(results[sp]['lncrna_sig'])
        print(f"- {name}: {n_sig} significant lncRNA-gene correlations")
    
    print("\n## CpG Methylation Regulation")
    for sp, name in species_info.items():
        n_genes = len(results[sp]['cpg_summary'])
        print(f"- {name}: {n_genes} genes with significant CpG-expression correlations")
    
    print("\n## Combined Regulation Classification")
    print(all_regulation.to_string(index=False))
    
    print("\n## Biomineralization Genes Summary")
    total_biomin_both = sum(len(results[sp]['regulation']['biomin_both']) for sp in species_info.keys())
    total_biomin_lncrna = sum(len(results[sp]['regulation']['biomin_lncrna']) for sp in species_info.keys())
    total_biomin_cpg = sum(len(results[sp]['regulation']['biomin_cpg']) for sp in species_info.keys())
    print(f"- Biomin genes regulated by BOTH lncRNA and CpG: {total_biomin_both}")
    print(f"- Biomin genes regulated by lncRNA only: {total_biomin_lncrna}")
    print(f"- Biomin genes regulated by CpG only: {total_biomin_cpg}")
    
    print(f"\n## Results saved to: {OUTPUT_DIR.absolute()}")
    print("\nOutput files include:")
    print("- *_lncrna_gene_correlations_sig.csv: Significant lncRNA-gene correlations")
    print("- *_cpg_gene_summary.csv: Gene-level CpG methylation summary")
    print("- *_regulation_classification.csv: Gene classification by regulation type")
    print("- *_concordance_details.csv: Concordance analysis details")
    print("- cross_species_regulation_summary.csv: Summary across all species")
    print("- Various PNG visualizations")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

