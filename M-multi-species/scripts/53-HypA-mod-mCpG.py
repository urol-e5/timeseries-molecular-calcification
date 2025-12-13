#!/usr/bin/env python3
"""
53-HypA-mod-mCpG.py

Test the hypothesis that genes with high gene expression variability (based on 
exon-level expression) at timepoints T2 and T3 will have corresponding distinct 
differences in mCpG methylation within the gene.

Analysis Overview:
1. For each species, identify calcification-related genes that change in activity 
   (expression level AND expression variability based on exon-level expression) 
   from T1→T2, T1→T3, T4→T2, and T4→T3.
2. Test whether genes with high exon-level expression variability at T2/T3 have 
   distinct mCpG differences within the gene.

PARALLELIZED - Uses all available CPUs for computationally intensive operations.

Data Sources:
-------------
Expression Matrices (Biomineralization Genes):
- Apul: M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv
- Peve: M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv
- Ptua: M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv

Exon-Level Expression:
- M-multi-species/output/40-exon-count-matrix/[species]-exon_gene_count_matrix.csv
- M-multi-species/output/40-exon-count-matrix/[species]-exon_summary_by_ortholog.csv

CpG Methylation (mCpG):
- Apul: https://gannet.fish.washington.edu/metacarcinus/E5/20250903_meth_Apul/merged-WGBS-CpG-counts_filtered_n20.csv
- Peve: https://gannet.fish.washington.edu/metacarcinus/E5/Pevermanni/20250821_meth_Peve/merged-WGBS-CpG-counts_filtered_n20.csv
- Ptua: https://gannet.fish.washington.edu/metacarcinus/E5/Ptuahiniensis/20250821_meth_Ptua/merged-WGBS-CpG-counts_filtered_n20.csv

IMPORTANT: Gene ID Format Differences
- Apul: FUN_XXXXXX format - direct matching works
- Peve: exon files have "gene-" prefix, biomin files do not - needs prefix removal
- Ptua: Completely different gene annotation formats - MUST use group_id for matching
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Data URLs
BIOMIN_URLS = {
    'apul': 'https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv',
    'peve': 'https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv',
    'ptua': 'https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv',
}

CPG_URLS = {
    'apul': 'https://gannet.fish.washington.edu/metacarcinus/E5/20250903_meth_Apul/merged-WGBS-CpG-counts_filtered_n20.csv',
    'peve': 'https://gannet.fish.washington.edu/metacarcinus/E5/Pevermanni/20250821_meth_Peve/merged-WGBS-CpG-counts_filtered_n20.csv',
    'ptua': 'https://gannet.fish.washington.edu/metacarcinus/E5/Ptuahiniensis/20250821_meth_Ptua/merged-WGBS-CpG-counts_filtered_n20.csv',
}

# Script and output directories
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / '53-HypA-mod-mCpG'
DATA_DIR = SCRIPT_DIR.parent / 'output'

# Exon data paths (local)
EXON_PATHS = {
    'apul': DATA_DIR / '40-exon-count-matrix' / 'apul-exon_gene_count_matrix.csv',
    'peve': DATA_DIR / '40-exon-count-matrix' / 'peve-exon_gene_count_matrix.csv',
    'ptua': DATA_DIR / '40-exon-count-matrix' / 'ptua-exon_gene_count_matrix.csv',
}

EXON_SUMMARY_PATHS = {
    'apul': DATA_DIR / '40-exon-count-matrix' / 'apul-exon_summary_by_ortholog.csv',
    'peve': DATA_DIR / '40-exon-count-matrix' / 'peve-exon_summary_by_ortholog.csv',
    'ptua': DATA_DIR / '40-exon-count-matrix' / 'ptua-exon_summary_by_ortholog.csv',
}

# Annotation file
ANNOTATION_PATH = DATA_DIR / '12-ortho-annot' / 'ortholog_groups_annotated.csv'

# Analysis parameters
CPG_GENE_BUFFER_BP = 2000  # Buffer region around genes for CpG mapping
CV_RATIO_THRESHOLD = 1.5   # Threshold for high variability classification
CV_DIFF_THRESHOLD = 0.2    # Alternative threshold for CV difference

# Species information
SPECIES_NAMES = {
    'apul': 'Acropora pulchra',
    'peve': 'Porites evermanni',
    'ptua': 'Pocillopora tuahiniensis',
}

SPECIES_COLORS = {
    'apul': '#E64B35',
    'peve': '#4DBBD5',
    'ptua': '#00A087',
}

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


def parse_cpg_ids(cpg_series: pd.Series) -> pd.DataFrame:
    """
    Parse CpG IDs to extract chromosome and position.
    
    Format: CpG_chromosome_position
    """
    stripped = cpg_series.str.replace('^CpG_', '', regex=True)
    pattern = r'^(.+)_(\d+)$'
    extracted = stripped.str.extract(pattern)
    
    return pd.DataFrame({
        'cpg_id': cpg_series,
        'cpg_chr': extracted[0],
        'cpg_pos': pd.to_numeric(extracted[1], errors='coerce')
    })


def extract_timepoint(sample_id: str) -> str:
    """Extract timepoint from sample ID (e.g., 'ACR-139-TP1' -> 'TP1')."""
    match = re.search(r'TP\d+', sample_id)
    return match.group() if match else None


def get_sample_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    """Get sample columns from dataframe (excluding metadata columns)."""
    return [c for c in df.columns if c not in exclude_cols]


def calculate_log2fc(mean1: float, mean2: float) -> float:
    """Calculate log2 fold change between two means."""
    if mean1 <= 0 or mean2 <= 0:
        return np.nan
    return np.log2(mean2 + 1) - np.log2(mean1 + 1)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_biomin_data(species: str) -> pd.DataFrame:
    """Load biomineralization gene expression counts for a species."""
    print(f"    Loading biomin counts for {species}...")
    df = pd.read_csv(BIOMIN_URLS[species])
    print(f"    Loaded {len(df)} genes")
    return df


def load_cpg_data(species: str) -> pd.DataFrame:
    """Load CpG methylation data for a species."""
    print(f"    Loading mCpG data for {species}...")
    df = pd.read_csv(CPG_URLS[species])
    print(f"    Loaded {len(df)} CpG sites")
    return df


def load_exon_data(species: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load exon expression data and summary for a species."""
    print(f"    Loading exon data for {species}...")
    
    exon_counts = pd.read_csv(EXON_PATHS[species])
    exon_summary = pd.read_csv(EXON_SUMMARY_PATHS[species])
    
    print(f"    Loaded {len(exon_counts)} exons, {len(exon_summary)} summary entries")
    return exon_counts, exon_summary


def load_annotation() -> pd.DataFrame:
    """Load ortholog group annotation file."""
    print("    Loading annotation file...")
    df = pd.read_csv(ANNOTATION_PATH)
    print(f"    Loaded {len(df)} annotations")
    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def calculate_expression_by_timepoint(biomin_df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    Calculate mean expression per gene and timepoint.
    Returns long-format dataframe with gene expression by timepoint.
    """
    meta_cols = ['group_id', 'gene_id']
    sample_cols = get_sample_columns(biomin_df, meta_cols)
    
    # Reshape to long format
    expr_long = biomin_df.melt(
        id_vars=meta_cols,
        value_vars=sample_cols,
        var_name='sample_id',
        value_name='count'
    )
    
    # Add timepoint info
    expr_long['timepoint'] = expr_long['sample_id'].apply(extract_timepoint)
    expr_long['log_count'] = np.log2(expr_long['count'] + 1)
    
    # Calculate mean per gene and timepoint
    expr_by_tp = expr_long.groupby(['group_id', 'gene_id', 'timepoint']).agg({
        'count': 'mean',
        'log_count': ['mean', 'std']
    }).reset_index()
    
    expr_by_tp.columns = ['group_id', 'gene_id', 'timepoint', 'mean_count', 'mean_log', 'sd_log']
    expr_by_tp['species'] = species
    
    return expr_by_tp


def calculate_expression_fold_changes(expr_by_tp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fold changes between timepoint comparisons:
    T1→T2, T1→T3, T4→T2, T4→T3
    """
    # Pivot to wide format
    expr_wide = expr_by_tp.pivot_table(
        index=['group_id', 'gene_id', 'species'],
        columns='timepoint',
        values='mean_log'
    ).reset_index()
    
    # Calculate fold changes
    expr_wide['FC_T1_to_T2'] = expr_wide.get('TP2', np.nan) - expr_wide.get('TP1', np.nan)
    expr_wide['FC_T1_to_T3'] = expr_wide.get('TP3', np.nan) - expr_wide.get('TP1', np.nan)
    expr_wide['FC_T4_to_T2'] = expr_wide.get('TP2', np.nan) - expr_wide.get('TP4', np.nan)
    expr_wide['FC_T4_to_T3'] = expr_wide.get('TP3', np.nan) - expr_wide.get('TP4', np.nan)
    
    return expr_wide


def calculate_exon_variability_by_timepoint(
    exon_counts: pd.DataFrame,
    exon_summary: pd.DataFrame,
    biomin_groups: set,
    species: str
) -> pd.DataFrame:
    """
    Calculate exon-level expression variability per gene and timepoint.
    
    Uses group_id for matching to handle gene ID format differences.
    """
    # Get gene_id to group_id mapping from exon_summary
    gene_to_group = exon_summary[['gene_id', 'group_id']].drop_duplicates()
    gene_to_group = gene_to_group.dropna(subset=['group_id'])
    
    # Filter to biomineralization genes using group_id
    biomin_gene_ids = gene_to_group[gene_to_group['group_id'].isin(biomin_groups)]['gene_id'].tolist()
    
    if len(biomin_gene_ids) == 0:
        print(f"    WARNING: No matching genes found for {species}")
        return pd.DataFrame()
    
    print(f"    Found {len(biomin_gene_ids)} biomin genes via group_id mapping")
    
    # Filter exon counts to biomin genes
    meta_cols = ['gene_id', 'e_id', 'chr', 'strand', 'start', 'end']
    sample_cols = get_sample_columns(exon_counts, meta_cols)
    
    exon_biomin = exon_counts[exon_counts['gene_id'].isin(biomin_gene_ids)]
    
    if len(exon_biomin) == 0:
        print(f"    WARNING: No exon data found for biomin genes in {species}")
        return pd.DataFrame()
    
    print(f"    Processing {len(exon_biomin)} exons from {exon_biomin['gene_id'].nunique()} genes")
    
    # Reshape to long format
    exon_long = exon_biomin.melt(
        id_vars=meta_cols,
        value_vars=sample_cols,
        var_name='sample_id',
        value_name='count'
    )
    
    exon_long['timepoint'] = exon_long['sample_id'].apply(extract_timepoint)
    exon_long['log_count'] = np.log2(exon_long['count'] + 1)
    
    # Calculate mean exon expression per exon per timepoint
    exon_means = exon_long.groupby(['gene_id', 'timepoint', 'e_id'])['log_count'].mean().reset_index()
    
    # Calculate CV of exon expression within each gene for each timepoint
    exon_var = exon_means.groupby(['gene_id', 'timepoint']).agg({
        'e_id': 'count',
        'log_count': ['mean', 'std', 'max', 'min']
    }).reset_index()
    
    exon_var.columns = ['gene_id', 'timepoint', 'n_exons', 'mean_exon_expr', 'sd_exon_expr', 'max_exon', 'min_exon']
    
    # Calculate CV
    exon_var['cv_exon'] = exon_var.apply(
        lambda row: row['sd_exon_expr'] / row['mean_exon_expr'] 
        if row['mean_exon_expr'] > 0 else np.nan,
        axis=1
    )
    exon_var['exon_range'] = exon_var['max_exon'] - exon_var['min_exon']
    
    # Add group_id mapping
    exon_var = exon_var.merge(gene_to_group, on='gene_id', how='left')
    exon_var['species'] = species
    
    return exon_var


def calculate_exon_variability_ratios(exon_var: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the relative exon variability at T2/T3 compared to T1/T4.
    """
    # Pivot to wide format
    exon_var_wide = exon_var.pivot_table(
        index=['gene_id', 'group_id', 'species'],
        columns='timepoint',
        values='cv_exon'
    ).reset_index()
    
    # Calculate baseline and stress means
    exon_var_wide['mean_cv_baseline'] = (
        exon_var_wide.get('TP1', np.nan) + exon_var_wide.get('TP4', np.nan)
    ) / 2
    
    exon_var_wide['mean_cv_stress'] = (
        exon_var_wide.get('TP2', np.nan) + exon_var_wide.get('TP3', np.nan)
    ) / 2
    
    # Calculate ratios and differences
    exon_var_wide['cv_ratio'] = exon_var_wide['mean_cv_stress'] / exon_var_wide['mean_cv_baseline']
    exon_var_wide['cv_diff'] = exon_var_wide['mean_cv_stress'] - exon_var_wide['mean_cv_baseline']
    
    # Classify high variability genes
    exon_var_wide['high_var_T2T3'] = (
        (exon_var_wide['cv_ratio'] > CV_RATIO_THRESHOLD) | 
        (exon_var_wide['cv_diff'] > CV_DIFF_THRESHOLD)
    )
    
    return exon_var_wide


def get_gene_coordinates(exon_counts: pd.DataFrame, biomin_gene_ids: List[str]) -> pd.DataFrame:
    """Extract gene coordinates from exon data."""
    exon_filtered = exon_counts[exon_counts['gene_id'].isin(biomin_gene_ids)]
    
    gene_coords = exon_filtered.groupby(['gene_id', 'chr']).agg({
        'start': 'min',
        'end': 'max',
        'e_id': 'count'
    }).reset_index()
    
    gene_coords.columns = ['gene_id', 'chr', 'gene_start', 'gene_end', 'n_exons']
    
    return gene_coords


def map_cpgs_to_genes(
    cpg_df: pd.DataFrame,
    gene_coords: pd.DataFrame,
    gene_to_group: pd.DataFrame,
    buffer_bp: int = CPG_GENE_BUFFER_BP
) -> pd.DataFrame:
    """
    Map CpGs to gene regions with a buffer.
    Returns CpGs that fall within gene body + buffer region.
    """
    # Parse CpG locations
    cpg_locations = parse_cpg_ids(cpg_df['CpG'])
    cpg_df = cpg_df.copy()
    cpg_df['cpg_chr'] = cpg_locations['cpg_chr'].values
    cpg_df['cpg_pos'] = cpg_locations['cpg_pos'].values
    
    # Remove CpGs with invalid positions
    cpg_df = cpg_df.dropna(subset=['cpg_chr', 'cpg_pos'])
    
    # Merge CpGs with genes on chromosome
    cpg_gene = cpg_df.merge(gene_coords, left_on='cpg_chr', right_on='chr', how='inner')
    
    # Filter to CpGs within gene region + buffer
    cpg_gene = cpg_gene[
        (cpg_gene['cpg_pos'] >= (cpg_gene['gene_start'] - buffer_bp)) &
        (cpg_gene['cpg_pos'] <= (cpg_gene['gene_end'] + buffer_bp))
    ]
    
    # Classify position
    cpg_gene['position_type'] = cpg_gene.apply(
        lambda row: 'upstream' if row['cpg_pos'] < row['gene_start']
        else ('downstream' if row['cpg_pos'] > row['gene_end'] else 'gene_body'),
        axis=1
    )
    
    # Add group_id
    cpg_gene = cpg_gene.merge(gene_to_group, on='gene_id', how='left')
    
    return cpg_gene


def calculate_mcpg_variability_by_timepoint(
    cpg_gene: pd.DataFrame,
    species: str
) -> pd.DataFrame:
    """
    Calculate mCpG variability per gene and timepoint.
    """
    # Get sample columns
    all_cols = cpg_gene.columns.tolist()
    meta_cols = ['CpG', 'cpg_chr', 'cpg_pos', 'gene_id', 'chr', 'gene_start', 
                 'gene_end', 'n_exons', 'position_type', 'group_id']
    sample_cols = [c for c in all_cols if c not in meta_cols and re.match(r'^(ACR|POR|POC|TES)-.*TP\d+', c)]
    
    if len(sample_cols) == 0:
        print(f"    WARNING: No valid sample columns found for {species}")
        return pd.DataFrame()
    
    # Reshape to long format
    cpg_long = cpg_gene.melt(
        id_vars=['CpG', 'gene_id', 'group_id', 'position_type'],
        value_vars=sample_cols,
        var_name='sample_id',
        value_name='meth'
    )
    
    cpg_long['timepoint'] = cpg_long['sample_id'].apply(extract_timepoint)
    cpg_long = cpg_long.dropna(subset=['meth', 'timepoint'])
    
    # Calculate mean mCpG per CpG per timepoint
    cpg_means = cpg_long.groupby(['gene_id', 'group_id', 'timepoint', 'CpG'])['meth'].mean().reset_index()
    
    # Calculate variability within genes by timepoint
    gene_mcpg_var = cpg_means.groupby(['gene_id', 'group_id', 'timepoint']).agg({
        'CpG': 'count',
        'meth': ['mean', 'std', 'max', 'min']
    }).reset_index()
    
    gene_mcpg_var.columns = ['gene_id', 'group_id', 'timepoint', 'n_cpgs', 
                             'mean_gene_meth', 'sd_gene_meth', 'max_meth', 'min_meth']
    
    # Calculate CV
    gene_mcpg_var['cv_meth'] = gene_mcpg_var.apply(
        lambda row: row['sd_gene_meth'] / row['mean_gene_meth'] 
        if row['mean_gene_meth'] > 0 else np.nan,
        axis=1
    )
    gene_mcpg_var['meth_range'] = gene_mcpg_var['max_meth'] - gene_mcpg_var['min_meth']
    gene_mcpg_var['species'] = species
    
    return gene_mcpg_var


def calculate_mcpg_changes(mcpg_var: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mCpG changes between timepoints.
    """
    # Pivot to wide format
    mcpg_wide = mcpg_var.pivot_table(
        index=['gene_id', 'group_id', 'species'],
        columns='timepoint',
        values='mean_gene_meth'
    ).reset_index()
    
    # Calculate delta methylation
    mcpg_wide['delta_meth_T1_T2'] = mcpg_wide.get('TP2', np.nan) - mcpg_wide.get('TP1', np.nan)
    mcpg_wide['delta_meth_T1_T3'] = mcpg_wide.get('TP3', np.nan) - mcpg_wide.get('TP1', np.nan)
    mcpg_wide['delta_meth_T4_T2'] = mcpg_wide.get('TP2', np.nan) - mcpg_wide.get('TP4', np.nan)
    mcpg_wide['delta_meth_T4_T3'] = mcpg_wide.get('TP3', np.nan) - mcpg_wide.get('TP4', np.nan)
    
    # Absolute changes
    mcpg_wide['abs_delta_T2'] = np.abs(mcpg_wide['delta_meth_T1_T2'])
    mcpg_wide['abs_delta_T3'] = np.abs(mcpg_wide['delta_meth_T1_T3'])
    
    return mcpg_wide


def merge_exon_mcpg_data(exon_var_ratios: pd.DataFrame, mcpg_changes: pd.DataFrame) -> pd.DataFrame:
    """Merge exon variability with mCpG change data using group_id."""
    merged = exon_var_ratios.merge(
        mcpg_changes,
        on=['group_id', 'species'],
        how='inner',
        suffixes=('_exon', '_cpg')
    )
    return merged


def categorize_variability(df: pd.DataFrame, high_threshold: float = 1.5, low_threshold: float = 0.67) -> pd.DataFrame:
    """
    Categorize genes by exon variability using fixed thresholds.
    
    High Variability: CV ratio > high_threshold (stress > 1.5x baseline)
    Low Variability: CV ratio < low_threshold (stress < 0.67x baseline, i.e., decreased)
    Medium Variability: everything else
    
    This uses meaningful thresholds rather than quantiles to allow different
    species to have different distributions of variability categories.
    """
    df = df.copy()
    
    # Filter out NaN and infinite values for categorization
    valid_mask = df['cv_ratio'].notna() & np.isfinite(df['cv_ratio']) & (df['cv_ratio'] > 0)
    
    df['var_category'] = 'Medium Variability'  # Default
    df.loc[valid_mask & (df['cv_ratio'] > high_threshold), 'var_category'] = 'High Variability'
    df.loc[valid_mask & (df['cv_ratio'] < low_threshold), 'var_category'] = 'Low Variability'
    
    return df


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_expression_fold_changes(all_fc: pd.DataFrame, output_dir: Path) -> None:
    """Plot expression fold changes across timepoint comparisons."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    fc_cols = ['FC_T1_to_T2', 'FC_T1_to_T3', 'FC_T4_to_T2', 'FC_T4_to_T3']
    
    for ax, species in zip(axes, ['apul', 'peve', 'ptua']):
        species_data = all_fc[all_fc['species'] == species]
        
        if species_data.empty:
            continue
        
        # Reshape for plotting
        fc_long = species_data.melt(
            id_vars=['group_id', 'gene_id', 'species'],
            value_vars=fc_cols,
            var_name='comparison',
            value_name='log2FC'
        )
        fc_long['comparison'] = fc_long['comparison'].str.replace('FC_', '').str.replace('_to_', '→')
        
        # Boxplot
        fc_long.boxplot(column='log2FC', by='comparison', ax=ax)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-1, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(SPECIES_NAMES.get(species, species))
        ax.set_xlabel('Comparison')
        ax.set_ylabel('log2 Fold Change')
    
    plt.suptitle('Gene Expression Changes Across Timepoints', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'expression_fold_changes.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_exon_variability_by_timepoint(all_exon_var: pd.DataFrame, output_dir: Path) -> None:
    """Plot exon-level expression variability by timepoint."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, species in zip(axes, ['apul', 'peve', 'ptua']):
        species_data = all_exon_var[all_exon_var['species'] == species]
        
        if species_data.empty:
            continue
        
        # Filter extreme values for visualization
        species_data = species_data[species_data['cv_exon'] < species_data['cv_exon'].quantile(0.95)]
        
        # Boxplot
        species_data.boxplot(column='cv_exon', by='timepoint', ax=ax)
        ax.set_title(SPECIES_NAMES.get(species, species))
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('CV of Exon Expression')
    
    plt.suptitle('Exon-Level Expression Variability by Timepoint', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'exon_variability_by_timepoint.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_exon_var_vs_mcpg(merged_data: pd.DataFrame, output_dir: Path) -> None:
    """Plot scatter of exon variability vs mCpG change."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, species in enumerate(['apul', 'peve', 'ptua']):
        species_data = merged_data[merged_data['species'] == species]
        
        if species_data.empty:
            continue
        
        # Filter extreme values
        species_data = species_data[
            (species_data['cv_ratio'] < 5) & 
            (species_data['cv_ratio'] > 0)
        ]
        
        # T2 plot
        ax1 = axes[0, idx]
        ax1.scatter(species_data['cv_ratio'], species_data['abs_delta_T2'], 
                   alpha=0.5, c=SPECIES_COLORS.get(species, 'gray'), s=20)
        ax1.axvline(x=1, linestyle=':', color='gray', alpha=0.5)
        ax1.set_xlabel('CV Ratio (T2+T3 / T1+T4)')
        ax1.set_ylabel('|Δ mCpG| at T2')
        ax1.set_title(f'{SPECIES_NAMES.get(species, species)} - T2')
        
        # Add correlation
        valid = species_data[['cv_ratio', 'abs_delta_T2']].dropna()
        if len(valid) > 5:
            corr, p = stats.spearmanr(valid['cv_ratio'], valid['abs_delta_T2'])
            ax1.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p:.3e}', 
                    transform=ax1.transAxes, verticalalignment='top', fontsize=9)
        
        # T3 plot
        ax2 = axes[1, idx]
        ax2.scatter(species_data['cv_ratio'], species_data['abs_delta_T3'], 
                   alpha=0.5, c=SPECIES_COLORS.get(species, 'gray'), s=20)
        ax2.axvline(x=1, linestyle=':', color='gray', alpha=0.5)
        ax2.set_xlabel('CV Ratio (T2+T3 / T1+T4)')
        ax2.set_ylabel('|Δ mCpG| at T3')
        ax2.set_title(f'{SPECIES_NAMES.get(species, species)} - T3')
        
        # Add correlation
        valid = species_data[['cv_ratio', 'abs_delta_T3']].dropna()
        if len(valid) > 5:
            corr, p = stats.spearmanr(valid['cv_ratio'], valid['abs_delta_T3'])
            ax2.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p:.3e}', 
                    transform=ax2.transAxes, verticalalignment='top', fontsize=9)
    
    plt.suptitle('Exon Expression Variability vs mCpG Change', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'exon_var_vs_mcpg.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_mcpg_by_category(merged_data: pd.DataFrame, output_dir: Path) -> None:
    """Plot mCpG changes by variability category."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, species in enumerate(['apul', 'peve', 'ptua']):
        species_data = merged_data[
            (merged_data['species'] == species) & 
            (merged_data['var_category'].isin(['High Variability', 'Low Variability']))
        ]
        
        if species_data.empty:
            continue
        
        # T2 plot
        ax1 = axes[0, idx]
        species_data.boxplot(column='abs_delta_T2', by='var_category', ax=ax1)
        ax1.set_title(f'{SPECIES_NAMES.get(species, species)} - T2')
        ax1.set_xlabel('Variability Category')
        ax1.set_ylabel('|Δ mCpG|')
        
        # T3 plot
        ax2 = axes[1, idx]
        species_data.boxplot(column='abs_delta_T3', by='var_category', ax=ax2)
        ax2.set_title(f'{SPECIES_NAMES.get(species, species)} - T3')
        ax2.set_xlabel('Variability Category')
        ax2.set_ylabel('|Δ mCpG|')
    
    plt.suptitle('mCpG Changes: High vs Low Exon Variability Genes', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'mcpg_by_variability_category.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_species_comparison(merged_data: pd.DataFrame, output_dir: Path) -> None:
    """Create cross-species comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of CV ratios by species
    ax1 = axes[0]
    species_order = ['apul', 'peve', 'ptua']
    data_by_species = [merged_data[merged_data['species'] == sp]['cv_ratio'].dropna() 
                       for sp in species_order]
    bp = ax1.boxplot(data_by_species, labels=[SPECIES_NAMES.get(sp, sp) for sp in species_order],
                     patch_artist=True)
    for patch, sp in zip(bp['boxes'], species_order):
        patch.set_facecolor(SPECIES_COLORS.get(sp, 'gray'))
        patch.set_alpha(0.7)
    ax1.axhline(y=1, linestyle='--', color='gray', alpha=0.5)
    ax1.set_ylabel('CV Ratio (T2+T3 / T1+T4)')
    ax1.set_title('Exon Variability Ratio by Species')
    
    # Box plot of mCpG changes by species
    ax2 = axes[1]
    data_by_species_mcpg = [merged_data[merged_data['species'] == sp]['abs_delta_T2'].dropna() 
                            for sp in species_order]
    bp2 = ax2.boxplot(data_by_species_mcpg, labels=[SPECIES_NAMES.get(sp, sp) for sp in species_order],
                      patch_artist=True)
    for patch, sp in zip(bp2['boxes'], species_order):
        patch.set_facecolor(SPECIES_COLORS.get(sp, 'gray'))
        patch.set_alpha(0.7)
    ax2.set_ylabel('|Δ mCpG| at T2')
    ax2.set_title('mCpG Change Magnitude by Species')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_species_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def perform_correlation_analysis(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between exon variability and mCpG changes."""
    results = []
    
    for species in merged_data['species'].unique():
        species_data = merged_data[merged_data['species'] == species]
        
        # CV ratio vs abs delta T2
        valid_t2 = species_data[['cv_ratio', 'abs_delta_T2']].dropna()
        if len(valid_t2) > 5:
            corr_t2, p_t2 = stats.spearmanr(valid_t2['cv_ratio'], valid_t2['abs_delta_T2'])
        else:
            corr_t2, p_t2 = np.nan, np.nan
        
        # CV ratio vs abs delta T3
        valid_t3 = species_data[['cv_ratio', 'abs_delta_T3']].dropna()
        if len(valid_t3) > 5:
            corr_t3, p_t3 = stats.spearmanr(valid_t3['cv_ratio'], valid_t3['abs_delta_T3'])
        else:
            corr_t3, p_t3 = np.nan, np.nan
        
        results.append({
            'species': species,
            'species_name': SPECIES_NAMES.get(species, species),
            'n_genes': len(species_data),
            'cor_cv_ratio_delta_T2': corr_t2,
            'p_value_T2': p_t2,
            'cor_cv_ratio_delta_T3': corr_t3,
            'p_value_T3': p_t3,
        })
    
    return pd.DataFrame(results)


def perform_wilcoxon_tests(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Perform Wilcoxon tests comparing high vs low variability genes."""
    results = []
    
    for species in merged_data['species'].unique():
        species_data = merged_data[
            (merged_data['species'] == species) & 
            (merged_data['var_category'].isin(['High Variability', 'Low Variability']))
        ]
        
        high = species_data[species_data['var_category'] == 'High Variability']
        low = species_data[species_data['var_category'] == 'Low Variability']
        
        result = {
            'species': species,
            'species_name': SPECIES_NAMES.get(species, species),
            'n_high': len(high),
            'n_low': len(low),
        }
        
        # T2 test
        high_t2 = high['abs_delta_T2'].dropna()
        low_t2 = low['abs_delta_T2'].dropna()
        if len(high_t2) > 1 and len(low_t2) > 1:
            stat, p = stats.mannwhitneyu(high_t2, low_t2, alternative='two-sided')
            result['wilcox_stat_T2'] = stat
            result['wilcox_p_T2'] = p
            result['median_high_T2'] = high_t2.median()
            result['median_low_T2'] = low_t2.median()
        else:
            result['wilcox_stat_T2'] = np.nan
            result['wilcox_p_T2'] = np.nan
            result['median_high_T2'] = np.nan
            result['median_low_T2'] = np.nan
        
        # T3 test
        high_t3 = high['abs_delta_T3'].dropna()
        low_t3 = low['abs_delta_T3'].dropna()
        if len(high_t3) > 1 and len(low_t3) > 1:
            stat, p = stats.mannwhitneyu(high_t3, low_t3, alternative='two-sided')
            result['wilcox_stat_T3'] = stat
            result['wilcox_p_T3'] = p
            result['median_high_T3'] = high_t3.median()
            result['median_low_T3'] = low_t3.median()
        else:
            result['wilcox_stat_T3'] = np.nan
            result['wilcox_p_T3'] = np.nan
            result['median_high_T3'] = np.nan
            result['median_low_T3'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Parallel Processing Wrapper
# =============================================================================

def process_species_parallel(args: Tuple) -> Dict:
    """Process a single species - wrapper for parallel execution."""
    species = args[0]
    return process_species(species)


def process_species(species: str) -> Dict:
    """
    Process a single species: load data, calculate metrics, merge datasets.
    """
    print(f"\n{'='*60}")
    print(f"Processing {SPECIES_NAMES.get(species, species)}")
    print('='*60)
    
    results = {
        'species': species,
        'species_name': SPECIES_NAMES.get(species, species),
    }
    
    try:
        # Load data
        biomin_df = load_biomin_data(species)
        cpg_df = load_cpg_data(species)
        exon_counts, exon_summary = load_exon_data(species)
        
        # Get biomin group IDs
        biomin_groups = set(biomin_df['group_id'].dropna().unique())
        print(f"    Biomin orthogroups: {len(biomin_groups)}")
        
        # Calculate expression by timepoint
        print("    Calculating expression by timepoint...")
        expr_by_tp = calculate_expression_by_timepoint(biomin_df, species)
        results['expr_by_tp'] = expr_by_tp
        
        # Calculate expression fold changes
        print("    Calculating expression fold changes...")
        expr_fc = calculate_expression_fold_changes(expr_by_tp)
        results['expr_fc'] = expr_fc
        
        # Calculate exon variability by timepoint
        print("    Calculating exon variability...")
        exon_var = calculate_exon_variability_by_timepoint(exon_counts, exon_summary, biomin_groups, species)
        results['exon_var'] = exon_var
        
        if exon_var.empty:
            print(f"    WARNING: No exon variability data for {species}")
            return results
        
        # Calculate exon variability ratios
        print("    Calculating exon variability ratios...")
        exon_var_ratios = calculate_exon_variability_ratios(exon_var)
        results['exon_var_ratios'] = exon_var_ratios
        
        # Get gene_id to group_id mapping
        gene_to_group = exon_summary[['gene_id', 'group_id']].drop_duplicates().dropna(subset=['group_id'])
        biomin_gene_ids = gene_to_group[gene_to_group['group_id'].isin(biomin_groups)]['gene_id'].tolist()
        
        # Get gene coordinates
        print("    Extracting gene coordinates...")
        gene_coords = get_gene_coordinates(exon_counts, biomin_gene_ids)
        print(f"    Gene coordinates extracted: {len(gene_coords)} genes")
        
        # Map CpGs to genes
        print("    Mapping CpGs to genes...")
        cpg_gene = map_cpgs_to_genes(cpg_df, gene_coords, gene_to_group)
        print(f"    CpG-gene mappings: {len(cpg_gene)}")
        
        if cpg_gene.empty:
            print(f"    WARNING: No CpG-gene mappings for {species}")
            return results
        
        # Calculate mCpG variability by timepoint
        print("    Calculating mCpG variability...")
        mcpg_var = calculate_mcpg_variability_by_timepoint(cpg_gene, species)
        results['mcpg_var'] = mcpg_var
        
        if mcpg_var.empty:
            print(f"    WARNING: No mCpG variability data for {species}")
            return results
        
        # Calculate mCpG changes
        print("    Calculating mCpG changes...")
        mcpg_changes = calculate_mcpg_changes(mcpg_var)
        results['mcpg_changes'] = mcpg_changes
        
        # Merge exon and mCpG data
        print("    Merging exon and mCpG data...")
        merged = merge_exon_mcpg_data(exon_var_ratios, mcpg_changes)
        results['merged'] = merged
        print(f"    Merged data: {len(merged)} genes")
        
        # Categorize by variability
        if not merged.empty:
            merged = categorize_variability(merged)
            results['merged'] = merged
            
            # Debug: check species in merged data
            species_in_merged = merged['species'].unique() if 'species' in merged.columns else ['unknown']
            print(f"    Species in merged data: {species_in_merged}")
        
        # Summary
        print(f"\n    === {SPECIES_NAMES.get(species, species)} Summary ===")
        print(f"    Biomin genes: {len(biomin_df)}")
        print(f"    CpG sites: {len(cpg_df)}")
        print(f"    Exons: {len(exon_counts)}")
        print(f"    Genes with both exon and mCpG data: {len(merged)}")
        
        if not merged.empty:
            # Check CV ratio distribution
            valid_cv = merged['cv_ratio'].dropna()
            if len(valid_cv) > 0:
                print(f"    CV ratio range: {valid_cv.min():.3f} - {valid_cv.max():.3f}")
                print(f"    CV ratio median: {valid_cv.median():.3f}")
            
            n_high = (merged['var_category'] == 'High Variability').sum()
            n_med = (merged['var_category'] == 'Medium Variability').sum()
            n_low = (merged['var_category'] == 'Low Variability').sum()
            print(f"    High variability genes (CV ratio > 1.5): {n_high}")
            print(f"    Medium variability genes: {n_med}")
            print(f"    Low variability genes (CV ratio < 0.67): {n_low}")
        
    except Exception as e:
        print(f"    ERROR processing {species}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the analysis."""
    print("="*70)
    print("Hypothesis Test: Exon Expression Variability vs mCpG Differences")
    print("="*70)
    print(f"Using {get_n_cpus()} CPUs for parallel processing")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each species in parallel
    species_list = ['apul', 'peve', 'ptua']
    all_results = {}
    
    n_workers = min(get_n_cpus(), len(species_list))
    
    if n_workers > 1:
        print(f"\nProcessing {len(species_list)} species in parallel with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_species, sp): sp for sp in species_list}
            
            for future in as_completed(futures):
                sp = futures[future]
                try:
                    result = future.result()
                    all_results[sp] = result
                except Exception as e:
                    print(f"ERROR processing {sp}: {e}")
    else:
        # Sequential processing
        for sp in species_list:
            all_results[sp] = process_species(sp)
    
    # Combine results across species
    print("\n" + "="*60)
    print("Combining Results Across Species")
    print("="*60)
    
    # Combine dataframes
    all_expr_fc = pd.concat([r.get('expr_fc', pd.DataFrame()) for r in all_results.values() if r.get('expr_fc') is not None], ignore_index=True)
    all_exon_var = pd.concat([r.get('exon_var', pd.DataFrame()) for r in all_results.values() if r.get('exon_var') is not None], ignore_index=True)
    all_exon_var_ratios = pd.concat([r.get('exon_var_ratios', pd.DataFrame()) for r in all_results.values() if r.get('exon_var_ratios') is not None], ignore_index=True)
    all_mcpg_changes = pd.concat([r.get('mcpg_changes', pd.DataFrame()) for r in all_results.values() if r.get('mcpg_changes') is not None], ignore_index=True)
    all_merged = pd.concat([r.get('merged', pd.DataFrame()) for r in all_results.values() if r.get('merged') is not None], ignore_index=True)
    
    # Save combined data
    print("Saving combined results...")
    
    if not all_expr_fc.empty:
        all_expr_fc.to_csv(OUTPUT_DIR / 'expression_fold_changes.csv', index=False)
    
    if not all_exon_var.empty:
        all_exon_var.to_csv(OUTPUT_DIR / 'exon_variability_by_timepoint.csv', index=False)
    
    if not all_exon_var_ratios.empty:
        all_exon_var_ratios.to_csv(OUTPUT_DIR / 'exon_variability_ratios.csv', index=False)
    
    if not all_mcpg_changes.empty:
        all_mcpg_changes.to_csv(OUTPUT_DIR / 'mcpg_changes_by_gene.csv', index=False)
    
    if not all_merged.empty:
        all_merged.to_csv(OUTPUT_DIR / 'exon_mcpg_merged.csv', index=False)
    
    # Statistical analysis
    print("\n" + "="*60)
    print("Statistical Analysis")
    print("="*60)
    
    if not all_merged.empty:
        # Correlation analysis
        print("\nCorrelation Analysis:")
        correlations = perform_correlation_analysis(all_merged)
        print(correlations.to_string(index=False))
        correlations.to_csv(OUTPUT_DIR / 'correlation_results.csv', index=False)
        
        # Wilcoxon tests
        print("\nWilcoxon Tests (High vs Low Variability):")
        wilcox_results = perform_wilcoxon_tests(all_merged)
        print(wilcox_results.to_string(index=False))
        wilcox_results.to_csv(OUTPUT_DIR / 'wilcoxon_test_results.csv', index=False)
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    if not all_expr_fc.empty:
        print("  - Expression fold changes plot...")
        plot_expression_fold_changes(all_expr_fc, OUTPUT_DIR)
    
    if not all_exon_var.empty:
        print("  - Exon variability by timepoint plot...")
        plot_exon_variability_by_timepoint(all_exon_var, OUTPUT_DIR)
    
    if not all_merged.empty:
        print("  - Exon variability vs mCpG scatter plot...")
        plot_exon_var_vs_mcpg(all_merged, OUTPUT_DIR)
        
        print("  - mCpG by variability category plot...")
        plot_mcpg_by_category(all_merged, OUTPUT_DIR)
        
        print("  - Cross-species comparison plot...")
        plot_cross_species_comparison(all_merged, OUTPUT_DIR)
    
    # Hypothesis evaluation
    print("\n" + "="*60)
    print("HYPOTHESIS EVALUATION")
    print("="*60)
    print("\nHypothesis: Genes with high exon-level expression variability at T2/T3")
    print("will have corresponding distinct differences in mCpG within the gene.\n")
    
    if not all_merged.empty and not correlations.empty:
        for _, row in correlations.iterrows():
            print(f"\n{row['species_name']}:")
            print(f"  N genes with both data types: {row['n_genes']}")
            print(f"  Correlation (CV ratio vs |Δ mCpG| at T2): ρ = {row['cor_cv_ratio_delta_T2']:.3f}, p = {row['p_value_T2']:.3e}")
            print(f"  Correlation (CV ratio vs |Δ mCpG| at T3): ρ = {row['cor_cv_ratio_delta_T3']:.3f}, p = {row['p_value_T3']:.3e}")
            
            # Check significance
            sig_t2 = "SIGNIFICANT" if row['p_value_T2'] < 0.05 else "not significant"
            sig_t3 = "SIGNIFICANT" if row['p_value_T3'] < 0.05 else "not significant"
            print(f"  T2: {sig_t2}, T3: {sig_t3}")
        
        if not wilcox_results.empty:
            print("\n\nWilcoxon Test Results (High vs Low Variability Genes):")
            for _, row in wilcox_results.iterrows():
                print(f"\n{row['species_name']}:")
                if not np.isnan(row['wilcox_p_T2']):
                    sig = "***" if row['wilcox_p_T2'] < 0.001 else ("**" if row['wilcox_p_T2'] < 0.01 else ("*" if row['wilcox_p_T2'] < 0.05 else ""))
                    print(f"  T2: median(high)={row['median_high_T2']:.3f} vs median(low)={row['median_low_T2']:.3f}, p={row['wilcox_p_T2']:.3e} {sig}")
                if not np.isnan(row['wilcox_p_T3']):
                    sig = "***" if row['wilcox_p_T3'] < 0.001 else ("**" if row['wilcox_p_T3'] < 0.01 else ("*" if row['wilcox_p_T3'] < 0.05 else ""))
                    print(f"  T3: median(high)={row['median_high_T3']:.3f} vs median(low)={row['median_low_T3']:.3f}, p={row['wilcox_p_T3']:.3e} {sig}")
    
    # Summary table
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    summary_data = []
    for sp in species_list:
        r = all_results.get(sp, {})
        merged = r.get('merged', pd.DataFrame())
        
        if not merged.empty and 'var_category' in merged.columns:
            n_high = int((merged['var_category'] == 'High Variability').sum())
            n_med = int((merged['var_category'] == 'Medium Variability').sum())
            n_low = int((merged['var_category'] == 'Low Variability').sum())
            
            # Calculate CV ratio stats
            valid_cv = merged['cv_ratio'].dropna()
            cv_median = float(valid_cv.median()) if len(valid_cv) > 0 else np.nan
            cv_mean = float(valid_cv.mean()) if len(valid_cv) > 0 else np.nan
        else:
            n_high, n_med, n_low = 0, 0, 0
            cv_median, cv_mean = np.nan, np.nan
        
        summary_data.append({
            'Species': SPECIES_NAMES.get(sp, sp),
            'Biomin Genes': len(r.get('expr_fc', [])),
            'Genes with Exon+mCpG': len(merged),
            'High Var (>1.5)': n_high,
            'Medium Var': n_med,
            'Low Var (<0.67)': n_low,
            'CV Ratio Median': round(cv_median, 3) if not np.isnan(cv_median) else 'N/A',
            'CV Ratio Mean': round(cv_mean, 3) if not np.isnan(cv_mean) else 'N/A',
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / 'analysis_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()

