
Develop a codebase to address the the following regarding calcification gene activity in corals:

For each species determine what calcification related genes change in activity (gene expression level AND gene expression variablilty based on exon level expression) from T1 to T2, T1 to T3, T4 to T2, and T4 to T3. 

For those genes that change in expression level and expression variability (based on exon level expression) at T2 adn T3 compared to other Time points determine how relates to this.


Hypotheses to test is that those genes that have high gene expression variability based on exon level expression at time point T2 and T3 will have corresponding distince differences in 
mCpG somewhere within the gene.

Data neccessary to test the hypotheses 

Expression matrices of calicification genes across timepoints and treatments for each species.

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv


Annotation file
M-multi-species/output/12-ortho-annot/ortholog_groups_annotated.csv


CpG methylation (mCpG)

Acropora pulchra
 https://gannet.fish.washington.edu/metacarcinus/E5/20250903_meth_Apul/merged-WGBS-CpG-counts_filtered_n20.csv 
 
Porites evermanni
 https://gannet.fish.washington.edu/metacarcinus/E5/Pevermanni/20250821_meth_Peve/merged-WGBS-CpG-counts_filtered_n20.csv 

Pocillopora tuahiniensis
 https://gannet.fish.washington.edu/metacarcinus/E5/Ptuahiniensis/20250821_meth_Ptua/merged-WGBS-CpG-counts_filtered_n20.csv 


Exon-Level Expression Data (for expression variability analysis)

Located at: M-multi-species/output/40-exon-count-matrix/

Raw exon count matrices:
- apul-exon_gene_count_matrix.csv
- peve-exon_gene_count_matrix.csv  
- ptua-exon_gene_count_matrix.csv

Exon summary files with ortholog group mappings:
- apul-exon_summary_by_ortholog.csv
- peve-exon_summary_by_ortholog.csv
- ptua-exon_summary_by_ortholog.csv


IMPORTANT: Gene ID Format Differences

The gene IDs in the exon count matrices have DIFFERENT formats than the biomin count matrices:

| Species | Exon Count Matrix gene_id | Biomin Counts gene_id |
|---------|---------------------------|----------------------|
| Apul | FUN_000001 | FUN_002435 | ✓ Same format - direct matching works |
| Peve | gene-Peve_00000001 | Peve_00000077 | Need to remove "gene-" prefix |
| Ptua | gene-Pocillopora_meandrina_HIv1___RNAseq.10273_t | Pocillopora_meandrina_HIv1___TS.g25680.t1b | Completely different gene annotation formats! |

SOLUTION: Use ortholog group IDs (group_id) for matching instead of gene_id

Both the biomin counts and the exon summary files contain a `group_id` column with ortholog group identifiers (e.g., OG_08948). These are consistent across datasets and should be used for joining exon variability data with expression and methylation data.

Steps for proper matching:
1. Load exon summary files (*-exon_summary_by_ortholog.csv) to get gene_id → group_id mapping
2. Add group_id to the raw exon count data using this mapping
3. Join exon variability results with biomin/methylation data using group_id (not gene_id)

