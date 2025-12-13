
Develop a codebase to address the the following regarding calcification gene activity in corals:

For each species determine what calcification related genes change in activity (gene expression level AND gene expression variablilty based on exon level expression) from T1 to T2, T1 to T3, T4 to T2, and T4 to T3. 

For those genes that change in expression level and expression variability based on exon level expression) determine how gene body methylation relates to this.

Hypotheses 1 to test is that those that change in activity in T2 adn T3 compared to T1 and T4 will have low gene body methylation. 

Hypotheses 2 to test is that those genes that are highly expressed in T2 and T3 that have high GBM overall - will have higher GBM at T2 and T3.

Hypotheses 3 to test is that those genes that have high gene expression variability based on exon level expressio at time point T2 and T3 will have differential GBM.

Data neccessary to test the hypotheses 

Expression matrices of calicification genes across timepoints and treatments for each species.

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/apul_biomin_counts.csv

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/peve_biomin_counts.csv

https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/M-multi-species/output/33-biomin-pathway-counts/ptua_biomin_counts.csv


Annotation file
M-multi-species/output/12-ortho-annot/ortholog_groups_annotated.csv



Gene Body Methylation 

Acropora pulchra
https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/D-Apul/output/40-Apul-Gene-Methylation/Apul-gene-methylation_75pct.tsv 

Porites evermanni
https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/E-Peve/output/15-Peve-Gene-Methylation/Peve-gene-methylation_75pct.tsv 

Pocillopora tuahiniensis
https://raw.githubusercontent.com/urol-e5/timeseries-molecular-calcification/refs/heads/main/F-Ptua/output/09-Ptua-Gene-Methylation/Ptua-gene-methylation_75pct.tsv


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

