
# Alternative Splicing Analysis Instructions


# Species-specific RNA-seq alignment output directories
apul_rnaseq_dir <- here("D-Apul/output/02.20-D-Apul-RNAseq-alignment-HiSat2")
peve_rnaseq_dir <- here("E-Peve/output/02.20-E-Peve-RNAseq-alignment-HiSat2")
ptua_rnaseq_dir <- here("F-Ptua/output/02.20-F-Ptua-RNAseq-alignment-HiSat2")



⸻

✔︎ Background

Each sample’s directory contains:

e_data.ctab      # exon-level counts and FPKM
i_data.ctab      # intron/junction coverage
t_data.ctab      # transcript-level FPKM
g_data.ctab      # gene-level FPKM

Alternative splicing happens when:
	•	Transcripts of the same gene change in relative abundance across conditions, or
	•	Certain exons or splice junctions are condition-specific.

⸻

✅ Strategy 1 — Differential Transcript Usage (DTU)

Goal: Identify genes whose isoform proportions change across conditions.
	1.	Read ballgown object:

library(ballgown)
library(dplyr)

bg <- ballgown(dataDir = "ballgown/", samplePattern = "sample", meas = "all")
pData(bg) <- read.csv("sample_metadata.csv")

	2.	Perform differential expression on transcripts:

stat_results <- stattest(bg,
                         feature = "transcript",
                         meas = "FPKM",
                         covariate = "condition",
                         getFC = TRUE,
                         libadjust = TRUE)

	3.	Collapse transcript-level results to genes with ≥2 transcripts and discordant behavior:

library(tidyr)

alt_splice_candidates <- stat_results %>%
  separate(t_id, into = c("transcript", "gene"), sep = "\\.") %>% # adjust if needed
  group_by(gene) %>%
  summarize(
    n_trans = n(),
    n_sig = sum(qval < 0.05),
    any_spliced = n_sig > 1
  ) %>%
  filter(any_spliced == TRUE)

Interpretation
Genes where multiple transcripts are differentially expressed → candidate alternative splicing events.

⸻

🔁 Strategy 2 — Exon-Level Tests (Differential Exon Usage)

Use e_data.ctab to identify exons whose expression changes independent of whole-gene expression.

exon_results <- stattest(bg,
                         feature = "exon",
                         meas = "FPKM",
                         covariate = "condition",
                         libadjust = TRUE)

exon_AS <- exon_results %>% filter(qval < 0.05)

Exons significant here while the gene is not → exon skipping / alternative exons.

⸻

🔀 Strategy 3 — Junction-Level Tests

i_data.ctab includes splice junction expression.

junction_results <- stattest(bg,
                             feature = "intron",
                             meas = "cov",
                             covariate = "condition")

junction_AS <- junction_results %>% filter(qval < 0.05)

Significant intron usage differences → splice site switching.

⸻

🎉 Best Practice: Integrate Results

A robust AS gene list is:

AS_genes <- unique(c(alt_splice_candidates$gene,
                     exon_AS$gene_id,
                     junction_AS$gene_id))


⸻


Visualization 

Use IsoformSwitchAnalyzeR to visualize isoform switches and predict functional consequences.
