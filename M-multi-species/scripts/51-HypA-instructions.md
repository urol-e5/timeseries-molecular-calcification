
Develop a codebase to test the following hypothesis regarding epigenetic regulation of calcification in corals:

“Plastic vs buffered calcification is encoded by epigenetic lability of ion-transport genes”
Species differ in how epigenetically labile key calcification genes are.
Acropora shows high epigenetic responsiveness (dynamic DNA methylation) at calcification genes, enabling rapid changes but causing instability.
Porites shows epigenetic buffering (stable methylation, reduced regulatory turnover), enabling sustained calcification under stress.
 "Frontloading" vs. "Reacting" Hypothesis
Ecological Basis: Porites evermanni is stress-tolerant and maintains homeostasis, while Acropora pulchra grows rapidly but crashes under stress. "Frontloading" refers to maintaining high constitutive levels of stress-response genes so the organism is always prepared.
Hypothesis: Porites evermanni exhibits constitutive high gene body methylation (GBM) in key calcification genes (e.g., Ca-ATPase, Carbonic Anhydrase), leading to stable, high-level transcription regardless of environmental fluctuations ("Frontloading"). In contrast, Acropora pulchra relies on promoter methylation changes to dynamically upregulate or downregulate these genes in response to immediate light and temperature conditions.
Expected Observation: Under stable conditions, Porites will show lower variance in methylation patterns and gene expression compared to Acropora. Under heat stress, Acropora will show massive epigenetic remodeling (DMRs) associated with a collapse in calcification, whereas Porites methylation profiles will remain rigid, supporting continued calcification
Target gene systems
Ca²⁺ transport: PMCAs, Ca²⁺ channels
H⁺ removal / pH regulation: V-type ATPases, SLC transporters
Carbon supply: carbonic anhydrases (CA)
Epigenetic mechanism
Environmentally induced shifts in gene-body methylation (GBM) and/or promoter accessibility regulate transcriptional responsiveness.
Acropora: low baseline GBM → high transcriptional variance
Porites: high baseline GBM → transcriptional canalization


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

