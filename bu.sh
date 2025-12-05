#!/bin/bash

# rsync -avz -e ssh . \
# --exclude='*.sam' \
# --exclude='tmp*' \
# --exclude='*gz_C_to_T.fastq' \
# --exclude='*gz_G_to_A.fastq' \
# --exclude='Non_CpG_context*' \
# --exclude='.*' --exclude='*/.*' \
# sr320@gannet.fish.washington.edu:/volume1/v1_web/owlshell/bu-github/

rsync -av ../timeseries-molecular-calcification \
--exclude='*.sam' \
--exclude='tmp*' \
--exclude='*gz_C_to_T.fastq' \
--exclude='*gz_G_to_A.fastq' \
--exclude='Non_CpG_context*' \
--exclude='.*' --exclude='*/.*' \
sr320@gannet.fish.washington.edu:/volume1/v1_web/owlshell/bu-github/


rsync -av ../timeseries-molecular-calcification \
--exclude='*.sam' \
--exclude='tmp*' \
--exclude='*gz_C_to_T.fastq' \
--exclude='*gz_G_to_A.fastq' \
--exclude='Non_CpG_context*' \
--exclude='.*' --exclude='*/.*' \
sr320@gannet.fish.washington.edu:/volume2/web/gitrepos/urol-e5/

