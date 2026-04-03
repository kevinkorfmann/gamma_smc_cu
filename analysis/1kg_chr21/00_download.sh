#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DIR"
cd "$DIR"

BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage"

# Chr21 phased VCF
VCF="working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
if [ ! -f chr21.vcf.gz ]; then
    echo "Downloading chr21 VCF (~407 MB)..."
    curl -L -o chr21.vcf.gz "$BASE/$VCF"
    curl -L -o chr21.vcf.gz.tbi "$BASE/${VCF}.tbi"
else
    echo "chr21.vcf.gz already exists"
fi

# Sample metadata
if [ ! -f samples.txt ]; then
    echo "Downloading sample metadata..."
    curl -L -o samples.txt "$BASE/20130606_g1k_3202_samples_ped_population.txt"
else
    echo "samples.txt already exists"
fi

echo "Done. Files in $DIR:"
ls -lh "$DIR"
