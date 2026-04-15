==========================================================================================
                               Akbari et al. (2026) Data Release
                                     Date: March 12, 2026
==========================================================================================


OVERVIEW
------------------------------------------------------------------------------------------

This release provides data generated and analyzed as part of Akbari et al. (2026). It
includes selection summary statistics and genome-wide genotype data for 15,836 ancient
individuals together with modern reference data from the 1000 Genomes Project (1KG).

Genome-wide imputed genotypes are provided in both BCF and PLINK2 formats. In addition,
non-imputed genotype data at the 1240k capture panel are provided in TGENO format used in
the Reich Lab pipeline.

The release also includes ancestry composition estimates based on previously published
qpAdm models, a frozen version of the PQLseqPy package used in the analyses, and an
embargoed version of Supplementary Tables 1-5.

Genome-wide genotype data include 52,382,873 variants. Of these, 9,739,624 variants passed
quality control and were used in Akbari et al. (2026). All variants are included in this
release.

The TGENO dataset contains 1,233,013 SNPs corresponding to the 1240k capture panel.


------------------------------------------------------------------------------------------

CONTENTS
------------------------------------------------------------------------------------------

  1. Selection summary statistics
  2. High-quality imputed variants
  3. 1000 Genomes Project reference panel (BCF)
  4. Ancient DNA genome-wide genotype data (imputed BCF)
  5. Genome-wide imputed genotypes for ancient and modern Europeans (PLINK2)
  6. 1240k capture panel genotypes for ancient and modern Europeans (non-imputed TGENO)
  7. Ancestry composition estimates (qpAdm models)
  8. Frozen version of the PQLseqPy package
  9. Embargoed version of Supplementary Tables 1-5


------------------------------------------------------------------------------------------

DATASETS
------------------------------------------------------------------------------------------


1. Selection summary statistics
------------------------------------------------------------------------------------------

Path:
  ./Selection_Summary_Statistics_01OCT2025.tsv.gz

Description:
  Selection summary statistics for 9,739,624 quality-controlled variants.

Format:
  Gzipped tab-delimited text file.

Notes:
  - Coordinates are in GRCh37
  - Variant IDs are defined as CHROM_POS_REF_ALT


2. High-quality imputed variants
------------------------------------------------------------------------------------------

Path:
  ./HighQualityVariants.tsv

Description:
  Tab-delimited list of 9,739,624 high-quality imputed variants (SNPs and indels) that
  passed quality control in Akbari et al. (2026).

Format:
  Tab-delimited text file.

Notes:
  - Coordinates are in GRCh37
  - Variant IDs are defined as CHROM_POS_REF_ALT


3. 1000 Genomes Project reference panel (BCF)
------------------------------------------------------------------------------------------

Path:
  ./1KG_RefPanel_bcf/
      1KG_30X_GRCh37_RefPanel_chr{CHROM}.{bcf,bcf.csi}

Description:
  Per-chromosome BCF files with CSI indices for the 1000 Genomes Project reference panel.

Files:
  44 files (22 chromosomes × BCF + index)

Sample composition:
  2,504 individuals

Variant composition:
  52,382,873 variants (SNPs and indels)

Notes:
  These data correspond to high-coverage (30X) whole-genome sequencing from the
  1000 Genomes Project (PMID: 36055201). The original data were generated on GRCh38
  and lifted over to GRCh37 using CrossMap.


4. Ancient DNA genome-wide genotype data (imputed BCF)
------------------------------------------------------------------------------------------

Path:
  ./AncientDNA_bcf/
      AncientDNA_AllVariants_NoQC_GRCh37_chr{CHROM}.{bcf,bcf.csi}

Description:
  Per-chromosome BCF files with CSI indices containing genotype likelihoods and imputed
  genotypes for ancient DNA samples.

Files:
  44 files (22 chromosomes × BCF + index)

Sample composition:
  15,836 ancient individuals

Variant composition:
  52,382,873 variants (SNPs and indels)

BCF fields:

  Imputation (GLIMPSE):
    GT   phased and imputed genotypes
    DS   genotype dosage
    GP   genotype posterior probabilities

  Genotype calling (mpileup):
    PL   phred-scaled genotype likelihoods
    AD   allelic depths (high-quality bases)

Notes:
  PL and AD values are present only when at least one read covers the variant. To
  minimize reference bias, only PL values for SNPs were used to build the imputation
  model for both SNPs and indels. PL and AD values for indels are reported when
  available.


5. Genome-wide imputed genotypes for ancient and modern Europeans (PLINK2)
------------------------------------------------------------------------------------------

Path:
  ./AncientDNA_plus_EUR1KG_plink2/
      AncientDNA_plus_EUR1KG_AllVariants_NoQC_GRCh37_autosomes.{pgen,psam,pvar}

Description:
  Autosomal genotype data in PLINK2 format used for genome-wide analyses in
  Akbari et al. (2026).

Sample composition:
  Ancient samples: 15,836 individuals (including related individuals)
  Modern samples :   503 European individuals from the 1000 Genomes Project

Variant composition:
  52,382,873 variants (SNPs and indels)

Notes:
  This directory contains the combined ancient and modern European dataset used
  in Akbari et al. (2026). UK Biobank data are not included, as redistribution is
  prohibited under their licensing terms.


6. 1240k capture panel genotypes for ancient and modern Europeans (non-imputed TGENO)
------------------------------------------------------------------------------------------

Path:
  ./AncientDNA_plus_EUR1KG_1240K_TGENO/
      AncientDNA_plus_EUR1KG_1240K_GRCh37_autosomes_plus_XY_TGENO.{geno,snp,ind}

Description:
  Genotype data in TGENO format derived from the Reich Lab internal 1240k dataset,
  containing ancient individuals together with modern European individuals from the
  1000 Genomes Project.

  These data were generated using a pipeline similar to that used for the Allen
  Ancient DNA Resource (AADR) described in Mallick et al. (2024) (PMID: 38341426).
  These data are distinct from the imputed genotype datasets provided in PLINK2
  and BCF format in this release.

  Due to the typically low coverage of ancient DNA data, genotypes for most
  ancient individuals are represented as pseudo-haploid calls. A single allele
  is recorded by randomly sampling one read when multiple reads are present.
  Some higher-coverage ancient samples may have diploid genotype calls. Modern
  1000 Genomes individuals are diploid.

Format:
  Packed binary genotype matrix with accompanying text files:

      .geno   genotype matrix (packed binary)
      .snp    variant information
      .ind    sample information

  TGENO is a packed binary genotype format used in the Reich Lab pipeline.
  Genotypes are stored using 2 bits per genotype in a transposed layout with
  one record per sample. The file contains a header with hash codes that
  verify that the corresponding .snp and .ind files match the genotype data.

  If SNP or sample identifiers are modified, the .geno file must be
  regenerated to maintain consistency with the hash stored in the file
  header.

  For details on this format see the convertf documentation in the
  AdmixTools repository:

      https://github.com/DReichLab/AdmixTools/blob/master/convertf/README

Sample composition:
  Ancient samples: 15,836 individuals
  Modern samples :   503 European individuals from the 1000 Genomes Project

Variant composition:
  1,233,013 SNPs from the 1240k capture panel


7. Ancestry composition estimates (qpAdm models)
------------------------------------------------------------------------------------------

Path:
  ./Ancestry_Composition_3way_qpAdm_Patterson2022.tsv
  ./Ancestry_Composition_4way_qpAdm_Fernandes2020.tsv

Description:
  Ancestry composition estimates computed using qpAdm under two previously
  published models.

  3-way model (Patterson et al. 2022, PMID: 34937049):
    - WHG (Western Hunter-Gatherer)
    - EEF (Early European Farmer)
    - STEPPE (Steppe pastoralists)

  4-way model (Fernandes et al. 2020, PMID: 32094539):
    - ANF (Anatolian Neolithic Farmer)
    - WHG (Western Hunter-Gatherer)
    - ICR (Iranian/Caucasian-related)
    - EHG (Eastern Hunter-Gatherer)

  Additional details are provided in Supplementary Information
  Section 7 of Akbari et al. (2026).

Format:
  Tab-delimited text files.


8. Frozen version of the PQLseqPy package
------------------------------------------------------------------------------------------

Path:
  ./PQLseqPy-v0.1.2.zip

Description:
  Archived version of the PQLseqPy Python package corresponding exactly
  to the version used in this study.


9. Embargoed version of Supplementary Tables 1-5
------------------------------------------------------------------------------------------

Path:
  ./selection_online_tables_1-5_full_archaeological_metadata_
    embargoed_until_2041.xlsx

Description:
  Complete archaeological metadata for all individuals.

  This file is identical to Supplementary Tables 1-5 except that
  Supplementary Table 1 contains additional columns (P-AB) with full
  archaeological metadata.

  These data will remain embargoed until January 1, 2041.


------------------------------------------------------------------------------------------

USAGE NOTES AND CAUTIONS
------------------------------------------------------------------------------------------

  - All data are aligned to GRCh37 (hg19) coordinates using the hs37d5
    reference sequence.

  - Raw sequencing data (BAM files) for the 10,016 newly generated
    ancient samples are available through the European Nucleotide
    Archive (ENA) under accession PRJEB106907.

  - Imputed genotypes are released as AllVariants_NoQC, meaning that no
    post-imputation or downstream quality control filtering has been
    applied.

  - Ancient DNA-specific artifacts may be present, and coverage varies
    substantially across samples.

  - Users should apply quality control procedures appropriate for
    their analyses.


------------------------------------------------------------------------------------------

CITATION
------------------------------------------------------------------------------------------

Akbari A. et al. (2026)
Ancient DNA reveals pervasive directional selection across West Eurasia.
Nature.


------------------------------------------------------------------------------------------

CONTACT
------------------------------------------------------------------------------------------

David Reich
reich@genetics.med.harvard.edu