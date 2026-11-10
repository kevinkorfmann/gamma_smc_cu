# Gamma-SMC CUDA

Estimate pairwise coalescence times from phased genomes on NVIDIA GPUs.

## Use the tool

[Install Gamma-SMC CUDA](install.md), then follow the
[VCF walkthrough](usage.md) to infer coalescence times from your own data.
The [Python API](python-api.md) also supports simulated tree sequences,
selected haplotype pairs, and gene or interval summaries.

## Explore the findings

[Browse the 143-locus findings list](findings.md),
[download the **143-locus CSV**](https://raw.githubusercontent.com/kevinkorfmann/gamma_smc_cu/findings-2026-09-27/findings_database/releases/2026-09-27/stage5_loci.csv),
or [download the full atlas and related gene lists](findings.md#download-gene-lists).
The atlas contains **19,119 gene rows across 26 populations** from the
high-coverage 1000 Genomes panel.

## Reproduce the analyses

The [reproduction guide](reproducing.md) describes the corrected workflow,
source versions, data downloads and validation requirements. It links the
separate historical recovery package and the later analysis/benchmark scripts.

## Latest update · 10 November 2026

- Analysis source now includes rank diagnostics, detection/localisation,
  selection/BGS follow-ups and exploratory profile matching.
- The reproduction guide replaces historical launch instructions with the
  corrected full-chromosome workflow and verified recovery entry points.
- The findings release remains **2026-09-27**; this update does not change its tables.

## Findings release · 27 September 2026

- Corrected gene atlas and candidate list, with versioned downloads.
- Full-context pair batching and direct export of dense coalescence estimates.
- Updated accuracy and runtime results: 15.91 seconds for the reported
  63,190-pair V100S benchmark.

[Source code](https://github.com/kevinkorfmann/gamma_smc_cu) ·
[Report an issue](https://github.com/kevinkorfmann/gamma_smc_cu/issues)

```{toctree}
:hidden:
:maxdepth: 1

install
usage
findings
python-api
methods
reproducing
releases
```
